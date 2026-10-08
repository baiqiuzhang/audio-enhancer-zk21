"""引擎输出率与内容保真：**长度必须能自证采样率**，且内容不能被静默改写。

这套件是为一个实测缺陷建的（2026-10-01）：

- `core/engines.py::_pad_tail` 早先把"比输入长"的输出 `y[:target_len]` 截掉。
  48 kHz 原生模型 `dpdfnet2_48khz_hr` 喂 16 kHz 素材时，sherpa 内部会把输入升采样到
  48 kHz、返回**3 倍**样本数；被截成 1/3 之后样本数与输入一致，pipeline 的长度守卫
  （`align_to` / `_stack`）与 `_effective_rate` 就再也看不见它——实测后果是
  「时长看起来一致、内容只剩前 1/3，且被当成 16 kHz 播放（3× 变调）」：
  16 kHz 的 1 kHz 正弦 → 输出谱峰落到 **333 Hz**。
- 顺带覆盖：分段路径（`MAX_FRAMES_SINGLE_PASS` 之上）此前按"输入长度"写输出缓冲，
  升采样模型下第二段就会越界写。

判据（对每个引擎、每条素材）：

1. **自洽**：`len(y) ≈ len(x) * rate / in_rate`（容差 1%，与 `LENGTH_TOLERANCE` 同量级）。
   即"引擎自报的率必须能解释它返回的样本数"。这一条直接钉住上面那个缺陷。
2. **内容**：1 kHz 正弦进来，按自报率解释的谱峰仍在 1 kHz（±5%）。
   降噪模型会削弱它，但不会把它搬走——所以只看频率、不看电平。
3. **端到端**：`pipeline.run` 之后输出时长与输入一致、采样率等于目标率、
   谱峰仍是 1 kHz（这一层才是用户真正看到的东西）。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_engine_rate.py
    .venv\\Scripts\\python.exe tests\\check_engine_rate.py --engine dpdfnet2_48khz_hr
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import engines, media_io, modelstore, pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TONE_HZ = 1000.0
SECONDS = 3.0

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def section(t: str) -> None:
    print()
    print(t)
    print("=" * len(t))


def tone(freq: float, seconds: float, rate: int, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def peak_hz(y: np.ndarray, fs: int) -> float:
    """谱峰频率（Hz）。只看峰值位置，所以降噪削弱幅度不影响判据。"""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size < 1024 or not np.any(y):
        return float("nan")
    seg = y[: min(len(y), fs * 2)]                 # 取最多 2 s，够分辨 1 kHz
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return float(np.fft.rfftfreq(len(seg), 1.0 / fs)[int(np.argmax(spec))])


def tone_dominance(y: np.ndarray, fs: int, freq: float = TONE_HZ) -> float:
    """1 kHz 邻域能量占总能量的比例——判断"这个引擎有没有把纯音当噪声压掉"。

    为什么需要它：`deepfilternet3` 面对**纯 1 kHz 正弦**输出的是低频主导的残留
    （实测谱峰 26 Hz），也就是说纯音不是它能"保真通过"的信号。这时用"谱峰是否还在
    1 kHz"去判它，是在测一个它本来就不保证的性质。所以内容判据只在纯音确实被保留
    （邻域占比 ≥ 0.2）时才生效；**长度↔自报率自洽**那条判据不受影响，永远生效。
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    seg = y[: min(len(y), fs * 2)]
    if seg.size < 1024 or not np.any(seg):
        return 0.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / fs)
    sel = (freqs >= freq * 0.95) & (freqs <= freq * 1.05)
    total = float(spec.sum())
    return float(spec[sel].sum()) / total if total > 0 else 0.0


def check_engine_direct(cat, only: str | None) -> None:
    section("1. 引擎直调：自报采样率必须能解释返回的样本数，且不改写内容")
    header = f"{'引擎':<22}{'入率':>7}{'入帧':>9}{'出帧':>9}{'自报率':>8}{'长度比':>8}{'谱峰Hz':>9}  判定"
    print(header)
    print("-" * len(header))

    for spec in cat.engines:
        if only and spec.id != only:
            continue
        if spec.kind == "bwe":
            continue          # 带宽扩展是另一套契约（16k→48k 生成式），由 check_enhance 覆盖
        if not modelstore.engine_ready(spec, cat):
            print(f"{spec.name:<22}  [跳过] 模型未就绪")
            continue

        eng = engines.build_engine(spec, cat)
        x = tone(TONE_HZ, SECONDS, 16000)
        try:
            y, rate = eng.process(x, 16000, pipeline.EngineParams())
        except Exception as exc:                        # noqa: BLE001
            check(f"{spec.id} 可运行", False, f"{type(exc).__name__}: {exc}")
            continue

        y = np.asarray(y, dtype=np.float32).reshape(-1)
        expect = len(x) * float(rate) / 16000.0
        tol = max(engine_tolerance(rate), 0.01 * expect)
        hz = peak_hz(y, int(rate))
        dom = tone_dominance(y, int(rate))
        consistent = abs(len(y) - expect) <= tol
        # 内容判据只在"纯音确实被保留"时生效（见 tone_dominance 的说明）
        content_ok = dom < 0.2 or abs(hz - TONE_HZ) <= 0.05 * TONE_HZ
        print(f"{spec.name:<22}{16000:>7}{len(x):>9}{len(y):>9}{rate:>8}"
              f"{len(y) / len(x):>8.2f}{hz:>9.0f}  "
              f"{'OK' if consistent and content_ok else 'FAIL'}")
        check(f"{spec.id}：长度与自报率自洽", consistent,
              f"len(y)={len(y)} 期望 {expect:.0f}±{tol:.0f}（自报 {rate} Hz）")
        if dom < 0.2:
            print(f"  [跳过]  {spec.id}：纯音未被保留（1 kHz 邻域能量占比 {dom:.3f}），"
                  f"该引擎不保证纯音保真，内容判据不适用")
        else:
            check(f"{spec.id}：1 kHz 内容未被搬走", content_ok,
                  f"谱峰 {hz:.0f} Hz @ {rate} Hz（邻域占比 {dom:.2f}）")


def engine_tolerance(rate: int) -> int:
    """允许的帧数偏差：模型固有的一帧对齐截断（≤100 ms）与分段边界。"""
    return max(1, int(0.1 * rate))


def check_segmented(cat, only: str | None) -> None:
    section("2. 分段路径（强制每段 2 s）：升采样模型也不能错位/越界")
    old = engines.MAX_FRAMES_SINGLE_PASS
    engines.MAX_FRAMES_SINGLE_PASS = 16000 * 2
    try:
        for spec in cat.engines:
            if only and spec.id != only:
                continue
            if spec.kind != "sherpa" or not modelstore.engine_ready(spec, cat):
                continue
            eng = engines.build_engine(spec, cat)
            x = tone(TONE_HZ, 9.0, 16000)               # 9 s → 至少 5 段
            try:
                y, rate = eng.process(x, 16000, pipeline.EngineParams(segment_seconds=2.0))
            except Exception as exc:                    # noqa: BLE001
                check(f"{spec.id} 分段可运行", False, f"{type(exc).__name__}: {exc}")
                continue
            y = np.asarray(y, dtype=np.float32).reshape(-1)
            expect = len(x) * float(rate) / 16000.0
            hz = peak_hz(y, int(rate))
            print(f"  {spec.name:<22} 出 {len(y)} 帧 @ {rate} Hz（期望 {expect:.0f}）谱峰 {hz:.0f} Hz")
            check(f"{spec.id} 分段：长度自洽",
                  abs(len(y) - expect) <= max(engine_tolerance(rate), 0.03 * expect),
                  f"{len(y)} vs {expect:.0f}")
            check(f"{spec.id} 分段：内容未被搬走", abs(hz - TONE_HZ) <= 0.06 * TONE_HZ,
                  f"{hz:.0f} Hz")
    finally:
        engines.MAX_FRAMES_SINGLE_PASS = old


def check_pipeline_end_to_end(cat, tmp: str) -> None:
    section("3. 端到端：48 kHz 原生模型 + 16 kHz 素材（用户可见的那一层）")
    src = os.path.join(tmp, "tone16k.wav")
    x = tone(TONE_HZ, SECONDS, 16000)
    media_io.write_audio(src, x, 16000)

    for eid in ("dpdfnet2_48khz_hr", "dpdfnet2"):
        spec = cat.engine(eid)
        if spec is None or not modelstore.engine_ready(spec, cat):
            print(f"  [跳过] {eid} 未就绪")
            continue
        try:
            res = pipeline.run(src, cat, pipeline.PipelineOptions(
                engine_id=eid, write_output=False, reuse_existing=False,
                highpass_hz=0.0, keep_channels=True,
            ))
        except Exception as exc:                        # noqa: BLE001
            check(f"{eid} 端到端可运行", False, f"{type(exc).__name__}: {exc}")
            continue
        y = np.asarray(res.enhanced.samples, dtype=np.float32).reshape(-1)
        dur_in = len(x) / 16000.0
        dur_out = len(y) / res.enhanced.sample_rate
        hz = peak_hz(y, res.enhanced.sample_rate)
        print(f"  {eid:<20} 输出 {len(y)} 帧 @ {res.enhanced.sample_rate} Hz"
              f"  时长 {dur_out:.2f}s（输入 {dur_in:.2f}s）谱峰 {hz:.0f} Hz")
        check(f"{eid}：输出采样率 = 目标率（源率）", res.enhanced.sample_rate == 16000,
              str(res.enhanced.sample_rate))
        check(f"{eid}：输出时长与输入一致（±2%）",
              abs(dur_out - dur_in) <= 0.02 * dur_in, f"{dur_out:.3f}s vs {dur_in:.3f}s")
        check(f"{eid}：1 kHz 内容未被搬走（±5%）",
              abs(hz - TONE_HZ) <= 0.05 * TONE_HZ, f"{hz:.0f} Hz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None, help="只测某个引擎 id")
    args = ap.parse_args()

    print(f"项目根：{ROOT}")
    cat = modelstore.load_catalog(ROOT)
    tmp = tempfile.mkdtemp(prefix="ae_engrate_")
    try:
        check_engine_direct(cat, args.engine)
        check_segmented(cat, args.engine)
        check_pipeline_end_to_end(cat, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"引擎输出率与内容保真验证结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("引擎输出率与内容保真验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
