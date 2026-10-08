"""加速改动的**等价性与闸门**验证（不追求跑得快，追求"改完还一样"）。

本套件回答三个问题：

1. `core/dsp.py` 的分块帧电平（`frame_rms_streaming`、`detect_active_span_and_gaps`）
   与旧的一次性矩阵实现**逐位一致**吗？（旧实现以参照实现的形式写在本文件里）
   顺带量出两者在长素材上的**峰值分配**差——那才是这次改动的真正目的：
   整链峰值驻留实测 300 s→1249 MB / 1200 s→4873 MB（线性外推到 4139 s 约 16.7 GB，
   超过本机 15.6 GB 物理内存，也就是在换页盘上跑）。
2. 声道的并行/串行**结果逐位相同**吗？内存闸门在内存不足时会不会正确回落到串行？
3. `dsp.spectral_process` 里向量化后的 3 帧平滑与 `np.convolve(mode="same")` 一致吗？

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_speed.py
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dsp, media_io, modelstore, pipeline, sysinfo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def section(t: str) -> None:
    print()
    print(t)
    print("=" * len(t))


# ---------------------------------------------------------------------------
# 参照实现（= 改动前的写法，只用于对照）
# ---------------------------------------------------------------------------

def _ref_frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        return x.reshape(1, -1)
    n_frames = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def _ref_frame_rms(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    frames = _ref_frames(np.asarray(x), win, hop)
    return np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))


def _ref_smooth3(gain: np.ndarray) -> np.ndarray:
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, gain)


# ---------------------------------------------------------------------------
# 1. 分块帧电平
# ---------------------------------------------------------------------------

def check_frame_rms() -> None:
    section("1. 分块帧电平：与旧实现逐位一致 + 峰值分配对比")
    rng = np.random.default_rng(2026)
    for seconds, rate, win_ms in ((3.0, 16000, 20.0), (20.0, 48000, 20.0), (5.0, 48000, 16.0)):
        x = rng.standard_normal(int(rate * seconds)).astype(np.float32)
        win = int(rate * win_ms / 1000.0)
        hop = win // 2
        ref = _ref_frame_rms(x, win, hop)
        new = dsp.frame_rms_streaming(x, win, hop)
        check(f"{rate} Hz / {seconds:g}s / {win_ms:g}ms：逐位相同",
              ref.shape == new.shape and bool(np.array_equal(ref, new)),
              f"{ref.shape} vs {new.shape}")

    # 短于 1 帧的边界
    tiny = np.arange(100, dtype=np.float32)
    check("长度不足一帧时行为一致",
          bool(np.array_equal(_ref_frame_rms(tiny, 960, 480),
                              dsp.frame_rms_streaming(tiny, 960, 480))))

    # 峰值分配：1200 s 单声道 48 kHz（模拟长素材）
    x = rng.standard_normal(48000 * 1200).astype(np.float32)

    def peak_mb(fn) -> tuple[float, float]:
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        secs = time.perf_counter() - t0
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / 1024**2, secs

    p_old, t_old = peak_mb(lambda: _ref_frame_rms(x, 960, 480))
    p_new, t_new = peak_mb(lambda: dsp.frame_rms_streaming(x, 960, 480))
    print(f"  1200 s 单声道一次遍历：旧 峰值 {p_old:8.1f} MB / {t_old:.2f}s"
          f"　→　新 峰值 {p_new:8.1f} MB / {t_new:.2f}s")
    check("分块后峰值分配降一个数量级以上", p_new * 10 < p_old,
          f"{p_old:.0f} MB → {p_new:.0f} MB")
    check("耗时没有明显变差（<2×）", t_new < t_old * 2.0, f"{t_old:.2f}s → {t_new:.2f}s")


def check_trim_equivalence() -> None:
    section("2. 静音裁剪合并成一趟：区间与静音段与旧实现完全相同")
    rng = np.random.default_rng(7)
    sr = 16000
    # 构造：1 s 数字静音 + 3 s 噪声 + 1.5 s 停顿 + 2 s 噪声 + 1 s 数字静音
    sig = np.concatenate([
        np.zeros(sr, np.float32),
        (rng.standard_normal(sr * 3) * 0.05).astype(np.float32),
        np.zeros(int(sr * 1.5), np.float32),
        (rng.standard_normal(sr * 2) * 0.05).astype(np.float32),
        np.zeros(sr, np.float32),
    ])
    span_new, gaps_new = dsp.detect_active_span_and_gaps(sig, sr)

    # 旧实现：两趟各算一次（用参照帧电平复刻旧逻辑）
    def old_span(samples, rate):
        win = max(1, int(rate * 20.0 / 1000.0))
        hop = max(1, win // 2)
        frame_db = 20.0 * np.log10(np.maximum(_ref_frame_rms(samples, win, hop), 1e-12))
        threshold = max(float(np.percentile(frame_db, 95)) - 40.0, -75.0)
        active = np.nonzero(frame_db > threshold)[0]
        if active.size == 0:
            return 0, len(samples), threshold
        pad = int(rate * 150.0 / 1000.0)
        return (max(0, int(active[0]) * hop - pad),
                min(len(samples), int(active[-1]) * hop + win + pad), threshold)

    def old_gaps(samples, rate):
        win = max(1, int(rate * 20.0 / 1000.0))
        hop = max(1, win // 2)
        frame_db = 20.0 * np.log10(np.maximum(_ref_frame_rms(samples, win, hop), 1e-12))
        threshold = max(float(np.percentile(frame_db, 95)) - 40.0, -75.0)
        quiet = frame_db <= threshold
        min_gap_frames = max(1, int(1000.0 / (20.0 / 2.0)))
        out, run_start = [], None
        for i, is_quiet in enumerate(quiet):
            if is_quiet and run_start is None:
                run_start = i
            elif not is_quiet and run_start is not None:
                if i - run_start >= min_gap_frames:
                    out.append((run_start * hop / rate, i * hop / rate))
                run_start = None
        if run_start is not None and len(quiet) - run_start >= min_gap_frames:
            out.append((run_start * hop / rate, len(quiet) * hop / rate))
        return out

    o_start, o_end, o_thr = old_span(sig, sr)
    check("裁剪区间逐位相同",
          (span_new.start_frame, span_new.end_frame) == (o_start, o_end),
          f"{(span_new.start_frame, span_new.end_frame)} vs {(o_start, o_end)}")
    check("阈值相同", abs(span_new.threshold_dbfs - o_thr) < 1e-9,
          f"{span_new.threshold_dbfs} vs {o_thr}")
    check("中间静音段列表相同", gaps_new == old_gaps(sig, sr),
          f"{gaps_new} vs {old_gaps(sig, sr)}")
    check("确实找到了首尾裁剪与中间停顿",
          span_new.start_frame > 0 and span_new.end_frame < len(sig) and len(gaps_new) >= 1,
          f"span=({span_new.start_frame},{span_new.end_frame}) gaps={len(gaps_new)}")

    # 薄封装仍然可用（老调用点不应被破坏）
    check("detect_active_span 薄封装与合并版一致",
          dsp.detect_active_span(sig, sr).start_frame == span_new.start_frame)
    check("count_internal_gaps 薄封装与合并版一致",
          dsp.count_internal_gaps(sig, sr) == gaps_new)


def check_smooth3() -> None:
    section("3. 3 帧平滑向量化：与 np.convolve(mode='same') 等价")
    rng = np.random.default_rng(11)
    for shape in ((3, 5), (17, 513), (5630, 129)):
        g = rng.random(shape).astype(np.float32)
        ref = _ref_smooth3(g)
        sm = g * 0.5
        sm[:-1] += 0.25 * g[1:]
        sm[1:] += 0.25 * g[:-1]
        d = float(np.max(np.abs(ref - sm)))
        check(f"形状 {shape}：最大偏差 ≤1e-6", d <= 1e-6, f"max|Δ|={d:.2e}")


# ---------------------------------------------------------------------------
# 4. 声道并行：结果逐位一致 + 内存闸门
# ---------------------------------------------------------------------------

def check_parallel_equivalence(cat) -> None:
    section("4. 声道并行：与串行结果逐位相同")
    import tempfile

    sr = 48000
    t = np.arange(sr * 2) / sr
    rng = np.random.default_rng(3)
    left = (0.08 * np.sin(2 * np.pi * 220 * t) + rng.standard_normal(len(t)) * 0.01).astype(np.float32)
    right = (0.05 * np.sin(2 * np.pi * 330 * t) + rng.standard_normal(len(t)) * 0.01).astype(np.float32)
    stereo = np.stack([left, right], axis=1)

    tmp = tempfile.mkdtemp(prefix="ae_speed_")
    src = os.path.join(tmp, "stereo48k.wav")
    media_io.write_audio(src, stereo, sr)

    def run(parallel: str):
        return pipeline.run(src, cat, pipeline.PipelineOptions(
            engine_id="dsp_baseline",
            highpass_hz=0.0,
            keep_channels=True,
            write_output=False,
            reuse_existing=False,
            parallel_channels=parallel,
        ))

    a = run("off")
    b = run("on")
    check("输出长度一致", len(a.enhanced.samples) == len(b.enhanced.samples))
    check("输出逐位相同（并行只是调度，不改数值）",
          np.array_equal(np.asarray(a.enhanced.samples), np.asarray(b.enhanced.samples)))
    check("统计量一致", a.enhanced_stats.rms_dbfs == b.enhanced_stats.rms_dbfs)


def check_gate() -> None:
    section("5. 内存闸门：内存不足时必须回落到串行并说明原因")
    logs: list[str] = []
    real = sysinfo.available_memory_mb

    # 充足 → 并行
    sysinfo.available_memory_mb = lambda: 99_000.0
    try:
        check("内存充足 → 2 个声道并行",
              pipeline.channel_workers(2, "auto", "dfn_exe", logs.append,
                                       samples_per_channel=48000 * 1200) == 2)
    finally:
        sysinfo.available_memory_mb = real

    # 不足 → 串行 + 说明
    logs.clear()
    sysinfo.available_memory_mb = lambda: 500.0
    try:
        n = pipeline.channel_workers(2, "auto", "dfn_exe", logs.append,
                                     samples_per_channel=48000 * 4139)
    finally:
        sysinfo.available_memory_mb = real
    check("内存不足 → 回落串行", n == 1, str(n))
    check("回落时写明了原因", any("串行" in m and "可用内存" in m for m in logs), str(logs[:1]))

    # 单声道 / off / on
    logs.clear()
    check("单声道永远串行",
          pipeline.channel_workers(1, "auto", "dfn_exe", logs.append, 48000) == 1)
    check("off 永远串行",
          pipeline.channel_workers(2, "off", "dfn_exe", logs.append, 48000) == 1)
    sysinfo.available_memory_mb = lambda: 100.0
    try:
        check("on 显式并行时跳过闸门",
              pipeline.channel_workers(2, "on", "dfn_exe", logs.append, 48000) == 2)
    finally:
        sysinfo.available_memory_mb = real
    check("跳过闸门也有明确提示", any("显式开启" in m for m in logs))

    # 长度越长要求越多：增量应接近「每采样 12 B」（69 min 比 1 min 多约 2.2 GB）
    short = pipeline.parallel_channel_need_mb("dfn_exe", 48000 * 60)
    long_ = pipeline.parallel_channel_need_mb("dfn_exe", 48000 * 4139)
    delta = long_ - short
    check("所需内存随素材长度增长（增量≈2.2 GB）", 1800 < delta < 2600,
          f"{short:.0f} MB → {long_:.0f} MB（+{delta:.0f} MB）")
    check("69 分钟立体声的闸门值在合理区间（1–6 GB）",
          1000 < long_ < 6000, f"{long_:.0f} MB")


def check_manifest_stage_times() -> None:
    section("6. manifest 必须带分阶段耗时（否则历史运行无法事后诊断）")
    import json
    import tempfile

    from core import outputs, modelstore

    cat = modelstore.load_catalog(ROOT)
    tmp = tempfile.mkdtemp(prefix="ae_speed_manifest_")
    sr = 16000
    sig = (0.05 * np.sin(2 * np.pi * 200 * np.arange(sr * 2) / sr)).astype(np.float32)
    src = os.path.join(tmp, "tone.wav")
    media_io.write_audio(src, sig, sr)
    out_dir = os.path.join(tmp, "out")
    res = pipeline.run(src, cat, pipeline.PipelineOptions(
        engine_id="dsp_baseline", output_dir=out_dir, highpass_hz=0.0,
        reuse_existing=False,
    ))
    check("写出了结果", bool(res.output_path) and os.path.isfile(res.output_path))
    manifest = outputs.load_manifest(out_dir)
    entry = manifest["entries"].get(res.fingerprint)
    check("manifest 里有这条登记", entry is not None)
    if entry:
        st = entry.get("stats", {})
        check("stats.stage_times 存在且含 decode", "decode" in (st.get("stage_times") or {}),
              str(list((st.get("stage_times") or {}).keys())))
        check("stage_times 含 write（写出耗时不再丢失）", "write" in (st.get("stage_times") or {}),
              str(list((st.get("stage_times") or {}).keys())))
        check("stats.wall_s ≥ elapsed_s",
              float(st.get("wall_s") or 0) >= float(st.get("elapsed_s") or 0),
              f"{st.get('wall_s')} vs {st.get('elapsed_s')}")
        check("stats.channels 已记录", int(st.get("channels") or 0) == 1, str(st.get("channels")))
        with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as fh:
            json.load(fh)          # 必须是合法 JSON


def check_inference_slots() -> None:
    section("7. 进程级推理槽位：总并发被钉住，且等待时可取消")
    # 为什么需要：批处理（2 worker）与声道并行（2）是两个**独立**闸门，各自只读空闲内存、
    # 互不预留 → 最多 4 个 deep-filter.exe 同时跑（每个 69 分钟声道按实测 2–3 GB）。
    import threading  # noqa: PLC0415

    from core import engines  # noqa: PLC0415
    from core.errors import Cancelled  # noqa: PLC0415

    real = sysinfo.available_memory_mb
    sysinfo.available_memory_mb = lambda: 6000.0
    try:
        engines.INFERENCE_SLOTS.configure(2)
        check("许可数可显式重设（测试用）", engines.INFERENCE_SLOTS.capacity == 2,
              str(engines.INFERENCE_SLOTS.capacity))

        running, peak = [0], [0]
        lock = threading.Lock()

        def task() -> None:
            engines.INFERENCE_SLOTS.acquire()
            try:
                with lock:
                    running[0] += 1
                    peak[0] = max(peak[0], running[0])
                time.sleep(0.15)
            finally:
                with lock:
                    running[0] -= 1
                engines.INFERENCE_SLOTS.release()

        threads = [threading.Thread(target=task) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        check("6 个任务争 2 个槽位时并发峰值 ≤2", peak[0] <= 2, f"峰值 {peak[0]}")

        engines.INFERENCE_SLOTS.configure(1)
        engines.INFERENCE_SLOTS.acquire()          # 占满
        raised = None
        try:
            engines.INFERENCE_SLOTS.acquire(cancel=lambda: True)
        except Cancelled as exc:
            raised = exc
        finally:
            engines.INFERENCE_SLOTS.release()
        check("等待槽位期间被取消会抛 Cancelled", raised is not None, str(raised))
    finally:
        sysinfo.available_memory_mb = real
        engines.INFERENCE_SLOTS.configure(2)       # 收尾：留一个宽裕的默认值


def check_predecoded_and_decode_peak() -> None:
    section("8. 预解码复用 + PyAV 解码不再双份缓冲")
    import tempfile  # noqa: PLC0415
    import tracemalloc  # noqa: PLC0415

    from core import media_io  # noqa: PLC0415

    cat = modelstore.load_catalog(ROOT)
    tmp = tempfile.mkdtemp(prefix="ae_speed_pre_")
    sr = 16000
    src = os.path.join(tmp, "t.wav")
    media_io.write_audio(src, (0.05 * np.sin(2 * np.pi * 300 * np.arange(sr * 3) / sr))
                         .astype(np.float32), sr)

    real_decode = media_io.decode
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return real_decode(*a, **k)

    media_io.decode = counting
    try:
        pre = real_decode(src)
        calls["n"] = 0
        a = pipeline.run(src, cat, pipeline.PipelineOptions(
            engine_id="dsp_baseline", write_output=False, highpass_hz=0.0, predecoded=pre))
        check("给了 predecoded 就不再解码第二次", calls["n"] == 0, f"调用 {calls['n']} 次")

        calls["n"] = 0
        b = pipeline.run(src, cat, pipeline.PipelineOptions(
            engine_id="dsp_baseline", write_output=False, highpass_hz=0.0))
        check("没给 predecoded 时照常解码一次", calls["n"] == 1, f"调用 {calls['n']} 次")
        check("两条路径结果逐位相同",
              np.array_equal(np.asarray(a.enhanced.samples), np.asarray(b.enhanced.samples)))

        calls["n"] = 0
        half = media_io.AudioData(samples=pre.samples[: sr], sample_rate=sr)   # 时长只有 1/3
        pipeline.run(src, cat, pipeline.PipelineOptions(
            engine_id="dsp_baseline", write_output=False, highpass_hz=0.0, predecoded=half))
        check("时长不吻合的 predecoded 被拒绝并回退解码", calls["n"] == 1,
              f"调用 {calls['n']} 次")
    finally:
        media_io.decode = real_decode

    # PyAV 解码峰值：预分配后应接近 1×，而不是旧的 2×
    import soundfile as sf  # noqa: PLC0415

    big = os.path.join(tmp, "big.wav")
    n = sr * 120
    sf.write(big, np.random.default_rng(1).standard_normal((n, 2)).astype(np.float32) * 0.05,
             sr, subtype="FLOAT")
    tracemalloc.start()
    data = media_io._decode_pyav(big, None, False)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ratio = peak / max(1, data.samples.nbytes)
    check("PyAV 解码峰值 ≤1.3× 结果大小（旧实现 2.01×）", ratio <= 1.3, f"{ratio:.2f}×")


def main() -> int:
    print(f"项目根：{ROOT}")
    cat = modelstore.load_catalog(ROOT)
    check_frame_rms()
    check_trim_equivalence()
    check_smooth3()
    check_parallel_equivalence(cat)
    check_gate()
    check_manifest_stage_times()
    check_inference_slots()
    check_predecoded_and_decode_peak()
    print()
    if failures:
        print(f"加速改动验证结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("加速改动验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
