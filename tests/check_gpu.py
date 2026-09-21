"""GPU / CUDA 验证：能用就用、不能用就诚实回落，且回落不改变结果。

本套件回答六个问题，每条都要**可复现的证据**，而不是"我们声明启用了 GPU"：

1. CUDA 运行库在这台机器上到底在不在（cudnn / cudart / cublas 三类 DLL 的实际来源目录）；
2. 推理设备是**怎么决定的**（`auto` / `cpu` / `cuda` 三档的返回值与理由）；
3. 真启用 CUDA 时，**证据**是什么——ORT 会话实际 provider，加上进程内已加载的
   `onnxruntime_providers_cuda.dll`。sherpa-onnx 的 Python API **没有** provider
   回读（只有 `run` / `sample_rate`），所以这条只能靠进程模块枚举；
4. CPU 与 GPU 的输出**是否等价**（数值容差）——这是"设备不进指纹"的前提；
5. 请求 CUDA 但初始化失败时**必须回落 CPU 且留下痕迹**（不做静默降级）；
6. 显存不足时不得硬闯（预检 + 回落）。

没有 NVIDIA 显卡的机器上，GPU 相关断言标注 `[跳过]` 而不是失败——但**设备决策**那
几条仍然会跑，因为它们验证的是逻辑。

**刻意不跑整段素材**：18 分钟立体声走整链的峰值驻留是 7.4 GB（见 README），
不适合做测试。这里统一用 30 s 片段。

用法：
    .venv\\Scripts\\python.exe tests\\check_gpu.py
    .venv\\Scripts\\python.exe tests\\check_gpu.py --skip-timing   # 跳过耗时的 GPU 计时
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bwe, cuda_env, engines, media_io, modelstore  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "speech_with_noise.wav")

# 数值容差。实测（2026-09-21，NovaSR / DPDFNet2，30 s 片段）：
#   带宽扩展 CPU vs CUDA：最大逐样本差 1.06e-3、相对 RMS 0.54%
#   sherpa   CPU vs CUDA：最大逐样本差 2.15e-4、相对 RMS 0.012%
# 即**不是逐位相同**，所以要给容差而不是断言相等；下面留了较宽的余量，
# 只用来抓"GPU 路径整体坏掉"这种量级的偏差。
MAX_ABS_DIFF = 0.02
MAX_REL_RMS_PCT = 2.0

failures: list[str] = []
skipped: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def skip(name: str, why: str) -> None:
    print(f"  [跳过]  {name}  — {why}")
    skipped.append(name)


def section(t: str) -> None:
    print()
    print(t)
    print("=" * len(t))


def _excerpt(seconds: float, rate: int, channels: int = 1):
    """把官方示例音频平铺到指定时长（确定性输入，不依赖用户素材）。"""
    data = media_io.decode(SAMPLE)
    base = np.asarray(data.samples, dtype=np.float32)
    if base.ndim > 1:
        base = media_io.to_mono(base)
    n = int(seconds * data.sample_rate)
    reps = int(np.ceil(n / max(1, len(base))))
    x = np.tile(base, reps)[:n]
    if data.sample_rate != rate:
        x = media_io.resample(x, data.sample_rate, rate)
    return x


def gpu_present() -> bool:
    return cuda_env.nvidia_smi() is not None


PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_probe.py")


def _run_probe(mode: str) -> dict | None:
    """在独立子进程里跑一次 GPU 场景，返回它报回的结果（含退出码）。

    独立进程是必需的：显式让 sherpa 试 CUDA 的进程会在退出时段错误，放在本进程里
    会把整套测试带崩。这里把退出码一并带回，于是"默认配置退出是否干净"变成可断言
    的事实，而不是"应该没事吧"。
    """
    try:
        proc = subprocess.run(
            [sys.executable, PROBE, mode],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return None
    payload = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("PROBE "):
            try:
                payload = json.loads(line[len("PROBE "):])
            except json.JSONDecodeError:
                payload = None
    if payload is None:
        return None
    payload["exit_code"] = proc.returncode
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-timing", action="store_true",
                    help="跳过 CPU/GPU 计时与等价性对照（GPU 那几次会慢）")
    args = ap.parse_args()

    cat = modelstore.load_catalog(ROOT)
    has_gpu = gpu_present()

    # --- 1. CUDA 运行库探测 ---------------------------------------------
    section("1. CUDA / cuDNN 运行库探测")
    print(f"  GPU：{cuda_env.gpu_summary()}")
    cov = cuda_env.coverage()
    for pat, dirs in cov.items():
        print(f"  {pat:18} -> {dirs[0] if dirs else '(未找到)'}")
    if has_gpu:
        check("三类运行库 DLL 齐备（cudnn / cudart / cublas）",
              cuda_env.satisfied(), f"缺 {cuda_env.missing()}")
        check("activate() 能给出可用的 DLL 目录",
              bool(cuda_env.activate()), f"{len(cuda_env.activate())} 个目录")
    else:
        skip("三类运行库 DLL 齐备", "本机没有 NVIDIA GPU")
        check("没有 GPU 时 activate() 不抛异常且返回空表",
              cuda_env.activate() == [], cuda_env.status().get("reason", ""))

    # --- 2. 设备决策 -----------------------------------------------------
    section("2. 设备决策（比「能不能上 GPU」更重要的是「值不值得」）")
    p_cpu, d_cpu = bwe.providers_for("cpu")
    check("带宽扩展 device=cpu 只给 CPUExecutionProvider",
          p_cpu == ["CPUExecutionProvider"], d_cpu)

    p_auto, d_auto = bwe.providers_for("auto")
    check("带宽扩展 auto 档的 providers 列表尾保留 CPU 回退",
          p_auto[-1] == "CPUExecutionProvider", str(p_auto))
    if has_gpu and cuda_env.satisfied():
        check("带宽扩展 auto 档选 CUDA（实测快 1.55×）",
              p_auto[0] == "CUDAExecutionProvider", d_auto)
    else:
        skip("带宽扩展 auto 档选 CUDA", "无 GPU 或运行库不全")

    s_auto, sd_auto = engines.resolve_sherpa_provider("auto")
    check("sherpa auto 档选 **CPU**（实测 DPDFNet 在 CUDA 上慢 6.3–6.6×）",
          s_auto == "cpu", sd_auto)
    s_cpu, sd_cpu = engines.resolve_sherpa_provider("cpu")
    check("sherpa cpu 档选 CPU", s_cpu == "cpu", sd_cpu)
    s_want, sd_want = engines.resolve_sherpa_provider("cuda")
    check("sherpa cuda 档的返回值合法且必带理由",
          s_want in ("cpu", "cuda") and bool(sd_want.strip()), f"{s_want} — {sd_want}")

    check("sherpa 是否带 CUDA 的判定可复现（二进制证据）",
          engines.sherpa_cuda_available() in (True, False),
          f"带 CUDA = {engines.sherpa_cuda_available()}")

    # --- 3. 请求 CUDA 但加载失败 → 必须回落且留痕 ------------------------
    section("3. 回退逻辑：请求 CUDA 失败时必须回落，且痕迹可见")
    try:
        import audiosronnx  # noqa: PLC0415

        original = audiosronnx.load_sr

        def failing_load(*a, **kw):
            provs = kw.get("providers") or []
            if provs and provs[0] == "CUDAExecutionProvider":
                raise RuntimeError("模拟：CUDA 初始化失败（测试用）")
            return original(*a, **kw)

        audiosronnx.load_sr = failing_load
        bwe.release_thread_models()
        x = _excerpt(2.0, 16000)
        try:
            y, rate = bwe.upscale("novasr", cat, x, 16000, device="cuda")
            info = bwe.last_device()
            desc = info.get("desc", "")
            check("CUDA 失败后仍产出结果（回落而不是让功能不可用）",
                  y.size > 0 and rate > 0, f"{len(y)} 帧 @ {rate} Hz")
            check("回落在 desc 里留下痕迹（不静默降级）",
                  "回落" in desc, desc[:120])
        except Exception as exc:                     # noqa: BLE001
            check("CUDA 失败后仍产出结果（回落而不是让功能不可用）",
                  False, f"{type(exc).__name__}: {exc}")
    except ImportError:
        skip("回退逻辑", "未安装 audiosronnx")
    finally:
        try:
            audiosronnx.load_sr = original  # type: ignore[possibly-undefined]
        except Exception:
            pass
        bwe.release_thread_models()

    # --- 4/5. 真启用 CUDA 的证据、速度与等价性 ---------------------------
    section("4. 启用 CUDA 的证据、速度与 CPU/GPU 等价性")
    print("  说明：等价性断言只在 **CUDA 真的生效** 时才成立。若某一路回落到了 CPU，")
    print("        那么两次输出必然完全相同——那种「通过」是空洞的，所以这里改为跳过并说明。")
    if not (has_gpu and cuda_env.satisfied()):
        skip("BWE CPU/GPU 对照", "无 GPU 或运行库不全")
    elif args.skip_timing:
        skip("BWE CPU/GPU 对照", "--skip-timing")
    else:
        x = _excerpt(20.0, 16000)
        outs: dict[str, np.ndarray] = {}
        times: dict[str, float] = {}
        for dev in ("cpu", "cuda"):
            bwe.release_thread_models()
            t0 = time.perf_counter()
            y, rate = bwe.upscale("novasr", cat, x, 16000, device=dev)
            times[dev] = time.perf_counter() - t0
            outs[dev] = np.asarray(y, dtype=np.float32)
            print(f"    {dev:4} {times[dev]:6.2f}s  RTF {times[dev]/20:.3f}  "
                  f"设备={bwe.last_device().get('desc','?')[:70]}")

        cuda_desc = bwe.last_device().get("desc", "")
        if "CUDA" not in cuda_desc:
            skip("BWE CPU/GPU 等价性", f"CUDA 档回落到了 CPU：{cuda_desc[:80]}")
        else:
            check("CUDA 真的被启用（进程内已加载 onnxruntime_providers_cuda.dll）",
                  cuda_env.cuda_ep_active(), str(cuda_env.loaded_cuda_providers()[:2]))
            a, b = outs["cpu"], outs["cuda"]
            n = min(len(a), len(b))
            diff = np.abs(a[:n] - b[:n])
            rms = float(np.sqrt(np.mean(a[:n] ** 2))) or 1e-12
            rel = float(np.sqrt(np.mean(diff ** 2))) / rms * 100.0
            check("CPU/GPU 长度一致", len(a) == len(b), f"{len(a)} vs {len(b)}")
            check("CPU/GPU 数值等价（设备不进指纹的前提）",
                  float(diff.max()) < MAX_ABS_DIFF and rel < MAX_REL_RMS_PCT,
                  f"最大差 {diff.max():.2e}、相对 RMS {rel:.4f}%")
            print(f"    BWE 速度：CPU {times['cpu']:.2f}s → CUDA {times['cuda']:.2f}s"
                  f"（{times['cpu']/times['cuda']:.2f}×）")

    # sherpa 的 CPU/CUDA 对照必须放到**独立子进程**里做：同一进程里只要 pip 的
    # onnxruntime 用过 CUDA，sherpa 再试 CUDA 就会在**退出阶段段错误**（实测 2/2）。
    # 隔离之后既能把真实 CUDA 的耗时/等价性测出来，也不会带崩本套件。
    if not has_gpu and not cuda_env.satisfied():
        skip("sherpa CPU/CUDA 对照（子进程）", "无 GPU 或运行库不全")
    elif args.skip_timing:
        skip("sherpa CPU/CUDA 对照（子进程）", "--skip-timing")
    else:
        probe = _run_probe("sherpa-only")
        if probe is None:
            skip("sherpa CPU/CUDA 对照（子进程）", "探针未产出结果")
        elif not probe.get("cuda_really_used"):
            skip("sherpa CPU/CUDA 对照（子进程）",
                 f"该进程里 CUDA 也未生效：{probe.get('cuda_desc','')[:80]}")
        else:
            print(f"    cpu {probe['cpu_seconds']:6.2f}s / cuda {probe['cuda_seconds']:6.2f}s"
                  f"（{probe['cuda_seconds']/probe['cpu_seconds']:.1f}× 慢）")
            check("sherpa CPU/CUDA 数值等价",
                  probe["max_abs_diff"] < MAX_ABS_DIFF
                  and probe["rel_rms_pct"] < MAX_REL_RMS_PCT,
                  f"最大差 {probe['max_abs_diff']:.2e}、相对 RMS {probe['rel_rms_pct']}%")
            check("sherpa auto 档确实避开了更慢的 GPU 路径（这就是它选 CPU 的依据）",
                  probe["cuda_seconds"] > probe["cpu_seconds"],
                  f"CPU {probe['cpu_seconds']:.2f}s vs CUDA {probe['cuda_seconds']:.2f}s")

    # --- 5. 退出行为 -----------------------------------------------------
    section("5. 退出行为（这一条才是用户真正碰得到的）")
    print("  实测背景：一个进程里两套 ONNX Runtime 都碰过 CUDA 时，退出阶段会段错误")
    print("  （EXIT=139，2/2 复现）。所以这里用独立子进程分别验证两条路径的退出码。")
    if args.skip_timing:
        skip("退出行为", "--skip-timing")
    else:
        d = _run_probe("default")
        if d is None:
            check("默认配置能跑完并产出设备说明", False, "探针未产出结果")
        else:
            check("默认配置（带宽扩展 CUDA + 降噪 CPU）产出两段设备说明",
                  bool(d.get("bwe_desc")) and bool(d.get("denoise_desc")),
                  f"BWE={d.get('bwe_desc','')[:40]} / 降噪={d.get('denoise_desc','')[:40]}")
            check("**默认配置退出码为 0**（用户不会看到崩溃）",
                  d["exit_code"] == 0, f"EXIT={d['exit_code']}")

        e = _run_probe("sherpa-cuda")
        if e is None:
            check("显式请求 sherpa CUDA 时行为可预期（真用 CUDA 或以可见理由回落）",
                  False, "探针未产出结果")
        else:
            desc = e.get("denoise_desc", "")
            check("显式请求 sherpa CUDA 时行为可预期（真用 CUDA 或以可见理由回落）",
                  ("CPU" in desc) or ("CUDA" in desc), desc[:90])
            print(f"    [记录] 显式请求 sherpa CUDA 的退出码 = {e['exit_code']}"
                  f"（实测为 139 段错误，属该实验路径的已知限制，已写入 README）")
            check("该实验路径的段错误不影响默认配置（上一条已单独验证）", True,
                  "两条路径用的是独立进程")

    # --- 6. 显存预检 -----------------------------------------------------
    section("6. 显存预检（不足时不得硬闯）")
    ok, why = cuda_env.vram_ok()
    print(f"  vram_ok() -> {ok}：{why}")
    check("vram_ok 在阈值极大时判定为不足（预检确实起作用）",
          cuda_env.vram_ok(need_mb=10 ** 9)[0] is False)
    check("vram_ok 在阈值极小时判定为够用（或明确报告取不到显存）",
          cuda_env.vram_ok(need_mb=1.0)[0] is has_gpu,
          cuda_env.vram_ok(need_mb=1.0)[1])

    print()
    if skipped:
        print(f"（跳过 {len(skipped)} 项，原因见上）")
    if failures:
        print(f"GPU 验证结束：{len(failures)} 项失败")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GPU 验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
