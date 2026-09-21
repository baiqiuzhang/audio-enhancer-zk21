"""DeepFilterNet3 独立 exe 冒烟测试。

重点确认两件计划里标注为「尚未验证」的事：
  1. 它是否接受任意采样率输入（README 声称仅 48 kHz，源码却显示会重采样）。
  2. 输出文件如何命名（它按输入 basename 派生，不提供自定义输出名）。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe tests\\smoke_dfn.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "models", "dfn", "deep-filter.exe")
SRC = os.path.join(ROOT, "samples", "speech_with_noise.wav")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


def run_case(rate: int, exe: str, src: str, tmp: str, extra: list[str] | None = None):
    """把源音频重采样到 rate → 跑 exe → 回读产物。"""
    import soxr

    in_dir = os.path.join(tmp, f"in{rate}")
    out_dir = os.path.join(tmp, f"out{rate}")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    x, sr = sf.read(src, dtype="float32")
    if sr != rate:
        x = soxr.resample(x, sr, rate, quality="HQ").astype(np.float32)

    # 输入名保持唯一，避免按 basename 派生的产物互相覆盖
    in_path = os.path.join(in_dir, f"probe_{rate}.wav")
    sf.write(in_path, x, rate, subtype="PCM_16")

    cmd = [exe, "-o", out_dir, "--pf", "-D", *(extra or []), in_path]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    elapsed = time.perf_counter() - t0

    produced = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.lower().endswith(".wav")
    )
    result = {
        "in_rate": rate,
        "rc": proc.returncode,
        "elapsed": elapsed,
        "n_produced": len(produced),
        "produced_name": os.path.basename(produced[0]) if produced else None,
        "stderr_tail": (proc.stderr or "").strip().splitlines()[-1:] ,
        "same_name": (os.path.basename(produced[0]) == os.path.basename(in_path)) if produced else None,
    }
    if produced:
        y, yr = sf.read(produced[0], dtype="float32")
        result.update(
            out_rate=int(yr),
            in_frames=int(len(x)),
            out_frames=int(len(y)),
            in_rms=rms(x),
            out_rms=rms(y) if len(y) else 0.0,
        )
        result["audio_s"] = len(x) / rate
        result["rtf"] = elapsed / result["audio_s"] if result["audio_s"] else float("nan")
    return result


def main() -> int:
    if not os.path.isfile(EXE):
        print(f"[FAIL] 找不到 deep-filter.exe：{EXE}")
        return 2
    if not os.path.isfile(SRC):
        print(f"[FAIL] 找不到测试音频：{SRC}")
        return 2

    print(f"exe: {EXE}  ({os.path.getsize(EXE):,} B)")
    print(f"源:  {SRC}\n")

    header = f"{'输入率':>9}{'退出码':>8}{'耗时':>9}{'产物':>6}{'输出率':>9}{'入帧':>9}{'出帧':>9}{'RTF':>8}{'入RMS':>10}{'出RMS':>10}"
    print(header)
    print("-" * len(header))

    failures = []
    with tempfile.TemporaryDirectory(prefix="dfn_probe_") as tmp:
        for rate in (16000, 44100, 48000):
            try:
                r = run_case(rate, EXE, SRC, tmp)
            except Exception as exc:
                failures.append((rate, str(exc)))
                print(f"{rate:>9}{'EXC':>8}  {str(exc)[:70]}")
                continue

            if not r["n_produced"]:
                failures.append((rate, "未产出任何 wav"))
                print(f"{rate:>9}{r['rc']:>8}{r['elapsed']:>8.2f}s{'0':>6}   "
                      f"[无产物] {r['stderr_tail']}")
                continue

            print(
                f"{rate:>9}{r['rc']:>8}{r['elapsed']:>8.2f}s{r['n_produced']:>6}"
                f"{r['out_rate']:>9}{r['in_frames']:>9}{r['out_frames']:>9}"
                f"{r['rtf']:>8.3f}{r['in_rms']:>10.5f}{r['out_rms']:>10.5f}"
            )
            if r["out_rate"] != rate:
                print(f"{'':>9}  [注意] 输出采样率 {r['out_rate']} != 输入 {rate}")
            if r["same_name"]:
                print(f"{'':>9}  [命名] 产物沿用输入 basename：{r['produced_name']}")

    print()
    if failures:
        print(f"失败 {len(failures)} 项：")
        for rate, msg in failures:
            print(f"  - {rate} Hz: {msg[:200]}")
        return 1
    print("三种采样率均处理成功。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
