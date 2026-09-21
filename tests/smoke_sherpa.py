"""引擎冒烟测试：确认模型文件能被 sherpa-onnx 加载并实际跑出结果。

同时验证 DenoiserAudio 的返回类型、采样率自适应、以及处理前后电平变化。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe tests\\smoke_sherpa.py
    .venv\\Scripts\\python.exe tests\\smoke_sherpa.py --model models/sherpa/dpdfnet2.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "sherpa")


def make_test_signal(sample_rate: int, seconds: float = 8.0, snr_db: float = 5.0):
    """合成"类语音"测试信号：谐波源 + 音节包络 + 共振峰 + 加性噪声。

    只为验证流程能跑通并观察电平变化，不代表真实语音质量。
    """
    rng = np.random.default_rng(20260921)
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate

    # 基频滑动 90→140 Hz，模拟语调起伏
    f0 = 90.0 + 50.0 * np.sin(2 * np.pi * 0.25 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    voice = np.zeros(n, dtype=np.float64)
    for k, amp in enumerate([1.0, 0.6, 0.4, 0.25, 0.15, 0.1], start=1):
        voice += amp * np.sin(k * phase)
    voice /= np.max(np.abs(voice))

    # 音节包络：约 4 Hz 起落
    syl = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t - np.pi / 2)
    voice *= np.clip(syl, 0.05, 1.0)

    # 简单共振峰塑形（一阶差分 + 低通混合，粗略抬高中频）
    voice = 0.7 * voice + 0.3 * np.gradient(voice) * sample_rate / (2 * np.pi * 800.0)

    # 加性白噪，按目标 SNR 定标
    sig_power = float(np.mean(voice ** 2))
    noise = rng.standard_normal(n)
    noise_power = float(np.mean(noise ** 2))
    target_noise = sig_power / (10 ** (snr_db / 10.0))
    noise *= np.sqrt(target_noise / max(noise_power, 1e-20))

    noisy = (voice + noise) * 0.25  # 留足余量，避免削顶
    return noisy.astype(np.float32), voice.astype(np.float32)


def build_denoiser(kind: str, model_path: str, threads: int = 2):
    import sherpa_onnx  # noqa: PLC0415

    if kind == "gtcrn":
        model = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=model_path),
            num_threads=threads,
            debug=False,
            provider="cpu",
        )
    elif kind == "dpdfnet":
        model = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            dpdfnet=sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(
                model=model_path, attenuation_limit_db=12.0
            ),
            num_threads=threads,
            debug=False,
            provider="cpu",
        )
    else:
        raise ValueError(f"未知模型类型：{kind}")

    cfg = sherpa_onnx.OfflineSpeechDenoiserConfig(model=model)
    if not cfg.validate():
        raise RuntimeError(f"配置校验失败：{cfg}")
    return sherpa_onnx.OfflineSpeechDenoiser(cfg)


def classify(path: str) -> str:
    name = os.path.basename(path).lower()
    if "gtcrn" in name:
        return "gtcrn"
    if "dpdfnet" in name or "baseline" in name:
        return "dpdfnet"
    raise ValueError(f"无法从文件名判断模型类型：{name}")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


def dbfs(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-12))


def run_one(path: str, sample_rate: int, threads: int = 2) -> dict:
    kind = classify(path)
    noisy, clean = make_test_signal(sample_rate)
    t0 = time.perf_counter()
    denoiser = build_denoiser(kind, path, threads=threads)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = denoiser(noisy, sample_rate)
    proc_s = time.perf_counter() - t0

    # 关键点：.samples 是 Python list，必须显式转 ndarray
    enhanced = np.asarray(out.samples, dtype=np.float32)
    out_rate = int(out.sample_rate)

    audio_s = len(noisy) / sample_rate
    return {
        "file": os.path.basename(path),
        "kind": kind,
        "bytes": os.path.getsize(path),
        "in_rate": sample_rate,
        "out_rate": out_rate,
        "out_type": type(out.samples).__name__,
        "in_len": len(noisy),
        "out_len": len(enhanced),
        "in_rms": rms(noisy),
        "out_rms": rms(enhanced),
        "clean_rms": rms(clean),
        "load_s": load_s,
        "proc_s": proc_s,
        "rtf": proc_s / audio_s if audio_s else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None,
                    help="指定模型路径（可重复）；默认扫描 models/sherpa/*.onnx")
    ap.add_argument("--rate", type=int, default=16000, help="测试信号采样率（默认 16000）")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    try:
        import sherpa_onnx  # noqa: PLC0415
        print(f"sherpa-onnx {sherpa_onnx.version}  |  onnxruntime {sherpa_onnx.onnxruntime_version}")
    except Exception as exc:
        print(f"[FAIL] 无法导入 sherpa-onnx：{exc}")
        return 2

    paths = args.model
    if not paths:
        if not os.path.isdir(MODEL_DIR):
            print(f"[FAIL] 模型目录不存在：{MODEL_DIR}")
            return 2
        paths = sorted(
            os.path.join(MODEL_DIR, f)
            for f in os.listdir(MODEL_DIR)
            if f.endswith(".onnx") and not f.startswith("_")
        )
    if not paths:
        print(f"[FAIL] 在 {MODEL_DIR} 下没找到任何 .onnx 模型")
        return 2

    print(f"测试信号：{args.rate} Hz，8.0 s，SNR 5 dB（合成类语音）\n")
    header = f"{'模型':<26}{'读取':>8}{'处理':>8}{'RTF':>8}{'输入RMS':>10}{'输出RMS':>10}{'样本':>10}"
    print(header)
    print("-" * len(header))

    failures = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"{os.path.basename(p):<26}  [跳过] 文件不存在")
            continue
        try:
            r = run_one(p, args.rate, threads=args.threads)
        except Exception as exc:
            failures.append((os.path.basename(p), str(exc)))
            print(f"{os.path.basename(p):<26}  [FAIL] {str(exc)[:70]}")
            continue

        print(
            f"{r['file']:<26}{r['load_s']:>7.2f}s{r['proc_s']:>7.2f}s{r['rtf']:>8.3f}"
            f"{r['in_rms']:>10.5f}{r['out_rms']:>10.5f}{r['out_len']:>10d}"
        )
        if r["out_len"] == 0:
            failures.append((r["file"], "输出长度为 0"))
        if r["out_rate"] != r["in_rate"]:
            print(f"{'':<26}  [注意] 采样率变化 {r['in_rate']} → {r['out_rate']}（应为模型原生率）")

    print()
    if failures:
        print(f"失败 {len(failures)} 项：")
        for name, msg in failures:
            print(f"  - {name}: {msg[:160]}")
        return 1

    print(f"全部 {len(paths)} 个模型加载并运行成功。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
