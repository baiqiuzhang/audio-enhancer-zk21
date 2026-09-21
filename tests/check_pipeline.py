"""流水线验证：逐个引擎跑通，并覆盖立体声/采样率/静音裁剪/长度对齐。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_pipeline.py
    .venv\\Scripts\\python.exe tests\\check_pipeline.py --engine dpdfnet2
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dsp, media_io, modelstore, pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "speech_with_noise.wav")

# 默认输出位置是「输入目录/AudioEnhancer_zk21/」。测试用临时目录，避免污染 samples/。
OUT_DIR = tempfile.mkdtemp(prefix="ae_pipeline_")


def section(t: str) -> None:
    print()
    print(t)
    print("=" * len(t))


def engine_sweep(cat: modelstore.Catalog, only: str | None) -> int:
    section("1. 引擎逐个跑通（输入：官方测试音频 speech_with_noise.wav）")
    print(f"源：{os.path.basename(SAMPLE)}\n")

    header = (f"{'引擎':<24}{'RTF':>7}{'耗时':>8}{'入RMSdB':>9}{'出RMSdB':>9}"
              f"{'入峰值dB':>10}{'出峰值dB':>10}{'长度':>7}")
    print(header)
    print("-" * len(header))

    failures = 0
    for spec in cat.engines:
        if only and spec.id != only:
            continue
        opts = pipeline.PipelineOptions(
            engine_id=spec.id,
            engine=pipeline.EngineParams(num_threads=2),
            output_dir=OUT_DIR,
        )
        try:
            r = pipeline.run(SAMPLE, cat, opts)
        except Exception as exc:
            failures += 1
            print(f"{spec.name:<24}  [FAIL] {str(exc)[:100]}")
            continue

        aligned = len(r.source.samples) == len(r.enhanced.samples)
        if not aligned:
            failures += 1
        print(
            f"{spec.name:<24}{r.rtf:>7.3f}{r.elapsed:>7.2f}s"
            f"{r.source_stats.rms_dbfs:>9.1f}{r.enhanced_stats.rms_dbfs:>9.1f}"
            f"{r.source_stats.peak_dbfs:>10.1f}{r.enhanced_stats.peak_dbfs:>10.1f}"
            f"{'OK' if aligned else 'MISMATCH':>7}"
        )
    return failures


def stereo_48k_case(cat: modelstore.Catalog, engine_id: str) -> int:
    section(f"2. 48 kHz 立体声 + 首尾静音 + 增益归一化（引擎：{engine_id}）")
    failures = 0

    sr = 48000
    n_sil = sr  # 1 秒静音
    t = np.arange(int(sr * 3.0)) / sr
    tone = 0.06 * np.sin(2 * np.pi * 220 * t)
    rng = np.random.default_rng(7)
    tone += rng.standard_normal(len(t)) * 0.01
    left = np.concatenate([np.zeros(n_sil, np.float32), tone.astype(np.float32), np.zeros(n_sil, np.float32)])
    right = left * 0.7   # 制造左右差异，验证声像是否保留
    stereo = np.stack([left, right], axis=1)

    path = os.path.join(ROOT, "out", "_test_48k_stereo.wav")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    media_io.write_audio(path, stereo, sr)

    info = media_io.probe(path)
    stats = dsp.analyze(stereo, sr)
    print(f"构造素材：{info.duration:.2f}s  {stats.describe()}")
    print(f"          首尾各 1.00s 数字零，中间 3.00s 含噪信号")

    opts = pipeline.PipelineOptions(
        engine_id=engine_id,
        engine=pipeline.EngineParams(num_threads=2),
        trim_silence=True,
        normalize=True,
        normalize_target_dbfs=-3.0,
        output_path=os.path.join(ROOT, "out", "_test_48k_stereo_enh.wav"),
    )
    try:
        r = pipeline.run(path, cat, opts, log=lambda m: print("  · " + m))
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return failures + 1

    checks = [
        ("裁剪生效", r.trim is not None and r.trim.start_frame > 0 and r.trim.end_frame < len(stereo)),
        ("裁剪时长合理", r.trim is not None and 2.5 < r.trim.active_s < 3.7),
        ("保留立体声", r.enhanced.channels == 2 and r.source.channels == 2),
        ("保留采样率", r.enhanced.sample_rate == 48000),
        ("长度对齐", len(r.source.samples) == len(r.enhanced.samples)),
        ("归一化到约 -3 dBFS", r.enhanced_stats.peak_dbfs > -5.0 and r.enhanced_stats.peak_dbfs < -1.0),
        ("输出文件存在", bool(r.output_path) and os.path.isfile(r.output_path)),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failures += 1

    print(f"\n  统计：入 {r.source_stats.describe()}")
    print(f"        出 {r.enhanced_stats.describe()}")
    print(f"  RTF {r.rtf:.3f}   引擎输出率 {r.engine_out_rate} Hz")
    return failures


def long_file_case(cat: modelstore.Catalog, engine_id: str) -> int:
    section(f"3. 长文件分段路径（引擎：{engine_id}，强制 2 秒分段）")
    sr = 16000
    t = np.arange(int(sr * 9.0)) / sr
    rng = np.random.default_rng(11)
    x = (0.05 * np.sin(2 * np.pi * 180 * t) + rng.standard_normal(len(t)) * 0.01).astype(np.float32)
    path = os.path.join(ROOT, "out", "_test_long.wav")
    media_io.write_audio(path, x, sr)

    spec = cat.engine(engine_id)
    from core import engines  # noqa: PLC0415

    eng = engines.build_engine(spec, cat)
    if not isinstance(eng, engines.SherpaEngine):
        print("（该引擎不走分段路径，跳过）")
        return 0

    # 把单次上限压到很小，强制走分段
    old = engines.MAX_FRAMES_SINGLE_PASS
    engines.MAX_FRAMES_SINGLE_PASS = sr * 2
    try:
        opts = pipeline.PipelineOptions(
            engine_id=engine_id,
            engine=pipeline.EngineParams(num_threads=2, segment_seconds=2.0),
            output_dir=OUT_DIR,
        )
        r = pipeline.run(path, cat, opts)
    finally:
        engines.MAX_FRAMES_SINGLE_PASS = old

    ok = len(r.source.samples) == len(r.enhanced.samples)
    print(f"  输入 {len(x)} 帧 → 输出 {len(r.enhanced.samples)} 帧  长度对齐：{'PASS' if ok else 'FAIL'}")
    print(f"  入 RMS {r.source_stats.rms_dbfs:.1f} dBFS → 出 RMS {r.enhanced_stats.rms_dbfs:.1f} dBFS")

    # 检查分段接缝是否有突变（相邻样本差的最大值应与非接缝处同量级）
    y = r.enhanced.samples
    if y.ndim == 2:
        y = y.mean(axis=1)
    d = np.abs(np.diff(y))
    seam_idx = [i for i in range(sr * 2, len(y) - 1, sr * 2)]
    if seam_idx and d.size:
        seam_max = max(float(d[i]) for i in seam_idx if i < d.size)
        p999 = float(np.percentile(d, 99.9))
        ok2 = seam_max <= max(p999 * 4.0, 1e-4)
        print(f"  接缝处最大跳变 {seam_max:.6f} vs 全局 p99.9 {p999:.6f}  → {'PASS' if ok2 else 'FAIL（可能有可闻接缝）'}")
        if not ok2:
            return 1
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None, help="只测某个引擎 id")
    ap.add_argument("--skip-long", action="store_true")
    args = ap.parse_args()

    cat = modelstore.load_catalog(ROOT)
    failures = 0
    failures += engine_sweep(cat, args.engine)

    primary = args.engine or "dpdfnet2"
    failures += stereo_48k_case(cat, primary)
    if not args.skip_long:
        failures += long_file_case(cat, primary)

    print()
    print(f"（输出写在临时目录：{OUT_DIR}）")
    if failures:
        print(f"流水线验证结束：{failures} 项失败")
        return 1
    print("流水线验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
