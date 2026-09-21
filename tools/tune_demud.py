"""去闷 EQ 的效果验证与调参工具。

对素材的**一段**施加与流水线相同的 EQ（`core.enhance.spectral_shape`），
用同一套客观指标量出前后的频谱倾斜变化，并可选写出处理后的片段供试听。

为什么不用 PESQ/STOI 当判据：它们与"发闷是否改善"相关性弱，且文献里
"客观指标涨、下游任务不涨"的反例很多。这里直接量**发闷指数**（200–500 Hz
相对 2–5 kHz 的电平差）——它就是"发闷"的物理定义。

用法：
    .venv\\Scripts\\python.exe tools\\tune_demud.py <文件>
    .venv\\Scripts\\python.exe tools\\tune_demud.py <文件> --start 60 --dur 90 \\
        --mud-cut-db -5 --shelf-hz 3000 --shelf-db 6 --out D:/tmp/tuned.wav
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import enhance, media_io  # noqa: E402
from tools.diagnose_audio import BANDS, band_metrics, octave_levels, spectra  # noqa: E402


def slice_of(path: str, start_s: float, dur_s: float) -> tuple[np.ndarray, int]:
    """取一段单声道。整体解码后再切（PyAV 没有按时间 seek 的简单接口）。"""
    data = media_io.decode(path, media_io.probe(path).default_audio_index)
    rate = int(data.sample_rate)
    mono = media_io.to_mono(np.asarray(data.samples, dtype=np.float32))
    a = max(0, int(start_s * rate))
    b = len(mono) if dur_s <= 0 else min(len(mono), a + int(dur_s * rate))
    return mono[a:b], rate


def report(tag: str, mono: np.ndarray, rate: int) -> dict:
    bm = band_metrics(mono, rate)["speech_only"]
    print(f"\n  [{tag}] 谱质心 {bm['centroid_hz']:.0f} Hz · "
          f"发闷指数 {bm['muddiness_db']:+.2f} dB")
    for name, _lo, _hi in BANDS:
        print(f"    {name:26s} 占比 {bm['shares_pct'][name]:6.2f}%   "
              f"相对中频 {bm['level_db_rel_mid'][name]:+6.2f} dB")
    return bm


def main() -> int:
    ap = argparse.ArgumentParser(description="去闷 EQ 效果验证 / 调参")
    ap.add_argument("file")
    ap.add_argument("--start", type=float, default=0.0, help="起始秒")
    ap.add_argument("--dur", type=float, default=60.0, help="时长秒；0 = 到结尾")
    ap.add_argument("--mud-cut-hz", type=float, default=350.0)
    ap.add_argument("--mud-cut-db", type=float, default=-5.0)
    ap.add_argument("--mud-cut-width-oct", type=float, default=0.9)
    ap.add_argument("--shelf-hz", type=float, default=3000.0)
    ap.add_argument("--shelf-db", type=float, default=6.0)
    ap.add_argument("--hp", type=float, default=0.0, help="高通截止 Hz（0=关）")
    ap.add_argument("--out", default="", help="写出处理后片段供试听")
    args = ap.parse_args()

    if not os.path.isfile(args.file):
        print(f"[FAIL] 文件不存在：{args.file}")
        return 2

    mono, rate = slice_of(args.file, args.start, args.dur)
    print(f"素材：{os.path.basename(args.file)}  "
          f"片段 {args.start:.0f}s 起 {len(mono) / rate:.1f}s @ {rate} Hz")

    print(f"\n参数：高通 {args.hp:g} Hz · 去闷 {args.mud_cut_db:+g} dB @ "
          f"{args.mud_cut_hz:g} Hz（{args.mud_cut_width_oct:g} 倍频程）· "
          f"搁架 {args.shelf_db:+g} dB @ {args.shelf_hz:g} Hz")

    before = report("处理前", mono, rate)

    y = enhance.spectral_shape(
        mono, rate,
        hp_cutoff_hz=args.hp, shelf_hz=args.shelf_hz, shelf_db=args.shelf_db,
        mud_cut_hz=args.mud_cut_hz, mud_cut_db=args.mud_cut_db,
        mud_cut_width_oct=args.mud_cut_width_oct,
    )
    after = report("处理后", y, rate)

    print(f"\n  --- 变化 ---")
    print(f"  发闷指数 {before['muddiness_db']:+.2f} → {after['muddiness_db']:+.2f} dB"
          f"   （改善 {before['muddiness_db'] - after['muddiness_db']:+.2f} dB）")
    print(f"  谱质心 {before['centroid_hz']:.0f} → {after['centroid_hz']:.0f} Hz")
    for name, _lo, _hi in BANDS:
        d = after["level_db_rel_mid"][name] - before["level_db_rel_mid"][name]
        print(f"  {name:26s} {d:+6.2f} dB")

    print("\n  --- 倍频程（相对最强，dB）---")
    for tag, sig in (("前", mono), ("后", y)):
        freqs, sp = spectra(sig, rate)
        vals = dict(octave_levels(freqs, sp["psd_speech"], rate))
        print(f"  {tag}: " + "  ".join(f"{int(c) if c >= 1 else c}:{v:+.0f}"
                                       for c, v in vals.items()))

    # 削顶检查：EQ 提升频段可能把峰值推过 0 dBFS
    peak_in = float(np.max(np.abs(mono)))
    peak_out = float(np.max(np.abs(y)))
    flag = "  ⚠ 已削顶，需要降增益或减小搁架" if peak_out >= 0.999 else ""
    print(f"\n  峰值 {peak_in:.4f} → {peak_out:.4f}{flag}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        media_io.write_audio(args.out, y, rate)
        print(f"  已写出片段供试听：{args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
