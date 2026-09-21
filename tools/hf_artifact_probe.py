"""带宽扩展的高频伪影测量台（单引擎单进程，避免一个引擎 OOM 带崩整个过程）。

用法：
    .venv\\Scripts\\python.exe tools\\hf_artifact_probe.py --engine novasr
    .venv\\Scripts\\python.exe tools\\hf_artifact_probe.py --engine lavasr
    .venv\\Scripts\\python.exe tools\\hf_artifact_probe.py --engine hifiganbwe

回答两个问题，都用**绝对电平**（dBFS），因为"高出局部中位数 N dB"不等于听得见：

1. 生成频段里有没有**离散窄带线**（啸叫/嗡声），各自多响；
2. 这些线相对通带低多少分贝——低于通带 40 dB 以上基本可以不管；
3. 附带做镜像检验：生成频段的谱形状是否只是通带在 16 kHz（模型输入 Nyquist）两侧的镜像。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bwe, media_io, modelstore  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA = r"C:\Users\10424\Desktop\bandicam 2026-09-21 13-54-07-025.mp4"
START_S, DUR_S = 300.0, 20.0
NFFT = 8192


def rms_dbfs(x: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))), 1e-12))


def band_signal_db(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """带内 RMS（用 FFT 掩码做零相位带通，够用且无相位失真）。"""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / sr)
    X[(f < lo) | (f >= min(hi, sr / 2))] = 0
    return rms_dbfs(np.fft.irfft(X, n))


def avg_psd(x: np.ndarray, sr: int, nfft: int = NFFT):
    win = np.hanning(nfft)
    hop = nfft // 2
    nfr = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(nfr)[:, None]
    fr = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
    p = fr.mean(axis=0) / np.sum(win ** 2)          # 正弦幅度 1 → 0.25
    return p, np.fft.rfftfreq(nfft, 1 / sr)


def line_level_dbfs(x: np.ndarray, sr: int, freq: float) -> float:
    """用整段做加窗投影，估计该频点的正弦幅度（dBFS）。"""
    t = np.arange(len(x)) / sr
    w = np.hanning(len(x))
    proj = np.sum(x * w * np.exp(-2j * np.pi * freq * t))
    amp = 2.0 * abs(proj) / np.sum(w)
    return 20 * np.log10(max(amp, 1e-12))


def find_lines(p, f, lo, hi, thresh_db=8.0, top=6):
    """在 [lo,hi) 找相对**局部平滑基线**高出 thresh_db 的谱线。"""
    sel = (f >= lo) & (f < hi)
    fb, pb = f[sel], p[sel]
    if len(pb) < 8:
        return []
    # 基线：用较宽的滑动中位数（去掉窄线的影响）
    k = max(3, int(round(600.0 / (f[1] - f[0]))))    # 约 600 Hz 窗
    base = np.array([
        np.median(pb[max(0, i - k):i + k + 1]) for i in range(len(pb))
    ])
    rel = 10 * np.log10(np.maximum(pb, 1e-30) / np.maximum(base, 1e-30))
    cand = np.argsort(rel)[::-1]
    out = []
    for i in cand:
        if rel[i] < thresh_db or len(out) >= top:
            break
        if any(abs(fb[i] - o[0]) < 120 for o in out):    # 合并邻近 bin
            continue
        out.append((float(fb[i]), float(rel[i])))
    return out


def ratio_stats(p_out, p_in, f_out, f_in, pairs, tag):
    """逐点比较两组谱（差值序列的均值与离散度）。用于镜像检验。"""
    do, di = [], []
    for fo, fi in pairs:
        so = (f_out >= fo - 100) & (f_out < fo + 100)
        si = (f_in >= fi - 100) & (f_in < fi + 100)
        if so.any() and si.any():
            do.append(10 * np.log10(max(p_out[so].mean(), 1e-30)))
            di.append(10 * np.log10(max(p_in[si].mean(), 1e-30)))
    d = np.array(do) - np.array(di)
    if not len(d):
        print(f"    {tag:<30} （无数据点）")
        return d
    print(f"    {tag:<30} 均值 {d.mean():+6.1f} dB  离散度(std) {d.std():5.2f} dB"
          f"  范围 {d.min():+6.1f}…{d.max():+6.1f} dB  ({len(d)} 点)")
    return d


def mirror_analysis(p_out, f_out, p_in, f_in, tag):
    """判定"补出来的高频"是不是通带在输入 Nyquist(16 kHz) 上的镜像副本。

    判据：若输出(16k−f) 相对 输入(f) 的差**大致平坦**（只差一个常数增益），那它就是
    镜像副本；"真的生成了新内容"会给出明显不平坦的差值。
    两个对照：通带自身（只经一次增益，其离散度即方法噪声下限）与输出(8k+f)（同向平移，
    非镜像的反例基线）。
    """
    print(f"    —— 镜像副本检验（{tag}）——")
    a = np.arange(200, 6000, 100)
    ratio_stats(p_out, p_in, f_out, f_in, [(f, f) for f in a], "通带自身（方法噪声下限）")
    b = np.arange(500, 7000, 100)
    ratio_stats(p_out, p_in, f_out, f_in, [(16000 - f, f) for f in b],
                "输出(16k−f) vs 输入(f) [镜像]")
    ratio_stats(p_out, p_in, f_out, f_in, [(16000 + f, f) for f in b],
                "输出(16k+f) vs 输入(f) [镜像]")
    ratio_stats(p_out, p_in, f_out, f_in, [(8000 + f, f) for f in b],
                "输出(8k+f) vs 输入(f)  [反例]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="novasr")
    ap.add_argument("--media", default=MEDIA)
    ap.add_argument("--duration", type=float, default=DUR_S)
    ap.add_argument("--start", type=float, default=START_S)
    ap.add_argument("--no-bwe", action="store_true", help="只分析输入（对照）")
    ap.add_argument("--mirror", action="store_true", help="额外做镜像副本检验")
    ap.add_argument("--compare-cpu-gpu", action="store_true",
                    help="同一输入分别用 CPU / CUDA 跑一遍，对比高频伪影（查 TF32 之类的影响）")
    args = ap.parse_args()

    info = media_io.probe(args.media)
    d = media_io.decode(args.media, stream_index=info.default_audio_index)
    sr = d.sample_rate
    seg = np.asarray(d.samples[int(args.start * sr):int((args.start + args.duration) * sr)])
    mono16 = media_io.resample(media_io.to_mono(seg), sr, 16000)

    def report(x, rate, tag, p_in=None, f_in=None):
        pb = band_signal_db(x, rate, 100, 5000)
        print(f"  [{tag}] 通带(100–5k) RMS {pb:.1f} dBFS")
        for lo, hi in ((5000, 8000), (8000, 16000), (16000, 24000)):
            if lo >= rate / 2:
                continue
            lv = band_signal_db(x, rate, lo, hi)
            print(f"    {lo//1000}–{hi//1000}k RMS {lv:7.1f} dBFS   相对通带 {lv-pb:+7.1f} dB")
        p, f = avg_psd(x, rate)
        for lo, hi, name in ((5000, 16000, "5–16k"), (16000, 24000, "16–24k")):
            if lo >= rate / 2:
                continue
            lines = find_lines(p, f, lo, min(hi, rate / 2))
            print(f"    {name} 离散线（相对局部基线 >8 dB）：")
            if not lines:
                print("      （无）")
            for freq, rel in lines:
                lvl = line_level_dbfs(x, rate, freq)
                print(f"      {freq:8.0f} Hz  +{rel:5.1f} dB(相对基线)  "
                      f"绝对 {lvl:7.1f} dBFS  相对通带 {lvl-pb:+.1f} dB")
        # 镜像检验
        def prof(lo, hi, flip=False):
            sel = (f >= lo) & (f < min(hi, rate / 2))
            v = 10 * np.log10(np.maximum(p[sel], 1e-30))
            return v[::-1] if flip else v
        a = prof(500, 7500); b = prof(8500, 15500, flip=True); c = prof(16500, 23500)
        n = min(len(a), len(b), len(c))
        if n > 16:
            print(f"    镜像相关：通带谱 vs 16k 下侧镜像 r={np.corrcoef(a[:n], b[:n])[0,1]:+.3f}"
                  f"；vs 上侧镜像 r={np.corrcoef(a[:n], c[:n])[0,1]:+.3f}")
        if args.mirror and p_in is not None:
            mirror_analysis(p, f, p_in, f_in, tag)

    print(f"素材：{os.path.basename(args.media)}  {args.start:.0f}s 起 {args.duration:.0f}s")
    print()
    p_in, f_in = avg_psd(mono16, 16000)
    if args.no_bwe:
        report(mono16, 16000, "输入 16k（对照）")
        return 0

    cat = modelstore.load_catalog(ROOT)
    if args.compare_cpu_gpu:
        outs = {}
        for dev in ("cpu", "cuda"):
            bwe.release_thread_models()
            t0 = time.perf_counter()
            y, out_rate = bwe.upscale(args.engine, cat, mono16, 16000, device=dev)
            t = time.perf_counter() - t0
            outs[dev] = np.asarray(y, dtype=np.float32).reshape(-1)
            print(f"\n===== {dev.upper()}（{t:.2f}s）=====")
            report(outs[dev], out_rate, f"{args.engine}/{dev}", p_in, f_in)
        a, b = outs["cpu"], outs["cuda"]
        n = min(len(a), len(b))
        diff = a[:n] - b[:n]
        print(f"\n===== CPU − CUDA 的差信号 =====")
        print(f"  长度 {len(a)} vs {len(b)}；最大逐样本差 {np.abs(diff).max():.3e}")
        print(f"  差信号 RMS {rms_dbfs(diff):.1f} dBFS；"
              f"相对 CPU 输出 RMS {rms_dbfs(diff) - rms_dbfs(a[:n]):+.1f} dB")
        for lo, hi in ((0, 5000), (5000, 8000), (8000, 16000), (16000, 24000)):
            print(f"    差信号 {lo//1000}–{hi//1000}k RMS {band_signal_db(diff, out_rate, lo, hi):7.1f} dBFS")
        p_d, f_d = avg_psd(diff, out_rate)
        for lo, hi, name in ((5000, 16000, "5–16k"), (16000, 24000, "16–24k")):
            lines = find_lines(p_d, f_d, lo, min(hi, out_rate / 2))
            print(f"    差信号 {name} 离散线：")
            for freq, rel in lines:
                print(f"      {freq:8.0f} Hz  +{rel:5.1f} dB(相对基线)  "
                      f"绝对 {line_level_dbfs(diff, out_rate, freq):7.1f} dBFS")
            if not lines:
                print("      （无）")
        return 0

    t0 = time.perf_counter()
    y, out_rate = bwe.upscale(args.engine, cat, mono16, 16000, device="auto")
    t = time.perf_counter() - t0
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    print(f"引擎 {args.engine}：{out_rate} Hz，耗时 {t:.2f}s（RTF {t/args.duration:.3f}），"
          f"设备 {bwe.last_device().get('desc','')[:60]}")
    report(y, out_rate, args.engine, p_in, f_in)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
