"""对真实素材出客观诊断：容器/流清单 + 音频指标（混响、频谱倾斜、带限、电平）。

只读输入，绝不改写任何文件。分块累积统计，避免把长素材的中间数组全留在内存里
（18 分钟 48 kHz 立体声 float32 约 414 MB）。

设计要点（踩过的两个测量坑，别再犯）：
1. **门限要自适应，并且要报告它有没有真的生效**。原始素材若被 AGC 压过动态范围，
   停顿段可能只低 20–30 dB，固定的 −50 dBFS 门限会把所有帧都当语音。
   本脚本报出语音帧占比，占比接近 100% 就说明门限失效、要改看倍频程表。
2. **"语音帧谱"与"全部帧谱"的比值指标是尺度不变的**，而静音帧能量≈0，
   所以两者只在形状上可能不同——报这两套的比值没有意义。真正有对比价值的是
   **语音帧 vs 停顿帧的谱形**：低频大头若出现在停顿帧里，那是房间轰鸣/电流噪声，
   处理手段（低切/高通）与处理"语音本身的低频"完全不同。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tools\\diagnose_audio.py <文件> [<文件> ...]
    .venv\\Scripts\\python.exe tools\\diagnose_audio.py --report out.md <文件> ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dsp, enhance, media_io  # noqa: E402

# 频段划分（Hz）。lowmid 是「发闷」的典型堆积区，presence 是「清晰度」区。
BANDS: list[tuple[str, float, float | None]] = [
    ("sub 20–200", 20.0, 200.0),
    ("lowmid 200–500（发闷区）", 200.0, 500.0),
    ("mid 500–2k", 500.0, 2000.0),
    ("presence 2k–5k", 2000.0, 5000.0),
    ("high 5k+", 5000.0, None),
]
MID_BAND = "mid 500–2k"
MUD_BAND = "lowmid 200–500（发闷区）"
PRES_BAND = "presence 2k–5k"

# 倍频程中心频率：带限的采集链会在这里表现为明显的滚降台阶
OCTAVE_CENTERS = [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]


# ---------------------------------------------------------------------------
# 分块平均功率谱：一趟里同时累积 全部帧 / 语音帧 / 停顿帧
# ---------------------------------------------------------------------------

def spectra(mono: np.ndarray, rate: int, *, n_fft: int = 4096,
            chunk_s: float = 30.0, dynamic_range_db: float = 40.0,
            floor_dbfs: float = -75.0) -> tuple[np.ndarray, dict]:
    """返回 (频率轴, 三套 PSD 与门限元信息)。

    门限沿用 core/dsp.detect_active_span 的思路：帧电平 p95 下压 dynamic_range_db，
    并夹在 floor_dbfs 之上。若 frame_p95 − p10 很窄（被压过动态范围），门限会失效，
    这时 speech_frame_pct 会接近 100%，元信息里会给出提示。
    """
    win = np.hanning(n_fft).astype(np.float32)
    hop = n_fft // 2
    nbins = n_fft // 2 + 1
    acc_all = np.zeros(nbins, dtype=np.float64)
    acc_sp = np.zeros(nbins, dtype=np.float64)
    acc_bg = np.zeros(nbins, dtype=np.float64)
    n_all = n_sp = n_bg = 0
    chunk = max(n_fft, int(chunk_s * rate))
    idx = np.arange(n_fft)[None, :] + hop * np.arange(max(1, chunk // hop))[:, None]

    def _chunks():
        for start in range(0, len(mono) - n_fft, chunk):
            seg = mono[start:start + chunk + n_fft]
            n_fr = 1 + (len(seg) - n_fft) // hop
            if n_fr < 1:
                break
            yield seg[idx[:n_fr]]

    # 第一趟只为定门限（不做 FFT，代价很小）
    levels = []
    for frames in _chunks():
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-12
        levels.append(20.0 * np.log10(rms))
    z = np.zeros(nbins)
    if not levels:
        return np.fft.rfftfreq(n_fft, 1.0 / rate), {
            "psd_all": z, "psd_speech": z, "psd_bg": z, "gate_db": None,
            "usable": False, "note": "素材太短，无法做频谱统计",
        }
    all_db = np.concatenate(levels)
    ref = float(np.percentile(all_db, 95))
    gate = max(ref - dynamic_range_db, floor_dbfs)
    span_db = ref - float(np.percentile(all_db, 10))

    # 第二趟：一趟里同时累积三套谱
    for frames in _chunks():
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-12
        fdb = 20.0 * np.log10(rms)
        spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
        acc_all += np.sum(spec, axis=0)
        n_all += len(fdb)
        hi = fdb >= gate
        lo = ~hi
        if np.any(hi):
            acc_sp += np.sum(spec[hi], axis=0)
            n_sp += int(np.sum(hi))
        if np.any(lo):
            acc_bg += np.sum(spec[lo], axis=0)
            n_bg += int(np.sum(lo))

    freqs = np.fft.rfftfreq(n_fft, 1.0 / rate)
    norm = 1.0 / (np.sum(win ** 2) * rate)
    pct = 100.0 * n_sp / max(n_all, 1)
    note = ""
    if pct >= 99.0:
        note = (f"语音帧占比 {pct:.1f}% —— 门限基本失效（动态范围只有 "
                f"{span_db:.1f} dB）。看倍频程表，不要看语音/停顿的对比。")
    return freqs, {
        "psd_all": acc_all / max(n_all, 1) * norm,
        "psd_speech": acc_sp / max(n_sp, 1) * norm,
        "psd_bg": acc_bg / max(n_bg, 1) * norm,
        "gate_db": round(gate, 1),
        "frame_p95_db": round(ref, 1),
        "frames_total": n_all,
        "frames_speech": n_sp,
        "frames_bg": n_bg,
        "speech_frame_pct": round(pct, 1),
        "dynamic_span_db": round(span_db, 1),
        "usable": pct < 99.0,
        "note": note,
    }


def band_stats(freqs: np.ndarray, psd: np.ndarray, rate: int) -> dict:
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    nyq = rate / 2.0
    powers: dict[str, float] = {}
    for name, lo, hi in BANDS:
        hi_eff = nyq if hi is None else min(hi, nyq)
        powers[name] = (float(np.sum(psd[(freqs >= lo) & (freqs < hi_eff)]) * df)
                        if hi_eff > lo else 0.0)
    total = sum(powers.values()) or 1.0
    mid = powers.get(MID_BAND, 0.0) or 1e-30
    rel = {k: 10.0 * np.log10(max(v, 1e-30) / mid) for k, v in powers.items()}
    return {
        "shares_pct": {k: 100.0 * v / total for k, v in powers.items()},
        "level_db_rel_mid": rel,
        "centroid_hz": round(float(np.sum(freqs * psd) / max(np.sum(psd), 1e-30)), 1),
        "muddiness_db": round(float(rel.get(MUD_BAND, 0.0) - rel.get(PRES_BAND, 0.0)), 2),
    }


def band_metrics(mono: np.ndarray, rate: int) -> dict:
    """便捷封装：直接对一段信号算**语音帧**的频段指标（供工具脚本复用）。"""
    freqs, sp = spectra(mono, rate)
    return {"speech_only": band_stats(freqs, sp["psd_speech"], rate), "gate": sp}


def octave_levels(freqs: np.ndarray, psd: np.ndarray, rate: int) -> list[tuple[float, float]]:
    """倍频程电平（dB，相对最强那一个倍频程）。带限会表现为明显的滚降台阶。"""
    nyq = rate / 2.0
    vals: list[tuple[float, float]] = []
    for c in OCTAVE_CENTERS:
        lo, hi = c / np.sqrt(2), c * np.sqrt(2)
        if lo >= nyq:
            continue
        sel = (freqs >= lo) & (freqs < min(hi, nyq))
        if not np.any(sel):
            continue
        vals.append((c, float(np.sum(psd[sel]))))
    if not vals:
        return []
    peak = max(v for _, v in vals)
    if peak <= 0.0:                     # 这一组没有任何能量（例如没有停顿帧）
        return []
    return [(c, round(10.0 * np.log10(max(v, 1e-30) / peak), 1)) for c, v in vals]


def spectrum_edges(freqs: np.ndarray, psd: np.ndarray, *, drop_db: float = 20.0
                   ) -> tuple[float | None, float | None]:
    """PSD 从峰值下降 drop_db 之后，还剩下内容的最低/最高频率。"""
    if not len(psd):
        return None, None
    peak = float(np.max(psd))
    if peak <= 0:
        return None, None
    thr = peak * (10.0 ** (-drop_db / 10.0))
    above = np.flatnonzero(psd >= thr)
    if not len(above):
        return None, None
    return float(freqs[above[0]]), float(freqs[above[-1]])


def bandpass(samples: np.ndarray, rate: int, lo: float, hi: float) -> np.ndarray:
    """巴特沃斯带通（因果，只用于诊断）。用途见 reverb_band 的说明。"""
    from scipy.signal import butter, sosfilt  # noqa: PLC0415 — 诊断工具，局部引入

    nyq = rate / 2.0
    sos = butter(4, [max(lo, 1.0) / nyq, min(hi, nyq * 0.99) / nyq],
                 btype="bandpass", output="sos")
    return sosfilt(sos, np.asarray(samples, dtype=np.float64)).astype(np.float32)


def reverb_band(mono: np.ndarray, rate: int, lo: float = 400.0,
                hi: float = 1000.0) -> tuple[dict, dict]:
    """全带 vs 带限的衰减斜率，用来判断全带读数是否被稳态噪声底拖慢。

    为什么要这么做：estimate_decay_slope_db_per_s 用**全带**帧电平拟合衰减，
    而稳态底噪会把尾巴"托住"，使读数比真实混响**更湿**。两个真实素材的底噪
    约在语音下 24 dB 且从不停下来，所以全带读数可疑。

    取 400–1000 Hz 是因为这两个文件在该带信噪比最高（语音接近峰值、噪声远低于
    它自己的峰值）：实测噪声 ≈ −67 dBFS、语音 ≈ −31 dBFS，SNR ≈ 36 dB；
    而在 2 kHz（噪声峰值处）SNR 只剩约 9 dB，那些带测衰减会被噪声污染。
    """
    full = enhance.estimate_decay_slope_db_per_s(mono, rate)
    band = enhance.estimate_decay_slope_db_per_s(bandpass(mono, rate, lo, hi), rate)
    return full, band


def stereo_correlation(stereo: np.ndarray, *, max_s: float = 120.0,
                       rate: int = 48000) -> float | None:
    if stereo.ndim != 2 or stereo.shape[1] < 2:
        return None
    seg = stereo[: int(max_s * rate)]
    left = seg[:, 0].astype(np.float64)
    right = seg[:, 1].astype(np.float64)
    if left.std() < 1e-9 or right.std() < 1e-9:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def describe_container(info: media_io.MediaInfo) -> list[str]:
    lines = [
        f"容器 {info.container} · 类型 {info.kind} · 时长 "
        f"{info.duration / 60:.2f} min ({info.duration:.1f}s) · 解码器 {info.decoder}",
    ]
    for v in info.video_streams:
        fps = f"{v.fps:.3f} fps" if v.fps else "fps 未知"
        lines.append(f"  视频流 #{v.index}: {v.codec} {v.width}×{v.height} {fps}")
    for a in info.audio_streams:
        br = f"{a.bit_rate / 1000:.0f} kbps" if a.bit_rate else "码率未知"
        lang = f" 语言={a.language}" if getattr(a, "language", None) else ""
        lines.append(f"  音轨流 #{a.index}: {a.codec} {a.sample_rate} Hz "
                     f"{a.channels}ch {br}{lang}")
    if not info.video_streams:
        lines.append("  （无视频流）")
    if not info.audio_streams:
        lines.append("  ⚠ 无音轨流")
    return lines


# ---------------------------------------------------------------------------

def diagnose(path: str) -> dict:
    print(f"\n{'=' * 78}")
    print(f"文件：{path}")
    print(f"大小：{os.path.getsize(path):,} B")
    print("=" * 78)

    t0 = time.perf_counter()
    info = media_io.probe(path)
    for line in describe_container(info):
        print(f"  {line}")

    data = media_io.decode(path, info.default_audio_index)
    rate = int(data.sample_rate)
    print(f"  解码完成：{data.frames:,} 帧 @ {rate} Hz / {data.channels}ch"
          f"（{len(data.samples) * data.samples.dtype.itemsize / 1e6:.0f} MB）"
          f"  用时 {time.perf_counter() - t0:.1f}s")

    stereo = np.asarray(data.samples, dtype=np.float32)
    corr = stereo_correlation(stereo, rate=rate)
    mono = media_io.to_mono(stereo)
    del stereo

    print("\n  --- 电平与活动度 ---")
    stats = dsp.analyze(mono, rate)
    print(f"  {stats.describe()}")
    print(f"  判断：{stats.verdict()}")
    print(f"  削顶样本占比 {stats.clipped_ratio * 100:.4f}%   直流偏移 {stats.dc_offset:+.2e}")
    print(f"  帧电平 p10/p50/p90：{stats.frame_db_p10:.1f} / {stats.frame_db_p50:.1f} / "
          f"{stats.frame_db_p90:.1f} dBFS   ← 动态范围 {stats.frame_db_p90 - stats.frame_db_p10:.1f} dB")
    print(f"  立体声相关性：{'—' if corr is None else f'{corr:.4f}'}"
          f"{'  ← 近似单声道' if corr is not None and corr > 0.999 else ''}")

    print("\n  --- 混响（全带 vs 400–1k 带限）---")
    decay, decay_band = reverb_band(mono, rate)
    print(f"  全带  ：{enhance.reverb_verdict(decay)}")
    if decay.get("slope_db_per_s") is not None:
        print(f"          斜率 {decay['slope_db_per_s']} dB/s"
              f"（p25 {decay.get('p25')} / p75 {decay.get('p75')}，"
              f"{decay.get('num_segments')} 段）")
    print(f"  400–1k：{enhance.reverb_verdict(decay_band)}")
    if decay_band.get("slope_db_per_s") is not None:
        print(f"          斜率 {decay_band['slope_db_per_s']} dB/s"
              f"（{decay_band.get('num_segments')} 段）")
    if decay.get("slope_db_per_s") and decay_band.get("slope_db_per_s"):
        gap = decay_band["slope_db_per_s"] - decay["slope_db_per_s"]
        verdict = ("差值大 → 全带读数主要被噪声底拖慢，房间比全带读数更干"
                   if gap > 8.0 else "差值不大 → 混响确实存在，不是噪声底造成的假象")
        print(f"  带限比全带快 {gap:+.1f} dB/s —— {verdict}")

    print("\n  --- 频谱 ---")
    freqs, sp = spectra(mono, rate)
    print(f"  门限 {sp['gate_db']} dBFS（帧电平 p95 {sp['frame_p95_db']}）→ "
          f"语音帧 {sp['speech_frame_pct']}% / 停顿帧 "
          f"{round(100 * sp['frames_bg'] / max(sp['frames_total'], 1), 1)}%")
    if sp.get("note"):
        print(f"  ⚠ {sp['note']}")

    for key, label in (("psd_speech", "语音帧"), ("psd_bg", "停顿帧")):
        bs = band_stats(freqs, sp[key], rate)
        lo, hi = spectrum_edges(freqs, sp[key])
        edge = f"−20dB 边缘 {lo:.0f}–{hi:.0f} Hz" if lo is not None else "（无可测内容）"
        print(f"\n  [{label}] 谱质心 {bs['centroid_hz']:.0f} Hz · "
              f"发闷指数 {bs['muddiness_db']:+.2f} dB · {edge}")
        for name, _l, _h in BANDS:
            print(f"    {name:26s} 占比 {bs['shares_pct'][name]:6.2f}%   "
                  f"相对中频 {bs['level_db_rel_mid'][name]:+6.2f} dB")

    print("\n  --- 倍频程电平（相对最强倍频程，dB）— 带限会在这里现出滚降台阶 ---")
    for label, key in (("语音帧", "psd_speech"), ("停顿帧", "psd_bg")):
        oct_ = octave_levels(freqs, sp[key], rate)
        row = "  ".join(f"{int(c) if c >= 1 else c}:{v:+.0f}" for c, v in oct_)
        print(f"  {label}: {row}")

    span = dsp.detect_active_span(mono, rate)
    print("\n  --- 有效内容 ---")
    print(f"  有效区间 {span.start_s:.2f}s–{span.end_s:.2f}s（总长 {span.total_s:.2f}s，"
          f"可用 {span.active_s:.2f}s，阈值 {span.threshold_dbfs:.1f} dBFS）")
    print(f"  诊断总用时 {time.perf_counter() - t0:.1f}s")

    return {
        "path": path, "size": os.path.getsize(path), "container": info.container,
        "kind": info.kind, "duration_s": round(info.duration, 2),
        "video_streams": len(info.video_streams), "audio_streams": len(info.audio_streams),
        "sample_rate": rate, "channels": data.channels, "stats": stats,
        "decay": decay, "decay_band": decay_band, "freqs": freqs, "sp": sp,
        "corr": corr, "span": span,
    }


def render_report(results: list[dict]) -> str:
    out = ["# 真实素材客观诊断", "",
           f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}",
           "工具：`tools/diagnose_audio.py`（只读输入，绝不改写素材）", "",
           "> 读法：**语音帧**那组看音色；**停顿帧**那组看房间/底噪；",
           "> 门限若失效（语音帧占比接近 100%），只看倍频程表与频谱边缘。", ""]
    for r in results:
        st, sp = r["stats"], r["sp"]
        corr_txt = "—" if r["corr"] is None else f"{r['corr']:.4f}"
        out += [
            f"## `{os.path.basename(r['path'])}`", "",
            f"- 大小 {r['size']:,} B · 容器 {r['container']} · 类型 {r['kind']}"
            f" · 时长 {r['duration_s']:.1f}s",
            f"- 流：视频 {r['video_streams']} 条 / 音轨 {r['audio_streams']} 条"
            f" · 分析音轨 {r['sample_rate']} Hz / {r['channels']}ch",
            "", "### 电平与活动度（实测）", "",
            "| 指标 | 值 |", "|---|---|",
            f"| 峰值 | {st.peak:.4f}（{st.peak_dbfs:+.1f} dBFS） |",
            f"| RMS | {st.rms:.5f}（{st.rms_dbfs:+.1f} dBFS） |",
            f"| 静音帧占比 | {st.silence_ratio * 100:.1f}% |",
            f"| 削顶样本占比 | {st.clipped_ratio * 100:.4f}% |",
            f"| 直流偏移 | {st.dc_offset:+.2e} |",
            f"| 帧电平 p10/p50/p90 | {st.frame_db_p10:.1f} / {st.frame_db_p50:.1f}"
            f" / {st.frame_db_p90:.1f} dBFS |",
            f"| 动态范围（p90−p10） | {st.frame_db_p90 - st.frame_db_p10:.1f} dB |",
            f"| 电平判断 | {st.verdict()} |",
            f"| 立体声相关性 | {corr_txt} |",
            "", "### 混响（实测）", "",
            f"- 全带衰减斜率：{r['decay'].get('slope_db_per_s')} dB/s"
            f"（p25 {r['decay'].get('p25')} / p75 {r['decay'].get('p75')}，"
            f"{r['decay'].get('num_segments')} 段）→ "
            f"{enhance.reverb_verdict(r['decay'])}",
            f"- **400–1k 带限衰减斜率**：{r['decay_band'].get('slope_db_per_s')} dB/s"
            f"（{r['decay_band'].get('num_segments')} 段）→ "
            f"{enhance.reverb_verdict(r['decay_band'])}",
        ]
        if (r["decay"].get("slope_db_per_s") and
                r["decay_band"].get("slope_db_per_s")):
            gap = r["decay_band"]["slope_db_per_s"] - r["decay"]["slope_db_per_s"]
            out.append(
                f"- 带限比全带快 **{gap:+.1f} dB/s** —— "
                + ("差值大，说明全带读数主要被稳态噪声底拖慢，"
                   "房间比全带读数显示的更干" if gap > 8.0
                   else "差值不大，说明混响确实存在，不是噪声底造成的假象"))
        out.append("- 口径：全带读数用整段帧电平拟合，**会被稳态底噪拖慢**"
                   "（底噪把尾巴托住 → 显得更湿）；带限读数取 400–1000 Hz，"
                   "该带信噪比最高（实测语音 ≈ −31 dBFS、噪声 ≈ −67 dBFS）。")
        out += [
            "", "### 频谱（实测）", "",
            f"门限 {sp['gate_db']} dBFS（帧电平 p95 {sp['frame_p95_db']}）→ "
            f"语音帧 {sp['speech_frame_pct']}% / 停顿帧 "
            f"{round(100 * sp['frames_bg'] / max(sp['frames_total'], 1), 1)}%"
            + (f"<br>⚠ {sp['note']}" if sp.get("note") else ""),
            "",
        ]
        for key, label in (("psd_speech", "语音帧（看音色）"),
                           ("psd_bg", "停顿帧（看房间/底噪）")):
            bs = band_stats(r["freqs"], sp[key], r["sample_rate"])
            lo, hi = spectrum_edges(r["freqs"], sp[key])
            if lo is not None:
                edge = f"−20dB 边缘 {lo:.0f}–{hi:.0f} Hz"
            else:
                edge = "（这一组没有可测内容）"
            out += [f"**{label}**：谱质心 **{bs['centroid_hz']:.0f} Hz** · "
                    f"发闷指数 **{bs['muddiness_db']:+.2f} dB** · {edge}", "",
                    "| 频段 | 能量占比 | 相对中频 |", "|---|---|---|"]
            for name, _l, _h in BANDS:
                out.append(f"| {name} | {bs['shares_pct'][name]:.2f}% |"
                           f" {bs['level_db_rel_mid'][name]:+.2f} dB |")
            out.append("")
        out += ["**倍频程电平**（相对最强倍频程，dB；带限表现为滚降台阶）", "",
                "| 帧组 | " + " | ".join(
                    f"{int(c) if c >= 1 else c}Hz" for c in OCTAVE_CENTERS) + " |",
                "|---" * (len(OCTAVE_CENTERS) + 1) + "|"]
        for label, key in (("语音帧", "psd_speech"), ("停顿帧", "psd_bg")):
            vals = dict(octave_levels(r["freqs"], sp[key], r["sample_rate"]))
            if not vals:
                out.append(f"| {label} | " + " | ".join(
                    "—" for _ in OCTAVE_CENTERS) + " |")
                continue
            cells = " | ".join(f"{vals[c]:+.0f}" if c in vals else "—"
                               for c in OCTAVE_CENTERS)
            out.append(f"| {label} | {cells} |")
        out += ["",
                f"### 有效内容（实测）", "",
                f"- 有效区间 {r['span'].start_s:.2f}s–{r['span'].end_s:.2f}s"
                f"（总长 {r['span'].total_s:.2f}s，可用 {r['span'].active_s:.2f}s）", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="真实素材客观诊断（只读）")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    missing = [f for f in args.files if not os.path.isfile(f)]
    if missing:
        for f in missing:
            print(f"[FAIL] 文件不存在：{f}")
        return 2

    results = []
    for path in args.files:
        try:
            results.append(diagnose(path))
        except Exception as exc:  # noqa: BLE001 — 单个文件失败不应中断其余
            print(f"[错误] {os.path.basename(path)}：{type(exc).__name__}: {exc}")

    if args.report and results:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(render_report(results))
        print(f"\n报告已写入：{args.report}")

    print("\n" + "=" * 78)
    print(f"诊断完成：{len(results)}/{len(args.files)} 个文件")
    return 0 if len(results) == len(args.files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
