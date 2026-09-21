"""增强链验证：证明"这次真的加了内容"，而不是只有降噪。

判据设计成与"降噪"相反的方向，所以无法用"少了点噪声"蒙混过去：
- 降噪的特征是**所有频段能量下降、谱质心下移**
- 带宽扩展的特征是**原来没有的 8 kHz 以上出现内容、谱质心上移**

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_enhance.py
    .venv\\Scripts\\python.exe tests\\check_enhance.py --skip-bwe   # 跳过较慢的带宽扩展
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import enhance, media_io, modelstore, outputs, pipeline, wpe  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "speech_with_noise.wav")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def section(t: str) -> None:
    print()
    print(t)
    print("=" * len(t))


# --- 测量工具：功率谱密度，使不同长度/采样率可比 ---------------------------
_TRAP = getattr(np, "trapezoid", None) or np.trapz


def psd(x: np.ndarray, sr: int, nfft: int = 8192):
    step = nfft // 2
    w = np.hanning(nfft)
    norm = np.sum(w**2)
    fs, Ps = [], []
    for i in range(0, max(0, len(x) - nfft), step):
        X = np.fft.rfft(x[i:i + nfft] * w)
        fs.append(np.fft.rfftfreq(nfft, 1 / sr))
        Ps.append(np.abs(X) ** 2 / norm)
    if not fs:
        return np.fft.rfftfreq(nfft, 1 / sr), np.zeros(nfft // 2 + 1)
    return np.mean(fs, axis=0), np.mean(Ps, axis=0)


def band_db(f, P, lo, hi) -> float:
    m = (f >= lo) & (f < hi)
    return float(10 * np.log10(_TRAP(P[m], f[m]) + 1e-30))


def tilt(x, sr, bands=((500, 2000), (2000, 4000), (4000, 8000), (8000, 16000))) -> dict:
    f, P = psd(x, sr)
    ref = band_db(f, P, *bands[0])
    return {f"{lo//1000}-{hi//1000}k": round(band_db(f, P, lo, hi) - ref, 1)
            for lo, hi in bands}


def centroid(x, sr) -> float:
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(np.sum(f * X) / max(np.sum(X), 1e-20))


def aligned_corr(a: np.ndarray, b: np.ndarray, sr: int) -> tuple[float, int]:
    """按互相关对齐后再算相关性——时延会让样本级相关性失去意义。"""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ml = int(sr * 0.05)
    cc = np.correlate(b, a, mode="full")
    mid = len(a) - 1
    lag = int(np.argmax(cc[mid - ml:mid + ml])) - ml
    p, q = (a[:n - lag], b[lag:n]) if lag >= 0 else (a[-lag:], b[:n + lag])
    m = min(len(p), len(q))
    return float(np.corrcoef(p[:m], q[:m])[0, 1]), lag


def lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """砖墙低通：用来把"低频是否被保留"从全带里隔离出来判。"""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    X = X.copy()
    X[f > cutoff] = 0
    return np.fft.irfft(X, n=len(x)).astype(np.float32)


def make_opts(**kw) -> pipeline.PipelineOptions:
    o = pipeline.PipelineOptions(engine_id="dpdfnet2", **kw)
    o.engine.num_threads = 2
    return o


# ---------------------------------------------------------------------------

def check_perceptual() -> None:
    section("1. 听感增强：搁架 EQ 与压缩（纯 numpy，无新依赖）")

    sr = 16000
    t = np.arange(int(sr * 4)) / sr
    rng = np.random.default_rng(1)
    sig = (0.3 * np.sin(2 * np.pi * 250 * t)
           + 0.10 * np.sin(2 * np.pi * 5500 * t)
           + rng.standard_normal(len(t)) * 0.01).astype(np.float32)

    y = enhance.spectral_shape(sig, sr, hp_cutoff_hz=0, shelf_hz=4000, shelf_db=3.0)
    d_high = band_db(*psd(y, sr), 5000, 8000) - band_db(*psd(sig, sr), 5000, 8000)
    d_low = band_db(*psd(y, sr), 100, 500) - band_db(*psd(sig, sr), 100, 500)
    check("搁架提升量接近设定值 ±1 dB", abs(d_high - 3.0) <= 1.0, f"{d_high:+.2f} dB")
    check("低频基本不动（±0.5 dB）", abs(d_low) <= 0.5, f"{d_low:+.2f} dB")
    check("谱质心上移（提亮的判据）", centroid(y, sr) > centroid(sig, sr),
          f"{centroid(sig,sr):.0f} → {centroid(y,sr):.0f} Hz")

    y2 = enhance.spectral_shape(sig, sr, hp_cutoff_hz=80, shelf_db=0)
    d_hp = band_db(*psd(y2, sr), 10, 60) - band_db(*psd(sig, sr), 10, 60)
    d_pass = band_db(*psd(y2, sr), 1000, 3000) - band_db(*psd(sig, sr), 1000, 3000)
    check("高通压低截止频率以下的能量", d_hp < -6.0, f"{d_hp:+.2f} dB")
    check("高通不影响通带", abs(d_pass) < 0.5, f"{d_pass:+.2f} dB")

    # 压缩：用真的超过阈值的信号
    bursts = []
    for _ in range(12):
        n = int(sr * 0.25)
        tt = np.arange(n) / sr
        seg = (0.4 * np.sin(2 * np.pi * 130 * tt) + 0.12 * np.sin(2 * np.pi * 390 * tt))
        bursts.append((seg * np.hanning(n) ** 0.3).astype(np.float32))
        bursts.append(np.zeros(int(sr * 0.25), np.float32))
    dry = np.concatenate(bursts) * 1.6

    yc, st = enhance.compress(dry, sr, threshold_dbfs=-18, ratio=2.0)
    check("压缩已生效", st.get("applied") is True, str(st))

    def short_rms(x, sr, ms=50):
        win = int(sr * ms / 1000)
        fr = np.abs(x[:len(x) // win * win]).reshape(-1, win)
        return 20 * np.log10(np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1)) + 1e-12)

    a, b = short_rms(dry, sr), short_rms(yc, sr)
    mask = a > -60
    check("语音段动态范围收窄", (b[mask].max() - b[mask].min()) < (a[mask].max() - a[mask].min()),
          f"{a[mask].max()-a[mask].min():.1f} → {b[mask].max()-b[mask].min():.1f} dB")
    check("增益曲线无突变（无抽吸）", st["gain_jump_max_db"] < 2.0,
          f"{st['gain_jump_max_db']} dB/帧")
    check("不产生削顶", float(np.max(np.abs(yc))) <= 1.0, f"{float(np.max(np.abs(yc))):.3f}")


def check_reverb_indicator() -> None:
    section("2. 混响指标：只在衰减段测量，干/湿可区分")

    sr = 16000
    rng = np.random.default_rng(3)
    bursts = []
    for _ in range(12):
        n = int(sr * 0.25)
        tt = np.arange(n) / sr
        f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * tt)
        seg = 0.4 * np.sin(2 * np.pi * f0 * tt) + 0.12 * np.sin(2 * np.pi * 3 * f0 * tt)
        bursts.append((seg * np.hanning(n) ** 0.3).astype(np.float32))
        bursts.append(np.zeros(int(sr * 0.25), np.float32))
    dry = np.concatenate(bursts)

    rir = np.zeros(int(sr * 0.8))
    rir[0] = 1.0
    rir[1:] = rng.standard_normal(len(rir) - 1) * np.exp(-np.arange(1, len(rir)) / (sr * 0.10))
    rir /= np.sqrt(np.sum(rir**2))
    wet = np.convolve(dry, rir)[:len(dry)].astype(np.float32)
    wet *= abs(dry).max() / max(abs(wet).max(), 1e-9)

    r_dry = enhance.estimate_decay_slope_db_per_s(dry, sr)
    r_wet = enhance.estimate_decay_slope_db_per_s(wet, sr)
    print(f"  干声 {enhance.reverb_verdict(r_dry)}")
    print(f"  混响 {enhance.reverb_verdict(r_wet)}")

    check("干声能测出衰减段", r_dry["slope_db_per_s"] is not None)
    check("混响能测出衰减段", r_wet["slope_db_per_s"] is not None)
    if r_dry["slope_db_per_s"] and r_wet["slope_db_per_s"]:
        check("干声的衰减明显快于混响（3 倍以上）",
              r_dry["slope_db_per_s"] > r_wet["slope_db_per_s"] * 3,
              f"{r_dry['slope_db_per_s']} vs {r_wet['slope_db_per_s']} dB/s")
        check("能区分出两个档位（不是同一档文案）",
              enhance.reverb_verdict(r_dry) != enhance.reverb_verdict(r_wet))


def check_wpe() -> None:
    section("3. 去混响：WPE 实测无效，流水线必须明确拒绝启用")

    # 这里**不是**在验证 WPE 有效（它无效），而是在锁定"不会放出坏开关"这个行为：
    # 实测该实现会把尾部能量抬高 15.7 dB、输出与干声相关性为负，
    # 所以正确做法是硬拒绝，而不是提供一个能毁掉素材的选项。
    with tempfile.TemporaryDirectory(prefix="ae_wpe_") as tmp:
        inp = os.path.join(tmp, "x.wav")
        shutil.copy2(SAMPLE, inp)
        cat = modelstore.load_catalog(ROOT)
        o = make_opts(dereverb_enabled=True, output_dir=os.path.join(tmp, "out"))
        try:
            pipeline.run(inp, cat, o)
            check("启用去混响时被明确拒绝", False, "居然没有拒绝")
        except NotImplementedError as exc:
            msg = str(exc)
            check("启用去混响时被明确拒绝", "停用" in msg or "无效" in msg, msg.splitlines()[0][:60])
            check("拒绝理由里给出了实测数字", "15.7" in msg)
            check("拒绝理由里给出了替代方案", "DPDFNet" in msg)
        except Exception as exc:
            check("启用去混响时被明确拒绝", False, f"抛的是 {type(exc).__name__}")

    # wpe 模块本身仍应可导入且不再产生 NaN（保留为将来实现起点）
    sr = 16000
    rng = np.random.default_rng(5)
    x = (0.3 * np.sin(2 * np.pi * 200 * np.arange(sr * 2) / sr)).astype(np.float32)
    y = wpe.dereverb(x, sr, taps=6, delay=2, iterations=2)
    check("wpe 模块可导入、输出长度一致且为有限值",
          len(y) == len(x) and bool(np.isfinite(y).all()), f"{len(y)} 帧")


def check_pipeline_stages(cat, skip_bwe: bool) -> None:
    section("4. 流水线：阶段顺序、增强签名、以及「没增强」时确实不动内容")

    with tempfile.TemporaryDirectory(prefix="ae_enh_") as tmp:
        inp = os.path.join(tmp, "采访.wav")
        shutil.copy2(SAMPLE, inp)

        # (a) 纯降噪（识别用）
        r_dn = pipeline.run(inp, cat, make_opts(
            trim_silence=True, highpass_hz=80, normalize=True,
            output_dir=os.path.join(tmp, "dn")))
        check("纯降噪：阶段列表里没有增强项",
              not any(s.startswith(("bwe", "shelf", "compress", "wpe")) for s in r_dn.stages),
              str(r_dn.stages))
        check("纯降噪：输出保持源采样率", r_dn.enhanced.sample_rate == 16000,
              f"{r_dn.enhanced.sample_rate} Hz")
        x = media_io.to_mono(r_dn.source.samples)
        y_dn = media_io.to_mono(r_dn.enhanced.samples)
        t_in, t_dn = tilt(x, 16000), tilt(y_dn, 16000)
        # 「有没有凭空造内容」的真正判据是**输入奈奎斯特频率以上**有没有出现能量。
        # 输入是 16 kHz，所以看 8-16 kHz：降噪后必须仍然空着。
        check("纯降噪：8 kHz 以上仍然没有内容（没造东西出来）",
              t_dn["8-16k"] < -100, f"{t_dn['8-16k']} dB")
        # 注意**不能**断言"质心不上移"：DPDFNet 在噪声集中的 1-2 kHz 压得最多，
        # 于是 2-4 kHz 的相对比重会上升、质心自然上移。那是模型特性，不是增强。
        check("纯降噪：4-8 kHz 没有明显抬升（±1 dB 内）",
              t_dn["4-8k"] <= t_in["4-8k"] + 1.0,
              f"{t_in['4-8k']} → {t_dn['4-8k']} dB")

        # (b) 听感增强（EQ + 压缩，不带宽带扩展）
        r_eq = pipeline.run(inp, cat, make_opts(
            trim_silence=True, highpass_hz=80, normalize=True,
            shelf_enabled=True, shelf_hz=4000, shelf_db=3.0,
            compressor_enabled=True,
            output_dir=os.path.join(tmp, "eq")))
        check("听感增强：阶段列表含 shelf + compress",
              "shelf" in r_eq.stages and "compress" in r_eq.stages, str(r_eq.stages))
        y_eq = media_io.to_mono(r_eq.enhanced.samples)
        t_eq = tilt(y_eq, 16000)
        check("听感增强：高频段能量上升",
              t_eq["4-8k"] > t_dn["4-8k"] + 1.0,
              f"{t_dn['4-8k']} → {t_eq['4-8k']} dB")
        check("听感增强：谱质心上移",
              centroid(y_eq, 16000) > centroid(y_dn, 16000),
              f"{centroid(y_dn,16000):.0f} → {centroid(y_eq,16000):.0f} Hz")

        if skip_bwe:
            print("  [跳过] 带宽扩展（--skip-bwe）")
            return

        # (c) 带宽扩展：这里是最关键的"真增强"判据
        for bwe_id in ("novasr", "hifiganbwe", "lavasr"):
            spec = cat.engine(bwe_id)
            if spec is None or not all(
                os.path.isfile(cat.resolve(f.path)) for f in spec.files
            ):
                print(f"  [跳过] {bwe_id}：权重未就绪")
                continue
            r_bwe = pipeline.run(inp, cat, make_opts(
                trim_silence=True, highpass_hz=80, normalize=True, bwe_id=bwe_id,
                output_dir=os.path.join(tmp, f"bwe_{bwe_id}")))
            y_b = media_io.to_mono(r_bwe.enhanced.samples)
            rate = r_bwe.enhanced.sample_rate
            check(f"{bwe_id}：输出为 48 kHz", rate == 48000, f"{rate} Hz")

            t_b = tilt(y_b, rate, bands=((500, 2000), (2000, 4000), (4000, 8000), (8000, 16000)))
            f_b, P_b = psd(y_b, rate)
            above = band_db(f_b, P_b, 8000, 16000) - band_db(f_b, P_b, 500, 2000)
            check(f"{bwe_id}：8 kHz 以上出现了内容（输入本来完全没有）",
                  above > -60, f"相对通带 {above:.1f} dB")
            check(f"{bwe_id}：谱质心明显上移",
                  centroid(y_b, rate) > centroid(x, 16000) * 1.1,
                  f"{centroid(x,16000):.0f} → {centroid(y_b,rate):.0f} Hz")

            # 「低频是否被保留」必须**把低频隔离出来**再判：
            # lavasr 生成的高频比通带高约 19 dB，会主导全带相关性
            # （全带 0.19 vs 低通后 0.79，实测）——用全带相关性会得出错误结论。
            # 参照也不是原始信号，而是"同一套参数但不做带宽扩展"的输出，
            # 这样才能把带宽扩展对低频的影响单独分离出来。
            r_dn48 = pipeline.run(inp, cat, make_opts(
                trim_silence=True, highpass_hz=80, normalize=True,
                output_rate=rate, output_dir=os.path.join(tmp, f"ref_{bwe_id}")))
            ref = media_io.to_mono(r_dn48.enhanced.samples)
            corr, lag = aligned_corr(lowpass(ref, rate, 5000), lowpass(y_b, rate, 5000), rate)
            check(f"{bwe_id}：低频被保留（<5 kHz 与「仅降噪」高度一致）", corr >= 0.95,
                  f"相关性 {corr:.4f}（时延 {lag} 帧）")

            # 顺带记录高频相对通带的电平。注意：**修复前**三个引擎的生成频段都
            # 远高于通带（lavasr 在 12–20 kHz 甚至高 +89 dB 且自身已削顶），
            # 听感就是"只剩高频"；现在 core/bwe.py 会把它压到通带之下
            # （HF_TARGET_DB，默认 −12 dB），所以这里的期望值已经反过来。
            f_hi, P_hi = psd(y_b, rate)
            hi = band_db(f_hi, P_hi, 8000, 16000) - band_db(f_hi, P_hi, 500, 2000)
            print(f"    （参考）8 kHz 以上相对通带 {hi:+.1f} dB，"
                  f"谱质心 {centroid(y_b, rate):.0f} Hz")

        # (d) 立体声 + 带宽扩展：这条组合**曾经一个套件都没覆盖**，结果上线了
        #     「听感只剩高频、而且有一个声道是坏的」的缺陷。这里用不变式守住：
        #     ① 同内容的双声道输入 → 输出两声道必须**逐样本一致**（任何"一路坏"
        #        都会立刻违反它）；② 时长必须不变（这条抓的是 align_to 曾按样本数
        #        比较、把 48 kHz 的扩展输出截成 1/3 时长那个静默数据丢失）；
        #     ③ 生成频段必须被压到通带之下（直接对应"只剩高频"这个症状）。
        section("4b. 立体声 + 带宽扩展（曾经的覆盖盲区）")
        src_audio = media_io.decode(inp)
        dual = np.stack([media_io.to_mono(src_audio.samples)] * 2, axis=1)
        dual_path = os.path.join(tmp, "dual_mono.wav")
        media_io.write_audio(dual_path, dual, src_audio.sample_rate)
        dur_in = len(dual) / float(src_audio.sample_rate)

        for bwe_id in ("novasr", "lavasr"):
            spec = cat.engine(bwe_id)
            if spec is None or not all(
                os.path.isfile(cat.resolve(f.path)) for f in spec.files
            ):
                print(f"  [跳过] {bwe_id}：权重未就绪")
                continue
            r_st = pipeline.run(dual_path, cat, make_opts(
                trim_silence=False, highpass_hz=80, normalize=False, bwe_id=bwe_id,
                keep_channels=True, output_dir=os.path.join(tmp, f"st_{bwe_id}")))
            y_st = np.asarray(r_st.enhanced.samples)
            rate_st = r_st.enhanced.sample_rate
            check(f"[立体声] {bwe_id}：输出保持双声道",
                  y_st.ndim == 2 and y_st.shape[1] == 2, str(y_st.shape))
            if y_st.ndim == 2 and y_st.shape[1] == 2:
                dmax = float(np.max(np.abs(y_st[:, 0] - y_st[:, 1])))
                check(f"[立体声] {bwe_id}：同内容双声道 → 输出两声道逐样本一致",
                      dmax < 1e-6, f"最大逐样本差 {dmax:.3e}")
            dur_out = len(y_st) / float(rate_st)
            check(f"[立体声] {bwe_id}：时长不变（带宽扩展只改采样率、不改时长）",
                  abs(dur_out - dur_in) < 0.05,
                  f"{dur_in:.3f}s@{src_audio.sample_rate} → {dur_out:.3f}s@{rate_st}")
            f_s, P_s = psd(np.asarray(y_st[:, 0], dtype=np.float32), rate_st)
            hi_st = band_db(f_s, P_s, 8000, 16000) - band_db(f_s, P_s, 500, 2000)
            check(f"[立体声] {bwe_id}：生成频段被压到通带之下（不再「只剩高频」）",
                  hi_st < 0.0, f"8k+ 相对通带 {hi_st:+.1f} dB（目标 ≈ −12）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bwe", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(SAMPLE):
        print(f"[FAIL] 找不到测试音频：{SAMPLE}")
        return 2
    cat = modelstore.load_catalog(ROOT)
    print(f"项目根：{ROOT}")

    check_perceptual()
    check_reverb_indicator()
    check_wpe()
    check_pipeline_stages(cat, args.skip_bwe)

    print()
    if failures:
        print(f"增强链验证结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("增强链验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
