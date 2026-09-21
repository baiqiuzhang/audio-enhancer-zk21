"""处理流水线（纯逻辑，不依赖 Qt，便于命令行测试）。

流程：
    decode → [可选] 静音裁剪 → [可选] 增益归一化 → 逐声道增强
           → 重采样回目标采样率 → 写出

两点设计取舍：

1. **响度匹配不在流水线里做。** 引擎去噪后 RMS 本来就会下降，若在此处强行对齐到原音
   响度，等于把残余噪声一起抬起来。A/B 的「增益匹配」是**监听侧**的补偿（见 ui/player.py），
   不污染输出文件。
2. **逐声道处理以保留立体声。** 增强模型都是单声道训练的，但把同一模型分别作用到左右
   声道是标准做法，能保住声像；比强行降混成单声道更符合直觉。单声道素材没有差别。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import dsp, enhance, media_io, outputs, wpe
from .engines import (
    HEAVY_STAGE,
    Cancelled,
    EngineParams,
    build_engine,
    engine_for,
    engine_kind,
    kind_of,
)
from .modelstore import Catalog

ProgressCb = Callable[[float, str], None]
CancelCb = Callable[[], bool]


@dataclass
class PipelineOptions:
    engine_id: str
    engine: EngineParams = field(default_factory=EngineParams)

    # --- 前处理 -------------------------------------------------------
    trim_silence: bool = False
    trim_dynamic_range_db: float = 40.0
    trim_pad_ms: float = 150.0
    # 高通（所有引擎共用的一个阶段；0 = 关闭）
    highpass_hz: float = 80.0

    # --- 后处理：带宽扩展（修复类，会改写内容；None = 关闭）-----------
    bwe_id: str | None = None
    bwe_cutoff_hz: float = 8000.0

    # --- 后处理：听感增强（修饰，不改写内容）-------------------------
    shelf_enabled: bool = False
    # 搁架中心默认 3000 Hz 而不是 4000：实测（真实素材，18 分钟 mp4，
    # 300s 起 90s 片段）中心 3000 Hz 时发闷指数改善 +7.21 dB、presence 提升
    # +5.60 dB；中心 4000 Hz 只有 +5.54 / +3.97 dB——因为亏空在 2–5 kHz，
    # 4 kHz 的搁架把 2–3 kHz 放过了。见 .zcode/diagnosis/。
    shelf_hz: float = 3000.0
    shelf_db: float = 3.0
    # 去闷：中低频**钟形**衰减（200–500 Hz 是实测的"发闷"堆积区；0 dB = 关闭）。
    # 不用搁架是因为发闷是一段带宽而不是"某个频率以上"——搁架会把 100 Hz 以下的
    # 基频也一起砍掉，声音会变薄。实测标定见 .zcode/diagnosis/。
    mud_cut_hz: float = 350.0
    mud_cut_db: float = 0.0
    mud_cut_width_oct: float = 0.9
    compressor_enabled: bool = False
    comp_threshold_dbfs: float = -18.0
    comp_ratio: float = 2.0
    comp_attack_ms: float = 10.0
    comp_release_ms: float = 200.0
    comp_makeup_db: float = 0.0

    # --- 后处理：去混响（实验性，单声道下收益有限）-------------------
    dereverb_enabled: bool = False
    dereverb_taps: int = 10
    dereverb_delay: int = 3
    dereverb_iterations: int = 3

    # --- 电平与输出 ---------------------------------------------------
    normalize: bool = False
    normalize_target_dbfs: float = -3.0
    normalize_mode: str = "peak"          # peak | rms
    keep_channels: bool = True
    output_rate: int | None = None        # None = 保持源采样率
    output_ext: str = ".wav"
    output_path: str | None = None        # 显式指定输出文件；None 则按命名规范生成
    output_dir: str | None = None         # 覆盖输出目录；None = 输入目录下的 AudioEnhancer_zk21/
    write_output: bool = True             # False = 只在内存里出结果，不按命名规范写盘、也不查复用
    reuse_existing: bool = True           # 参数完全相同则复用已有文件，不重算


@dataclass
class PipelineResult:
    source_path: str
    source: media_io.AudioData           # 原音（已裁剪到同一区间，便于对齐 A/B）
    enhanced: media_io.AudioData         # 处理后
    engine_id: str
    engine_name: str
    source_stats: dsp.LevelStats
    enhanced_stats: dsp.LevelStats
    source_info: media_io.MediaInfo
    audio_stream_index: int
    trim: dsp.ActiveSpan | None
    internal_gaps: list[tuple[float, float]]
    gain_db: float
    elapsed: float
    output_path: str | None
    engine_out_rate: int
    notes: list[str] = field(default_factory=list)

    # --- 命名、登记与复用（Phase 0 起）---------------------------------
    output_dir: str | None = None
    fingerprint: str = ""
    variant_label: str = ""
    reused: bool = False                 # True = 直接复用了已有文件，没有重算
    stages: list[str] = field(default_factory=list)
    # --- 混响指标（Phase 5）--------------------------------------------
    reverb_slope_db_s: float | None = None

    @property
    def rtf(self) -> float:
        dur = self.source_stats.duration
        return self.elapsed / dur if dur else float("nan")


def align_to(enhanced: np.ndarray, source: np.ndarray) -> np.ndarray:
    """把 enhanced 对齐到 source 的长度。

    模型可能截掉末尾不足一帧的部分（sherpa 约一个 hop、DeepFilterNet 约 30 ms），
    补齐时用源信号对应尾部，避免在结尾留一段可听见的静音。
    """
    if len(enhanced) == len(source):
        return enhanced
    if len(enhanced) > len(source):
        return enhanced[:len(source)]
    pad = len(source) - len(enhanced)
    tail = source[len(enhanced):]
    if enhanced.ndim == 2 and tail.ndim == 1:
        tail = media_io.to_stereo(tail)
    elif enhanced.ndim == 1 and tail.ndim == 2:
        tail = media_io.to_mono(tail)
    return np.concatenate([enhanced, tail[:pad]])


def _tick(progress: ProgressCb | None, frac: float, msg: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, frac)), msg)


def _check(cancel: CancelCb | None) -> None:
    if cancel and cancel():
        raise Cancelled("已取消")


def _as_channels(samples: np.ndarray) -> list[np.ndarray]:
    """(frames, ch) 或 (frames,) → 单声道数组列表。"""
    if samples.ndim == 1:
        return [np.ascontiguousarray(samples, dtype=np.float32)]
    return [np.ascontiguousarray(samples[:, i], dtype=np.float32) for i in range(samples.shape[1])]


def _stack(channels: list[np.ndarray]) -> np.ndarray:
    n = min(len(c) for c in channels)
    if len(channels) == 1:
        return channels[0][:n]
    return np.ascontiguousarray(np.stack([c[:n] for c in channels], axis=1), dtype=np.float32)


def run(
    media_path: str,
    catalog: Catalog,
    options: PipelineOptions,
    *,
    stream_index: int | None = None,
    progress: ProgressCb | None = None,
    cancel: CancelCb | None = None,
    log: Callable[[str], None] | None = None,
) -> PipelineResult:
    """对媒体文件跑一遍完整流水线。视频文件只取其音轨。"""

    def note(msg: str) -> None:
        if log:
            log(msg)

    t_start = time.perf_counter()

    # --- 1. 探测与解码 -------------------------------------------------
    _tick(progress, 0.01, "探测媒体…")
    _check(cancel)
    info = media_io.probe(media_path)
    if info.is_video:
        resolved = stream_index if stream_index is not None else info.default_audio_index
        note(f"视频容器：只提取音轨 {resolved}，画面不参与处理。")
    else:
        resolved = stream_index if stream_index is not None else info.default_audio_index

    _tick(progress, 0.03, "解码音频…")
    _check(cancel)
    # 纯音频文件交给 soundfile（更快、更精确）；只有需要按容器选流时才传 stream_index，
    # 否则会把 wav/mp3 也推给 PyAV，白白损失速度与精度。
    decode_index = resolved if info.decoder == "pyav" else None
    decoded = media_io.decode(media_path, stream_index=decode_index)
    note(f"解码完成：{decoded.frames} 帧 @ {decoded.sample_rate} Hz，{decoded.channels} 声道。")

    sr = decoded.sample_rate
    original = decoded.samples

    # --- 2. 静音裁剪与增益归一化 --------------------------------------
    trim_span: dsp.ActiveSpan | None = None
    gaps: list[tuple[float, float]] = []
    source = original

    if options.trim_silence:
        _tick(progress, 0.06, "检测静音段…")
        _check(cancel)
        mono_for_detect = media_io.to_mono(original)
        trim_span = dsp.detect_active_span(
            mono_for_detect, sr,
            dynamic_range_db=options.trim_dynamic_range_db,
            pad_ms=options.trim_pad_ms,
        )
        gaps = dsp.count_internal_gaps(
            mono_for_detect, sr, dynamic_range_db=options.trim_dynamic_range_db
        )
        if trim_span.start_frame > 0 or trim_span.end_frame < len(mono_for_detect):
            note(
                f"裁剪静音：保留 {trim_span.start_s:.2f}s–{trim_span.end_s:.2f}s"
                f"（原长 {trim_span.total_s:.2f}s，阈值 {trim_span.threshold_dbfs:.1f} dBFS）"
            )
            source = np.ascontiguousarray(original[trim_span.start_frame:trim_span.end_frame])
        else:
            note("未检测到需要裁剪的首尾静音。")
        if gaps:
            total_gap = sum(b - a for a, b in gaps)
            note(f"另外发现 {len(gaps)} 段中间静音（合计 {total_gap:.1f}s）——默认保留，不改变时间轴。")

    source_stats = dsp.analyze(source, sr)

    # --- 3. 输出位置与复用判定 -----------------------------------------
    # 指纹只编码「输入 + 会影响结果的参数」，所以同参数第二次处理可以直接复用文件。
    # 判据是**文件名里的指纹**，不依赖 manifest —— manifest 被删也能正常工作。
    spec = catalog.engine(options.engine_id)
    if spec is None:
        raise ValueError(f"未知引擎：{options.engine_id}")
    eng_kind = engine_kind(spec)
    target_rate = int(options.output_rate or sr)
    project_root = os.path.dirname(catalog.models_root)

    fp = ""
    out_dir: str | None = None
    # write_output=False 时整段跳过：既不解析输出目录，也不算指纹，因此不查复用。
    # 显式给了 output_path 的情况不受影响——那是调用方明确要一个文件。
    if options.output_path is None and options.write_output:
        if options.output_dir:
            out_dir = options.output_dir
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir, fallback = outputs.output_dir_for(media_path, project_root)
            if fallback:
                note(f"[注意] {fallback}")

        fp = outputs.fingerprint(media_path, options, resolved, eng_kind)
        expected = os.path.join(
            out_dir, outputs.build_filename(media_path, options, fp, eng_kind)
        )
        if options.reuse_existing and os.path.isfile(expected):
            note(f"参数与已有结果完全一致，直接复用（不重算）：{os.path.basename(expected)}")
            _tick(progress, 0.5, "复用已有结果…")
            prev = media_io.decode(expected)
            # 复用分支**不重采样**：指纹相同即参数完全相同（指纹含 bwe.id 与 output_rate），
            # 这个文件本身就是本次要产出的东西。复用发生在带宽扩展阶段之前，此时
            # target_rate 还是源采样率，按它重采样会把上一轮写出的 48 kHz 带宽扩展结果
            # 降回 16 kHz —— 等于把带宽扩展白做，而且这个偏差会被记录下来一直复用下去。
            y = align_to(np.asarray(prev.samples, dtype=np.float32), source)
            elapsed = time.perf_counter() - t_start
            _tick(progress, 1.0, "已复用已有结果")
            return PipelineResult(
                source_path=media_path,
                source=media_io.AudioData(samples=source, sample_rate=sr),
                enhanced=media_io.AudioData(samples=y, sample_rate=prev.sample_rate),
                engine_id=options.engine_id,
                engine_name=spec.name,
                source_stats=source_stats,
                enhanced_stats=dsp.analyze(y, prev.sample_rate),
                source_info=info,
                audio_stream_index=resolved,
                trim=trim_span,
                internal_gaps=gaps,
                gain_db=0.0,
                elapsed=elapsed,
                output_path=expected,
                engine_out_rate=prev.sample_rate,
                notes=["已复用同参数的已有结果，未重新计算。"],
                output_dir=out_dir,
                fingerprint=fp,
                variant_label=outputs.display_label(options),
                reused=True,
                stages=["reuse"],
            )

    gain_pre = 0.0
    working = source
    stages: list[str] = []

    # 去混响放在最前面：WPE 是线性前处理，让后面的降噪去收拾它留下的痕迹，
    # 顺序反过来会更难处理。实验性、默认关 —— 单通道 WPE 会连带把语音变薄。
    if options.dereverb_enabled:
        # 实测（2026-09-21，合成混响素材，taps=10/delay=3/iterations=3）：
        #   · 偏移后 100–400 ms 的尾部能量相对起始**升高 15.7 dB**（应下降）
        #   · 输出与干声的样本级相关性 **−0.35**（负值 → 输出已与原始信号无关）
        #   · 衰减斜率反而变缓（90.8 → 68.4 dB/s）
        # 即实现无效且会产生垃圾音频。宁可明确拒绝，也不给一个会毁掉素材的开关。
        raise NotImplementedError(
            "去混响（WPE）已停用：实测实现无效——尾部能量反而升高 15.7 dB、"
            "输出与干声相关性为负（−0.35）。\n"
            "本轮只提供「混响衰减斜率」检测指标；混响抑制请依赖 DPDFNet"
            "（它的训练信号模型含房间冲激响应，本来就在压混响）。\n"
            "详见 core/wpe.py 顶部的说明与 README「已知限制」。"
        )

    # 高通：所有引擎共用的一个阶段（去低频轰鸣与直流），对识别也有利。
    # 注意 source 保持为裁剪后的**原始**信号 —— 它是 A/B 的"原音"参照。
    if options.highpass_hz and options.highpass_hz > 0:
        _tick(progress, 0.07, f"高通 {options.highpass_hz:g} Hz…")
        _check(cancel)
        working = enhance.spectral_shape(working, sr, hp_cutoff_hz=options.highpass_hz)
        stages.append("highpass")

    if options.normalize:
        _check(cancel)
        # 前置归一化是给模型的电平调理：素材过低（例如峰值只有 0.16）时，
        # 先把电平拉到正常范围再送进引擎，结果更稳定。
        working, gain_pre = dsp.normalize(
            working,
            target_dbfs=options.normalize_target_dbfs,
            mode=options.normalize_mode,
        )
        stages.append("normalize-pre")

    # --- 4. 引擎降噪 ---------------------------------------------------
    _tick(progress, 0.12, "准备引擎…")
    # 用线程局部实例：批处理的多个 worker 可能同时处理不同文件，而
    # sherpa 的 denoiser 对象不可跨线程共享。
    engine = engine_for(spec, catalog)
    if not engine.ready():
        raise FileNotFoundError(
            f"引擎「{spec.name}」所需的模型文件尚未就绪，请先在模型列表里下载。"
        )

    channels = _as_channels(working) if options.keep_channels else [media_io.to_mono(working)]
    if options.keep_channels and len(channels) > 1:
        note(f"逐声道处理 {len(channels)} 个声道（保留声像）。")

    engine_frac = 0.76          # 0.12 → 0.88 留给引擎
    outs: list[np.ndarray] = []
    engine_out_rate = sr
    for i, ch in enumerate(channels):
        _check(cancel)
        base = 0.12 + engine_frac * (i / len(channels))

        def sub_progress(f: float, msg: str, _base=base, _i=i) -> None:
            span = engine_frac / len(channels)
            prefix = f"[声道 {_i + 1}/{len(channels)}] " if len(channels) > 1 else ""
            _tick(progress, _base + span * f, prefix + msg)

        y, engine_out_rate = engine.process(
            ch, sr, options.engine,
            progress=sub_progress,
            cancel=cancel,
        )
        outs.append(np.asarray(y, dtype=np.float32))

    enhanced = _stack(outs)
    cur_rate = engine_out_rate
    stages.append("denoise")

    # --- 4b. 后处理：带宽扩展（修复类，会改写内容）--------------------
    if options.bwe_id:
        _check(cancel)
        bwe_spec = catalog.engine(options.bwe_id)
        if bwe_spec is None:
            raise ValueError(f"未知的带宽扩展引擎：{options.bwe_id}")
        if bwe_spec.kind != "bwe":
            raise ValueError(f"引擎「{bwe_spec.name}」不是带宽扩展模型。")
        bwe_eng = engine_for(bwe_spec, catalog)
        if not bwe_eng.ready():
            raise FileNotFoundError(
                f"带宽扩展模型「{bwe_spec.name}」尚未就绪，请先在模型列表里下载。"
            )
        note(f"带宽扩展：{bwe_spec.name}　**生成式，会编造高频内容，不要用于识别**")

        # 重阶段互斥：带宽扩展的内存占用远高于降噪，多个 worker 同时跑极易 OOM
        with HEAVY_STAGE:
            bchans = _as_channels(enhanced) if options.keep_channels else [media_io.to_mono(enhanced)]
            b_frac = 0.62
            b_outs: list[np.ndarray] = []
            for bi, bch in enumerate(bchans):
                _check(cancel)

                def b_progress(f: float, msg: str, _i=bi) -> None:
                    span = (0.88 - b_frac) / len(bchans)
                    prefix = f"[声道 {_i + 1}/{len(bchans)}] " if len(bchans) > 1 else ""
                    _tick(progress, b_frac + span * f, prefix + msg)

                by, cur_rate = bwe_eng.process(
                    bch, cur_rate, options.engine,
                    progress=b_progress, cancel=cancel,
                )
                b_outs.append(np.asarray(by, dtype=np.float32))
            enhanced = _stack(b_outs)
        stages.append(f"bwe:{options.bwe_id}")
        # 带宽扩展的输出采样率就是要交付的采样率。默认目标率是**源采样率**，
        # 若不在这里覆盖，刚补出来的高频会在下一步被重新采样掉，等于白做
        # —— 实测踩过这个坑（输出还是 16 kHz、8 kHz 以上依然为空）。
        if not options.output_rate:
            target_rate = cur_rate
            note(f"输出采样率取带宽扩展的原生率：{target_rate} Hz")

    # --- 4c. 后处理：听感增强（修饰，不改写内容）----------------------
    # 搁架提升与去闷合成**同一条**增益曲线，一次 STFT/ISTFT 就同时做完，
    # 所以任一项非零就跑。shelf_enabled 只管搁架；去闷用「0 dB = 关闭」表示。
    eq_shelf = bool(options.shelf_enabled and options.shelf_db)
    if eq_shelf or options.mud_cut_db:
        _check(cancel)
        bits = []
        if eq_shelf:
            bits.append(f"搁架 {options.shelf_db:+g} dB @ {options.shelf_hz:g} Hz")
        if options.mud_cut_db:
            bits.append(f"去闷 {options.mud_cut_db:+g} dB @ {options.mud_cut_hz:g} Hz")
        _tick(progress, 0.89, "听感 EQ：" + " + ".join(bits) + "…")
        enhanced = enhance.spectral_shape(
            enhanced, cur_rate,
            shelf_hz=options.shelf_hz,
            shelf_db=options.shelf_db if eq_shelf else 0.0,
            mud_cut_hz=options.mud_cut_hz,
            mud_cut_db=options.mud_cut_db,
            mud_cut_width_oct=options.mud_cut_width_oct,
            progress=progress, cancel=cancel,
        )
        if eq_shelf:
            stages.append("shelf")
        if options.mud_cut_db:
            stages.append("mudcut")

    if options.compressor_enabled:
        _check(cancel)
        _tick(progress, 0.90, "温和压缩 / AGC…")
        enhanced, cstats = enhance.compress(
            enhanced, cur_rate,
            threshold_dbfs=options.comp_threshold_dbfs,
            ratio=options.comp_ratio,
            attack_ms=options.comp_attack_ms,
            release_ms=options.comp_release_ms,
            makeup_db=options.comp_makeup_db,
        )
        if cstats.get("applied"):
            note(
                f"压缩：最大衰减 {cstats['max_reduction_db']} dB，"
                f"中位 {cstats['median_reduction_db']} dB，"
                f"增益跳变 {cstats['gain_jump_max_db']} dB/帧"
            )
        stages.append("compress")

    # --- 5. 采样率对齐与写出 -------------------------------------------
    _check(cancel)
    if cur_rate != target_rate:
        _tick(progress, 0.91, f"重采样 {cur_rate} → {target_rate} Hz…")
        note(f"处理后为 {cur_rate} Hz，重采样回 {target_rate} Hz。")
        enhanced = media_io.resample(enhanced, cur_rate, target_rate)

    # 与原音对齐长度：引擎可能截掉末尾不足一帧的部分
    enhanced = align_to(enhanced, source)

    enhanced_stats = dsp.analyze(enhanced, target_rate)

    # 后置归一化：引擎会改变电平（降噪后 RMS 自然下降），所以「归一化」这个开关的契约
    # 必须是**输出**达到目标电平，而不是输入。这里再补一次以兑现该契约。
    gain_db = gain_pre
    if options.normalize:
        _tick(progress, 0.95, "输出电平归一化…")
        enhanced, gain_post = dsp.normalize(
            enhanced,
            target_dbfs=options.normalize_target_dbfs,
            mode=options.normalize_mode,
        )
        gain_db += gain_post
        enhanced_stats = dsp.analyze(enhanced, target_rate)
        note(
            f"增益归一化：输入调理 {gain_pre:+.2f} dB + 输出校正 {gain_post:+.2f} dB"
            f"（{options.normalize_mode} → {options.normalize_target_dbfs:.1f} dBFS）"
        )

    # --- 混响指标 -------------------------------------------------------
    # 只在衰减段测量，因此不受语音内容干扰。注意它会被**稳态噪声底**拖慢，
    # 所以它说明的是"衰减有多慢"，不等于"一定是混响"——降噪前后的对比能区分二者。
    reverb = {"slope_db_per_s": None, "num_segments": 0}
    try:
        _tick(progress, 0.93, "测量混响衰减…")
        reverb = enhance.estimate_decay_slope_db_per_s(media_io.to_mono(source), sr)
        note(enhance.reverb_verdict(reverb))
        after = enhance.estimate_decay_slope_db_per_s(media_io.to_mono(enhanced), target_rate)
        delta = enhance.decay_slope_delta(reverb, after)
        if delta:
            note(delta)
    except Exception as exc:                     # 指标失败不应影响主流程
        note(f"[注意] 混响指标计算失败（不影响结果）：{type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - t_start

    # --- 6. 写出与登记 -------------------------------------------------
    out_path = None
    if options.output_path:
        # 显式指定输出路径 = 导出语义：不查重、不登记
        _tick(progress, 0.97, "写出文件…")
        out_path = media_io.write_audio(options.output_path, enhanced, target_rate)
        note(f"已写出：{out_path}")
    elif out_dir:
        _tick(progress, 0.97, "写出文件…")
        out_path = os.path.join(
            out_dir, outputs.build_filename(media_path, options, fp, eng_kind)
        )
        media_io.write_audio(out_path, enhanced, target_rate)
        note(f"已写出：{out_path}")
        try:
            outputs.register(out_dir, outputs.VariantEntry(
                fp=fp,
                output=os.path.basename(out_path),
                input_name=os.path.basename(media_path),
                input_size=os.path.getsize(media_path),
                input_mtime=os.path.getmtime(media_path),
                stream_index=resolved,
                chain_label=outputs.display_label(options),
                params=outputs.spec_dict(media_path, options, resolved, eng_kind),
                stats={
                    "duration_s": round(enhanced_stats.duration, 3),
                    "sample_rate": target_rate,
                    "peak_dbfs": round(enhanced_stats.peak_dbfs, 2),
                    "rms_dbfs": round(enhanced_stats.rms_dbfs, 2),
                    "source_peak_dbfs": round(source_stats.peak_dbfs, 2),
                    "source_rms_dbfs": round(source_stats.rms_dbfs, 2),
                    "elapsed_s": round(elapsed, 2),
                    "rtf": round(elapsed / source_stats.duration, 4)
                    if source_stats.duration else None,
                },
                trim_span=(trim_span.start_s, trim_span.end_s) if trim_span else None,
            ))
        except OSError as exc:
            note(f"[注意] 结果文件已写出，但登记 manifest 失败：{exc}")

    _tick(progress, 1.0, f"完成（{elapsed:.1f}s）")

    return PipelineResult(
        source_path=media_path,
        source=media_io.AudioData(samples=source, sample_rate=sr),
        enhanced=media_io.AudioData(samples=enhanced, sample_rate=target_rate),
        engine_id=options.engine_id,
        engine_name=spec.name,
        source_stats=source_stats,
        enhanced_stats=enhanced_stats,
        source_info=info,
        audio_stream_index=resolved,
        trim=trim_span,
        internal_gaps=gaps,
        gain_db=gain_db,
        elapsed=elapsed,
        output_path=out_path,
        engine_out_rate=engine_out_rate,
        output_dir=out_dir,
        fingerprint=fp,
        variant_label=outputs.display_label(options),
        stages=stages,
        reverb_slope_db_s=reverb.get("slope_db_per_s"),
    )


def default_output_path(source_path: str, engine_id: str, project_root: str, ext: str = ".wav") -> str:
    stem = os.path.splitext(os.path.basename(source_path))[0]
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem).strip() or "output"
    out_dir = os.path.join(project_root, "out")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{safe}_{engine_id}{ext}")
