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

# 逐声道合并 / 对齐时允许的长度差（帧）。超过就抛错而不是静默截断：
# 声道之间长度不一致一定是缺陷（曾经发生过"第二路只有第一路 1/3 长"，而
# _stack 静默截断 + align_to 用原音补尾把它掩盖成了"后半段是原音"）。
# 取 4800 帧（48 kHz 下 100 ms）是为了留足余量容纳模型固有的帧对齐截断——
# DeepFilterNet 的延迟补偿会丢末尾约 30 ms（48 kHz 下 1440 帧），必须容得下；
# 同时远小于任何真实缺陷（那次是 1,000,000 帧量级）。
LENGTH_TOLERANCE = 4800


def _effective_rate(in_rate: int, n_in: int, n_out: int,
                    reported: int | float | None = None,
                    tol: float = 0.02) -> int:
    """确定引擎的**真实输出采样率**：以长度比为准，但只在必要时覆盖上报值。

    实测（2026-09-21）：sherpa 的 DPDFNet 喂 48 kHz 时**上报 16000 Hz，却返回与
    输入相同样本数**的 48 kHz 数据——灌 1 kHz 正弦进去，按 48000 Hz 回读仍是
    1000.1 Hz，长度比 1.0000。（早先把「输出率 = 模型原生率」当成事实办事，
    那条推断是错的；本文档从未写过这句。）

    这个不一致正是下游带宽扩展被搞坏的源头：拿 16000 当 48 kHz 数据的率去重采样，
    等于把整个频谱上移 3 倍。用长度比推断对 DeepFilterNet（16k 输入→48k 数据，
    比值 3.0）与 DSP 基线（比值 1.0）同样成立。

    但**不能无条件用推导值**：带宽扩展的输出样本数不会是精确的整数倍（实测
    novasr 给 115,085 而非 115,089），四舍五入会把 48000 变成 47998，反而引入
    新错误。所以只在「推导值与上报值相差超过 tol」时才用推导值。
    """
    if n_in <= 0:
        return int(in_rate)
    derived = float(in_rate) * float(n_out) / float(n_in)
    if reported and abs(derived - float(reported)) <= max(1.0, tol * float(reported)):
        return int(reported)
    return int(round(derived))


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
    # 生成频段相对通带的目标电平（dB，负值）。三个 BWE 引擎的原始输出都把生成
    # 频段抬到远高于通带（实测 lavasr 在 12–20 kHz 高 +89 dB，且它自己已削顶到
    # 满刻度），听感就是"只剩高频"。默认压到通带之下 12 dB（语音在 500 Hz 以上
    # 约每倍频程 −6 dB，生成频段本就不该压过通带）。
    bwe_hf_target_db: float = -12.0

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
    video_out: bool = False              # True = 把增强后的音轨写回视频（画面 stream copy，不重编码）
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
    # 分阶段耗时（秒），键见 _STAGE_LABELS。用于定位瓶颈，不进指纹。
    stage_times: dict[str, float] = field(default_factory=dict)
    # 各推理阶段**实际**用的设备说明（如 "CUDA（空闲显存…）" / "CPU（…回落…）"）。
    # 单独成字段而不是只走 note()：单文件模式不传 log=，note() 会被丢弃，
    # 那样"实际用了哪个设备"在用户最常用的路径上就看不见了。
    device_notes: list[str] = field(default_factory=list)
    # --- 混响指标（Phase 5）--------------------------------------------
    reverb_slope_db_s: float | None = None

    @property
    def rtf(self) -> float:
        dur = self.source_stats.duration
        return self.elapsed / dur if dur else float("nan")


def align_to(enhanced: np.ndarray, source: np.ndarray, *,
             src_rate: int, enh_rate: int,
             note: Callable[[str], None] | None = None) -> np.ndarray:
    """把 enhanced 对齐到 source 的**时长**（不是样本数）。

    为什么必须按时间对齐：带宽扩展会把采样率从 16 kHz 提到 48 kHz —— 样本数变
    3 倍、**时长不变**。早先这里直接比较样本数，于是 48 kHz 的扩展输出被
    `enhanced[:len(source)]` 截成 **1/3 时长**；因为随后的断言只看频段内容、
    不看时长，这个静默的数据丢失一直没被发现（是加了长度守卫之后才暴露的）。

    模型可能截掉末尾不足一帧的部分（sherpa 约一个 hop、DeepFilterNet 约 30 ms），
    补齐时用源信号对应尾部，避免在结尾留一段可听见的静音。但补尾量过大时必须抛错：
    那说明上游出了实质缺陷，而"用原音补尾"会把它掩盖成"后半段是原音"。
    """
    target_n = int(round(len(source) * float(enh_rate) / float(src_rate)))
    if len(enhanced) == target_n:
        return enhanced
    if len(enhanced) > target_n:
        over = len(enhanced) - target_n
        if over > LENGTH_TOLERANCE:
            raise RuntimeError(
                f"处理后比源长 {over} 帧（源 {len(source)}@{src_rate}Hz → 期望 "
                f"{target_n} 帧@{enh_rate}Hz，实际 {len(enhanced)}），超出容差 "
                f"{LENGTH_TOLERANCE}；这是上游缺陷，已拒绝静默裁掉。"
            )
        return enhanced[:target_n]
    pad = target_n - len(enhanced)
    if pad > max(LENGTH_TOLERANCE, int(0.01 * target_n)):
        raise RuntimeError(
            f"处理后比源短 {pad} 帧（源 {len(source)}@{src_rate}Hz → 期望 "
            f"{target_n} 帧@{enh_rate}Hz，实际 {len(enhanced)}），补尾量超过 1%"
            f"（或 {LENGTH_TOLERANCE} 帧）。用原音补尾会把上游缺陷掩盖成"
            "「后半段是原音」，已拒绝。"
        )
    if pad == 0:
        return enhanced
    # 补尾取自源信号尾部；两边采样率不同时先把这段尾巴重采样到目标率
    take_src = max(1, int(round(pad * float(src_rate) / float(enh_rate))))
    tail = source[len(source) - take_src:]
    if int(src_rate) != int(enh_rate):
        tail = media_io.resample(np.asarray(tail, dtype=np.float32),
                                 int(src_rate), int(enh_rate))
    if enhanced.ndim == 2 and np.asarray(tail).ndim == 1:
        tail = media_io.to_stereo(tail)
    elif enhanced.ndim == 1 and np.asarray(tail).ndim == 2:
        tail = media_io.to_mono(tail)
    if note:
        note(f"[注意] 处理后比源短 {pad} 帧（期望 {target_n}，实际 {len(enhanced)}），"
             "已用原音尾部补齐。")
    return np.concatenate([enhanced, np.asarray(tail)[:pad]])


def _tick(progress: ProgressCb | None, frac: float, msg: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, frac)), msg)


def _check(cancel: CancelCb | None) -> None:
    if cancel and cancel():
        raise Cancelled("已取消")


# 分阶段耗时的显示名。键是机器友好的 ASCII（写进 manifest / 测试断言用），
# 中文只用于日志展示。
_STAGE_LABELS = {
    "decode": "解码",
    "trim": "静音裁剪",
    "prep": "前处理",
    "denoise": "降噪推理",
    "bwe": "带宽扩展",
    "eq": "听感EQ",
    "compress": "压缩",
    "resample": "重采样",
    "align": "时长对齐",
    "stats": "电平统计",
    "normalize_out": "输出归一化",
    "reverb": "混响指标",
    "write": "写出",
    "reuse": "复用读取",
}


class _StageTimer:
    """分阶段耗时累计器。

    此前流水线只记录了**总**耗时，于是"瓶颈在推理、还是在解码 / 重采样 / 被内存
    压到单并发"无法回答——而 GPU 选型（值不值得为此换 sherpa 的 GPU wheel）恰恰
    要靠这个数字决定。所以先补齐测量，再动 GPU。

    另一处口径澄清：`PipelineResult.elapsed` 与 `rtf` 取的是**写出之前**的时刻，
    所以它们不含写盘时间（310 MB 的 wav 写出可观）。写出时间单独记在 "write" 里，
    要比较"总墙钟"就用 summary() 里的合计值。
    """

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.times: dict[str, float] = {}

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.times[name] = self.times.get(name, 0.0) + (now - self.last)
        self.last = now

    def summary(self) -> str:
        return stage_summary(self.times)


def stage_summary(times: dict[str, float]) -> str:
    """把 stage_times 渲染成一行中文明细（日志与界面共用同一实现）。"""
    return " · ".join(
        f"{_STAGE_LABELS.get(k, k)} {v:.1f}s" for k, v in times.items()
    )


def _as_channels(samples: np.ndarray) -> list[np.ndarray]:
    """(frames, ch) 或 (frames,) → 单声道数组列表。"""
    if samples.ndim == 1:
        return [np.ascontiguousarray(samples, dtype=np.float32)]
    return [np.ascontiguousarray(samples[:, i], dtype=np.float32) for i in range(samples.shape[1])]


def _stack(channels: list[np.ndarray], *, what: str = "该阶段") -> np.ndarray:
    """把逐声道结果合并回 (frames, ch)。

    **长度差超过 LENGTH_TOLERANCE 帧就抛错**，而不是静默截断到较短那一路。
    这条守卫是有来历的：带宽扩展的声道循环曾把上一声道的输出采样率当成下一
    声道的输入采样率，导致第二路长度只有第一路的 1/3；而这里静默截断、下游
    `align_to` 又用原始音频补尾，两者叠加把「一路正常一路坏」的文件**无声地**
    写进了磁盘（用户实测到了坏文件，日志里却什么都没有）。
    """
    if not channels:
        raise ValueError("没有声道可合并")
    if len(channels) == 1:
        return channels[0]
    lens = [len(c) for c in channels]
    spread = max(lens) - min(lens)
    if spread > LENGTH_TOLERANCE:
        raise RuntimeError(
            f"{what}的各声道长度不一致：{lens}（相差 {spread} 帧，"
            f"容差 {LENGTH_TOLERANCE}）。逐声道处理必须对每个声道产生同样的长度，"
            "静默截断会产出「一路正常一路坏」的文件，已拒绝写出。"
        )
    n = min(lens)
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

    tm = _StageTimer()
    t_start = tm.t0

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
    tm.mark("decode")

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
    tm.mark("trim")

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
            y = align_to(np.asarray(prev.samples, dtype=np.float32), source,
                         src_rate=sr, enh_rate=int(prev.sample_rate), note=note)
            tm.mark("reuse")
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
                stage_times=dict(tm.times),
            )

    gain_pre = 0.0
    working = source
    stages: list[str] = []
    device_notes: list[str] = []

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

    tm.mark("prep")

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

    enhanced = _stack(outs, what="降噪")
    # 不信引擎上报的率，用长度比推真实率（见 _effective_rate 的实测记录）
    cur_rate = _effective_rate(sr, len(source), len(enhanced), reported=engine_out_rate)
    if cur_rate != int(engine_out_rate):
        note(f"[注意] 引擎上报输出率 {engine_out_rate} Hz，但按长度比实际是 "
             f"{cur_rate} Hz，已按实际值处理（sherpa 的 DPDFNet 会这样错报）。")
    stages.append("denoise")
    tm.mark("denoise")
    _dev = engine.provider_desc()
    if _dev:
        device_notes.append(f"降噪推理设备：{_dev}")
        note(device_notes[-1])

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
            # ★ 所有声道的**输入**率是同一个值，循环前快照一次，循环内绝不改写它。
            # 之前这里写成 `by, cur_rate = bwe_eng.process(bch, cur_rate, ...)`，
            # 于是把上一声道的**输出**率当成了下一声道的输入率。三个 BWE 引擎的
            # 输出率恒为 48 kHz，所以只要上游降噪的输出率不是 48 kHz，第二路就会
            # 被宣告错误的输入率——实测后果是整个频谱被上移 3 倍、长度变成 1/3、
            # 两声道互不相关，听感就是「一路正常、一路坏」。
            b_in_rate = cur_rate
            b_out_rates: set[int] = set()
            for bi, bch in enumerate(bchans):
                _check(cancel)

                def b_progress(f: float, msg: str, _i=bi) -> None:
                    span = (0.88 - b_frac) / len(bchans)
                    prefix = f"[声道 {_i + 1}/{len(bchans)}] " if len(bchans) > 1 else ""
                    _tick(progress, b_frac + span * f, prefix + msg)

                by, ch_rate = bwe_eng.process(
                    bch, b_in_rate, options.engine,
                    cutoff_hz=options.bwe_cutoff_hz,
                    hf_target_db=options.bwe_hf_target_db,
                    progress=b_progress, cancel=cancel,
                )
                by = np.asarray(by, dtype=np.float32).reshape(-1)
                # 同样用长度比推真实率，不信上报值（三个引擎都上报 48 kHz，
                # 但喂进去的率不同时它们的长度比不同）
                b_out_rates.add(_effective_rate(b_in_rate, len(bch), len(by),
                                                reported=ch_rate))
                b_outs.append(by)

            if len(b_out_rates) != 1:
                raise RuntimeError(
                    f"带宽扩展在各声道上报了不同的输出采样率 {sorted(b_out_rates)}，"
                    "这会让声道之间不一致（听感上是「一路正常、一路坏」），已拒绝写出。"
                )
            cur_rate = b_out_rates.pop()
            enhanced = _stack(b_outs, what="带宽扩展")
            note(f"带宽扩展输出 {cur_rate} Hz；生成频段已压到通带之下 "
                 f"{options.bwe_hf_target_db:g} dB（修正引擎原始输出的电平失衡）。")
        stages.append(f"bwe:{options.bwe_id}")
        tm.mark("bwe")
        _bdev = bwe_eng.provider_desc()
        if _bdev:
            device_notes.append(f"带宽扩展推理设备：{_bdev}")
            note(device_notes[-1])
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

    tm.mark("eq")

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

    tm.mark("compress")

    # --- 5. 采样率对齐与写出 -------------------------------------------
    _check(cancel)
    if cur_rate != target_rate:
        _tick(progress, 0.91, f"重采样 {cur_rate} → {target_rate} Hz…")
        note(f"处理后为 {cur_rate} Hz，重采样回 {target_rate} Hz。")
        enhanced = media_io.resample(enhanced, cur_rate, target_rate)

    tm.mark("resample")

    # 与原音对齐长度：引擎可能截掉末尾不足一帧的部分
    enhanced = align_to(enhanced, source, src_rate=sr, enh_rate=target_rate, note=note)
    tm.mark("align")

    enhanced_stats = dsp.analyze(enhanced, target_rate)
    tm.mark("stats")

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

    tm.mark("normalize_out")

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

    tm.mark("reverb")

    elapsed = time.perf_counter() - t_start

    # --- 6. 写出与登记 -------------------------------------------------
    out_path = None
    if options.output_path:
        # 显式指定输出路径 = 导出语义：不查重、不登记
        _tick(progress, 0.97, "写出文件…")
        out_path = media_io.write_audio(options.output_path, enhanced, target_rate)
        note(f"已写出：{out_path}")
    elif out_dir:
        out_path = os.path.join(
            out_dir, outputs.build_filename(media_path, options, fp, eng_kind)
        )
        if options.video_out:
            check_video_out_ok(info, media_path, trim_span, len(original))
            _tick(progress, 0.97, "写回视频（画面不重编码）…")
            media_io.write_video_with_audio(
                media_path, out_path, enhanced, target_rate,
                replace_stream_index=resolved,
            )
            note(f"已写回视频（画面原样复制）：{out_path}")
        else:
            _tick(progress, 0.97, "写出文件…")
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

    # 写出时间不计入 elapsed / rtf（沿用既有口径，便于与历史 RTF 比较），
    # 但它单独出现在分阶段明细里，所以"总墙钟 = summary() 的合计"。
    tm.mark("write")
    note(f"分阶段耗时（合计 {time.perf_counter() - t_start:.1f}s）：{tm.summary()}")

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
        stage_times=dict(tm.times),
        device_notes=device_notes,
        reverb_slope_db_s=reverb.get("slope_db_per_s"),
    )


def check_video_out_ok(info, media_path: str, trim_span, original_frames: int) -> None:
    """写回视频前的全部拒绝条件，集中在这里以便逐条测试。

    宁可明确拒绝，也不降级成静默重编码或产出音画不同步的文件——与本项目
    「实测无效就硬拒绝，不给会毁素材的开关」的既有取向一致（见 core/wpe.py）。
    """
    if not info.is_video:
        raise media_io.MediaError(
            "勾了「输出视频」但输入不是视频文件。请取消勾选，或改选输出音频格式。"
        )
    if trim_span is not None and (
        trim_span.start_frame > 0 or trim_span.end_frame < original_frames
    ):
        raise media_io.MediaError(
            "开了「裁剪首尾静音」且实际裁掉了内容：此时增强音频比视频短，"
            "写回会产生音画时长不一致的文件。请取消裁剪，或改为输出音频。"
        )
    reason = media_io.remux_refusal_reason(os.path.splitext(media_path)[1])
    if reason:
        raise media_io.MediaError(reason + "　请改为输出音频。")


def default_output_path(source_path: str, engine_id: str, project_root: str, ext: str = ".wav") -> str:
    stem = os.path.splitext(os.path.basename(source_path))[0]
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem).strip() or "output"
    out_dir = os.path.join(project_root, "out")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{safe}_{engine_id}{ext}")
