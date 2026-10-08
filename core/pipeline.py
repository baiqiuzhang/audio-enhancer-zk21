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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import dsp, enhance, media_io, outputs, sysinfo, wpe
from .engines import (
    HEAVY_STAGE,
    INFERENCE_SLOTS,
    Cancelled,
    EngineParams,
    build_engine,
    engine_for,
    engine_kind,
    release_thread_engines,
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


# ---------------------------------------------------------------------------
# 逐声道调度：串行还是并行
# ---------------------------------------------------------------------------
#
# **实测依据（2026-10-01，本机 20 逻辑核）**：用户素材（4139 s / 48 kHz / 立体声 /
# demud 预设 / DeepFilterNet3）在 1200 s 片段上的分阶段占比是
# 「降噪 83.0% + 听感 EQ 5.5% + 前处理 5.5% + 其它 6%」；而 `deep-filter.exe`
# **只用约 1 个核**（采样 6 s 的 CPU 增量 = 0.98 核，pid 级实测）——它是单线程的。
# 声道之间又完全独立（同一模型分别作用于左右声道，本来就是为了保住声像），
# 所以两个声道同时跑就是最直接的 ~2×，且**不改变任何数值**（每个声道的计算原样保留）。
#
# 代价是内存：DFN 路径一个声道要用「单声道副本 + 输出 float32 + 子进程自己的整文件副本」
# （69 分钟素材约 2–3 GB），两个声道同时跑就要多一份。所以按空闲内存闸门，
# 不够就回落到串行并**说明原因**（不静默降级，也不硬闯 OOM——本机常驻空闲只有 3–7 GB）。
# auto 模式下的并发上限。刻意只到 2：立体声是唯一被实测过的场景，
# 而 >2 声道素材（soundfile 路径不会降混）一次拉起 6 个引擎实例会直接吃爆内存。
_PARALLEL_MAX_WORKERS_AUTO = 2

# 并行声道的内存闸门按**素材长度**算，而不是一个常数——因为多拉一个声道的代价
# 就是"再复制一份这个声道"。系数含义：每声道每采样点的额外驻留（字节）= 输入副本
# 4 + 输出 float32 4 + 引擎侧整文件副本（DFN 子进程实测吃满输入 f32）。
# 标定方式：1200 s 立体声单声道长度 57.6M 采样 → 预测 691 MB；
# 整链实测峰值在并行后仍低于串行（见检查点实测表），所以这里的取值偏保守。
_PARALLEL_BYTES_PER_SAMPLE = {"dfn_exe": 12.0, "sherpa": 6.0, "dsp": 3.0}
_PARALLEL_BYTES_PER_SAMPLE_DEFAULT = 8.0
_PARALLEL_FIXED_MB = {"dfn_exe": 600.0, "sherpa": 400.0, "dsp": 200.0}
_PARALLEL_FIXED_DEFAULT_MB = 400.0
_PARALLEL_HEADROOM_MB = 600.0


def parallel_channel_need_mb(engine_kind_name: str, samples_per_channel: int) -> float:
    """再并行一个声道预计要多少空闲内存（MB）。"""
    per_sample = _PARALLEL_BYTES_PER_SAMPLE.get(
        engine_kind_name, _PARALLEL_BYTES_PER_SAMPLE_DEFAULT)
    fixed = _PARALLEL_FIXED_MB.get(engine_kind_name, _PARALLEL_FIXED_DEFAULT_MB)
    return fixed + per_sample * max(0, int(samples_per_channel)) / 1024.0 / 1024.0 \
        + _PARALLEL_HEADROOM_MB


def channel_workers(n_channels: int, policy: str, engine_kind_name: str,
                    note: Callable[[str], None],
                    samples_per_channel: int = 0) -> int:
    """决定声道级并发数。返回 1 = 串行。

    policy：`off` 永远串行；`on` 显式并行（跳过内存闸门，用户自担）；
    `auto` 按空闲内存决定（读不到内存时按"未知"放行，与 bwe._check_memory 同口径）。
    """
    if n_channels <= 1 or policy == "off":
        return 1
    if policy == "on":
        note(f"[注意] 已显式开启声道并行（{n_channels} 个声道同时处理），未做内存闸门。")
        return n_channels
    need = parallel_channel_need_mb(engine_kind_name, samples_per_channel)
    free = sysinfo.available_memory_mb()
    if free and free < need:
        note(
            f"[注意] 可用内存 {free:.0f} MB，低于声道并行所需 {need:.0f} MB"
            f"（引擎 {engine_kind_name}，每声道 {samples_per_channel / 48000:.0f}s 素材）："
            "本文件逐声道**串行**处理——会慢一些，但不会换页或 OOM。"
        )
        return 1
    return min(n_channels, _PARALLEL_MAX_WORKERS_AUTO)


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
    # 逐声道调度：auto（按空闲内存决定）/ on（强制并行）/ off（强制串行）。
    # **不进指纹**：声道只是调度方式，不改任何数值（与 num_threads 同理）。
    parallel_channels: str = "auto"
    # 界面上一步"导入后分析"已经完整解码过一次的音频。给了它就**不再解码第二次**
    # （69 分钟视频实测解码 14.61 s / 1.59 GB）。流水线会校验时长与目标流是否吻合，
    # 不吻合就照常解码；本字段**不进指纹**（它只是同一份输入的复用提示）。
    predecoded: media_io.AudioData | None = None
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
    # 复用界面上一步已经解码好的音频（O3，2026-10-01）：导入后分析已经完整解过一遍，
    # 这里再解一次等于同一段素材读两遍（69 分钟视频实测 14.61 s）并**同时留两份 1.59 GB**。
    # 校验足够严：时长必须与容器上报的时长吻合（2% 或 1 s 内），否则视为"不是同一个流"
    # 而照常解码——宁可多解一次，也不要把另一条音轨的数据当成这条。
    decoded = None
    pre = getattr(options, "predecoded", None)
    if pre is not None and pre.frames > 0 and pre.sample_rate > 0:
        want_s = float(info.duration or 0.0)
        have_s = pre.frames / float(pre.sample_rate)
        if want_s <= 0 or abs(have_s - want_s) <= max(1.0, 0.02 * want_s):
            decoded = pre
            note(f"复用导入时已解码的音频（跳过第二次解码）：{pre.frames} 帧 @ "
                 f"{pre.sample_rate} Hz，{pre.channels} 声道。")
        else:
            note(f"[注意] 预解码音频的时长 {have_s:.2f}s 与容器上报 {want_s:.2f}s 不符，"
                 "已改为重新解码（避免把另一条音轨当成目标音轨）。")
    if decoded is None:
        decoded = media_io.decode(media_path, stream_index=decode_index, cancel=cancel)
    note(f"解码完成：{decoded.frames} 帧 @ {decoded.sample_rate} Hz，{decoded.channels} 声道。")

    sr = decoded.sample_rate
    original = decoded.samples
    # 只留一个整数：`original` 在裁剪之外没有别的用处，而 69 分钟立体声是 1.5 GB。
    # 保留 `decoded` 这个名字就等于把那份数组钉在栈上陪跑整条流水线。
    original_frames = int(original.shape[0])
    del decoded
    tm.mark("decode")

    # --- 2. 静音裁剪与增益归一化 --------------------------------------
    trim_span: dsp.ActiveSpan | None = None
    gaps: list[tuple[float, float]] = []
    source = original

    if options.trim_silence:
        _tick(progress, 0.06, "检测静音段…")
        _check(cancel)
        mono_for_detect = media_io.to_mono(original)
        # 一次遍历同时给出「首尾可裁剪区间」与「中间长静音段」：两者共用同一套
        # 帧划分与同一道阈值，此前各跑一遍等于把最贵的那趟帧电平计算白付一次
        # （实测见 core/dsp.py::frame_rms_streaming）。
        trim_span, gaps = dsp.detect_active_span_and_gaps(
            mono_for_detect, sr,
            dynamic_range_db=options.trim_dynamic_range_db,
            pad_ms=options.trim_pad_ms,
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
        # 检测用的单声道副本（69 分钟 = 0.8 GB）到此为止：不删就会一直挂在栈上。
        del mono_for_detect

    # 裁剪过的副本已经独立存在，原始整段可以放掉（同样是为了别把 1.5 GB 拖到最后）。
    # `source is original` 时（未裁剪）这里什么都不做——名字换不掉同一份数据。
    if source is not original:
        del original

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
            prev = media_io.decode(expected)
            # 复用前做**完整性校验**：文件名里的指纹只证明"参数一样"，不证明这份文件是
            # 完整写出来的。历史上有两处会留下半截文件——旧版 `write_audio` 是直接写最终
            # 路径（现在已改成"临时文件 + os.replace"原子写），以及任何在写出过程中被强杀/
            # 断电的运行。轻度截断会被下面的 align_to 用原音尾部静默补齐，此后一直复用；
            # 重度截断则抛「这是上游缺陷」，用户却不知道该删哪个文件、每次重试都卡在同一处。
            # 这里既然**本来就要完整解码一次**，就顺手比一下时长：不符就放弃复用并重算。
            want_s = len(source) / float(sr) if sr else 0.0
            have_s = prev.frames / float(prev.sample_rate) if prev.sample_rate else 0.0
            # 容差按容器定：无损容器（wav/flac）写出时 align_to 会把长度对齐到 ±1 帧，
            # 所以正常产物的时长与期望差不到 0.5 帧 —— 容差取 2 ms 就足够严（实测 0.5%
            # 截断会落在 2 ms 之外被发现）。mp3 有编码器延迟/补齐（一个帧最长约 26 ms），
            # 放宽到 60 ms，否则会把**正常**的 mp3 产物误判成损坏而每次重算。
            tol_s = 0.060 if expected.lower().endswith(".mp3") else 0.002
            if prev.frames <= 0 or abs(have_s - want_s) > tol_s:
                note(
                    f"[注意] 已有结果 {os.path.basename(expected)} 的时长 {have_s:.3f}s 与期望 "
                    f"{want_s:.3f}s 不符（疑似半截/损坏），已放弃复用并重新计算。"
                    f"若确认它无用，可删除：{expected}"
                )
            else:
                note(f"参数与已有结果完全一致，直接复用（不重算）：{os.path.basename(expected)}")
                _tick(progress, 0.5, "复用已有结果…")
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
            del prev

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
    # `working` 到这里已经没用了（各声道的副本已独立存在）。69 分钟立体声是 1.59 GB，
    # 而下面正是整条流水线内存压力最大的阶段（推理 + 并行声道），所以显式放掉：
    # 未开高通/归一化时 `working is source`，这句是无操作，不影响正确性。
    if working is not source:
        del working

    engine_frac = 0.76          # 0.12 → 0.88 留给引擎
    outs: list[np.ndarray] = []
    engine_out_rate = sr
    channel_devs: dict[int, str | None] = {}
    sub_gates = [0.0] * len(channels)       # 各声道的局部进度，用于并行时的聚合
    sub_lock = threading.Lock()

    def _run_channel(i: int, ch: np.ndarray) -> tuple[int, np.ndarray, int, str | None]:
        """跑一个声道。并行时本函数在 worker 线程里执行——注意这里不能再碰非线程安全的
        共享状态（引擎实例取自线程局部缓存，见 engines.engine_for）。"""
        eng = engine if workers == 1 else engine_for(spec, catalog)
        base = 0.12 + engine_frac * (i / len(channels))

        def sub_progress(f: float, msg: str, _base=base, _i=i) -> None:
            span = engine_frac / len(channels)
            if workers == 1:
                frac = _base + span * f
            else:
                with sub_lock:
                    sub_gates[_i] = max(sub_gates[_i], float(f))
                    frac = 0.12 + engine_frac * (sum(sub_gates) / len(sub_gates))
            prefix = f"[声道 {_i + 1}/{len(channels)}] " if len(channels) > 1 else ""
            _tick(progress, frac, prefix + msg)

        try:
            # 进程级推理槽位：批处理并发 × 声道并行会让最多 4 个重型子进程同时跑，
            # 而两个闸门各自只看空闲内存、互不预留（见 engines.INFERENCE_SLOTS 的说明）。
            INFERENCE_SLOTS.acquire(cancel)
            try:
                out, rate = eng.process(
                    ch, sr, options.engine, progress=sub_progress, cancel=cancel
                )
            finally:
                INFERENCE_SLOTS.release()
            return i, np.asarray(out, dtype=np.float32), int(rate), eng.provider_desc()
        finally:
            if workers > 1:
                # 并行时 worker 线程持有的引擎句柄必须自己收掉，否则会跨任务堆积
                release_thread_engines()

    workers = channel_workers(
        len(channels), getattr(options, "parallel_channels", "auto"), eng_kind, note,
        samples_per_channel=len(channels[0]),
    )
    if workers > 1:
        note(
            f"声道并行：{len(channels)} 个声道同时处理（{eng_kind} 单实例只吃约 1 个核，"
            f"实测这是立体声最直接的加速；数值与串行逐位相同）。"
        )
    if workers > 1:
        # 线程池由 with 收尾：任一声道抛错（含取消）都会让整体失败（与串行同语义），
        # 已在跑的声道自然跑完，不会留下半截产物。
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ae-ch") as pool:
            results = list(pool.map(_run_channel, range(len(channels)), channels))
    else:
        results = [_run_channel(i, ch) for i, ch in enumerate(channels)]

    for i, y, rate, dev in sorted(results, key=lambda r: r[0]):
        outs.append(y)
        channel_devs[i] = dev
        engine_out_rate = rate
    if len({r[2] for r in results}) > 1:
        note(f"[注意] 各声道上报的输出采样率不一致：{sorted(r[2] for r in results)}")

    enhanced = _stack(outs, what="降噪")
    # 逐声道的输入副本与结果列表到此为止：69 分钟立体声下这两个各约 1.59 GB，
    # 而下面还要分配「带宽扩展 / 听感 EQ」的同尺寸新数组。别让它们挂到函数结束。
    del channels, outs, results
    # 不信引擎上报的率，用长度比推真实率（见 _effective_rate 的实测记录）
    cur_rate = _effective_rate(sr, len(source), len(enhanced), reported=engine_out_rate)
    if cur_rate != int(engine_out_rate):
        note(f"[注意] 引擎上报输出率 {engine_out_rate} Hz，但按长度比实际是 "
             f"{cur_rate} Hz，已按实际值处理（sherpa 的 DPDFNet 会这样错报）。")
    stages.append("denoise")
    tm.mark("denoise")
    # 设备说明从**每个声道的实际执行线程**里取（`bwe.last_device()` 是线程局部的，
    # 并行时主线程那份是空的）。去重后逐条记录——不做"声明式"报告。
    for _d in sorted({d for d in channel_devs.values() if d}):
        device_notes.append(f"降噪推理设备：{_d}")
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

                INFERENCE_SLOTS.acquire(cancel)
                try:
                    by, ch_rate = bwe_eng.process(
                        bch, b_in_rate, options.engine,
                        cutoff_hz=options.bwe_cutoff_hz,
                        hf_target_db=options.bwe_hf_target_db,
                        progress=b_progress, cancel=cancel,
                    )
                finally:
                    INFERENCE_SLOTS.release()
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
            # 逐声道的输入副本与结果列表（各约 1.59 GB @69 分钟）到此为止
            del bchans, b_outs
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
        # 听感 EQ 同样是 FFT 主导、声道之间独立 → 可以并行（实测 1200 s 双声道 1.52×，
        # 逐位相同）。它的内存代价比降噪小得多（每声道一份工作缓冲），所以单独过一遍
        # 闸门时按 "dsp" 档（每采样 3 B）而不是引擎档，免得被降噪的 3.5 GB 门槛连带禁掉。
        eq_workers = channel_workers(
            1 if enhanced.ndim == 1 else int(enhanced.shape[1]),
            getattr(options, "parallel_channels", "auto"), "dsp", note,
            samples_per_channel=len(enhanced),
        )
        enhanced = enhance.spectral_shape(
            enhanced, cur_rate,
            shelf_hz=options.shelf_hz,
            shelf_db=options.shelf_db if eq_shelf else 0.0,
            mud_cut_hz=options.mud_cut_hz,
            mud_cut_db=options.mud_cut_db,
            mud_cut_width_oct=options.mud_cut_width_oct,
            progress=progress, cancel=cancel,
            workers=eq_workers,
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
    # 写出之前再看一眼取消标志：写回视频是分钟级操作（AAC 编码实测约 30× 实时），
    # 已经取消的任务不该再往磁盘上落一个几百 MB 的文件。
    _check(cancel)
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
            check_video_out_ok(info, media_path, trim_span, original_frames)
            _tick(progress, 0.97, "写回视频（画面不重编码）…")
            media_io.write_video_with_audio(
                media_path, out_path, enhanced, target_rate,
                replace_stream_index=resolved, cancel=cancel,
            )
            note(f"已写回视频（画面原样复制）：{out_path}")
        else:
            _tick(progress, 0.97, "写出文件…")
            media_io.write_audio(out_path, enhanced, target_rate)
            note(f"已写出：{out_path}")

    # 写出时间不计入 elapsed / rtf（沿用既有口径，便于与历史 RTF 比较），
    # 但它单独出现在分阶段明细里，所以"总墙钟 = summary() 的合计"。
    #
    # **必须在登记之前结算**：manifest 里要带上这一段。此前是"先登记、后 mark('write')"，
    # 于是写盘/回写视频的耗时**从来没进过 manifest**——用户一次 734.72 s 的整段运行只留下
    # elapsed，瓶颈拆解永久丢失（只能靠外推猜）。顺序换过来之后，历史运行才可事后诊断。
    if out_path is not None:
        tm.mark("write")

    # 登记：只有"按命名规范自动落盘"才有登记语义；显式 output_path（导出）不登记。
    if out_dir and out_path:
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
                    # 下面三项是"事后诊断"用的，不参与任何判定：
                    #   stage_times —— 每段耗时（键见 _STAGE_LABELS，含 write）
                    #   wall_s      —— 含写出的总墙钟（elapsed 刻意不含写出）
                    #   channels    —— 声道数（立体声是逐声道串行/并行处理的倍率所在）
                    "stage_times": {k: round(v, 2) for k, v in tm.times.items()},
                    "wall_s": round(time.perf_counter() - t_start, 2),
                    "channels": 1 if enhanced.ndim == 1 else int(enhanced.shape[1]),
                    "device_notes": list(device_notes),
                },
                trim_span=(trim_span.start_s, trim_span.end_s) if trim_span else None,
            ))
        except OSError as exc:
            note(f"[注意] 结果文件已写出，但登记 manifest 失败：{exc}")

    _tick(progress, 1.0, f"完成（{elapsed:.1f}s）")

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
