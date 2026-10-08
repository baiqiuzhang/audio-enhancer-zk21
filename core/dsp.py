"""电平分析、静音检测裁剪、归一化，以及 DSP 基线处理。

纯 numpy 实现（不依赖 scipy/librosa）。注意 venv 里**其实装着 scipy**——它不是我们声明的
依赖，而是 `audiosronnx` 带进来的（见 requirements.txt 第 24 行的说明）。所以这里刻意
**不使用** scipy：它可能随可选的 audiosronnx 一起消失（`scipy.fft(workers=-1)` 实测确实
能把这边的 FFT 从 28.0 ms 降到 13.4 ms/块，但那是"可选依赖才能用的加速"，见 _istft_block
的注释）。
长文件一律分块处理，避免一次性物化整个 STFT 矩阵或整张帧矩阵——后者的实测代价见
`frame_rms_streaming`（4139 s 素材约 8 GB/次，整链外推峰值 16.7 GB，超过本机物理内存）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .errors import Cancelled

ProgressCb = Callable[[float, str], None]
CancelCb = Callable[[], bool]

# `Cancelled` 现在只有一个定义（core/errors.py），见该模块顶部的说明：
# 早先 dsp 与 engines 各定义一个同名异常，导致"取消发生在 DSP/EQ 阶段"时
# pipeline/worker 的 `except Cancelled` 捕不到，界面会把取消显示成处理失败。


def _tick(progress: ProgressCb | None, frac: float, msg: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, frac)), msg)


def _check(cancel: CancelCb | None) -> None:
    if cancel and cancel():
        raise Cancelled("已取消")


# ---------------------------------------------------------------------------
# dB 换算
# ---------------------------------------------------------------------------

_FLOOR = 1e-12


def to_dbfs(amplitude: float) -> float:
    return 20.0 * float(np.log10(max(abs(float(amplitude)), _FLOOR)))


def from_dbfs(db: float) -> float:
    return float(10.0 ** (db / 20.0))


# ---------------------------------------------------------------------------
# 电平分析
# ---------------------------------------------------------------------------

@dataclass
class LevelStats:
    duration: float
    peak: float
    rms: float
    peak_dbfs: float
    rms_dbfs: float
    dc_offset: float
    silence_ratio: float        # 低于阈值的帧占比
    clipped_ratio: float        # |x| >= 0.999 的样本占比
    frame_db_p10: float
    frame_db_p50: float
    frame_db_p90: float

    def describe(self) -> str:
        return (
            f"峰值 {self.peak:.4f}（{self.peak_dbfs:+.1f} dBFS） · "
            f"RMS {self.rms:.5f}（{self.rms_dbfs:+.1f} dBFS） · "
            f"静音帧 {self.silence_ratio * 100:.1f}%"
        )

    def verdict(self) -> str:
        """给出人话判断，因为「电平太低」往往比噪声更拖累识别。"""
        if self.peak_dbfs < -20.0:
            return "电平偏低，建议开启增益归一化"
        if self.peak_dbfs > -0.5 or self.clipped_ratio > 1e-4:
            return "电平接近满刻度，可能有削顶，建议先降增益"
        return "电平正常"


# ---------------------------------------------------------------------------
# 整段统计：分块累加（不物化 float64 副本）
# ---------------------------------------------------------------------------
#
# 为什么需要这三个：`np.square(x.astype(np.float64))` 这类写法会把整段放大成
# float64（4 字节 → 8 字节）并再复制一份平方结果，**瞬时内存是素材字节数的 4 倍**。
# 实测（主代理 2026-10-01，1200 s 立体声 = 439 MB）：
#
#   · `dsp.analyze`（改前）          峰值 1099 MB
#   · `dsp.normalize`（改前，peak）   峰值  879 MB（一次调用）
#   · `engines._pad_tail`（改前）     峰值  879 MB（**每个声道一次**）
#   · `ui/player` 的增益匹配 rms      峰值 1758 MB（每轨一次，且在 GUI 线程持锁执行）
#
# 直接 rms 对照（同一 439 MB 数组）：改前 634 ms / 1758 MB → 分块 645 ms / 122 MB，
# 数值相对差 0（逐位相同）。也就是说这笔内存是纯浪费，代价只有 ~2% 的时间。
# 分块尺寸取 400 万元素：float64 下约 32 MB，足够小到可以和"素材本身"共存。
_REDUCE_BLOCK = 4_000_000


def rms_blockwise(x: np.ndarray, *, block: int = _REDUCE_BLOCK) -> float:
    """整段 RMS，按块累加（float64 精度），瞬时分配 O(block) 而不是 O(n)。"""
    a = np.asarray(x)
    n = a.size
    if n == 0:
        return 0.0
    flat = a.reshape(-1)
    total = 0.0
    for i in range(0, n, block):
        seg = flat[i:i + block].astype(np.float64)
        total += float(np.square(seg).sum())
    return float(np.sqrt(total / n))


def peak_blockwise(x: np.ndarray, *, block: int = _REDUCE_BLOCK) -> float:
    """整段峰值 |x|，按块取最大。结果与 `np.max(np.abs(x))` **逐位相同**（max 与顺序无关）。"""
    a = np.asarray(x)
    n = a.size
    if n == 0:
        return 0.0
    flat = a.reshape(-1)
    best = 0.0
    for i in range(0, n, block):
        m = float(np.max(np.abs(flat[i:i + block])))
        if m > best:
            best = m
    return best


def clipped_ratio_blockwise(x: np.ndarray, threshold: float = 0.999,
                            *, block: int = _REDUCE_BLOCK) -> float:
    """|x| >= threshold 的样本占比，按块计数（结果与整段 `np.mean` 相同）。"""
    a = np.asarray(x)
    n = a.size
    if n == 0:
        return 0.0
    flat = a.reshape(-1)
    hits = 0
    for i in range(0, n, block):
        hits += int(np.count_nonzero(np.abs(flat[i:i + block]) >= threshold))
    return hits / float(n)


def analyze(
    samples: np.ndarray,
    sample_rate: int,
    *,
    silence_dbfs: float = -50.0,
    frame_ms: float = 20.0,
) -> LevelStats:
    """统计电平与静音占比。"""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    n = len(x)
    if n == 0:
        return LevelStats(0, 0, 0, -240.0, -240.0, 0, 1.0, 0, -240.0, -240.0, -240.0)

    # 三项整段统计都走分块实现（见上面 _REDUCE_BLOCK 的实测表）：
    # 改前这里一次 `np.square(x.astype(np.float64))` 在 4139 s 素材上要约 3.2 GB 瞬时。
    peak = peak_blockwise(x)
    rms = rms_blockwise(x)
    dc = float(np.mean(x, dtype=np.float64))      # numpy 原地按 float64 累加，无整段副本
    clipped = clipped_ratio_blockwise(x)

    win = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, win // 2)
    if n >= win:
        frame_rms = frame_rms_streaming(x, win, hop)
        frame_db = 20.0 * np.log10(np.maximum(frame_rms, _FLOOR))
        silence_ratio = float(np.mean(frame_db < silence_dbfs))
        p10, p50, p90 = (float(v) for v in np.percentile(frame_db, [10, 50, 90]))
    else:
        silence_ratio = 0.0
        p10 = p50 = p90 = to_dbfs(rms)

    return LevelStats(
        duration=n / float(sample_rate),
        peak=peak,
        rms=rms,
        peak_dbfs=to_dbfs(peak),
        rms_dbfs=to_dbfs(rms),
        dc_offset=dc,
        silence_ratio=silence_ratio,
        clipped_ratio=clipped,
        frame_db_p10=p10,
        frame_db_p50=p50,
        frame_db_p90=p90,
    )


def _frames_at(x: np.ndarray, first: int, count: int, win: int, hop: int) -> np.ndarray:
    """取第 first..first+count-1 帧（每帧从 first*hop 起跳），返回 (count, win)。"""
    idx = np.arange(win)[None, :] + hop * np.arange(first, first + count)[:, None]
    return x[idx]


# 每批计算的帧数。取 4096 帧是为了让临时数组远小于素材本身：
# 48 kHz 下 4096 帧 × 768 采样 × 8 B ≈ 25 MB，而下面的 "为什么必须分块" 里的
# 旧口径是同量级的 1/100。
_FRAME_BLOCK = 4096


def frame_rms_streaming(x: np.ndarray, win: int, hop: int, *,
                        progress: ProgressCb | None = None,
                        cancel: CancelCb | None = None,
                        block_frames: int = _FRAME_BLOCK) -> np.ndarray:
    """逐帧 RMS（float64，长度 n_frames）。**分块**计算，不物化整张帧矩阵。

    **为什么必须分块（实测，2026-10-01）**：旧写法 `_frames(x, win, hop)` 会把
    `(n_frames, win)` 的索引矩阵与采样矩阵一次性物化，成本随素材时长**线性增长**
    （20 ms 帧 / 10 ms hop，48 kHz 单声道，一次遍历）：

    | 素材 | 索引 int64 | 采样 float32 | 平方 float64 | 合计 |
    | --- | --- | --- | --- | --- |
    | 1200 s | 0.9 GB | 0.5 GB | 0.9 GB | **≈ 2.3 GB** |
    | 4139 s（用户那次 69 分钟素材） | 3.2 GB | 1.6 GB | 3.2 GB | **≈ 8.0 GB** |

    它是 `analyze` / `detect_active_span` / `count_internal_gaps` /
    `estimate_decay_slope_db_per_s`（source 与 enhanced 各一次）/ `compress` 共用的底层，
    一次链路上要跑 5–7 遍。整链峰值驻留实测（同一预设、同一素材、含 DFN3 推理）：

    | 素材时长 | 进程峰值驻留 |
    | --- | --- |
    | 300 s | 1249 MB |
    | 1200 s | 4873 MB |

    线性拟合 **≈ 4.03 MB × 素材秒数 + 41 MB**，外推到用户的 4139 s 素材是 **约 16.7 GB**
    —— 超过本机 15.6 GB 物理内存，也就是说那一次运行**在换页盘上跑**（这既拖慢速度，
    也让长素材随时可能 MemoryError）。分块之后同一趟只剩常数级的窗口。

    数值上与旧实现**逐元素相同**：每帧的 float64 求和顺序不变（一行内仍是同样的
    win 个元素、同样的顺序），只是不再一次摆放全部行。
    """
    x = np.asarray(x)
    n = len(x)
    if n < win:
        return np.sqrt(np.mean(np.square(x.astype(np.float64)), axis=0, keepdims=True)).reshape(-1)
    n_frames = 1 + (n - win) // hop
    out = np.empty(n_frames, dtype=np.float64)
    done = 0
    while done < n_frames:
        _check(cancel)
        count = min(block_frames, n_frames - done)
        frames = _frames_at(x, done, count, win, hop)
        out[done:done + count] = np.sqrt(
            np.mean(np.square(frames.astype(np.float64)), axis=1)
        )
        done += count
        if progress and n_frames:
            _tick(progress, done / n_frames, f"帧电平 {done * 100 // n_frames}%")
    return out


# ---------------------------------------------------------------------------
# 静音检测与裁剪
# ---------------------------------------------------------------------------

@dataclass
class ActiveSpan:
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    threshold_dbfs: float
    total_s: float

    @property
    def active_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def detect_active_span(
    samples: np.ndarray,
    sample_rate: int,
    *,
    dynamic_range_db: float = 40.0,
    floor_dbfs: float = -75.0,
    pad_ms: float = 150.0,
    frame_ms: float = 20.0,
) -> ActiveSpan:
    """找出「有内容」的区间（首尾裁剪用）。

    阈值取「帧电平的高分位 - dynamic_range_db」，再夹到 floor_dbfs 以上。
    这样对整体电平很低的素材（例如峰值只有 0.16 的录音）也能自适应，
    而不是套一个固定的绝对 dBFS 阈值。
    """
    span, _gaps = detect_active_span_and_gaps(
        samples, sample_rate,
        dynamic_range_db=dynamic_range_db, floor_dbfs=floor_dbfs,
        pad_ms=pad_ms, frame_ms=frame_ms,
    )
    return span


def detect_active_span_and_gaps(
    samples: np.ndarray,
    sample_rate: int,
    *,
    dynamic_range_db: float = 40.0,
    floor_dbfs: float = -75.0,
    pad_ms: float = 150.0,
    min_gap_ms: float = 1000.0,
    frame_ms: float = 20.0,
    progress: ProgressCb | None = None,
    cancel: CancelCb | None = None,
) -> tuple[ActiveSpan, list[tuple[float, float]]]:
    """一次帧电平遍历同时给出 (首尾可裁剪区间, 中间长静音段)。

    两件事本来就共用**同一套**帧划分（20 ms / hop 10 ms）、同一个参考分位（p95）与
    同一道阈值（`max(p95 - dynamic_range_db, floor_dbfs)`），此前却各跑一遍：等于把
    最贵的那趟帧电平计算（见 `frame_rms_streaming` 的实测）白付一次。合并后结果
    与两条旧路径**逐一相同**（同 win/hop/阈值、同 `>` 与 `<=` 的边界处理），
    `detect_active_span` / `count_internal_gaps` 保留为薄封装。
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    n = len(x)
    win = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, win // 2)
    total_s = n / float(sample_rate)
    if n < win:
        return (
            ActiveSpan(0, n, 0.0, total_s, floor_dbfs, total_s),
            [],
        )

    frame_db = 20.0 * np.log10(
        np.maximum(
            frame_rms_streaming(x, win, hop, progress=progress, cancel=cancel),
            _FLOOR,
        )
    )
    threshold = max(float(np.percentile(frame_db, 95)) - dynamic_range_db, floor_dbfs)

    # --- 首尾可裁剪区间 ---
    active = np.nonzero(frame_db > threshold)[0]
    if active.size == 0:
        span = ActiveSpan(0, n, 0.0, total_s, threshold, total_s)
    else:
        pad = int(sample_rate * pad_ms / 1000.0)
        start = max(0, int(active[0]) * hop - pad)
        end = min(n, int(active[-1]) * hop + win + pad)
        span = ActiveSpan(
            start_frame=start,
            end_frame=end,
            start_s=start / float(sample_rate),
            end_s=end / float(sample_rate),
            threshold_dbfs=threshold,
            total_s=total_s,
        )

    # --- 中间的长静音段（只报告，默认不删）---
    quiet = frame_db <= threshold
    gaps: list[tuple[float, float]] = []
    min_gap_frames = max(1, int(min_gap_ms / (frame_ms / 2.0)))
    run_start: int | None = None
    for i, is_quiet in enumerate(quiet):
        if is_quiet and run_start is None:
            run_start = i
        elif not is_quiet and run_start is not None:
            if i - run_start >= min_gap_frames:
                gaps.append((run_start * hop / sample_rate, i * hop / sample_rate))
            run_start = None
    if run_start is not None and len(quiet) - run_start >= min_gap_frames:
        gaps.append((run_start * hop / sample_rate, len(quiet) * hop / sample_rate))
    return span, gaps


def count_internal_gaps(
    samples: np.ndarray,
    sample_rate: int,
    *,
    min_gap_ms: float = 1000.0,
    dynamic_range_db: float = 40.0,
    floor_dbfs: float = -75.0,
    frame_ms: float = 20.0,
) -> list[tuple[float, float]]:
    """列出中间的长静音段（仅用于报告，默认不删除——删掉会改变时间轴）。

    与首尾裁剪共用一次帧电平遍历，见 `detect_active_span_and_gaps`。
    """
    _span, gaps = detect_active_span_and_gaps(
        samples, sample_rate,
        dynamic_range_db=dynamic_range_db, floor_dbfs=floor_dbfs,
        min_gap_ms=min_gap_ms, frame_ms=frame_ms,
    )
    return gaps


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

def normalize(
    samples: np.ndarray,
    *,
    target_dbfs: float = -3.0,
    mode: str = "peak",
    ceiling_dbfs: float = -0.5,
) -> tuple[np.ndarray, float]:
    """增益归一化。返回 (处理后, 实际施加的增益 dB)。

    mode='peak'：把峰值提到 target_dbfs。
    mode='rms' ：把 RMS 提到 target_dbfs，再用 ceiling_dbfs 限制峰值避免削顶。
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return x, 0.0

    if mode == "rms":
        cur = rms_blockwise(x)                 # 分块：避免整段 float64 副本（见实测表）
    else:
        cur = peak_blockwise(x)

    if cur <= _FLOOR:
        return x, 0.0

    gain_db = target_dbfs - to_dbfs(cur)
    gain = 10.0 ** (gain_db / 20.0)
    y = x * gain

    # RMS 模式下峰值可能越界，退让增益
    peak_after = peak_blockwise(y)
    ceiling = from_dbfs(ceiling_dbfs)
    if peak_after > ceiling > 0:
        shrink = ceiling / peak_after
        y = y * shrink
        gain_db += 20.0 * np.log10(shrink)

    return np.ascontiguousarray(y, dtype=np.float32), float(gain_db)


# ---------------------------------------------------------------------------
# STFT / ISTFT（分块互不重叠的 Hann + 加权 OLA）
# ---------------------------------------------------------------------------

def _hann(n: int) -> np.ndarray:
    return np.hanning(n + 1)[:n].astype(np.float32)   # 周期性 Hann（COLA 更好）


def _stft_block(x: np.ndarray, win: np.ndarray, hop: int) -> np.ndarray:
    n_fft = len(win)
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad + n_fft), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[idx] * win, axis=1)


def _istft_block(spec: np.ndarray, win: np.ndarray, hop: int, length: int) -> np.ndarray:
    """重叠相加重建（加权 OLA）。

    **这里刻意保留逐帧循环，因为"向量化"实测更慢**（2026-10-01，同一台机器）：
    把输出按 hop 分块、把帧矩阵 reshape 成 `(n_frames, n_fft/hop, hop)`、用 4 次整块
    加法替代循环之后，30 s 块上 median **62.5 ms → 53.8 ms（即新实现慢 15%）**；
    只有把 OLA 单独摘出来量才是"新的略快"（17.7 vs 20.1 ms）——整块加法每次都要
    重新读写整张输出数组，而逐帧只碰 8 KB 的连续切片（对缓存友好）。同一块里真正的
    大头是 `irfft`（24 ms）而不是这个循环。结论：**别为了"看起来更向量化"改这里**。

    （另一条同期实测、可作为后续选项的加速路径：`scipy.fft.rfft/irfft(workers=-1)`
    在同一块上是 13.4 ms vs numpy 28.0 ms（2.1×），但会引入对 scipy 的**显式**依赖，
    本轮不做。）
    """
    n_fft = len(win)
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * win
    out_len = hop * (len(frames) - 1) + n_fft
    out = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    w2 = (win.astype(np.float64)) ** 2
    for i in range(len(frames)):
        s = i * hop
        out[s:s + n_fft] += frames[i]
        wsum[s:s + n_fft] += w2
    out /= np.maximum(wsum, 1e-8)
    pad = n_fft // 2
    return out[pad:pad + length].astype(np.float32)


def spectral_process(
    samples: np.ndarray,
    sample_rate: int,
    *,
    hp_cutoff_hz: float = 80.0,
    gate_reduction_db: float = 12.0,
    gate_strength: float = 1.5,
    noise_percentile: float = 20.0,
    n_fft: int = 1024,
    block_s: float = 30.0,
    progress: ProgressCb | None = None,
    cancel: CancelCb | None = None,
) -> np.ndarray:
    """DSP 基线：高通 + 自适应谱门限（谱减法）。

    做法：每块做 STFT → 以该块内各频点幅度的低分位估计噪声底 → 按超出量计算增益 →
    增益在时间轴上做 3 帧平滑以压制"音乐噪声" → 逆变换。

    不做任何"内容推断"，所以不会凭空造出高频，但也去不掉非平稳噪声。
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    n = len(x)
    if n == 0:
        return x

    win = _hann(n_fft)
    hop = n_fft // 4
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # 高通：对截止频率以下做平滑滚降（不是硬切，避免振铃）
    hp_gain = np.ones_like(freqs)
    if hp_cutoff_hz > 0:
        hp_gain = np.clip(freqs / max(hp_cutoff_hz, 1e-6), 0.0, 1.0).astype(np.float32)
        hp_gain = hp_gain ** 2

    floor_gain = float(10.0 ** (-abs(gate_reduction_db) / 20.0))
    block = max(n_fft, int(sample_rate * block_s))
    out = np.empty(n, dtype=np.float32)
    pos = 0
    prev_gain: np.ndarray | None = None

    while pos < n:
        _check(cancel)
        end = min(n, pos + block)
        chunk = x[pos:end]
        # 块尾补一点上下文，抵消 STFT 的边界影响
        if end < n:
            chunk = x[pos:min(n, end + n_fft)]

        spec = _stft_block(chunk, win, hop)
        mag = np.abs(spec)

        if gate_reduction_db > 0 and mag.size:
            floor = np.percentile(mag, noise_percentile, axis=0)   # (bins,)
            safe = np.maximum(mag, 1e-10)
            ratio = floor[None, :] / safe
            gain = 1.0 - gate_strength * ratio
            gain = np.clip(gain, floor_gain, 1.0)

            # 时间轴 3 帧平滑，压制音乐噪声。
            # 等价于逐列 `np.convolve(col, [0.25,0.5,0.25], mode="same")`（两端按零填充），
            # 但不再对每个频点做一次 Python 调用：4096 块长下每块 513 个频点，
            # 4139 s 素材两声道就是 14 万次 Python 调用（实测 1200 s 单声道约 4 s 量级）。
            if gain.shape[0] >= 3:
                sm = gain * 0.5                      # 中心项
                sm[:-1] += 0.25 * gain[1:]           # 右邻
                sm[1:] += 0.25 * gain[:-1]           # 左邻
                gain = sm
            # 帧数 < 3 时与旧实现一致地**不做**平滑（长度 2 的 same 卷积边界行为不同，
            # 这条路径在 30 s 块长下走不到，但不要顺手改语义）
            # 与上一块衔接，避免块边界出现增益跳变
            if prev_gain is not None and prev_gain.shape == gain.shape:
                gain = 0.5 * gain + 0.5 * prev_gain
            prev_gain = gain
        else:
            gain = np.ones_like(mag)

        spec = spec * (gain * hp_gain[None, :])
        y = _istft_block(spec, win, hop, len(chunk))
        take = min(block, n - pos)
        out[pos:pos + take] = y[:take]
        pos += take

        _tick(progress, pos / n, f"DSP 处理 {pos / n * 100:.0f}%")

    return np.ascontiguousarray(out, dtype=np.float32)
