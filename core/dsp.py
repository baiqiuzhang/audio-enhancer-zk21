"""电平分析、静音检测裁剪、归一化，以及 DSP 基线处理。

纯 numpy 实现，不依赖 scipy/librosa——venv 里刻意没有它们。
长文件一律分块处理，避免一次性物化整个 STFT 矩阵（16 分钟 @16k 的全量 STFT 会吃掉数百 MB）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ProgressCb = Callable[[float, str], None]
CancelCb = Callable[[], bool]


class Cancelled(RuntimeError):
    """调用方请求取消。"""


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

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    dc = float(np.mean(x, dtype=np.float64))
    clipped = float(np.mean(np.abs(x) >= 0.999))

    win = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, win // 2)
    if n >= win:
        frames = _frames(x, win, hop)
        frame_rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
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


def _frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """按 win/hop 切帧，返回 (n_frames, win)。"""
    if len(x) < win:
        return x.reshape(1, -1)
    n_frames = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


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
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    n = len(x)
    win = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, win // 2)

    if n < win:
        return ActiveSpan(0, n, 0.0, n / float(sample_rate), floor_dbfs, n / float(sample_rate))

    frames = _frames(x, win, hop)
    frame_rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, _FLOOR))

    ref = float(np.percentile(frame_db, 95))
    threshold = max(ref - dynamic_range_db, floor_dbfs)

    active = np.nonzero(frame_db > threshold)[0]
    if active.size == 0:
        return ActiveSpan(0, n, 0.0, n / float(sample_rate), threshold, n / float(sample_rate))

    pad = int(sample_rate * pad_ms / 1000.0)
    start = max(0, int(active[0]) * hop - pad)
    end = min(n, int(active[-1]) * hop + win + pad)

    return ActiveSpan(
        start_frame=start,
        end_frame=end,
        start_s=start / float(sample_rate),
        end_s=end / float(sample_rate),
        threshold_dbfs=threshold,
        total_s=n / float(sample_rate),
    )


def trim_to_span(samples: np.ndarray, span: ActiveSpan) -> np.ndarray:
    return np.ascontiguousarray(samples[span.start_frame:span.end_frame])


def count_internal_gaps(
    samples: np.ndarray,
    sample_rate: int,
    *,
    min_gap_ms: float = 1000.0,
    dynamic_range_db: float = 40.0,
    floor_dbfs: float = -75.0,
    frame_ms: float = 20.0,
) -> list[tuple[float, float]]:
    """列出中间的长静音段（仅用于报告，默认不删除——删掉会改变时间轴）。"""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    win = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, win // 2)
    if len(x) < win:
        return []

    frames = _frames(x, win, hop)
    frame_rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, _FLOOR))
    threshold = max(float(np.percentile(frame_db, 95)) - dynamic_range_db, floor_dbfs)

    quiet = frame_db <= threshold
    gaps: list[tuple[float, float]] = []
    run_start: int | None = None
    min_gap_frames = max(1, int(min_gap_ms / (frame_ms / 2.0)))

    for i, is_quiet in enumerate(quiet):
        if is_quiet and run_start is None:
            run_start = i
        elif not is_quiet and run_start is not None:
            if i - run_start >= min_gap_frames:
                gaps.append((run_start * hop / sample_rate, i * hop / sample_rate))
            run_start = None
    if run_start is not None and len(quiet) - run_start >= min_gap_frames:
        gaps.append((run_start * hop / sample_rate, len(quiet) * hop / sample_rate))
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
        cur = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    else:
        cur = float(np.max(np.abs(x)))

    if cur <= _FLOOR:
        return x, 0.0

    gain_db = target_dbfs - to_dbfs(cur)
    gain = 10.0 ** (gain_db / 20.0)
    y = x * gain

    # RMS 模式下峰值可能越界，退让增益
    peak_after = float(np.max(np.abs(y)))
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

            # 时间轴 3 帧平滑，压制音乐噪声
            if gain.shape[0] >= 3:
                kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
                gain = np.apply_along_axis(
                    lambda col: np.convolve(col, kernel, mode="same"), 0, gain
                )
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
