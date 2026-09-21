"""听感增强与混响分析。

这里的东西**只做修饰**：改变已有频点的能量分布、稳定动态，不产生新内容。
与「带宽扩展」那种会编造高频的修复类模型是两回事，界面上必须分开标注。

纯 numpy，不引入 scipy / numba —— 复用 dsp.py 里已验证的分块 STFT 基础设施。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .dsp import _check, _frames, _hann, _istft_block, _stft_block, _tick

ProgressCb = Callable[[float, str], None]
CancelCb = Callable[[], bool]

# 判定"素材有多混响"的斜率分档（dB/s，越大越干）
SLOPE_DRY = 100.0
SLOPE_SLIGHT = 60.0
SLOPE_NOTICEABLE = 35.0


# ---------------------------------------------------------------------------
# 高通 + 高频搁架 EQ
# ---------------------------------------------------------------------------

def shelf_gain_curve(freqs: np.ndarray, *, hp_cutoff_hz: float = 0.0,
                     shelf_hz: float = 4000.0, shelf_db: float = 0.0) -> np.ndarray:
    """合成一条频点增益曲线：低频滚降 + 高频搁架提升。

    两个滤波器合成一条曲线，所以只需要一次 STFT/ISTFT：
    - 高通用 `(f/fc)^2` 平滑滚降，不用硬切（硬切会在截止点产生振铃）
    - 搁架用 tanh 过渡，避免拐点突兀、听起来"有台阶"
    """
    g = np.ones_like(freqs, dtype=np.float64)
    if hp_cutoff_hz and hp_cutoff_hz > 0:
        g *= np.clip(freqs / float(hp_cutoff_hz), 0.0, 1.0) ** 2
    if shelf_db:
        target = float(10.0 ** (shelf_db / 20.0))
        width = max(float(shelf_hz) * 0.5, 1.0)
        t = np.clip((freqs - float(shelf_hz)) / width, -1.0, 1.0)
        w = 0.5 * (np.tanh(t * 1.5) + 1.0)
        g *= (1.0 - w) + w * target
    return g.astype(np.float32)


def spectral_shape(
    samples: np.ndarray,
    sample_rate: int,
    *,
    hp_cutoff_hz: float = 0.0,
    shelf_hz: float = 4000.0,
    shelf_db: float = 0.0,
    n_fft: int = 1024,
    block_s: float = 30.0,
    progress: ProgressCb | None = None,
    cancel: CancelCb | None = None,
) -> np.ndarray:
    """高通 + 高频搁架 EQ。分块处理，长文件内存受控。"""
    x = np.asarray(samples, dtype=np.float32)
    if hp_cutoff_hz <= 0 and not shelf_db:
        return x.copy()

    multi = x.ndim == 2
    channels = (media_channels(x) if multi else [x])

    win = _hann(n_fft)
    hop = n_fft // 4
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    curve = shelf_gain_curve(freqs, hp_cutoff_hz=hp_cutoff_hz,
                             shelf_hz=shelf_hz, shelf_db=shelf_db)

    out: list[np.ndarray] = []
    total = len(channels)
    for ci, ch in enumerate(channels):
        _check(cancel)
        n = len(ch)
        block = max(n_fft, int(sample_rate * block_s))
        y = np.empty(n, dtype=np.float32)
        pos = 0
        while pos < n:
            _check(cancel)
            end = min(n, pos + block)
            spec = _stft_block(ch[pos:end], win, hop) * curve[None, :]
            y[pos:end] = _istft_block(spec, win, hop, end - pos)
            pos = end
            _tick(progress, (ci + pos / max(1, n)) / total,
                  f"EQ 处理 {ci + 1}/{total} 声道")
        out.append(y)

    return _join(out) if multi else out[0]


def media_channels(x: np.ndarray) -> list[np.ndarray]:
    return [np.ascontiguousarray(x[:, i], dtype=np.float32) for i in range(x.shape[1])]


def _join(channels: list[np.ndarray]) -> np.ndarray:
    n = min(len(c) for c in channels)
    return np.ascontiguousarray(np.stack([c[:n] for c in channels], axis=1), dtype=np.float32)


# ---------------------------------------------------------------------------
# 温和压缩 / AGC
# ---------------------------------------------------------------------------

def _exp_smooth(signal: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    """因果指数平滑（单极跟随器的 FIR 近似）—— 全向量化，不需要 scipy。

    用 FIR 近似而不是真正的递归 IIR：递归在 numpy 里只能写 Python 循环，
    而这里信号长度是"帧数"量级，FIR 的精度对包络平滑完全够用。
    前置填第一个值，避免开头被拖到 0。
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return signal
    tau_s = max(float(tau_s), 1e-4)
    n = max(1, int(tau_s * fs * 4))
    t = np.arange(n) / fs
    k = np.exp(-t / tau_s)
    k /= k.sum()
    padded = np.concatenate([np.full(n, signal[0]), signal])
    return np.convolve(padded, k, mode="full")[n - 1: n - 1 + len(signal)]


def compress(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float = -18.0,
    ratio: float = 2.0,
    attack_ms: float = 10.0,
    release_ms: float = 200.0,
    makeup_db: float = 0.0,
    block_ms: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """温和压缩 / AGC，让语音更"贴耳"、动态更稳。

    攻击/释放用**两个时间常数的包络取 min** 实现（全向量化）：
    - 衰减量刚出现：快包络先变负 → min 取快包络 = 快速攻击
    - 衰减量消失：快包络先回升，但慢包络还压着 → min 取慢包络 = 缓慢释放

    多声道共用一条增益曲线（检测信号取声道混合），以免破坏声像。
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0 or ratio <= 1.0:
        return x, {"applied": False}

    multi = x.ndim == 2
    detect = x.mean(axis=1) if multi else x

    hop = max(1, int(sample_rate * block_ms / 1000.0))
    frames = _frames(detect, hop, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-12
    level_db = 20.0 * np.log10(rms)

    over = np.maximum(level_db - float(threshold_dbfs), 0.0)
    gain_db = -(over * (1.0 - 1.0 / float(ratio)))          # 目标衰减（≤ 0）

    fs_frames = sample_rate / hop
    fast = _exp_smooth(gain_db, fs_frames, attack_ms / 1000.0)
    slow = _exp_smooth(gain_db, fs_frames, release_ms / 1000.0)
    smoothed = np.minimum(fast, slow) + float(makeup_db)    # 快攻击 + 慢释放

    gain_curve = 10.0 ** (smoothed / 20.0)
    gain = np.repeat(gain_curve, hop)[: len(detect)]
    if len(gain) < len(detect):
        gain = np.pad(gain, (0, len(detect) - len(gain)), mode="edge")

    y = np.clip(x * (gain[:, None] if multi else gain), -1.0, 1.0)
    y = np.ascontiguousarray(y, dtype=np.float32)

    jumps = np.abs(np.diff(smoothed)) if smoothed.size > 1 else np.zeros(1)
    return y, {
        "applied": True,
        "max_reduction_db": round(float(-np.min(smoothed)), 2),
        "median_reduction_db": round(float(-np.median(smoothed)), 2),
        "gain_jump_max_db": round(float(np.max(jumps)), 3),
    }


# ---------------------------------------------------------------------------
# 混响检测：语音 offset 之后的能量衰减斜率
# ---------------------------------------------------------------------------

def estimate_decay_slope_db_per_s(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 16.0,
    tail_ms: tuple[float, float] = (40.0, 300.0),
    dynamic_range_db: float = 40.0,
    floor_dbfs: float = -75.0,
    max_segments: int = 400,
) -> dict:
    """估计"语音停止后的能量衰减斜率"（dB/s）—— 判断素材有没有明显房间回声。

    只在**衰减段**测量，所以不受语音内容干扰：找出帧电平从较高处跌落的位置，
    对之后的几十到几百毫秒做线性拟合。近讲干声的衰减近似阶跃（斜率很大，通常
    ≥100–200 dB/s），有明显混响的房间只有 10–40 dB/s。

    不提供 C50 / DRR：它们需要已知的房间冲激响应，从纯录音里算不准，给了也是误导。
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    n = len(x)
    win = max(1, int(sample_rate * frame_ms / 1000.0))
    if n < win * 6:
        return {"slope_db_per_s": None, "num_segments": 0, "reason": "素材太短，无法测量衰减"}

    hop = max(1, win // 2)
    frames = _frames(x, win, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-12
    frame_db = 20.0 * np.log10(rms)

    ref = float(np.percentile(frame_db, 95))
    threshold = max(ref - dynamic_range_db, floor_dbfs)
    lo_f = max(1, int(tail_ms[0] / (frame_ms / 2.0)))
    hi_f = max(lo_f + 2, int(tail_ms[1] / (frame_ms / 2.0)))

    slopes: list[float] = []
    for i in range(1, len(frame_db) - hi_f):
        if frame_db[i] <= threshold or frame_db[i] - threshold < 10.0:
            continue                                   # 起点要高出噪声底足够多
        if frame_db[i] < frame_db[i - 1]:
            continue                                   # 只在能量开始下降处取样
        seg = frame_db[i + lo_f: i + hi_f]
        if len(seg) < 3 or seg[-1] > frame_db[i] - 6.0:
            continue                                   # 没真的跌下去，不算衰减段
        t = np.arange(len(seg)) * (frame_ms / 2.0) / 1000.0
        slope = float(np.polyfit(t, seg, 1)[0])         # dB/s，负值
        if np.isfinite(slope):
            slopes.append(-slope)                      # 转成正的：越大越"干"
        if len(slopes) >= max_segments:
            break

    if not slopes:
        return {"slope_db_per_s": None, "num_segments": 0,
                "reason": "没找到可测量的衰减段"}

    arr = np.asarray(slopes)
    return {
        "slope_db_per_s": round(float(np.median(arr)), 1),
        "p25": round(float(np.percentile(arr, 25)), 1),
        "p75": round(float(np.percentile(arr, 75)), 1),
        "num_segments": int(len(arr)),
        "threshold_dbfs": round(threshold, 1),
    }


def reverb_verdict(result: dict) -> str:
    """把衰减斜率翻成人话，用来决定"值不值得开去混响"。"""
    slope = result.get("slope_db_per_s")
    if slope is None:
        return "混响：无法判断（没测到足够的衰减段）"
    if slope >= SLOPE_DRY:
        return f"混响：接近近讲干声（衰减 {slope:.0f} dB/s），去混响基本无必要"
    if slope >= SLOPE_SLIGHT:
        return f"混响：略有房间感（衰减 {slope:.0f} dB/s），去混响收益有限"
    if slope >= SLOPE_NOTICEABLE:
        return f"混响：有明显房间回声（衰减 {slope:.0f} dB/s），可试开去混响"
    return f"混响：房间回声明显（衰减 {slope:.0f} dB/s），去混响可能有用"


def decay_slope_delta(before: dict, after: dict) -> str | None:
    """降噪前后衰减斜率的变化说明。

    DPDFNet 的训练信号模型里含房间冲激响应（x = s*r + n），所以它本来就在压混响。
    如果降噪后斜率已被显著拉陡，说明混响大多已被吃掉，再去混响意义不大。
    """
    b = before.get("slope_db_per_s")
    a = after.get("slope_db_per_s")
    if b is None or a is None:
        return None
    gain = a - b
    if gain >= 25:
        return f"降噪已把衰减斜率从 {b:.0f} 拉到 {a:.0f} dB/s（+{gain:.0f}），混响大多已被压掉"
    if gain <= -10:
        return f"降噪后衰减斜率反而变缓（{b:.0f} → {a:.0f} dB/s），去混响可能仍有价值"
    return f"降噪前后衰减斜率基本不变（{b:.0f} → {a:.0f} dB/s）"
