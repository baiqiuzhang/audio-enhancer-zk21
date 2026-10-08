"""听感增强与混响分析。

这里的东西**只做修饰**：改变已有频点的能量分布、稳定动态，不产生新内容。
与「带宽扩展」那种会编造高频的修复类模型是两回事，界面上必须分开标注。

纯 numpy，不引入 scipy / numba —— 复用 dsp.py 里已验证的分块 STFT 基础设施。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from .dsp import (
    _check,
    _hann,
    _istft_block,
    _stft_block,
    _tick,
    frame_rms_streaming,
)

ProgressCb = Callable[[float, str], None]
CancelCb = Callable[[], bool]

# 判定"素材有多混响"的斜率分档（dB/s，越大越干）
SLOPE_DRY = 100.0
SLOPE_SLIGHT = 60.0
SLOPE_NOTICEABLE = 35.0


# ---------------------------------------------------------------------------
# 高通 + 高频搁架 EQ
# ---------------------------------------------------------------------------

def _bell_gain(freqs: np.ndarray, center_hz: float, db: float,
               width_oct: float) -> np.ndarray:
    """对数频率上的钟形增益，用于中低频「去闷」衰减。

    用高斯钟形而不是搁架：发闷是**一段带宽**（实测 200–500 Hz）而不是"某个频率以上"，
    用搁架会把 100 Hz 以下的基频也一起砍掉，声音会变薄。
    `width_oct` 以倍频程为单位；0 dB 时返回全 1，不做任何事。
    """
    if not db or center_hz <= 0:
        return np.ones_like(freqs, dtype=np.float64)
    f = np.maximum(np.asarray(freqs, dtype=np.float64), 1e-6)
    d = np.log2(f / float(center_hz))
    w = max(float(width_oct), 0.05)
    shape = np.exp(-0.5 * (d / w) ** 2)
    target = float(10.0 ** (db / 20.0))
    return (1.0 - shape) + shape * target


def shelf_gain_curve(freqs: np.ndarray, *, hp_cutoff_hz: float = 0.0,
                     shelf_hz: float = 3000.0, shelf_db: float = 0.0,
                     mud_cut_hz: float = 350.0, mud_cut_db: float = 0.0,
                     mud_cut_width_oct: float = 0.9) -> np.ndarray:
    """合成一条频点增益曲线：低频滚降 + 高频搁架提升 + 中低频钟形衰减。

    三项合成一条曲线，所以只需要一次 STFT/ISTFT：
    - 高通用 `(f/fc)^2` 平滑滚降，不用硬切（硬切会在截止点产生振铃）
    - 搁架用 tanh 过渡，避免拐点突兀、听起来"有台阶"
    - 去闷用对数频率上的高斯钟形（见 _bell_gain），只压"发闷"那一段带宽

    默认 `mud_cut_db=0` 即不做去闷，所以不传该参数时行为与加它之前完全一致。
    """
    g = np.ones_like(freqs, dtype=np.float64)
    if hp_cutoff_hz and hp_cutoff_hz > 0:
        g *= np.clip(freqs / float(hp_cutoff_hz), 0.0, 1.0) ** 2
    if shelf_db:
        target = float(10.0 ** (shelf_db / 20.0))
        width = max(float(shelf_hz) * 0.5, 1.0)
        # 刻意**不**对 t 做 np.clip：截断会让 tanh(1.5)≈0.905 永远到不了 1，
        # 于是 w 的残余值给整个低频端留下约 +0.4 dB 的虚假提升（+6 dB 搁架时），
        # 这个"高频搁架"就不再是纯搁架了。不截断时 tanh 会自己饱和到 ±1。
        t = (freqs - float(shelf_hz)) / width
        w = 0.5 * (np.tanh(t * 1.5) + 1.0)
        g *= (1.0 - w) + w * target
    if mud_cut_db:
        g *= _bell_gain(freqs, mud_cut_hz, mud_cut_db, mud_cut_width_oct)
    return g.astype(np.float32)


def spectral_shape(
    samples: np.ndarray,
    sample_rate: int,
    *,
    hp_cutoff_hz: float = 0.0,
    shelf_hz: float = 3000.0,
    shelf_db: float = 0.0,
    mud_cut_hz: float = 350.0,
    mud_cut_db: float = 0.0,
    mud_cut_width_oct: float = 0.9,
    n_fft: int = 1024,
    block_s: float = 30.0,
    progress: ProgressCb | None = None,
    cancel: CancelCb | None = None,
    workers: int = 1,
) -> np.ndarray:
    """高通 + 高频搁架 EQ + 中低频去闷。分块处理，长文件内存受控。

    `workers > 1` 时在**声道之间**并行。实测（2026-10-01，1200 s 双声道 48 kHz）：
    串行 15.09 s → 线程并行 9.92 s，**1.52×**，且逐位相同。可行的原因是这段是 FFT 主导，
    而 numpy 的 pocketfft 在 C 层释放 GIL，线程真能并行；每声道内部仍是原来的分块循环。
    额外内存只有每声道一份工作缓冲（远小于"再复制一遍整段"），所以调用方可以直接把它接到
    已有的声道并行开关上。
    """
    x = np.asarray(samples, dtype=np.float32)
    if hp_cutoff_hz <= 0 and not shelf_db and not mud_cut_db:
        # 无操作分支：**不**返回 x.copy()。69 分钟立体声那是一次 1.59 GB 的白拷贝，
        # 而调用方（pipeline）拿到后只读它；本函数从不原地修改输入，所以直接返回是安全的。
        return x

    multi = x.ndim == 2
    channels = (media_channels(x) if multi else [x])

    win = _hann(n_fft)
    hop = n_fft // 4
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    curve = shelf_gain_curve(freqs, hp_cutoff_hz=hp_cutoff_hz,
                             shelf_hz=shelf_hz, shelf_db=shelf_db,
                             mud_cut_hz=mud_cut_hz, mud_cut_db=mud_cut_db,
                             mud_cut_width_oct=mud_cut_width_oct)

    total = len(channels)

    def run_one(ci: int, ch: np.ndarray) -> tuple[int, np.ndarray]:
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
        return ci, y

    if workers > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=min(workers, total)) as pool:
            results = list(pool.map(lambda a: run_one(*a), list(enumerate(channels))))
        results.sort(key=lambda r: r[0])
        out = [y for _ci, y in results]
    else:
        out = [run_one(ci, ch)[1] for ci, ch in enumerate(channels)]

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
    # 帧电平走分块实现：这是 `_frames` 一次性物化整张矩阵的替代品。
    # 5 ms 帧在 48 kHz 上是 240 采样，4139 s 素材会产生 82.8 万帧——
    # 旧口径的矩阵是 0.8 GB(float32) + 1.6 GB(float64)，本预设用不到压缩，
    # 但一旦勾上压缩就会白付这笔内存。
    rms = frame_rms_streaming(detect, hop, hop) + 1e-12
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
    # 分块帧电平：旧写法把 (n_frames, win) 整张矩阵物化，成本随素材时长线性增长
    # （见 core/dsp.py 的 frame_rms_streaming 实测表）。这段被 source 与 enhanced 各跑一次。
    rms = frame_rms_streaming(x, win, hop) + 1e-12
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

    注意：早先这里写过「DPDFNet 的训练信号模型里含房间冲激响应，所以它本来就在
    压混响」——那是**未验证的推断被当成事实写了**，已更正。DPDFNet 论文的定位是
    causal single-channel speech enhancement，并未声称去混响。

    斜率变陡说明有"尾巴"被压掉，但**不能**据此断言是去混响：降噪器的硬门限
    （把停顿压成数字静音）同样会让斜率变陡。实测同一素材降噪前后 42 → 62 dB/s，
    但发闷指数几乎不变（+16.13 → +16.68），说明它压的是噪声尾巴而非房间混响本身。
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
