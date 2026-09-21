"""单通道去混响（WPE）—— **本实现实测无效，已在流水线中停用**。

实测（2026-09-21，合成混响素材，taps=10 / delay=3 / iterations=3）：
  · 语音偏移后 100–400 ms 的尾部能量相对起始**升高 15.7 dB**（期望是下降）
  · 输出与干声的样本级相关性 **−0.35**（负值说明输出已与原始信号无关）
  · 衰减斜率反而变缓（90.8 → 68.4 dB/s）
也就是说它不是"效果有限"，而是**错的**。继续沿这个方向调试需要重做数值实现
（含归一化、对角加载与迭代收敛判据），本轮没有做，所以流水线里显式拒绝启用它，
避免给用户一个会毁掉素材的开关。

本模块保留下来是为了：① 记录已排除的路线与实测数字；② 若将来要接真正的
单通道去混响（例如神经去混响模型），这里的接口与文档可作为起点。

以下原文说明仍然成立，供将来参考：

单通道下 WPE 退化为"用信号自己的过去帧预测当前帧"，无法区分"过去可预测是因为
混响尾巴"还是"因为语音本身高度可预测"，因此会连带把语音白化。
公开文献里 VACE-WPE（arXiv:2102.05259）存在的唯一理由就是单通道 WPE 不够好。

**先读这段再决定要不要开**：WPE 是为**多麦克风**设计的。单通道下它退化为
"用信号自己的过去帧预测当前帧"，无法区分"过去可预测是因为混响尾巴"还是
"因为语音本身高度可预测"，因此会连带把语音白化（听起来更"薄"、更干）。
公开文献里 VACE-WPE（arXiv:2102.05259）存在的唯一理由就是单通道 WPE 不够好。
它被验证过的收益场景是**远场 ASR 的 WER**，而不是听感。

纯 ONNX 的单通道去混响模型目前不存在（镜像上唯一候选是 456 MB 的 OpenVINO IR、
GPL-3.0、且是音乐向训练），sherpa-onnx 也完全没有去混响能力（已从 Python 绑定、
C API 头文件、DLL 字符串三处确认）。所以这里提供的是"可测 + 可选 + 默认关"，
而不是一个可靠的卖点。

另外注意：DPDFNet 的训练信号模型本身就含房间冲激响应（x = s·r + n），
它已经在做一定程度的"把混响当噪声压掉"。建议先看 dsp 的衰减斜率指标，
并与降噪后的斜率对比，再决定是否需要 WPE。

算法：Nakatani 等人的 WPE（加权预测误差）。实现按 `nara_wpe` 的 MIT 许可纯 numpy
核心（`wpe_v8` / 相关矩阵 / 滤波器求解 / 滤波操作）重写，**不安装 nara_wpe 包**——
那个包的 `utils.py` 在顶层无条件 import scipy，与本项目的轻量依赖策略冲突。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ProgressCb = Callable[[float, str], None]
CancelCb = Callable[[], bool]


def _stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """(T,) → (F, T') 复数谱。"""
    win = np.hanning(n_fft + 1)[:n_fft].astype(np.float64)
    pad = n_fft // 2
    xp = np.pad(x.astype(np.float64), (pad, pad + n_fft), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[idx] * win, axis=1).T


def _istft(spec: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    win = np.hanning(n_fft + 1)[:n_fft].astype(np.float64)
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1) * win
    out_len = hop * (len(frames) - 1) + n_fft
    out = np.zeros(out_len)
    wsum = np.zeros(out_len)
    w2 = win**2
    for i in range(len(frames)):
        s = i * hop
        out[s:s + n_fft] += frames[i]
        wsum[s:s + n_fft] += w2
    out /= np.maximum(wsum, 1e-10)
    pad = n_fft // 2
    return out[pad:pad + length].astype(np.float32)


def _wpe_filter(Y: np.ndarray, *, taps: int, delay: int, iterations: int,
                eps: float = 1e-6) -> np.ndarray:
    """对单个频点的复数谱 Y (T,) 做 WPE，返回去混响后的谱。

    信号模型：y[t] = d[t] + Σ_k g_k^H · y[t - delay - k]
    迭代估计 g 并减掉预测出的晚期分量。
    """
    T = Y.shape[0]
    if T < taps + delay + 2:
        return Y.copy()

    # 延迟堆叠：Yt[k, t] = Y[t - delay - k]（不足处补零）
    idx = np.arange(delay, delay + taps)[:, None]
    cols = np.arange(T)[None, :] - idx
    Yt = np.where(cols >= 0, Y[np.clip(cols, 0, T - 1)], 0.0)

    d = Y.copy()
    eye = np.eye(taps)
    for _ in range(max(1, iterations)):
        R = (Yt @ Yt.conj().T) / T                       # E[ỹ ỹ^H]
        r = (Yt @ d.conj()) / T                          # E[ỹ d*]
        load = eps * max(float(np.trace(R).real) / taps, 1e-12)
        try:
            g = np.linalg.solve(R + load * eye, r)
        except np.linalg.LinAlgError:
            break
        d = Y - (g.conj()[:, None] * Yt).sum(axis=0)
    return d


def dereverb(
    mono: np.ndarray,
    sample_rate: int,
    *,
    taps: int = 10,
    delay: int = 3,
    iterations: int = 3,
    n_fft: int = 512,
    block_s: float = 30.0,
    progress: ProgressCb | None = None,
    cancel: CancelCb | None = None,
) -> np.ndarray:
    """单通道 WPE 去混响。分块处理以约束内存。

    内存提醒：nara_wpe 的整段实现里，中间张量随音频长度线性增长——60 分钟 16 kHz 素材
    约 9 GB，本机 3–4 GB 空闲内存下必然 OOM，所以这里**必须**分块。
    块间直接拼接（不交叉淡化）：WPE 是逐频点的线性滤波，块边界不会产生增益突变。
    """
    x = np.asarray(mono, dtype=np.float32).reshape(-1)
    n = len(x)
    if n < n_fft * 4:
        return x

    hop = n_fft // 2
    block = max(n_fft * 4, int(sample_rate * block_s))
    out = np.empty(n, dtype=np.float32)
    pos = 0
    total = max(1, int(np.ceil(n / block)))
    idx = 0

    while pos < n:
        if cancel and cancel():
            raise RuntimeError("已取消")
        end = min(n, pos + block)
        spec = _stft(x[pos:end], n_fft, hop)              # (F, T)
        for f in range(spec.shape[0]):
            spec[f] = _wpe_filter(spec[f], taps=taps, delay=delay, iterations=iterations)
        out[pos:end] = _istft(spec, n_fft, hop, end - pos)
        pos = end
        idx += 1
        if progress:
            progress(min(0.99, idx / total), f"去混响 {idx}/{total} 块")

    if progress:
        progress(1.0, "去混响完成")
    return out

