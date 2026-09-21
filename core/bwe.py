"""带宽扩展（修复类）：把高频"补回来"。

**这是编造内容**：生成式模型会合成原始素材里不存在的高频。因此它默认关闭，
且界面必须标红——有公开实测显示这类处理可能让识别变差（详见 README）。

用 `audiosronnx`（Apache-2.0，纯 ONNX，运行时只依赖 numpy/onnxruntime/scipy，**零 torch**）。

两个关键工程点：

1. **完全不依赖 HuggingFace。** 本机 huggingface.co 被墙，而且 audiosronnx 的权重解析是
   **惰性**的——直到第一次 `upscale()` 才去解析路径。所以不能只在加载时做手脚。
   这里给各引擎模块的 `resolve` 打补丁：先查项目自己的模型目录，缺失就抛一个
   "可在界面里下载"的明确错误。**不用 os.chdir**——chdir 是进程级的，批处理多线程会互相干扰。

2. **自己分块 + 交叉淡化。** 引擎不承诺处理任意长度，长音频必须切块，否则要么 OOM、
   要么内部行为未定义。
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

import numpy as np

from .modelstore import Catalog

# 每线程一套模型实例（ORT 会话），与 denoise 引擎同样的策略
_local = threading.local()

# 各引擎的模块路径 + 权重文件名，用于打补丁与本地查找
ENGINE_MODULES = {
    "novasr": ("audiosronnx.engines.novasr", ["novasr.onnx"]),
    "lavasr": ("audiosronnx.engines.lavasr", ["backbone.onnx", "spec_head.onnx", "denoiser_core.onnx"]),
    "hifiganbwe": ("audiosronnx.engines.hifiganbwe", ["hifiganbwe_wavenet.onnx"]),
}

# 分块参数：20 秒一块、0.5 秒交叉淡化（长音频内存与接缝的折中）
CHUNK_S = 20.0
OVERLAP_S = 0.5

# ---------------------------------------------------------------------------
# 生成频段的电平平衡
# ---------------------------------------------------------------------------
# 三个 BWE 引擎的**原始输出都把生成频段抬得远高于通带**。实测（2026-09-21，真实素材
# 60 秒片段，绝对频段对比：输出各频段功率 ÷ 输入同频段功率）：
#     novasr      5k–8k −2.7 dB / 8k–12k +13.7 / 12k–20k +24.2
#     hifiganbwe  5k–8k +1.6 dB / 8k–12k +26.2 / 12k–20k +29.4
#     lavasr      5k–8k +14.0 dB / 8k–12k +71.1 / 12k–20k +89.1（它自己还 np.clip 到 ±1，已削顶）
# 低频其实**保住了**（0–5 kHz 三个引擎都是 ≈0 dB），所以"听上去只剩高频"是
# **混音增益失衡**，不是丢低频——这一点很容易用"频段占比"读反（相对占比会被
# 高得离谱的生成频段压到 1%），必须用绝对值判断。
#
# 语音的自然频谱在 500 Hz 以上约每倍频程 −6 dB，所以生成频段本就该明显低于通带。
# 这里把它压到通带之下 HF_TARGET_DB，同时把通带响度对齐回输入——一条规则适用于
# 所有引擎，并且可以用同一套频段指标验证。
HF_TARGET_DB = -12.0
BALANCE_CUTOFF_HZ = 8000.0        # 与 audiosronnx 各引擎的默认合并截止一致
_PRIMARY_BANDS = (20.0, BALANCE_CUTOFF_HZ)


class BweUnavailable(RuntimeError):
    """audiosronnx 不可用，或权重尚未下载。"""


def _band_power(x: np.ndarray, sr: int, lo: float, hi: float,
                n_fft: int = 4096) -> float:
    """[lo, hi) 频段的功率。只用于算增益（输入是片段，不是整段），内存可控。"""
    from .dsp import _hann  # noqa: PLC0415 — 复用已有窗口实现

    sig = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(sig) < n_fft * 2:
        sig = np.pad(sig, (0, n_fft * 2 - len(sig)))
    hop = n_fft // 2
    n_fr = 1 + (len(sig) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_fr)[:, None]
    frames = sig[idx] * _hann(n_fft)
    psd = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    sel = (freqs >= lo) & (freqs < min(hi, sr / 2.0))
    return float(psd[sel].sum() * df)


def balance_gains(y: np.ndarray, out_rate: int, x_ref: np.ndarray, ref_rate: int,
                  *, cutoff_hz: float = BALANCE_CUTOFF_HZ,
                  hf_target_db: float = HF_TARGET_DB) -> tuple[float, float]:
    """算出 (通带增益, 生成频段增益)，让输出响度对齐输入、且高频落在通带之下。

    返回 (1.0, 1.0) 表示不做任何改动（测不出有效功率时保守地不动）。
    """
    lo, hi = _PRIMARY_BANDS
    p_in = _band_power(x_ref, ref_rate, lo, min(hi, cutoff_hz))
    p_low = _band_power(y, out_rate, lo, min(hi, cutoff_hz))
    p_hf = _band_power(y, out_rate, cutoff_hz, out_rate / 2.0)
    if p_in <= 0 or p_low <= 0 or p_hf <= 0:
        return 1.0, 1.0
    g_low = float(np.sqrt(p_in / p_low))
    # 目标是「生成频段 = 通带 × 10^(hf_target_db/10)」
    target_hf = p_in * (10.0 ** (hf_target_db / 10.0))
    g_hf = float(np.sqrt(target_hf / p_hf))
    return g_low, g_hf


def apply_balance(x: np.ndarray, sr: int, g_low: float, g_hf: float, *,
                  cutoff_hz: float = BALANCE_CUTOFF_HZ, n_fft: int = 1024,
                  block_s: float = 30.0) -> np.ndarray:
    """按两条增益把通带与生成频段分别缩放，过渡区用 tanh 平滑（避免台阶感）。"""
    if abs(g_low - 1.0) < 1e-6 and abs(g_hf - 1.0) < 1e-6:
        return np.asarray(x, dtype=np.float32)
    from .dsp import _hann, _istft_block, _stft_block  # noqa: PLC0415

    win = _hann(n_fft)
    hop = n_fft // 4
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    width = max(cutoff_hz * 0.15, 1.0)
    w = 0.5 * (np.tanh(((freqs - cutoff_hz) / width) * 1.5) + 1.0)
    curve = ((1.0 - w) * g_low + w * g_hf).astype(np.float32)

    sig = np.asarray(x, dtype=np.float32).reshape(-1)
    n = len(sig)
    out = np.empty(n, dtype=np.float32)
    block = max(n_fft, int(sr * block_s))
    pos = 0
    while pos < n:
        end = min(n, pos + block)
        spec = _stft_block(sig[pos:end], win, hop) * curve[None, :]
        out[pos:end] = _istft_block(spec, win, hop, end - pos)
        pos = end
    return out


def _import_audiosronnx():
    try:
        import audiosronnx  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        raise BweUnavailable(
            f"带宽扩展不可用：未安装 audiosronnx（{type(exc).__name__}: {exc}）。\n"
            "请运行：.venv\\Scripts\\python.exe -m pip install --pre audiosronnx"
        ) from exc
    return audiosronnx


def local_weight_dir(engine_id: str, catalog: Catalog) -> str | None:
    """该项目里该引擎权重所在目录（取清单里第一个文件所在目录）。"""
    spec = catalog.engine(engine_id)
    if spec is None or not spec.files:
        return None
    return os.path.dirname(catalog.resolve(spec.files[0].path))


def missing_weights(engine_id: str, catalog: Catalog) -> list[str]:
    from .modelstore import file_ready  # noqa: PLC0415

    spec = catalog.engine(engine_id)
    if spec is None:
        return []
    return [os.path.basename(f.path) for f in spec.files if not file_ready(f, catalog)]


def _install_local_resolver(engine_id: str, catalog: Catalog) -> None:
    """把该引擎模块里的 `resolve` 换成"本地优先"版本（幂等）。"""
    module_name, filenames = ENGINE_MODULES[engine_id]
    mod = __import__(module_name, fromlist=["resolve"])
    if getattr(mod, "_ae_patched", False):
        return

    original = mod.resolve
    weight_dir = local_weight_dir(engine_id, catalog)
    local = {
        name: os.path.join(weight_dir, name)
        for name in filenames
    } if weight_dir else {}

    def patched(ref, *, hf_repo=None, revision=None, cache_dir=None):
        name = os.path.basename(str(ref))
        candidate = local.get(name)
        if candidate and os.path.isfile(candidate):
            return candidate
        missing_weights(engine_id, catalog)
        raise BweUnavailable(
            f"缺少带宽扩展权重：{name}\n"
            f"请在界面的模型列表里选中「{engine_id}」后点「下载所选模型」，"
            "下载走 hf-mirror 镜像，不需要访问 huggingface.co。"
        )

    mod.resolve = patched
    mod._ae_patched = True


def _model_for(engine_id: str, catalog: Catalog):
    """当前线程的 SRModel（惰性加载并缓存）。"""
    cache = getattr(_local, "models", None)
    if cache is None:
        cache = OrderedDict()
        _local.models = cache
    if engine_id in cache:
        cache.move_to_end(engine_id)
        return cache[engine_id]

    audiosronnx = _import_audiosronnx()
    missing = missing_weights(engine_id, catalog)
    if missing:
        raise BweUnavailable(
            f"带宽扩展权重尚未就绪：{', '.join(missing)}\n"
            f"请在界面的模型列表里选中「{engine_id}」后点「下载所选模型」。"
        )
    _install_local_resolver(engine_id, catalog)

    try:
        model = audiosronnx.load_sr(engine=engine_id, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise BweUnavailable(
            f"带宽扩展模型加载失败（{engine_id}）：{type(exc).__name__}: {exc}"
        ) from exc

    cache[engine_id] = model
    while len(cache) > 2:                      # 与 denoise 一样限制驻留数量
        cache.popitem(last=False)
    return model


def release_thread_models() -> None:
    cache = getattr(_local, "models", None)
    if cache:
        cache.clear()


def _rebalance(y: np.ndarray, out_rate: int, x_ref: np.ndarray, ref_rate: int,
               cutoff_hz: float, hf_target_db: float) -> np.ndarray:
    """算增益并施加——`upscale` 里三个出口共用这一处，避免逻辑分叉。"""
    g_low, g_hf = balance_gains(y, out_rate, x_ref, ref_rate,
                                cutoff_hz=cutoff_hz, hf_target_db=hf_target_db)
    return apply_balance(y, out_rate, g_low, g_hf, cutoff_hz=cutoff_hz)


def upscale(
    engine_id: str,
    catalog: Catalog,
    mono: np.ndarray,
    sample_rate: int,
    *,
    cutoff_hz: float = BALANCE_CUTOFF_HZ,
    hf_target_db: float = HF_TARGET_DB,
    progress=None,
    cancel=None,
) -> tuple[np.ndarray, int]:
    """把单声道音频做带宽扩展，返回 (处理后, 输出采样率)。

    传入必须是**已经降噪过**的信号——MossFormer2_SR 一类的官方说明都要求输入是干净语音，
    所以流水线里的顺序是"先降噪、再上采样"。

    输出会经过 `balance_gains`/`apply_balance` 做频段平衡：模型原始输出会把生成频段
    抬到远高于通带（见文件头的实测数字），不修的话听感就是"只剩高频"。
    """
    x = np.ascontiguousarray(np.asarray(mono, dtype=np.float32).reshape(-1))
    if x.size == 0:
        return x, sample_rate

    model = _model_for(engine_id, catalog)
    n = len(x)
    chunk = int(CHUNK_S * sample_rate)

    if n <= chunk:
        if cancel and cancel():
            raise RuntimeError("已取消")
        y, rate = model.upscale(x, sample_rate)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        y = _rebalance(y, int(rate), x, sample_rate, cutoff_hz, hf_target_db)
        if progress:
            progress(1.0, "带宽扩展完成")
        return y, int(rate)

    # 长音频：分块 + 交叉淡化
    overlap = int(OVERLAP_S * sample_rate)
    step = max(1, chunk - overlap)
    out_rate = None
    out = None
    weights = None
    gains: tuple[float, float] | None = None   # 由**首段**算出，之后所有段共用，
    pos = 0                                    # 这样块与块之间不会有电平台阶
    total = max(1, int(np.ceil((n - overlap) / step)))
    idx = 0

    while pos < n:
        if cancel and cancel():
            raise RuntimeError("已取消")
        seg = x[pos:min(n, pos + chunk)]
        y, rate = model.upscale(seg, sample_rate)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if gains is None:
            g_low, g_hf = balance_gains(y, int(rate), seg, sample_rate,
                                        cutoff_hz=cutoff_hz,
                                        hf_target_db=hf_target_db)
            gains = (g_low, g_hf)
        y = apply_balance(y, int(rate), gains[0], gains[1], cutoff_hz=cutoff_hz)

        if out is None:
            out_rate = int(rate)
            total_len = int(round(n * out_rate / sample_rate))
            out = np.zeros(total_len, dtype=np.float32)
            weights = np.zeros(total_len, dtype=np.float32)

        start = int(round(pos * out_rate / sample_rate))
        end = min(len(out), start + len(y))
        take = end - start
        if take <= 0:
            break

        w = np.ones(take, dtype=np.float32)
        fade = min(int(OVERLAP_S * out_rate), take // 2)
        if fade > 0:
            if pos > 0:
                w[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            if pos + chunk < n:
                w[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        out[start:end] += y[:take] * w
        weights[start:end] += w

        pos += step
        idx += 1
        if progress:
            progress(min(0.99, idx / total), f"带宽扩展 {idx}/{total} 段")

    if out is None:                            # 理论上不会发生
        y, rate = model.upscale(x, sample_rate)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        y = _rebalance(y, int(rate), x, sample_rate, cutoff_hz, hf_target_db)
        return y, int(rate)

    out /= np.maximum(weights, 1e-6)
    if progress:
        progress(1.0, "带宽扩展完成")
    return out, int(out_rate or sample_rate)
