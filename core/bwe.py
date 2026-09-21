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


class BweUnavailable(RuntimeError):
    """audiosronnx 不可用，或权重尚未下载。"""


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


def upscale(
    engine_id: str,
    catalog: Catalog,
    mono: np.ndarray,
    sample_rate: int,
    *,
    progress=None,
    cancel=None,
) -> tuple[np.ndarray, int]:
    """把单声道音频做带宽扩展，返回 (处理后, 输出采样率)。

    传入必须是**已经降噪过**的信号——MossFormer2_SR 一类的官方说明都要求输入是干净语音，
    所以流水线里的顺序是"先降噪、再上采样"。
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
        if progress:
            progress(1.0, "带宽扩展完成")
        return y, int(rate)

    # 长音频：分块 + 交叉淡化
    overlap = int(OVERLAP_S * sample_rate)
    step = max(1, chunk - overlap)
    out_rate = None
    out = None
    weights = None
    pos = 0
    total = max(1, int(np.ceil((n - overlap) / step)))
    idx = 0

    while pos < n:
        if cancel and cancel():
            raise RuntimeError("已取消")
        seg = x[pos:min(n, pos + chunk)]
        y, rate = model.upscale(seg, sample_rate)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

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
        return np.asarray(y, dtype=np.float32).reshape(-1), int(rate)

    out /= np.maximum(weights, 1e-6)
    if progress:
        progress(1.0, "带宽扩展完成")
    return out, int(out_rate or sample_rate)
