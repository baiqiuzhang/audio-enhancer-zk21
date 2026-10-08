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
from . import cuda_env, sysinfo
from .errors import Cancelled

# 每线程一套模型实例（ORT 会话），与 denoise 引擎同样的策略
_local = threading.local()


def _last_device_store() -> dict:
    """**线程局部**的"最近一次实际设备"记录。

    用线程局部而不是模块全局：批处理会并发跑多个文件，全局变量会让 A 文件记上
    B 文件的设备——日志里就会把设备归错文件（本项目最不想要的假信息）。
    """
    store = getattr(_local, "last_device", None)
    if store is None:
        store = {}
        _local.last_device = store
    return store

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
# 分块长度与内存预检
# ---------------------------------------------------------------------------
# 各引擎的峰值驻留内存模型：`slope × 块长(秒) + intercept`（MB）。
# 全部由**独立进程实测拟合**（2026-09-22，48 kHz 单声道真实素材，峰值取自 Windows 的
# GetProcessMemoryInfo，基线已扣除）：
#     hifiganbwe  2 s→2042、10 s→4340、20 s→7509 MB   ⇒ slope 304、intercept 1435（三点拟合，最大偏差 3%）
#     novasr      2 s→ 213、 5 s→ 379、20 s→1242 MB   ⇒ slope  57、intercept   99（最大偏差 2%）
#     lavasr      2 s→ 168、 5 s→ 191、20 s→ 327 MB   ⇒ slope   9、intercept  150（最大偏差 2%）
#
# 也就是说 **hifiganbwe 是个异类**：默认 20 s 块要 7.5 GB，在空闲内存 4 GB 的机器上
# 必然失败（实测两次单次分配失败：498 MB 与 1.48 GB，CPU arena），而批处理的内存守卫
# 只按"单实例 200–500 MB"估，完全不知道这个尖峰。
#
# 两条修法都用上：
#   1. 按引擎给不同块长——hifiganbwe 降到 2 s（峰值约 2.0 GB，实测 30 s/60 s 输入都能跑完）。
#      代价经实测可忽略：2 s 块 vs 20 s 块的输出**相对 RMS 只差 0.07%**、逐带频谱一致
#      （5 s 块 0.08%），与"CPU/GPU 差异 0.5%"同一量级。
#   2. 跑之前先按下面这个模型预检；内存不够就**明确拒绝**并给出可执行建议，而不是把失败
#      留给 ORT 深处的一次随机分配。
_MEM_MODEL = {
    "hifiganbwe": (304.0, 1435.0),
    "novasr": (57.0, 99.0),
    "lavasr": (9.0, 150.0),
}
_DEFAULT_MEM_MODEL = (200.0, 800.0)     # 未知引擎：取一个偏保守的值
# 预检的余量：拟合值本身的误差在 3% 内，但运行时还有别的分配波动，所以再留一档。
MEM_SAFETY = 1.15

CHUNK_S_BY_ENGINE = {"hifiganbwe": 2.0}       # 未列出的引擎用 CHUNK_S


def chunk_seconds(engine_id: str) -> float:
    """该引擎应使用的分块长度（秒）。"""
    return float(CHUNK_S_BY_ENGINE.get(engine_id, CHUNK_S))


def estimate_peak_mb(engine_id: str, chunk_s: float | None = None) -> float:
    """估算该引擎在给定块长下的峰值驻留内存（MB，已含安全余量）。"""
    slope, intercept = _MEM_MODEL.get(engine_id, _DEFAULT_MEM_MODEL)
    secs = chunk_seconds(engine_id) if chunk_s is None else float(chunk_s)
    return (slope * secs + intercept) * MEM_SAFETY


def _check_memory(engine_id: str, seconds: float) -> None:
    """事前内存预检：不够就明确拒绝（并把替代方案写清楚）。

    只在**能读到内存信息且确实不够**时拒绝——`available_memory_mb()` 取不到时返回 0，
    那种情况按"未知"处理、放行，否则会在探测失败的机器上把功能全拒掉。
    """
    need = estimate_peak_mb(engine_id, seconds)
    free = sysinfo.available_memory_mb()
    if free and free < need:
        slope, intercept = _MEM_MODEL.get(engine_id, _DEFAULT_MEM_MODEL)
        raise BweUnavailable(
            f"带宽扩展「{engine_id}」这一档预计需要约 {need:.0f} MB 空闲内存"
            f"（块长 {seconds:.0f}s；该引擎实测成本约 {slope:.0f} MB/秒 + {intercept:.0f} MB 固定开销），"
            f"当前只有 {free:.0f} MB。\n"
            "请先关闭其它占用内存的程序再重试；"
            "或改用 NovaSR（同时长下只需约 1/6 的内存）。"
        )

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

# 生成频段增益的**放大上限**（dB）。0 = 只许压、不许抬。
#
# 为什么必须有这一条：原规则「把生成频段压到通带之下 hf_target_db」是一个**绝对电平
# 目标**，本身不含方向性；当模型原始输出本来就安静时，它会反过来变成巨额放大。
#
# 实测（2026-09-21，真实素材经 deepfilternet3 降噪后的 30 s 片段）：
#   · novasr **原始**输出在 9–16 kHz 比通带低约 **55 dB**（模型确实几乎没补什么）；
#   · 平衡步骤据此算出 **g_hf = +31.56 dB**，把模型的宽带本底整体抬高 31.6 dB；
#   · 抬高后的 1 kHz 细带在 **13–14 kHz 达到峰值（比通带只低 16.5 dB）**，且形状随频率
#     **上升**，而源素材在 9 kHz 以上就是 −85 dB 的硬悬崖 —— 听感正是用户报告的那层
#     "贴在原本音频上的高频嗡嗡声"（CPU / CUDA 两侧一致，所以与设备无关）。
#
# 因此平衡只允许**衰减**：模型产得多就压到目标，产得少就诚实地少加，绝不把它的本底放大。
# 想改成允许少量放大就调大这个常量（单位 dB）。
MAX_HF_BOOST_DB = 0.0


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

    **生成频段的增益只压不抬**（上限见 MAX_HF_BOOST_DB）：这条是实测加的护栏，
    否则"绝对电平目标"会在模型本来就安静时变成巨额放大，听感就是一层高频嗡嗡声。
    """
    lo, hi = _PRIMARY_BANDS
    p_in = _band_power(x_ref, ref_rate, lo, min(hi, cutoff_hz))
    p_low = _band_power(y, out_rate, lo, min(hi, cutoff_hz))
    p_hf = _band_power(y, out_rate, cutoff_hz, out_rate / 2.0)
    # 早先这里是 `if p_in <= 0 or p_low <= 0 or p_hf <= 0: return 1.0, 1.0`
    # ——而 `apply_balance` 见 (1,1) 会**原样交付模型的原始输出**。这条路径上模型的
    # 高频可能比通带高出几十 dB（本文件顶部实测：novasr +24.2 / hifiganbwe +29.4 /
    # lavasr +89.1 dB），把它未经平衡直接交给用户正是"只剩高频/嗡嗡声"那一类事故。
    # 所以：**平衡做不了就明确拒绝**（与本项目对去混响、对不支持容器的取向一致），
    # 只有"生成频段本身测不到能量"（p_hf==0）才是安全的——那时原始输出在该频段也没东西。
    if p_in <= 0 or p_low <= 0:
        raise BweUnavailable(
            "带宽扩展无法做频段平衡："
            + ("输入在 20 Hz–8 kHz 内没有可测能量（素材近乎静音或只有直流）"
               if p_in <= 0 else "模型输出在通带内没有可测能量")
            + "。此时若原样交付模型输出，其生成频段可能远高于通带（实测最高 +89 dB），"
              "听感就是一层刺耳高频——已拒绝，而不是交付一份未平衡的结果。"
        )
    if p_hf <= 0:
        return 1.0, 1.0          # 生成频段为空：原始输出在那里也没能量，压不压都一样
    g_low = float(np.sqrt(p_in / p_low))
    # 目标是「生成频段 = 通带 × 10^(hf_target_db/10)」
    target_hf = p_in * (10.0 ** (hf_target_db / 10.0))
    g_hf = float(np.sqrt(target_hf / p_hf))
    g_hf = min(g_hf, 10.0 ** (MAX_HF_BOOST_DB / 20.0))
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


def _model_for(engine_id: str, catalog: Catalog, device: str = "auto"):
    """当前线程的 SRModel（惰性加载并缓存，键含 device）。"""
    cache = getattr(_local, "models", None)
    if cache is None:
        cache = OrderedDict()
        _local.models = cache
    key = (engine_id, device)
    if key in cache:
        cache.move_to_end(key)
        return cache[key]

    audiosronnx = _import_audiosronnx()
    missing = missing_weights(engine_id, catalog)
    if missing:
        raise BweUnavailable(
            f"带宽扩展权重尚未就绪：{', '.join(missing)}\n"
            f"请在界面的模型列表里选中「{engine_id}」后点「下载所选模型」。"
        )
    _install_local_resolver(engine_id, catalog)

    providers, desc = providers_for(device)
    try:
        model = audiosronnx.load_sr(engine=engine_id, providers=providers)
    except Exception as exc:
        if providers and providers[0] == "CUDAExecutionProvider":
            # GPU 初始化失败（实测最常见的原因是 CUDA/cuDNN 的 DLL 加载不到，
            # 例如驱动太旧、或缺 cudnn64_9.dll）。**回落到 CPU 而不是让整个功能
            # 不可用**，但要留下痕迹：desc 会写明"回落 + 原因"，上层会记进日志。
            # 静默降级正是本项目一直在防的失败方式。
            fallback_providers = ["CPUExecutionProvider"]
            try:
                model = audiosronnx.load_sr(engine=engine_id, providers=fallback_providers)
            except Exception as exc2:
                raise BweUnavailable(
                    f"带宽扩展模型加载失败（{engine_id}）：CUDA 与 CPU 都失败。\n"
                    f"CUDA: {type(exc).__name__}: {str(exc)[:300]}\n"
                    f"CPU : {type(exc2).__name__}: {str(exc2)[:300]}"
                ) from exc2
            desc = (f"CPU（**请求 CUDA 失败后回落**：{type(exc).__name__}: "
                    f"{str(exc)[:200]}）")
        else:
            raise BweUnavailable(
                f"带宽扩展模型加载失败（{engine_id}）：{type(exc).__name__}: {exc}"
            ) from exc

    try:
        model._ae_provider_desc = desc
    except Exception:
        pass
    store = _last_device_store()
    store.clear()
    store.update({
        "engine": engine_id,
        "device": device,
        "desc": desc,
        "providers": list(providers),
        "cuda_ep_loaded": cuda_env.cuda_ep_active(),
    })

    cache[key] = model
    while len(cache) > 2:                      # 与 denoise 一样限制驻留数量
        cache.popitem(last=False)
    return model


def last_device() -> dict:
    """本线程最近一次模型加载**实际**用的设备信息（供日志与界面诚实显示）。

    不是"我们请求了什么"，而是"结果是什么"——包括 CUDA 初始化失败后是否回落。
    """
    return dict(_last_device_store())


def providers_for(device: str = "auto") -> tuple[list[str], str]:
    """决定 ONNX Runtime 的 (providers 列表, 人类可读的说明)。

    - `"cpu"`  ：强制纯 CPU。
    - `"auto"` ：有 CUDA 且**显存够**就用 CUDA，否则纯 CPU（**不报错、不中断**）。
      实测（2026-09-21，RTX 4060 Laptop，NovaSR，28.8 s）：CPU RTF 0.083 →
      CUDA RTF 0.053，**快 1.55×**，所以这一档确实该上 GPU。
    - `"cuda"` ：明确要求 CUDA；拿不到时返回纯 CPU 并把原因写进说明。

    列表尾始终保留 `CPUExecutionProvider`：ORT 会在 CUDA 不支持的算子/情形上
    自动回退到 CPU 版实现，这是官方支持的用法。
    """
    if device == "cpu":
        return ["CPUExecutionProvider"], "CPU（显式指定）"

    cuda_env.activate()                        # 让 ORT 找得到 CUDA/cuDNN 的 DLL
    try:
        import onnxruntime as ort              # noqa: PLC0415
        avail = list(ort.get_available_providers())
    except Exception as exc:
        return ["CPUExecutionProvider"], (
            f"CPU（onnxruntime 导入失败：{type(exc).__name__}: {str(exc)[:120]}）"
        )

    if "CUDAExecutionProvider" in avail:
        vram_ok, why = cuda_env.vram_ok()
        if not vram_ok:
            return ["CPUExecutionProvider"], f"CPU（**显存不足，未上 GPU**：{why}）"
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], (
            f"CUDA（{why}；providers 列表尾保留 CPU 回退）"
        )

    if device == "cuda":
        return ["CPUExecutionProvider"], (
            "CPU（**请求了 CUDA，但当前 onnxruntime 不带 CUDAExecutionProvider**："
            f"可用 {avail}）"
        )
    return ["CPUExecutionProvider"], f"CPU（自动判定：可用 {avail}）"


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
    device: str = "auto",
    progress=None,
    cancel=None,
) -> tuple[np.ndarray, int]:
    """把单声道音频做带宽扩展，返回 (处理后, 输出采样率)。

    传入必须是**已经降噪过**的信号——MossFormer2_SR 一类的官方说明都要求输入是干净语音，
    所以流水线里的顺序是"先降噪、再上采样"。

    输出会经过 `balance_gains`/`apply_balance` 做频段平衡：模型原始输出会把生成频段
    抬到远高于通带（见文件头的实测数字），不修的话听感就是"只剩高频"。

    `device` 取 `"auto"` / `"cpu"` / `"cuda"`，只影响用哪个执行提供者，**不改变语义**；
    拿不到 CUDA 时会回落 CPU 并通过 `last_device()` 诚实报告。
    """
    x = np.ascontiguousarray(np.asarray(mono, dtype=np.float32).reshape(-1))
    if x.size == 0:
        return x, sample_rate

    n = len(x)
    chunk_s = chunk_seconds(engine_id)
    # 事前内存预检放在**加载模型之前**：内存不够就直接拒绝，不去分配 ORT 会话。
    # 短输入按实际时长估算（比一个整块更省），长输入按一个整块估（峰值由单块决定）。
    _check_memory(engine_id, min(chunk_s, n / max(sample_rate, 1)))
    model = _model_for(engine_id, catalog, device)
    chunk = int(chunk_s * sample_rate)

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
            # 与 core/errors.py 的 Cancelled 保持同一类型：抛 RuntimeError 会让
            # pipeline/worker 的 `except Cancelled` 捕不到，取消被显示成"处理失败"。
            raise Cancelled("已取消")
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
