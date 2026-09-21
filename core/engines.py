"""增强引擎：统一接口 + 三种实现。

已实测结论（2026-09-21，本机）：
- sherpa-onnx 离线接口输出**无延迟**，但比输入短约一个 hop（帧对齐截断），
  这里用源信号尾部按增益补齐，保证长度一致（A/B 对齐需要）。
- sherpa-onnx 流式接口（OnlineSpeechDenoiser）有 40 ms 固定延迟，故不采用；
  长文件改用「分段调用离线接口 + 交叉淡化」来限制内存。
- DeepFilterNet3 exe 保持输入采样率，输出名沿用输入 basename，
  且因 -D 延迟补偿会丢掉末尾约 30 ms，同样用尾部补偿。

所有引擎都接收**单声道 float32**，返回 (处理后单声道 float32, 输出采样率)。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import cuda_env, dsp, media_io
from .modelstore import (
    Catalog,
    DownloadError,
    EngineSpec,
    download_files,
    engine_ready,
    file_ready,
)

ProgressCb = Callable[[float, str], None]
CancelCb = Callable[[], bool]

class Cancelled(RuntimeError):
    """调用方请求取消。"""


# 单次送进离线接口的样本上限（float32）。
# 48 kHz 单声道下约 25 分钟 ≈ 690 MB，是这台机器（空闲内存 3 GB 上下）的保守上限。
MAX_FRAMES_SINGLE_PASS = 72_000_000


@dataclass
class EngineParams:
    num_threads: int = 2
    # 推理设备。"auto" = 让**实测**决定，而不是"有 GPU 就上 GPU"。实测（2026-09-21，
    # RTX 4060 Laptop，真实素材，各重复 2–3 次）：
    #   · 带宽扩展（NovaSR）：CPU RTF 0.083 → CUDA 0.053，**快 1.55×** → auto 选 CUDA；
    #   · sherpa 降噪（DPDFNet2）：CPU RTF 0.126 → CUDA 0.842，**慢 6.6×** → auto 选 CPU
    #     （逐帧流式模型 + 极小 kernel，GPU 启动/同步延迟盖过算力收益）。
    # 两个参数都**不进指纹**：设备只影响速度，结果差异在 0.01%–0.5% RMS 量级
    # （见 README「GPU / CUDA」一节与 tests/check_gpu.py 的实测）。
    bwe_device: str = "auto"          # 带宽扩展（audiosronnx → onnxruntime）
    sherpa_provider: str = "auto"     # sherpa-onnx（GTCRN / DPDFNet）
    # sherpa DPDFNet
    attenuation_limit_db: float = 12.0
    # DeepFilterNet
    dfn_post_filter: bool = True
    dfn_atten_lim_db: float = 100.0
    dfn_compensate_delay: bool = True
    # DSP 基线
    dsp_hp_cutoff_hz: float = 80.0
    dsp_gate_reduction_db: float = 12.0
    # 长文件分段（秒）
    segment_seconds: float = 600.0


def _tick(progress: ProgressCb | None, frac: float, msg: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, frac)), msg)


def _check(cancel: CancelCb | None) -> None:
    if cancel and cancel():
        raise Cancelled("已取消")


def _pad_tail(y: np.ndarray, x: np.ndarray, target_len: int) -> np.ndarray:
    """把 y 补到 target_len：缺口用源信号对应尾部按增益补齐。

    模型会截掉末尾不足以构成一帧的部分（约 10 ms），直接补零会在结尾留一个
    可听见的静音，所以用源信号的尾巴按能量比缩放来接上。
    """
    if len(y) >= target_len:
        return y[:target_len]
    missing = target_len - len(y)
    if missing <= 0 or x.size == 0:
        return y
    src_tail = x[len(y):len(y) + missing]
    if src_tail.size < missing:
        src_tail = np.pad(src_tail, (0, missing - src_tail.size))
    out_rms = float(np.sqrt(np.mean(np.square(y.astype(np.float64))))) if y.size else 0.0
    in_rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64))))) if x.size else 0.0
    gain = (out_rms / in_rms) if in_rms > 1e-12 else 0.0
    return np.concatenate([y, (src_tail * gain).astype(np.float32)])


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class BaseEngine:
    kind = "base"

    def __init__(self, spec: EngineSpec, catalog: Catalog):
        self.spec = spec
        self.catalog = catalog

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def sample_rate(self) -> int | None:
        return self.spec.sample_rate

    def ready(self) -> bool:
        return engine_ready(self.spec, self.catalog)

    def ensure(self, progress: ProgressCb | None = None, log=None) -> None:
        """确保权重就绪；缺失则下载。"""
        missing = [f for f in self.spec.files if not file_ready(f, self.catalog)]
        if not missing:
            return
        download_files(
            missing,
            self.catalog,
            progress=lambda done, total: _tick(
                progress, (done / total) if total else 0.0,
                f"下载 {self.name} 权重…",
            ),
            log=log,
        )

    # 子类实现
    def process(
        self,
        mono: np.ndarray,
        sample_rate: int,
        params: EngineParams,
        *,
        progress: ProgressCb | None = None,
        cancel: CancelCb | None = None,
    ) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def release(self) -> None:
        """释放常驻资源（模型句柄）。"""

    def provider_desc(self) -> str | None:
        """**实际**生效的推理设备说明；None = 该实现不涉及（DSP / exe）。"""
        return None


# ---------------------------------------------------------------------------
# 推理设备：拿不到 GPU 就回落 CPU，并把"实际用了什么"报出来
# ---------------------------------------------------------------------------

def _probe_sherpa_cuda() -> bool:
    """当前 sherpa-onnx 是否带 CUDA 支持——用**二进制证据**判断，不猜版本号。

    实测（2026-09-21）：CPU-only wheel 的包里没有任何 `*providers_cuda*` 文件，
    二进制里是 "Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON"；GPU wheel 才会
    带 `onnxruntime_providers_cuda.dll`。
    """
    try:
        import sherpa_onnx  # noqa: PLC0415

        pkg = os.path.dirname(os.path.abspath(sherpa_onnx.__file__))
    except Exception:
        return False
    for _root, _dirs, files in os.walk(pkg):
        if any("providers_cuda" in f.lower() for f in files):
            return True
    return False


_sherpa_cuda_cache: bool | None = None


def sherpa_cuda_available() -> bool:
    global _sherpa_cuda_cache
    if _sherpa_cuda_cache is None:
        _sherpa_cuda_cache = _probe_sherpa_cuda()
    return _sherpa_cuda_cache


def resolve_sherpa_provider(want: str = "auto") -> tuple[str, str]:
    """(provider 字符串, 说明)。sherpa 只接受**单个** provider，所以回退在这一层做。

    **实测（2026-09-21，RTX 4060 Laptop，DPDFNet2，48 kHz 素材的 30 s 与 60 s 片段，
    各重复 2–3 次）：**

    | 设备 | 30 s 片段 | 60 s 片段 | RTF |
    | --- | --- | --- | --- |
    | CPU | 3.79 / 3.90 s | 12.08 s | 0.126–0.130 |
    | CUDA | 25.18 / 25.26 / 25.41 s | 75.50 s | 0.839–0.847 |

    **即 GPU 慢 6.3–6.6 倍**，三次重复的方差只有 ±0.5%，且比值与音频长度无关——所以
    这不是首次初始化开销，而是模型性质：DPDFNet 是逐帧带状态的流式模型，每帧要发
    大量极小的 kernel，GPU 的 kernel 启动与同步延迟盖过了算力收益。

    因此 `"auto"` **明确选择 CPU**（不做"有 GPU 就上 GPU"的傻事），要覆盖必须显式
    传 `"cuda"`。这条结论比"能不能跑在 GPU 上"更重要——能跑不等于值得跑。
    """
    if want == "cpu":
        return "cpu", "CPU（显式指定）"
    if want == "auto":
        return "cpu", (
            "CPU（自动判定：**实测 DPDFNet 在 CUDA 上比 CPU 慢 6.3–6.6 倍**，"
            "所以不自动上 GPU；确实要在 GPU 上跑请显式选 cuda）"
        )
    if not sherpa_cuda_available():
        return "cpu", (
            "CPU（**请求了 CUDA，但当前 sherpa-onnx 是 CPU-only 构建**："
            "包里没有 onnxruntime_providers_cuda.dll，需换 +cuda12.cudnn9 的 GPU wheel）"
        )
    cuda_env.activate()
    dirs = cuda_env.usable_dirs()
    if not dirs:
        return "cpu", ("CPU（sherpa 带 CUDA，但缺 CUDA/cuDNN 运行库："
                       + ", ".join(cuda_env.missing()) + "）")
    vram_ok, why = cuda_env.vram_ok()
    if not vram_ok:
        return "cpu", f"CPU（**显存不足，未上 GPU**：{why}）"
    return "cuda", (f"CUDA（{why}；CUDA/cuDNN 运行库取自 {dirs[0]}）"
                    "——**注意实测这一档反而慢 6.3–6.6 倍**，仅用于对比与实验")


# ---------------------------------------------------------------------------
# DSP 基线
# ---------------------------------------------------------------------------

class DspBaselineEngine(BaseEngine):
    """传统处理：高通 + 自适应谱门限。不需要任何权重，永远可用。"""

    kind = "dsp"

    def ready(self) -> bool:
        return True

    def ensure(self, progress=None, log=None) -> None:
        return

    def process(self, mono, sample_rate, params, *, progress=None, cancel=None):
        y = dsp.spectral_process(
            mono,
            sample_rate,
            hp_cutoff_hz=params.dsp_hp_cutoff_hz,
            gate_reduction_db=params.dsp_gate_reduction_db,
            progress=progress,
            cancel=cancel,
        )
        return y, sample_rate

    def release(self) -> None:
        return


# ---------------------------------------------------------------------------
# sherpa-onnx（GTCRN / DPDFNet）
# ---------------------------------------------------------------------------

class SherpaEngine(BaseEngine):
    kind = "sherpa"

    def __init__(self, spec: EngineSpec, catalog: Catalog):
        super().__init__(spec, catalog)
        self._denoiser = None
        self._key: tuple | None = None
        self._out_rate: int | None = None
        self._provider_desc: str = ""
        # CUDA 会话创建失败过就不再重试。否则每次 process() 都会按 "cuda" 重建
        # 会话、每次都失败、每次再回落一遍 —— 白白多付一次失败的开销。
        self._cuda_failed = False
        self._cuda_fail_reason = ""

    def provider_desc(self) -> str | None:
        return self._provider_desc or None

    def _build(self, params: EngineParams):
        want = getattr(params, "sherpa_provider", "auto")
        if want == "cuda" and self._cuda_failed:
            provider, prov_desc = "cpu", (
                f"CPU（**CUDA 会话创建失败后回落**：{self._cuda_fail_reason}）"
            )
        else:
            provider, prov_desc = resolve_sherpa_provider(want)
        key = (params.num_threads, float(params.attenuation_limit_db), provider)
        if self._denoiser is not None and self._key == key:
            return self._denoiser

        import sherpa_onnx  # noqa: PLC0415

        path = self.catalog.resolve(self.spec.files[0].path)
        if not os.path.isfile(path):
            raise DownloadError(f"模型文件缺失：{path}")

        def make_cfg(prov: str):
            common = dict(num_threads=int(params.num_threads), debug=False, provider=prov)
            if self.spec.sherpa_type == "gtcrn":
                model = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                    gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=path),
                    **common,
                )
            else:
                model = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                    dpdfnet=sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(
                        model=path, attenuation_limit_db=float(params.attenuation_limit_db)
                    ),
                    **common,
                )
            return sherpa_onnx.OfflineSpeechDenoiserConfig(model=model)

        cfg = make_cfg(provider)
        if not cfg.validate():
            raise DownloadError(f"模型配置校验失败：{cfg}")

        try:
            self._denoiser = sherpa_onnx.OfflineSpeechDenoiser(cfg)
        except Exception as exc:
            if provider != "cuda":
                raise
            # GPU 会话创建失败（驱动太旧 / DLL 缺失 / 显存不足）。**回落 CPU 而不是
            # 让整个降噪不可用**，但把原因写进 provider_desc —— 上层会记进日志，
            # 不做静默降级。
            cfg_cpu = make_cfg("cpu")
            if not cfg_cpu.validate():
                raise
            self._denoiser = sherpa_onnx.OfflineSpeechDenoiser(cfg_cpu)
            key = (params.num_threads, float(params.attenuation_limit_db), "cpu")
            self._cuda_failed = True
            self._cuda_fail_reason = f"{type(exc).__name__}: {str(exc)[:200]}"
            prov_desc = f"CPU（**CUDA 会话创建失败后回落**：{self._cuda_fail_reason}）"

        self._key = key
        self._provider_desc = prov_desc
        return self._denoiser

    def _run_once(self, mono: np.ndarray, sample_rate: int, params: EngineParams) -> np.ndarray:
        denoiser = self._build(params)
        out = denoiser(np.ascontiguousarray(mono, dtype=np.float32), int(sample_rate))
        # .samples 是 Python list，必须显式转 ndarray
        y = np.asarray(out.samples, dtype=np.float32)
        self._out_rate = int(out.sample_rate)
        return _pad_tail(y, mono, len(mono))

    def process(self, mono, sample_rate, params, *, progress=None, cancel=None):
        _check(cancel)
        n = len(mono)
        if n == 0:
            return mono, sample_rate

        if n <= MAX_FRAMES_SINGLE_PASS:
            _tick(progress, 0.05, f"{self.name} 加载模型…")
            y = self._run_once(mono, sample_rate, params)
            _tick(progress, 0.98, f"{self.name} 完成")
            return y, (self._out_rate or sample_rate)

        # 超长文件：分段 + 交叉淡化，限制内存峰值
        seg = max(1, int(params.segment_seconds * sample_rate))
        overlap = int(0.5 * sample_rate)
        step = max(1, seg - overlap)
        out = np.zeros(n, dtype=np.float32)
        weight = np.zeros(n, dtype=np.float32)
        fade = np.ones(overlap, dtype=np.float32)
        if overlap > 1:
            fade = np.linspace(0.0, 1.0, overlap, dtype=np.float32)

        pos = 0
        idx = 0
        total_seg = max(1, int(np.ceil((n - overlap) / step)))
        while pos < n:
            _check(cancel)
            end = min(n, pos + seg)
            chunk = mono[pos:end]
            y = self._run_once(chunk, sample_rate, params)

            w = np.ones(len(y), dtype=np.float32)
            if overlap > 1 and len(y) >= overlap:
                if pos > 0:
                    w[:overlap] = fade              # 淡入
                if end < n:
                    w[-overlap:] *= fade[::-1]     # 淡出
            out[pos:pos + len(y)] += y * w
            weight[pos:pos + len(y)] += w

            pos += step
            idx += 1
            _tick(progress, 0.05 + 0.93 * idx / total_seg,
                  f"{self.name} 分段处理 {idx}/{total_seg}")

        return out / np.maximum(weight, 1e-6), (self._out_rate or sample_rate)

    def release(self) -> None:
        self._denoiser = None
        self._key = None


# ---------------------------------------------------------------------------
# DeepFilterNet3 独立可执行
# ---------------------------------------------------------------------------

class DeepFilterEngine(BaseEngine):
    kind = "dfn_exe"

    def _exe(self) -> str:
        return self.catalog.resolve(self.spec.files[0].path)

    def process(self, mono, sample_rate, params, *, progress=None, cancel=None):
        _check(cancel)
        exe = self._exe()
        if not os.path.isfile(exe):
            raise DownloadError(f"缺少可执行文件：{exe}")

        n = len(mono)
        if n == 0:
            return mono, sample_rate

        # exe 按输入 basename 派生输出名，所以用隔离的临时目录 + 固定输入名，
        # 避免产物与其它文件撞名，也免去解析它的命名规则。
        with tempfile.TemporaryDirectory(prefix="ae_dfn_") as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            in_path = os.path.join(in_dir, "input.wav")

            # 必须写 float32：deep-filter 的 Rust 侧只支持到 16-bit，
            # 喂 PCM_24 会 panic（TooWide）——已实测。float32 无精度损失且被接受。
            # 它自身的输出固定是 PCM_16，这是该工具的限制，不是我们的选择。
            media_io.write_audio(in_path, mono, sample_rate, subtype="FLOAT")

            cmd = [exe, "-o", out_dir]
            if params.dfn_post_filter:
                cmd.append("--pf")
            if params.dfn_compensate_delay:
                cmd.append("-D")
            cmd += ["-a", str(float(params.dfn_atten_lim_db))]
            cmd.append(in_path)

            _tick(progress, 0.1, f"{self.name} 运行中（无进度输出）…")
            _check(cancel)
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=3600,
                )
            except FileNotFoundError as exc:
                raise DownloadError(f"无法执行 deep-filter：{exc}") from exc
            except subprocess.TimeoutExpired as exc:
                raise DownloadError("deep-filter 运行超过 1 小时，已中止。") from exc

            _check(cancel)
            produced = [
                os.path.join(out_dir, f) for f in os.listdir(out_dir)
                if f.lower().endswith(".wav")
            ]
            if not produced:
                raise DownloadError(
                    f"deep-filter 未产出音频（退出码 {proc.returncode}）。\n"
                    f"stderr: {(proc.stderr or '').strip()[:400]}"
                )
            _tick(progress, 0.85, f"{self.name} 读回结果…")
            data = media_io.decode(sorted(produced)[0])

        y = media_io.to_mono(data.samples)
        if data.sample_rate != sample_rate:
            y = media_io.resample(y, data.sample_rate, sample_rate)
        y = _pad_tail(np.asarray(y, dtype=np.float32), mono, n)
        _tick(progress, 1.0, f"{self.name} 完成")
        return y, sample_rate


# 重阶段（带宽扩展 / 去混响）的全局互斥：它们的内存占用远高于降噪，
# 多个 worker 同时跑极易 OOM，所以同一时刻只允许一个。
# 放在这里而不是 worker.py：pipeline 需要用它，而 pipeline 不能导入 worker（会成环）。
HEAVY_STAGE = threading.Semaphore(1)


# ---------------------------------------------------------------------------
# 带宽扩展 / 修复类
# ---------------------------------------------------------------------------

class BweEngine(BaseEngine):
    """带宽扩展（生成式，会编造高频）。默认关闭，界面上必须与降噪分区并标注风险。"""

    kind = "bwe"

    def provider_desc(self) -> str | None:
        from . import bwe  # noqa: PLC0415

        info = bwe.last_device()
        return info.get("desc") or None

    def process(self, mono, sample_rate, params, *, cutoff_hz=None, hf_target_db=None,
                progress=None, cancel=None):
        from . import bwe  # 局部导入：audiosronnx 未安装时其余功能仍可用

        kw = {}
        if cutoff_hz:
            kw["cutoff_hz"] = float(cutoff_hz)
        if hf_target_db is not None:
            kw["hf_target_db"] = float(hf_target_db)
        y, rate = bwe.upscale(
            self.spec.id, self.catalog, mono, sample_rate,
            device=getattr(params, "bwe_device", "auto"),
            progress=progress, cancel=cancel, **kw,
        )
        return np.asarray(y, dtype=np.float32), int(rate)

    def release(self) -> None:
        from . import bwe

        bwe.release_thread_models()


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

_ENGINE_CLASSES = {
    "dsp": DspBaselineEngine,
    "sherpa": SherpaEngine,
    "dfn_exe": DeepFilterEngine,
    "bwe": BweEngine,
}


def build_engine(spec: EngineSpec, catalog: Catalog) -> BaseEngine:
    cls = _ENGINE_CLASSES.get(spec.kind)
    if cls is None:
        raise ValueError(f"未知引擎类型：{spec.kind}（引擎 {spec.id}）")
    return cls(spec, catalog)


def engine_kind(spec: EngineSpec) -> str:
    """引擎类别字符串。

    用途是让「结果指纹」只收录该类别**真正会用**的旋钮——否则会出现
    "用 GTCRN 时拖了一下降噪强度滑块，指纹就变了、结果被判定要重算"这种假差异。
    """
    if spec.kind == "sherpa":
        return f"sherpa_{spec.sherpa_type or 'unknown'}"
    return spec.kind


def kind_of(catalog: Catalog, engine_id: str) -> str | None:
    spec = catalog.engine(engine_id)
    return None if spec is None else engine_kind(spec)


# ---------------------------------------------------------------------------
# 线程局部的引擎实例
#
# sherpa 的 OfflineSpeechDenoiser 之类的对象**不可跨线程共享**，而批处理会在多个
# worker 线程里同时跑。所以给每个线程一份实例，并用很小的 LRU 限制驻留数量——
# 否则 8 个引擎 × N 个线程会把本来就紧张的内存吃满。
# ---------------------------------------------------------------------------

_local = threading.local()
_ENGINES_PER_THREAD = 2


def engine_for(spec: EngineSpec, catalog: Catalog) -> BaseEngine:
    """取当前线程专属的引擎实例（超出上限时淘汰最久未用的并释放其模型句柄）。"""
    cache = getattr(_local, "cache", None)
    if cache is None:
        cache = OrderedDict()
        _local.cache = cache

    eng = cache.get(spec.id)
    if eng is not None:
        cache.move_to_end(spec.id)
        return eng

    eng = build_engine(spec, catalog)
    cache[spec.id] = eng
    while len(cache) > _ENGINES_PER_THREAD:
        _evicted_id, old = cache.popitem(last=False)
        try:
            old.release()
        except Exception:
            pass
    return eng


def release_thread_engines() -> None:
    """释放当前线程持有的引擎。批处理 worker 收尾时调用，避免句柄跨任务堆积。"""
    cache = getattr(_local, "cache", None)
    if not cache:
        return
    for eng in cache.values():
        try:
            eng.release()
        except Exception:
            pass
    cache.clear()


def build_all(catalog: Catalog) -> list[BaseEngine]:
    return [build_engine(e, catalog) for e in catalog.engines]
