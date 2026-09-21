"""CUDA / cuDNN 运行库的可见性（Windows）。

本项目上 GPU 只有一处真问题：**onnxruntime-gpu 与 sherpa-onnx 的 GPU 版都不捆绑
CUDA / cuDNN 的 DLL**——它们运行时按裸文件名加载 `cudnn64_9.dll`、`cublas64_12.dll`、
`cudart64_12.dll`。而 Windows 的默认搜索顺序里既没有 venv 的 `site-packages`，
也没有 conda base。于是"装了 GPU 版却只在 CPU 上跑"是默认结果，不是异常。

本文件只做一件事：把这些 DLL 所在的目录加进**本进程**的搜索路径。全部是只读使用，
不复制 DLL、不改 PATH 环境变量、不动 conda base，也不写任何系统配置。

DLL 目录按优先级（`AE_CUDA_DLL_DIR` 可覆盖，分号或逗号分隔，应急用）：

1. `AE_CUDA_DLL_DIR` 指定的目录；
2. venv 自己的 `nvidia-*-cu12` pip 包（`site-packages/nvidia/*/bin`）；
3. **conda base 的 `torch/lib`** —— 本机 base 里的 torch 2.5.1+cu121 自带完整一套
   CUDA 12.1 + cuDNN 9.1 DLL（实测：`cudart64_12` / `cublas64_12` / `cublasLt64_12` /
   `cudnn64_9` / `cudnn_ops64_9` / `cudnn_cnn64_9` / `cudnn_adv64_9` / `cufft64_11`）。
   路径由 `sys.base_prefix` 推导，**不写死盘符或用户名**。

为什么 PATH 与 `os.add_dll_directory` 两套都加：ORT / sherpa 内部用 `LoadLibrary`
按裸文件名加载，走的是标准搜索顺序（其中包含 PATH）；而 Python 3.8+ 的
`os.add_dll_directory` 影响的是另一条搜索路径。两套都加上，避免依赖"哪一套恰好命中"。

上游调用的正确姿势（**必须在导入 onnxruntime / 创建 ORT 会话之前**）：

    from . import cuda_env
    cuda_env.activate()
    import onnxruntime as ort
"""

from __future__ import annotations

import glob
import os
import sys

# 判定"这套环境有没有 CUDA + cuDNN 运行库"用**通配**而不是写死版本号：
# CUDA 12 的运行时叫 `cudart64_12.dll` / `cublas64_12.dll`，CUDA 13 叫 `cudart64_13.dll`，
# 而 cuDNN 9 在两个 CUDA 大版本下都叫 `cudnn64_9.dll`。写死版本号会在换 CUDA 大版本时
# **静默**判定"不可用"——正是本项目最不想要的那种失败方式。
_REQUIRED_PATTERNS = ("cudnn64_*.dll", "cudart64_*.dll", "cublas64_*.dll")

_activated: list[str] = []
_status: dict = {"activated": False, "dirs": [], "reason": ""}


def _has_pattern(d: str, pat: str) -> bool:
    return bool(glob.glob(os.path.join(d, pat)))


def candidate_dirs() -> list[str]:
    """所有可能放着 CUDA / cuDNN DLL 的目录（顺序即优先级，已去重）。"""
    out: list[str] = []

    override = os.environ.get("AE_CUDA_DLL_DIR", "")
    for part in override.replace(",", ";").split(";"):
        part = part.strip()
        if part:
            out.append(part)

    # venv 里 nvidia-*-cu12 / cu13 的 pip 包。注意它们**按包分目录**：
    # cudnn 在 nvidia/cudnn/bin、cublas 在 nvidia/cublas/bin、cudart 在
    # nvidia/cuda_runtime/bin —— 所以不能要求"三类 DLL 在同一目录"（实测踩过：
    # 那样写会把已装好的 CUDA 运行库判定成"不存在"）。
    for pattern in (
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin"),
        os.path.join(sys.prefix, "lib", "site-packages", "nvidia", "*", "bin"),
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "lib"),
    ):
        out.extend(sorted(glob.glob(pattern)))

    # conda base 的 torch/lib（由 base_prefix 推导，不写死路径）
    base = getattr(sys, "base_prefix", "") or sys.prefix
    for sub in ("Lib", "lib"):
        out.append(os.path.join(base, sub, "site-packages", "torch", "lib"))

    # ORT 自己的 capi 目录（某些发行版会把 cudnn 放在这里）
    for pattern in (
        os.path.join(sys.prefix, "Lib", "site-packages", "onnxruntime", "capi"),
        os.path.join(sys.prefix, "lib", "site-packages", "onnxruntime", "capi"),
    ):
        out.append(pattern)

    seen: set[str] = set()
    uniq: list[str] = []
    for d in out:
        key = os.path.normcase(os.path.abspath(d))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    return uniq


def coverage() -> dict[str, list[str]]:
    """三类运行库 DLL 各自能在哪些候选目录里找到（跨目录取并集）。"""
    dirs = [d for d in candidate_dirs() if os.path.isdir(d)]
    return {pat: [d for d in dirs if _has_pattern(d, pat)] for pat in _REQUIRED_PATTERNS}


def satisfied() -> bool:
    """三类 DLL 是否都能找到（不要求同目录）。"""
    cov = coverage()
    return all(cov[pat] for pat in _REQUIRED_PATTERNS)


def usable_dirs() -> list[str]:
    """含任意一类运行库 DLL 的候选目录——**全部**都要加进搜索路径。"""
    dirs = [d for d in candidate_dirs() if os.path.isdir(d)]
    return [d for d in dirs if any(_has_pattern(d, p) for p in _REQUIRED_PATTERNS)]


def missing() -> list[str]:
    """缺哪一类 DLL（用于诚实报错）。齐全时返回空表。"""
    cov = coverage()
    return [pat for pat in _REQUIRED_PATTERNS if not cov[pat]]


def activate() -> list[str]:
    """把可用目录加进本进程的 DLL 搜索路径。幂等。返回实际加入的目录。"""
    global _activated
    if _status["activated"]:
        return _activated

    dirs = usable_dirs() if satisfied() else []
    if not dirs:
        _status.update(
            activated=True, dirs=[],
            reason="未找到 CUDA/cuDNN 运行库（缺 " + ", ".join(missing()) + "）",
        )
        return []

    for d in dirs:
        # 1) 标准搜索顺序（ORT / sherpa 内部按裸名 LoadLibrary 走这条）
        try:
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass
        # 2) Python 侧的另一条搜索路径
        try:
            os.add_dll_directory(d)
        except Exception:
            pass

    _activated = dirs
    _status.update(activated=True, dirs=list(dirs), reason="")
    return dirs


def status() -> dict:
    """当前状态（供界面/日志**诚实显示**到底有没有真的开启 CUDA）。"""
    if not _status["activated"]:
        return {"activated": False, "dirs": usable_dirs(),
                "reason": "尚未调用 activate()"}
    return dict(_status)


def describe() -> str:
    """一行人类可读的状态（日志用）。"""
    st = status()
    if st["dirs"]:
        return f"CUDA/cuDNN 运行库已就绪（{len(st['dirs'])} 个目录：{st['dirs'][0]}）"
    return f"CUDA/cuDNN 运行库不可用：{st.get('reason') or '未找到'}"


# ---------------------------------------------------------------------------
# 真实显存查询
# ---------------------------------------------------------------------------
# 一律以 nvidia-smi 为准。本机实测：Windows 的"显示适配器"枚举与
# ai-process-manager 的 get_system_status 报的都是 **GameViewer Virtual Display
# Adapter**（网易 UU 远程的虚拟适配器），拿它做显存预检会得出错误结论。只有
# nvidia-smi 报的是物理卡（RTX 4060 Laptop，8188 MiB）。
import subprocess  # noqa: E402


def nvidia_smi(timeout: int = 10) -> dict | None:
    """返回 {name, total_mb, free_mb}；没有 NVIDIA 卡或取不到就返回 None。"""
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return None
    first = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return None
    try:
        return {"name": parts[0], "total_mb": float(parts[1]), "free_mb": float(parts[2])}
    except ValueError:
        return None


def vram_free_mb() -> float | None:
    info = nvidia_smi()
    return info["free_mb"] if info else None


# 上 GPU 之前要求的**最低空闲显存**。定得保守：这些模型本身很小（DPDFNet2 9.7 MB、
# NovaSR 228 KB），真正的显存开销来自激活与中间缓冲。阈值只是为了避免在显存已被
# 别的东西占满时硬闯成 OOM——那种失败比"回落 CPU"难看得多。可用
# AE_GPU_MIN_FREE_MB 覆盖。
GPU_MIN_FREE_MB = 512.0


def vram_ok(need_mb: float | None = None) -> tuple[bool, str]:
    """显存是否够用。返回 (是否够, 说明)。取值失败视为"不够"（不硬闯）。"""
    need = GPU_MIN_FREE_MB if need_mb is None else float(need_mb)
    override = os.environ.get("AE_GPU_MIN_FREE_MB", "").strip()
    if override:
        try:
            need = float(override)
        except ValueError:
            pass
    info = nvidia_smi()
    if info is None:
        return False, "取不到显存信息（nvidia-smi 不可用）"
    if info["free_mb"] < need:
        return False, (f"空闲显存 {info['free_mb']:.0f} MiB < 需要 {need:.0f} MiB"
                       f"（{info['name']}）")
    return True, f"空闲显存 {info['free_mb']:.0f} MiB ≥ {need:.0f} MiB"


def gpu_summary() -> str:
    """一行 GPU 摘要（界面与日志用）。取不到就明说，不假装。"""
    info = nvidia_smi()
    if not info:
        return "未检测到 NVIDIA GPU（nvidia-smi 不可用）"
    return (f"{info['name']}，显存 {info['free_mb']:.0f}/{info['total_mb']:.0f} MiB 可用")


# ---------------------------------------------------------------------------
# "到底有没有真的用上 GPU" —— 只能靠证据，不能靠声明
# ---------------------------------------------------------------------------
# sherpa-onnx 的 Python API 只暴露 run / sample_rate，**没有 provider 回读**；
# onnxruntime 的 get_available_providers() 说的是"这个 DLL 支持什么"，不是
# "本次会话真的用了什么"。所以这里直接枚举**本进程已加载的模块**：ORT 在注册
# CUDA EP 时会加载 onnxruntime_providers_cuda.dll，它出现了才算真的启用。

_CUDA_PROVIDER_HINTS = ("providers_cuda", "providers_tensorrt", "cudnn")


def loaded_modules() -> list[str]:
    """本进程已加载的模块完整路径（失败返回空表）。"""
    try:
        import ctypes
        from ctypes import wintypes

        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.EnumProcessModules.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        psapi.EnumProcessModules.restype = wintypes.BOOL
        psapi.GetModuleFileNameExW.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD
        ]
        psapi.GetModuleFileNameExW.restype = wintypes.DWORD

        handle = kernel32.GetCurrentProcess()
        cap = 2048
        buf = (ctypes.c_void_p * cap)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModules(handle, buf, ctypes.sizeof(buf),
                                        ctypes.byref(needed)):
            return []
        count = min(cap, int(needed.value) // ctypes.sizeof(ctypes.c_void_p))
        names: list[str] = []
        for i in range(count):
            nb = ctypes.create_unicode_buffer(1024)
            if psapi.GetModuleFileNameExW(handle, buf[i], nb, 1024):
                names.append(nb.value)
        return names
    except Exception:
        return []


def loaded_cuda_providers() -> list[str]:
    """进程里已加载的 ORT CUDA 相关 provider DLL（非空 = GPU 真的启用了）。"""
    return [
        os.path.basename(m) for m in loaded_modules()
        if any(h in os.path.basename(m).lower() for h in _CUDA_PROVIDER_HINTS)
    ]


def cuda_ep_active() -> bool:
    """CUDA 执行提供者是否真的在本进程里加载过。"""
    return bool(loaded_cuda_providers())
