"""本机资源探测（CPU 之外的那几项：物理内存）。

单独成模块的理由：**批处理的内存守卫**与**带宽扩展的事前内存预检**都要读空闲内存，
而它们分别住在 `core/worker.py` 与 `core/bwe.py` 里——后者不能导入前者
（`worker` 导入 `pipeline`，会成环）。所以把实现放在这个最底层、无依赖的模块里，
两边共用一份（而不是各写一遍）。

只做只读探测：不改系统状态、不改进程环境。
"""

from __future__ import annotations

import ctypes

__all__ = ["available_memory_bytes", "available_memory_mb", "human_gb"]


def available_memory_bytes() -> int:
    """可用物理内存（字节）。用 ctypes 直调，避免引入 psutil 依赖。

    取不到时返回 0——调用方必须把 0 当作"未知"而不是"没有内存"，否则会在
    探测失败的机器上把功能全部拒掉。
    """

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys)
    except Exception:
        pass
    return 0


def available_memory_mb() -> float:
    """可用物理内存（MB）；取不到时返回 0.0（= 未知）。"""
    return available_memory_bytes() / (1024.0 * 1024.0)


def human_gb(n: int) -> str:
    return f"{n / 1024**3:.2f} GB"
