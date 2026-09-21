"""诊断 PySide6 无法导入的原因（DLL 依赖问题）。"""
import ctypes
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(ROOT, ".venv", "Lib", "site-packages", "PySide6")


class V(ctypes.Structure):
    _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint32),
                ("c", ctypes.c_uint32), ("d", ctypes.c_uint32)]


def file_version(path: str) -> str:
    if not os.path.isfile(path):
        return "缺失"
    try:
        ver = ctypes.WinDLL("version")
        size = ver.GetFileVersionInfoSizeW(path, None)
        if not size:
            return "无版本信息"
        buf = ctypes.create_string_buffer(size)
        ver.GetFileVersionInfoW(path, 0, size, buf)
        ptr = ctypes.c_void_p()
        length = ctypes.c_uint()
        if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)):
            return "查询失败"
        v = ctypes.cast(ptr, ctypes.POINTER(V)).contents
        return f"{v.a}.{v.b}.{v.c}.{v.d}"
    except Exception as exc:  # noqa: BLE001
        return f"错误 {exc}"


print("=== VC 运行库版本（PySide6 6.11 需要较新的 VS2022 运行库）===")
for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
    print(f"  {name:<22}{file_version(os.path.join(r'C:\\Windows\\System32', name))}")

print()
print(f"=== PySide6 目录：{SP} ===")
print(f"  存在: {os.path.isdir(SP)}")
print(f"  Qt6Core.dll 版本: {file_version(os.path.join(SP, 'Qt6Core.dll'))}")

print()
print("=== 直接用 ctypes 加载，拿精确错误 ===")
os.add_dll_directory(SP)
for dll in ("Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll"):
    p = os.path.join(SP, dll)
    try:
        ctypes.WinDLL(p)
        print(f"  OK   {dll}")
    except OSError as exc:
        print(f"  FAIL {dll}: {exc}")

print()
print("=== 尝试只加载 QtCore.pyd（Python 扩展） ===")
try:
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("QtCore_probe", os.path.join(SP, "QtCore.pyd"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("  OK   QtCore.pyd")
except Exception as exc:  # noqa: BLE001
    print(f"  FAIL QtCore.pyd: {type(exc).__name__}: {exc}")

print()
print("=== PATH 中可能抢先的 Qt DLL ===")
seen = []
for d in os.environ.get("PATH", "").split(os.pathsep):
    if not d or not os.path.isdir(d):
        continue
    for f in ("Qt6Core.dll", "Qt5Core.dll"):
        if os.path.isfile(os.path.join(d, f)):
            seen.append(os.path.join(d, f))
for s in seen or ["（无）"]:
    print("  " + s)
print(f"  sys.version_info = {sys.version_info[:3]}")
