"""AudioEnhancer_zk21 启动入口。

用法（项目根目录）：
    .venv\\Scripts\\python.exe  run.py     # 带控制台，便于看输出
    .venv\\Scripts\\pythonw.exe run.py     # 无控制台窗口（桌面快捷方式用的就是这个）
"""

from __future__ import annotations

import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REQUIRED = [
    ("PyQt5", "PyQt5"),               # 见 README：本机 Anaconda 的 VC 运行库过旧，PySide6/Qt6 无法加载
    ("numpy", "numpy"),
    ("soundfile", "soundfile"),
    ("soxr", "soxr"),
    ("av", "av"),
    ("sherpa_onnx", "sherpa-onnx"),
    ("sounddevice", "sounddevice"),
]

LOG_PATH = os.path.join(ROOT, "out", "startup_error.log")

# Windows 任务栏用它来给窗口分组。不显式设置的话，进程会继承启动者（例如 ZCode）的
# AppUserModelID，任务栏就把本应用显示成启动者的图标——窗口图标设得再对也没用，
# 因为那是"分组图标"。必须在创建任何窗口之前设置。
APP_USER_MODEL_ID = "zju.AudioEnhancer_zk21"


def _set_app_user_model_id() -> str:
    """设置任务栏分组标识。返回实际读回的值（便于测试与诊断）。"""
    try:
        import ctypes  # noqa: PLC0415

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        return _get_app_user_model_id()
    except Exception:
        return ""


def _get_app_user_model_id() -> str:
    """回读当前进程的 AppUserModelID（未设置时返回空串）。"""
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        ptr = ctypes.c_wchar_p()
        ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(ptr))
        return ptr.value or ""
    except Exception:
        return ""


def _check_dependencies() -> list[str]:
    missing = []
    for module, package in REQUIRED:
        try:
            __import__(module)
        except Exception:
            missing.append(package)
    return missing


def _report_fatal(title: str, message: str) -> None:
    """让启动失败可见。

    用 pythonw 启动时没有控制台，异常会静默消失。所以这里除了落盘日志，
    在没有控制台（sys.stdout is None）时再弹一个系统消息框——
    用 ctypes 直调 MessageBoxW，不依赖 Qt，即使 Qt 本身导入失败也能显示。
    """
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {title} ===\n")
            fh.write(message.rstrip() + "\n")
        hint = LOG_PATH
    except Exception:
        hint = "（日志写入失败）"

    if sys.stdout is not None:
        return   # 有控制台，错误已经打印出来了，不必再弹模态框打断

    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None, f"{message[-1200:]}\n\n详情已写入：\n{hint}", title, 0x10  # MB_ICONERROR
        )
    except Exception:
        pass


def main() -> int:
    missing = _check_dependencies()
    if missing:
        lines = [
            "缺少依赖，无法启动：",
            *(f"  - {m}" for m in missing),
            "",
            "请先建立独立环境并安装依赖（不会触碰 Anaconda base）：",
            "  powershell -ExecutionPolicy Bypass -File setup_env.ps1",
            "",
            f"当前解释器：{sys.executable}",
        ]
        print("\n".join(lines))
        _report_fatal("AudioEnhancer_zk21 — 启动失败", "\n".join(lines))
        return 2

    try:
        from PyQt5.QtGui import QIcon  # noqa: PLC0415
        from PyQt5.QtWidgets import QApplication  # noqa: PLC0415

        from ui.main_window import MainWindow  # noqa: PLC0415

        # 必须在 QApplication（以及任何窗口）之前设置，否则任务栏分组已经定型
        aumid = _set_app_user_model_id()

        app = QApplication(sys.argv)
        app.setApplicationName("AudioEnhancer_zk21")
        app.setApplicationDisplayName("AudioEnhancer_zk21")

        # 设到 QApplication 上，这样所有对话框/消息框也会带上图标（不只是主窗口）
        icon_path = os.path.join(ROOT, "icon.ico")
        if os.path.isfile(icon_path):
            app.setWindowIcon(QIcon(icon_path))

        window = MainWindow(ROOT)   # 深色主题由 MainWindow 兜底应用
        if os.path.isfile(icon_path):
            window.setWindowIcon(QIcon(icon_path))
        window.show()
        if os.environ.get("AUDIOENHANCER_PRINT_AUMID"):
            print(f"AppUserModelID = {aumid!r}", flush=True)
        return app.exec_()
    except Exception:  # noqa: BLE001 — 顶层兜底，保证失败可见
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        _report_fatal("AudioEnhancer_zk21 — 启动异常", tb)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
