"""任务栏图标相关验证。

问题背景：任务栏显示的是**分组图标**，不是窗口图标。进程若没有显式设置
AppUserModelID，就会继承启动它那个进程（例如 ZCode）的标识，于是窗口图标设得再对，
任务栏仍显示成启动者的图标。

因此这里验三件必需的事：
  1. `icon.ico` 本身可用；
  2. 进程的 AppUserModelID 确实被设置成自己的标识（并且**在创建 QApplication 之前**）；
  3. 主窗口上确实挂了图标 —— 用 `WM_GETICON` 回读，与任务栏取图标是同一套机制。

最后一项用**真实平台 + `WA_DontShowOnScreen`**：窗口会被真正创建（有 HWND、有图标），
但不会显示到屏幕上。不能用 offscreen 平台——那个平台下没有真实窗口，读不到图标。

说明：**"看起来是否终于对了"仍需人眼确认一次**；脚本能证明的是三个机制都到位。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_icon.py
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
sys.path.insert(0, ROOT)

WM_GETICON, ICON_SMALL, ICON_BIG = 0x007F, 0, 1

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def main() -> int:
    print("1. 图标文件本身")
    icon = os.path.join(ROOT, "icon.ico")
    check("icon.ico 存在且非空", os.path.isfile(icon) and os.path.getsize(icon) > 1000,
          f"{os.path.getsize(icon):,} B" if os.path.isfile(icon) else "缺失")

    from PyQt5.QtGui import QImage, QIcon  # noqa: PLC0415
    img = QImage(icon)
    check("icon.ico 可被解析", not img.isNull(), f"{img.width()}×{img.height()}")

    print("\n2. 真实平台上创建窗口，并回读窗口图标")
    os.environ.pop("QT_QPA_PLATFORM", None)      # 必须用真实平台，否则没有真实窗口

    import run  # noqa: PLC0415

    # 与 run.py 相同的顺序：先设 AppUserModelID，再建 QApplication
    aumid = run._set_app_user_model_id()
    check("AppUserModelID 已设置且可回读", aumid == run.APP_USER_MODEL_ID,
          f"期望 {run.APP_USER_MODEL_ID!r}，实得 {aumid!r}")

    from PyQt5.QtCore import Qt  # noqa: PLC0415
    from PyQt5.QtWidgets import QApplication  # noqa: PLC0415

    from ui.main_window import MainWindow  # noqa: PLC0415

    app = QApplication.instance() or QApplication([])
    app.setWindowIcon(QIcon(icon))

    win = MainWindow(ROOT)
    win.setWindowIcon(QIcon(icon))
    win.setAttribute(Qt.WA_DontShowOnScreen, True)   # 创建但不显示到屏幕上
    win.show()
    app.processEvents()

    hwnd = int(win.winId())
    check("主窗口已获得原生句柄（HWND）", hwnd != 0, hex(hwnd))
    check("窗口图标对象非空", not win.windowIcon().isNull())

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    big = int(user32.SendMessageW(hwnd, WM_GETICON, ICON_BIG, 0))
    small = int(user32.SendMessageW(hwnd, WM_GETICON, ICON_SMALL, 0))
    check("窗口已挂大图标（WM_GETICON / ICON_BIG 非零）", big != 0, hex(big))
    check("窗口已挂小图标（WM_GETICON / ICON_SMALL 非零）", small != 0, hex(small))
    check("大/小图标是不同句柄（说明两个尺寸都设了）", big != small)

    win.close()
    app.processEvents()

    print("\n3. 源码层面的必要条件")
    src = open(os.path.join(ROOT, "run.py"), encoding="utf-8").read()
    idx_aumid = src.find("_set_app_user_model_id()")
    idx_app = src.find("QApplication(sys.argv)")
    check("AppUserModelID 在创建 QApplication 之前设置",
          0 < idx_aumid < idx_app, f"字符位置 {idx_aumid} < {idx_app}")
    check("窗口图标也被显式设置（不只是 app 级）", "window.setWindowIcon" in src)

    print("\n4. 集成：run.py 作为独立进程启动后保持驻留")
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"     # 这里只要确认不崩，不需要真实窗口
    env["AUDIOENHANCER_PRINT_AUMID"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [PY, os.path.join(ROOT, "run.py")],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    time.sleep(6)
    alive = proc.poll() is None
    check("启动后稳定驻留", alive)
    if alive:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    else:
        print("    输出：", (proc.stdout.read() if proc.stdout else "").strip()[:400])

    print()
    if failures:
        print(f"图标检查结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("图标检查结束：全部通过")
    print("（提醒：任务栏上最终显示什么图标，请人眼看一次确认）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
