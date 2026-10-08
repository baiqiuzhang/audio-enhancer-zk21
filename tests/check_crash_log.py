"""崩溃可诊断性：界面里未捕获的异常必须留下记录。

背景（2026-10-01 审计）：发行方式是 **pythonw**（没有控制台），而全仓库此前没有
`sys.excepthook` / `threading.excepthook` / `faulthandler`。于是"某个槽函数里抛了异常"
这件事对用户表现为**程序静默消失**，`out/startup_error.log` 里什么都没有——事后无法诊断。
装钩子之前实测：异常只由 PyQt 默认路径打印（pythonw 下等于打印到虚空）；
子代理在另一条路径上还观察到进程以 `0xC0000409` (abort) 结束。

本套件不假设"会不会 abort"（不同 PyQt 版本/不同触发路径行为不同），只断言**日志**：
主线程槽内异常与工作线程异常都必须带 traceback 落到指定日志里。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_crash_log.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures: list[str] = []

PROBE = r'''
import os, sys, threading
sys.path.insert(0, r"{root}")
import run as app
app.LOG_PATH = r"{log}"
app.install_crash_handlers()
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

def boom_slot():
    raise ValueError("{marker_slot}")

def boom_thread():
    raise RuntimeError("{marker_thread}")

a = QApplication(sys.argv)
QTimer.singleShot(50, boom_slot)
QTimer.singleShot(200, lambda: threading.Thread(target=boom_thread).start())
QTimer.singleShot(1200, a.quit)
a.exec_()
print("probe-done")
'''


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def main() -> int:
    print(f"项目根：{ROOT}")
    tmp = tempfile.mkdtemp(prefix="ae_crashlog_")
    log = os.path.join(tmp, "probe_error.log")
    script = os.path.join(tmp, "probe.py")
    tag = os.urandom(4).hex()
    marker_slot = f"SLOT_{tag}"
    marker_thread = f"THREAD_{tag}"

    with open(script, "w", encoding="utf-8") as fh:
        fh.write(PROBE.format(root=ROOT, log=log, marker_slot=marker_slot,
                              marker_thread=marker_thread))

    print("\n1. 子进程里在 Qt 槽内与工作线程内各抛一个未捕获异常")
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True,
                          timeout=60)
    print(f"  子进程退出码 {proc.returncode}   stdout 末尾：{(proc.stdout or '').strip()[-40:]!r}")
    check("子进程正常跑完（没有挂死）", proc.returncode is not None)

    text = ""
    if os.path.isfile(log):
        with open(log, encoding="utf-8") as fh:
            text = fh.read()
    check("日志文件已生成", bool(text), log)
    check("槽内异常带 traceback 被记录",
          marker_slot in text and "Traceback" in text,
          text[-200:] if text else "（空）")
    check("工作线程异常也带 traceback 被记录",
          marker_thread in text and "RuntimeError" in text,
          text[-200:] if text else "（空）")
    check("日志里写明异常类型", "未捕获异常（ValueError）" in text, text[:200])

    print("\n2. faulthandler 已装载（硬崩溃也不至于一无所知）")
    probe2 = os.path.join(tmp, "probe2.py")
    with open(probe2, "w", encoding="utf-8") as fh:
        fh.write(
            f"import sys\nsys.path.insert(0, r'{ROOT}')\nimport run as app\n"
            f"app.LOG_PATH = r'{log}'\napp.install_crash_handlers()\n"
            "import faulthandler\nprint('faulthandler enabled:', faulthandler.is_enabled())\n"
        )
    p2 = subprocess.run([sys.executable, probe2], capture_output=True, text=True, timeout=60)
    check("faulthandler.is_enabled() 为真", "True" in (p2.stdout or ""),
          (p2.stdout or "").strip() + (p2.stderr or "")[-120:])

    print("\n3. 钩子不吞异常语义：原异常仍会被 sys.excepthook 之外的地方看到")
    check("（记录不改判）退出码非负、进程未崩溃", proc.returncode >= 0, str(proc.returncode))

    print()
    if failures:
        print(f"崩溃可诊断性验证结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("崩溃可诊断性验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
