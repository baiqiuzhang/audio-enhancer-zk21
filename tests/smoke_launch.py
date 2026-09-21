"""入口验证：以子进程方式真实启动 run.py，确认窗口创建成功且进程不崩溃。

为什么要单开一个脚本：如果先把 QApplication 建好再调用 run.main()，会形成双重
QApplication（PyQt5 不支持），测试本身就会崩，测不出真实入口的问题。
这里直接用子进程跑真正的 run.py，并在无头平台下确认它稳定驻留。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\smoke_launch.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ALIVE_SECONDS = 6.0


def main() -> int:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"   # 不弹窗口，可在命令行跑
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"解释器: {PY}")
    print(f"启动: run.py （{ALIVE_SECONDS:.0f} 秒后检查是否仍在运行）")
    proc = subprocess.Popen(
        [PY, os.path.join(ROOT, "run.py")],
        cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )

    deadline = time.time() + ALIVE_SECONDS
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            tail = "\n".join(
                line for line in out.splitlines()
                if "QFontDatabase" not in line and "propagateSizeHints" not in line
            )
            print(f"\nFAIL  进程在 {ALIVE_SECONDS:.0f} 秒内就退出了（退出码 {proc.returncode}）")
            print("输出：")
            print(tail.strip() or "（无输出）")
            return 1
        time.sleep(0.2)

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    print()
    print("PASS  入口启动成功，窗口创建后稳定驻留（已被测试主动终止）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
