"""变体发现的三条 UI 路径 + 进程重启后仍可发现。

交接文件指出这套用例一直缺着：后端能力（`outputs.scan_variants`）已在
`check_outputs` / `check_batch` 里被间接验证过，但**没有**端到端用例证明
「界面里多选跑过、先后单跑、先单跑再多选」这三种真实操作序列都能在列表里被看到。
这一套就是补这个缺口。

三条路径的定义：
    路径一 先后单选   —— 逐个文件用「开始处理」（_start_process）跑，换引擎再跑一次
    路径二 多选后     —— 勾选多个文件用「批量处理」（_start_batch）一次跑完
    路径三 先后多选   —— 先单跑、再对同一批做多选，验证不会重复生成
外加：把变体发现交给一个**全新进程**，证明它只依赖磁盘而不是内存状态。

所有输入与输出都发生在临时目录里，不碰 samples/ 与项目 out/。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_variants.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from core import outputs  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "speech_with_noise.wav")
PY = sys.executable

# 两个轻量引擎：dsp_baseline 不需要权重，gtcrn 只有 535 KB。
# 用不同引擎就能得到不同指纹，从而在同一文件上产生多个变体，而不必跑重模型。
ENGINE_A = "dsp_baseline"
ENGINE_B = "gtcrn"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def section(t: str) -> None:
    print()
    print(t)
    print("=" * len(t))


# ---------------------------------------------------------------------------
# 与界面交互的小工具
# ---------------------------------------------------------------------------

def pump(app, predicate, timeout_s: float = 180.0, label: str = "") -> bool:
    """转事件循环直到条件成立（Qt 信号是队列投递的，必须转循环才会送达）。"""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    print(f"    [超时] 等待 {label} 超过 {timeout_s}s")
    return False


def engine_row(win, engine_id: str) -> int:
    for row in range(win.list_engines.count()):
        if win.list_engines.item(row).data(Qt.UserRole) == engine_id:
            return row
    return -1


def select_engine(win, app, engine_id: str) -> bool:
    row = engine_row(win, engine_id)
    if row < 0:
        return False
    win.list_engines.setCurrentRow(row)
    app.processEvents()
    return win._selected_engine().id == engine_id


def open_file(win, app, path: str) -> bool:
    """把界面的「当前文件」切到 path 并等分析完成。"""
    win._import_file(path)
    if not pump(app, lambda: win._stats is not None and
                os.path.abspath(win._media_path or "") == os.path.abspath(path),
                label=f"分析 {os.path.basename(path)}"):
        return False
    app.processEvents()
    return True


def current_variants(win) -> list:
    return [win.list_variants.item(i).data(Qt.UserRole)
            for i in range(win.list_variants.count())]


def variant_outputs(win) -> set[str]:
    return {e.output for e in current_variants(win)}


def process_one(win, app) -> bool:
    """走单文件路径（_start_process），并等拿到**新**结果。"""
    win._result = None          # 不清空的话，等 _result 会立刻命中上一次的旧结果
    win._start_process()
    return pump(app, lambda: win._result is not None, timeout_s=300.0, label="单文件处理")


def process_batch(win, app) -> bool:
    win._start_batch()
    if win._batch is None:
        print("    [跳过] 批处理被内存守卫拒绝启动（当前可用内存不足）")
        return False
    return pump(app, lambda: win._batch is None, timeout_s=600.0, label="批处理")


def wav_count(out_dir: str) -> int:
    if not os.path.isdir(out_dir):
        return 0
    return len([n for n in os.listdir(out_dir) if n.lower().endswith(".wav")])


# ---------------------------------------------------------------------------
# 路径一 / 二 / 三
# ---------------------------------------------------------------------------

def run_paths(app, work: dict) -> None:
    a_path, b_path = work["a"], work["b"]
    out_dir = work["out_dir"]

    win = MainWindow(ROOT)

    section("1. 路径一：先后单选（换引擎再跑一次）")
    if not open_file(win, app, a_path):
        check("打开文件 A", False)
        win.close()
        return
    check("打开文件 A", True)
    check("初始无变体", win.list_variants.count() == 0, f"count={win.list_variants.count()}")

    if not select_engine(win, app, ENGINE_A):
        check(f"选中引擎 {ENGINE_A}", False)
        win.close()
        return
    check(f"选中引擎 {ENGINE_A}", True)
    check("单文件处理完成", process_one(win, app))
    check("处理后立刻出现一个变体", win.list_variants.count() == 1,
          f"count={win.list_variants.count()}")

    # 换引擎再跑：同一文件的第二个变体
    check(f"选中引擎 {ENGINE_B}", select_engine(win, app, ENGINE_B))
    check("第二次单文件处理完成", process_one(win, app))
    check("同一文件出现两个变体", win.list_variants.count() == 2,
          f"count={win.list_variants.count()}")
    outs_a = variant_outputs(win)
    a_stem = outputs.safe_stem(a_path)
    check("两个变体都属于 A（没串到别的文件）",
          all(n.startswith(a_stem) for n in outs_a), str(sorted(outs_a)))
    check("磁盘上确有对应的 wav", wav_count(out_dir) == 2, f"{wav_count(out_dir)} 个")

    section("2. 路径二：多选后批量处理")
    win._add_files([a_path, b_path])
    check("列表里是两个文件", win.list_files.count() == 2, f"count={win.list_files.count()}")
    win.list_files.selectAll()
    check("两个都被勾选", len(win._selected_files()) == 2)
    check("批量处理完成", process_batch(win, app))
    check("日志报告批处理完成且未被拒绝",
          "批处理完成" in win.txt_log.toPlainText()

          and "批处理未启动" not in win.txt_log.toPlainText())
    check("磁盘上两个文件各有产出（A 复用 + B 新增 = 3）", wav_count(out_dir) == 3,
          f"{wav_count(out_dir)} 个")

    # B 是第一次跑，应当能看到自己的变体
    check("打开文件 B", open_file(win, app, b_path))
    outs_b = variant_outputs(win)
    b_stem = outputs.safe_stem(b_path)
    check("B 能看到自己的变体", len(outs_b) == 1, str(sorted(outs_b)))
    check("B 的变体都属于 B", all(n.startswith(b_stem) for n in outs_b), str(sorted(outs_b)))

    section("3. 路径三：先单跑再多选，不重复生成")
    before = wav_count(out_dir)
    check("回到文件 A", open_file(win, app, a_path))
    check("A 仍是两个变体（列表按输入名过滤，不串用）",
          variant_outputs(win) == outs_a, str(sorted(variant_outputs(win))))
    win.list_files.selectAll()
    check("再批量一次（参数与已有结果相同）", process_batch(win, app))
    check("没有新增文件（同参数被复用，不二次生成）", wav_count(out_dir) == before,
          f"{before} → {wav_count(out_dir)}")
    check("A 的变体集合不变", variant_outputs(win) == outs_a, str(sorted(variant_outputs(win))))

    win.close()
    work["expect_a"] = sorted(outs_a)
    work["expect_b"] = sorted(outs_b)


# ---------------------------------------------------------------------------
# 重启后仍可发现
# ---------------------------------------------------------------------------

CHILD = r"""
import os, sys
sys.path.insert(0, os.environ["AE_ROOT"])
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

app = QApplication([])
win = MainWindow(os.environ["AE_ROOT"])
win._import_file(os.environ["AE_PROBE_PATH"])
for _ in range(6000):
    app.processEvents()
    if win._stats is not None:
        break
app.processEvents()
names = sorted(win.list_variants.item(i).data(0x0100).output
               for i in range(win.list_variants.count()))
print("VARIANTS=" + ",".join(names))
win.close()
"""


def run_restart_check(work: dict) -> None:
    section("4. 全新进程里仍能发现同一批变体")

    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONIOENCODING"] = "utf-8"
    env["AE_ROOT"] = ROOT
    env["AE_PROBE_PATH"] = work["a"]

    try:
        proc = subprocess.run(
            [PY, "-c", CHILD], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", timeout=300,
        )
    except subprocess.TimeoutExpired:
        check("子进程在 300s 内结束", False)
        return

    line = ""
    for raw in (proc.stdout or "").splitlines():
        if raw.startswith("VARIANTS="):
            line = raw[len("VARIANTS="):]
    check("子进程正常退出", proc.returncode == 0,
          f"exit={proc.returncode}")
    if not line:
        check("子进程输出了变体列表", False, (proc.stdout or "")[-400:])
        return
    got = sorted(n for n in line.split(",") if n)
    check("新进程发现同一批变体", got == work["expect_a"],
          f"子进程 {got} vs 主进程 {work['expect_a']}")


# ---------------------------------------------------------------------------

def main() -> int:
    if not os.path.isfile(SAMPLE):
        print(f"[FAIL] 找不到测试音频：{SAMPLE}")
        return 2

    print(f"项目根：{ROOT}")
    app = QApplication([])

    tmp = tempfile.mkdtemp(prefix="ae_variants_")
    try:
        a_path = os.path.join(tmp, "会议 A.wav")
        b_path = os.path.join(tmp, "会议 B.wav")
        shutil.copyfile(SAMPLE, a_path)
        shutil.copyfile(SAMPLE, b_path)
        out_dir, fallback = outputs.output_dir_for(a_path, ROOT)
        print(f"临时工作目录：{tmp}")
        print(f"输出目录：{out_dir}")
        if fallback:
            print(f"（注意）{fallback}")

        work = {"a": a_path, "b": b_path, "out_dir": out_dir}
        run_paths(app, work)
        run_restart_check(work)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"变体发现验证结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("变体发现验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
