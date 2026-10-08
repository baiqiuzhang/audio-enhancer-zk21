"""取消的**响应延迟**：点「取消」之后必须秒级真的停下，且不留半截产物。

背景（2026-10-01 审计）：原先只有少数几处在循环里检查取消标志——

- `DeepFilterEngine` 用 `subprocess.run(timeout=3600)`：子进程结束前没人能插手。
  DFN 是单线程的，实测 69 分钟素材约 **237 s/声道**，也就是用户点取消后最坏要等这么久；
- `media_io.decode`（69 分钟视频实测 14.61 s）与 `write_video_with_audio`（实测约 136 s）
  完全没有取消入口；
- 而且 `dsp.Cancelled` 与 `engines.Cancelled` 曾是**两个不同的类**，于是取消发生在
  DSP/听感 EQ 阶段时，`pipeline`/`worker` 的 `except Cancelled` 捕获不到，
  界面会把"已取消"显示成"处理失败"并弹错误框。

现在这些都收敛到 `core/errors.py::Cancelled`，并且每个长循环都检查取消标志。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_cancel.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from core import dsp, engines, media_io, modelstore, pipeline  # noqa: E402
from core.errors import Cancelled  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def section(t: str) -> None:
    print()
    print(t)
    print("=" * len(t))


def tone(seconds: float, freq: float = 300.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.05 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def main() -> int:
    print(f"项目根：{ROOT}")
    cat = modelstore.load_catalog(ROOT)
    tmp = tempfile.mkdtemp(prefix="ae_cancel_")
    try:
        section("1. 取消异常只有一个类型（否则取消会被显示成「处理失败」）")
        check("engines.Cancelled 就是 core.errors.Cancelled",
              engines.Cancelled is Cancelled)
        check("dsp.Cancelled 就是 core.errors.Cancelled",
              dsp.Cancelled is Cancelled)
        check("DownloadCancelled 是 Cancelled 的子类",
              issubclass(modelstore.DownloadCancelled, Cancelled))
        raised = False
        try:
            dsp.spectral_process(tone(1.0), SR, cancel=lambda: True)
        except Cancelled:
            raised = True
        except Exception as exc:                      # noqa: BLE001
            raised = f"错类型：{type(exc).__name__}"
        check("DSP 阶段取消抛的就是 Cancelled", raised is True, str(raised))

        section("2. DFN 子进程可以被取消（延迟 ≤2 s；改前最坏 ≈237 s/声道）")
        spec = cat.engine("deepfilternet3")
        if not modelstore.engine_ready(spec, cat):
            print("  [跳过] deepfilternet3 未就绪")
        else:
            eng = engines.build_engine(spec, cat)
            state = {"t0": time.perf_counter(), "at": None}

            def cancel():
                if state["at"] is None and time.perf_counter() - state["t0"] > 0.2:
                    state["at"] = time.perf_counter()
                return state["at"] is not None

            t0 = time.perf_counter()
            caught = None
            try:
                eng.process(tone(30.0), SR, pipeline.EngineParams(), cancel=cancel)
            except Cancelled as exc:
                caught = exc
            dt = time.perf_counter() - (state["at"] or t0)
            check("DFN 子进程被取消", caught is not None, str(caught))
            check("取消延迟 ≤2 s", dt <= 2.0, f"{dt * 1000:.0f} ms")
            leftovers = [n for n in os.listdir(os.environ["TEMP"]) if n.startswith("ae_dfn_")]
            check("临时目录已清理（无 ae_dfn_* 残留）", not leftovers, str(leftovers[:4]))

        section("3. 解码与写回视频可以中断，且不留半成品")
        from media_fixtures import make_video  # noqa: PLC0415

        vid = make_video(os.path.join(tmp, "clip.mp4"), [tone(1.0)], seconds=1.0)
        caught = None
        try:
            media_io.decode(vid, cancel=lambda: True)
        except Cancelled as exc:
            caught = exc
        except Exception as exc:                      # noqa: BLE001
            caught = f"错类型：{type(exc).__name__}: {exc}"
        check("解码可被取消（且类型正确）", isinstance(caught, Cancelled), str(caught))

        out = os.path.join(tmp, "out.mp4")
        caught = None
        try:
            media_io.write_video_with_audio(vid, out, tone(1.0), SR, cancel=lambda: True)
        except Cancelled as exc:
            caught = exc
        except Exception as exc:                      # noqa: BLE001
            caught = f"错类型：{type(exc).__name__}: {exc}"
        check("写回视频可被取消（且类型正确）", isinstance(caught, Cancelled), str(caught))
        check("被取消后不产生输出文件", not os.path.exists(out))
        check("也不留 .part 残片",
              not [n for n in os.listdir(tmp) if ".part" in n],
              str([n for n in os.listdir(tmp) if ".part" in n]))

        section("4. 流水线层面：任意阶段取消都在 2 s 内结束，且不写盘")
        class SlowEngine(engines.BaseEngine):
            """假引擎：忙等到被取消为止（模拟一个不可中断的长阶段）。"""

            def ready(self) -> bool:
                return True

            def process(self, mono, sample_rate, params, *, progress=None, cancel=None):
                while not (cancel and cancel()):
                    time.sleep(0.05)
                raise Cancelled("已取消（假引擎）")

        src = os.path.join(tmp, "t.wav")
        media_io.write_audio(src, tone(2.0), SR)
        out_dir = os.path.join(tmp, "out")
        real_for = engines.engine_for
        engines.engine_for = lambda spec, catalog: SlowEngine(spec, catalog)
        result: dict = {}

        def run_pipeline():
            t0 = time.perf_counter()
            try:
                pipeline.run(src, cat, pipeline.PipelineOptions(
                    engine_id="dsp_baseline", output_dir=out_dir, highpass_hz=0.0,
                ), cancel=lambda: True)
                result["outcome"] = "未取消（意外）"
            except Cancelled as exc:
                result["outcome"] = f"Cancelled: {exc}"
            result["dt"] = time.perf_counter() - t0

        try:
            th = threading.Thread(target=run_pipeline, daemon=True)
            th.start()
            th.join(timeout=10)
            check("流水线在 2 s 内结束", result.get("dt", 99) <= 2.0,
                  f"{result.get('dt', -1) * 1000:.0f} ms")
            check("结果是 Cancelled", str(result.get("outcome", "")).startswith("Cancelled"),
                  str(result.get("outcome")))
            check("被取消的运行没有写出文件",
                  not os.path.isdir(out_dir) or not os.listdir(out_dir),
                  str(os.listdir(out_dir) if os.path.isdir(out_dir) else []))
        finally:
            engines.engine_for = real_for

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"取消响应验证结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("取消响应验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
