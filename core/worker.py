"""Qt 线程包装：把耗时的流水线、批处理与下载放到后台，保持界面响应。

core/ 下的模块刻意不依赖 Qt，这里只做信号转发与取消标志。

用 PyQt5 而非 PySide6：本机 Python 来自 Anaconda，其 python312.dll 在进程启动时
就从 Anaconda 目录加载了 VC 运行库 14.44；PySide6 6.11 自带的 Qt 6.11.2 要求
≥14.50，导入即报 WinError 127（已实测）。Qt5 只要求 14.2x，故 PyQt5 是本机可行路径。
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from PyQt5.QtCore import QObject, QThread, pyqtSignal as Signal

from . import dsp, media_io, outputs, pipeline
from .sysinfo import available_memory_bytes, human_gb
from .engines import Cancelled
from .modelstore import Catalog, DownloadCancelled, DownloadError, download_files
# 内存守护阈值（本机空闲内存常年只有 3–4 GB，而单个降噪实例约 200–500 MB、
# 带宽扩展模型 400–700 MB，所以并发数必须按内存收敛，而不是按 CPU 核数）
MIN_FREE_FOR_ANY = 0.8 * 1024**3      # 低于此值直接拒绝启动批次
MIN_FREE_FOR_PARALLEL = 1.5 * 1024**3  # 低于此值降到 1 个 worker


def safe_worker_count(requested: int) -> tuple[int, str | None]:
    """按当前可用内存收敛并发数。

    返回 (实际并发数, 说明)。返回 0 表示拒绝启动——**只报告占用，不删除任何文件**。
    """
    avail = available_memory_bytes()
    if avail == 0:                       # 读不到就按请求值走，不误判
        return max(1, requested), None
    if avail < MIN_FREE_FOR_ANY:
        return 0, (
            f"可用内存仅 {human_gb(avail)}，低于 {human_gb(int(MIN_FREE_FOR_ANY))} 的安全下限，"
            "已拒绝启动批处理。请关闭其它占用内存的程序后重试。"
        )
    if avail < MIN_FREE_FOR_PARALLEL and requested > 1:
        return 1, (
            f"可用内存 {human_gb(avail)} 偏紧，已自动把并发从 {requested} 降到 1，"
            "以免多个模型实例同时驻留导致内存不足。"
        )
    return max(1, requested), None


# 重阶段的全局互斥在这个模块里重新导出，方便调用方从一处取；
# 真正的定义在 engines.py（pipeline 需要它，而 pipeline 不能导入 worker，会成环）。
from .engines import HEAVY_STAGE  # noqa: E402,F401


class PipelineWorker(QThread):
    """在后台线程跑一遍完整流水线。"""

    progressed = Signal(float, str)
    succeeded = Signal(object)          # pipeline.PipelineResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, media_path, catalog, options, stream_index=None, parent=None):
        super().__init__(parent)
        self._path = media_path
        self._catalog = catalog
        self._options = options
        self._stream_index = stream_index
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:
        try:
            result = pipeline.run(
                self._path,
                self._catalog,
                self._options,
                stream_index=self._stream_index,
                progress=lambda f, m: self.progressed.emit(f, m),
                cancel=self.is_cancelled,
            )
        except Cancelled:
            self.cancelled.emit()
        except DownloadError as exc:
            self.failed.emit(f"模型下载失败：\n{exc}")
        except FileNotFoundError as exc:
            self.failed.emit(f"缺少模型文件：\n{exc}")
        except MemoryError:
            self.failed.emit(
                "内存不足。请改用更小的模型（GTCRN / DPDFNet2）、"
                "开启「裁剪静音」缩短素材，或关闭其它占用内存的程序后重试。"
            )
        except Exception as exc:  # noqa: BLE001 — 兜底，避免线程内异常静默丢失
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.succeeded.emit(result)


class AnalyzeWorker(QThread):
    """导入后立即在后台解码并统计电平、探测静音区间，用于信息面板与波形预览。

    这里会完整解码一次；处理时会再解码一次。多一次解码换来「处理前就能看到电平与
    待裁剪区间」，对长文件也只是一次几秒的开销，值得。

    `cancel()` 是**协作取消**（2026-10-01 加）：界面在用户切文件时调用它，worker 在解码
    循环里看到标志就尽快退出。此前界面用的是 `QThread.terminate()`——在线程可能正跑着
    PyAV/soundfile 的 C 代码时硬杀，PyQt 官方明确反对（死锁/堆损坏风险），而且对 69 分钟
    素材 `wait(2000)` 必然超时。
    """

    progressed = Signal(str)
    analyzed = Signal(object, object, object, object)   # (MediaInfo, AudioData, LevelStats, ActiveSpan)
    failed = Signal(str)

    def __init__(self, media_path, stream_index=None, parent=None):
        super().__init__(parent)
        self._path = media_path
        self._stream_index = stream_index
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:
        try:
            if self._cancel:
                return
            self.progressed.emit("探测媒体…")
            info = media_io.probe(self._path)
            index = self._stream_index if self._stream_index is not None else info.default_audio_index
            decode_index = index if info.decoder == "pyav" else None

            self.progressed.emit("解码并统计电平…")
            data = media_io.decode(self._path, stream_index=decode_index,
                                   cancel=self.is_cancelled)
            if self._cancel:
                return

            mono = media_io.to_mono(data.samples)
            stats = dsp.analyze(mono, data.sample_rate)
            if self._cancel:
                return

            self.progressed.emit("检测静音区间…")
            span = dsp.detect_active_span(mono, data.sample_rate)
        except Cancelled:
            return                      # 用户切了文件：静默退出，不要报"失败"
        except Exception as exc:  # noqa: BLE001
            if not self._cancel:
                self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            if not self._cancel:
                self.analyzed.emit(info, data, stats, span)


@dataclass
class BatchJob:
    """批处理里的一项：一个文件 + 一套参数。"""

    index: int
    media_path: str
    options: pipeline.PipelineOptions
    stream_index: int | None = None
    root: str = ""                       # 用于算输出目录（默认取输入文件所在目录）


class BatchProcessor(QObject):
    """并行批处理。

    并发数**按可用内存收敛**而不是按 CPU 核数：本机 20 个线程但常年只有 3–4 GB 空闲，
    而单个降噪实例约 200–500 MB、带宽扩展模型 400–700 MB。内存不足时降为 1 个 worker，
    再不足就直接拒绝启动并说明原因（只报告占用，绝不删除任何文件）。

    每个 worker 线程持有自己的引擎实例（见 engines.engine_for）——sherpa 的
    denoiser 对象不可跨线程共享。
    """

    job_progress = Signal(int, float, str)    # 序号, 0..1, 说明
    job_done = Signal(int, object)            # 序号, PipelineResult
    job_failed = Signal(int, str)             # 序号, 错误信息
    overall = Signal(int, int)                # 已完成, 总数
    log = Signal(str)
    finished_all = Signal(bool, object)       # (是否被取消, 汇总)

    def __init__(self, jobs: list[BatchJob], catalog: Catalog,
                 max_workers: int = 2, parent=None):
        super().__init__(parent)
        self._jobs = list(jobs)
        self._catalog = catalog
        self._requested = max(1, int(max_workers))
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self.results: dict[int, pipeline.PipelineResult] = {}
        self.errors: dict[int, str] = {}
        self.reused_count = 0
        self.workers_used = 0
        self.refused = False
        self.refusal_reason = ""

    # --- 控制 ---------------------------------------------------------
    def cancel(self) -> None:
        if not self._cancel.is_set():
            self._cancel.set()
            self.log.emit("正在取消…（已在处理的文件会尽快收尾）")

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def start(self) -> None:
        workers, note = safe_worker_count(self._requested)
        if note:
            self.log.emit(note)
        if workers <= 0:
            self.refused = True
            self.refusal_reason = note or "内存不足"
            self.finished_all.emit(False, {
                "refused": True, "reason": self.refusal_reason,
                "total": len(self._jobs), "done": 0, "failed": 0, "reused": 0,
            })
            return
        self.workers_used = workers
        self.log.emit(f"批处理开始：{len(self._jobs)} 个文件，并发 {workers}")
        self._thread = threading.Thread(
            target=self._coordinate, args=(workers,), name="ae-batch", daemon=True)
        self._thread.start()

    # --- 内部 ---------------------------------------------------------
    def _coordinate(self, workers: int) -> None:
        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ae-worker") as pool:
                futures = [pool.submit(self._run_job, job) for job in self._jobs]
                for fut in futures:
                    try:
                        fut.result()
                    except Exception as exc:      # noqa: BLE001 — _run_job 内部已兜底
                        self.log.emit(f"[批处理内部错误] {type(exc).__name__}: {exc}")
        finally:
            elapsed = time.perf_counter() - t0
            summary = {
                "refused": False,
                "cancelled": self._cancel.is_set(),
                "total": len(self._jobs),
                "done": len(self.results),
                "failed": len(self.errors),
                "reused": self.reused_count,
                "workers": workers,
                "elapsed_s": round(elapsed, 2),
            }
            self.finished_all.emit(self._cancel.is_set(), summary)

    def _run_job(self, job: BatchJob) -> None:
        if self._cancel.is_set():
            with self._lock:
                self.errors[job.index] = "已取消"
            self.job_failed.emit(job.index, "已取消")
            self._bump()
            return

        def progress(frac: float, msg: str) -> None:
            self.job_progress.emit(job.index, float(frac), msg or "")

        def log(msg: str) -> None:
            self.log.emit(f"[{os.path.basename(job.media_path)}] {msg}")

        try:
            res = pipeline.run(
                job.media_path, self._catalog, job.options,
                stream_index=job.stream_index,
                progress=progress, cancel=self._cancel.is_set, log=log,
            )
        except Cancelled:
            with self._lock:
                self.errors[job.index] = "已取消"
            self.job_failed.emit(job.index, "已取消")
        except MemoryError:
            msg = "内存不足，已跳过该文件（可改用更小的模型或减少并发）"
            with self._lock:
                self.errors[job.index] = msg
            self.job_failed.emit(job.index, msg)
        except Exception as exc:                  # noqa: BLE001 — 单个文件失败不应带崩整批
            msg = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.errors[job.index] = msg
            self.job_failed.emit(job.index, msg)
        else:
            with self._lock:
                self.results[job.index] = res
                if res.reused:
                    self.reused_count += 1
            self.job_done.emit(job.index, res)
        self._bump()

    def _bump(self) -> None:
        with self._lock:
            done = len(self.results) + len(self.errors)
        self.overall.emit(done, len(self._jobs))


class DownloadWorker(QThread):
    """在后台下载引擎所需的模型文件。"""
    progressed = Signal(float, str)
    log = Signal(str)
    finished_all = Signal(bool, str)     # (成功?, 错误信息)

    def __init__(self, files, catalog: Catalog, parent=None):
        super().__init__(parent)
        self._files = files
        self._catalog = catalog
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:
        try:
            download_files(
                self._files,
                self._catalog,
                progress=lambda done, total: self.progressed.emit(
                    (done / total) if total else 0.0, ""
                ),
                cancel=self.is_cancelled,
                log=lambda m: self.log.emit(m),
            )
        except DownloadCancelled:
            self.finished_all.emit(False, "已取消下载。")
        except DownloadError as exc:
            self.finished_all.emit(False, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.finished_all.emit(False, f"{type(exc).__name__}: {exc}")
        else:
            self.finished_all.emit(True, "")
