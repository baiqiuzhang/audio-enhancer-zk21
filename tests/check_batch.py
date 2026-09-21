"""批处理验证：多文件 × 多模型并行、复用、内存守卫、取消、单文件失败隔离。

核心逻辑先脱离 GUI 验透，再接界面 —— 这样出问题时定位快得多。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_batch.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication  # noqa: E402

from core import engines, modelstore, outputs, pipeline, worker  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "speech_with_noise.wav")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def section(t: str) -> None:
    print()
    print(t)
    print("=" * len(t))


def run_batch(app, proc: worker.BatchProcessor, timeout_s: float = 300.0) -> dict:
    """启动批处理并转事件循环直到结束（信号是从工作线程排队过来的）。

    注意：**start() 由这里统一调用**。先前把它漏在调用方，导致等待一个永不触发的
    信号、白等满超时——把启动和等待绑在一起就不会再犯。
    """
    box: dict = {}

    def on_done(cancelled, summary):
        box["cancelled"] = cancelled
        box["summary"] = summary

    proc.finished_all.connect(on_done)
    proc.start()

    t0 = time.perf_counter()
    while "summary" not in box and time.perf_counter() - t0 < timeout_s:
        app.processEvents()
        time.sleep(0.02)
    if "summary" not in box:
        box["summary"] = {"timeout": True, "done": len(proc.results), "failed": len(proc.errors)}
        print(f"  [超时] 批处理 {timeout_s:.0f}s 未结束")
    return box


def make_jobs(paths, engine_ids, **opts_kw):
    jobs = []
    idx = 0
    for path in paths:
        for eid in engine_ids:
            o = pipeline.PipelineOptions(engine_id=eid, **opts_kw)
            o.engine.num_threads = 2
            jobs.append(worker.BatchJob(index=idx, media_path=path, options=o))
            idx += 1
    return jobs


# ---------------------------------------------------------------------------

def check_memory_guard(app) -> None:
    section("1. 内存守卫")

    avail = worker.available_memory_bytes()
    check("能读到可用物理内存", avail > 0, worker.human_gb(avail))

    n, note = worker.safe_worker_count(4)
    check("内存正常时按请求并发", n == 4 and note is None, f"n={n} note={note}")

    real = worker.available_memory_bytes
    try:
        worker.available_memory_bytes = lambda: int(1.2 * 1024**3)
        n2, note2 = worker.safe_worker_count(4)
        check("内存偏紧时降到 1 个 worker 并说明原因",
              n2 == 1 and bool(note2), f"n={n2}")

        worker.available_memory_bytes = lambda: int(0.3 * 1024**3)
        n3, note3 = worker.safe_worker_count(4)
        check("内存严重不足时拒绝启动（返回 0）", n3 == 0 and bool(note3), f"n={n3}")

        # 拒绝时不得真的跑任何任务
        with tempfile.TemporaryDirectory(prefix="ae_guard_") as tmp:
            inp = os.path.join(tmp, "x.wav")
            shutil.copy2(SAMPLE, inp)
            proc = worker.BatchProcessor(make_jobs([inp], ["dsp_baseline"]),
                                        modelstore.load_catalog(ROOT), max_workers=4)
            box = run_batch(app, proc, timeout_s=60)
            summary = box.get("summary", {})
            check("拒绝启动时立刻结束且不产出结果",
                  summary.get("refused") is True and summary.get("done") == 0
                  and not proc.results)
            check("拒绝原因可读", "内存" in (summary.get("reason") or ""),
                  (summary.get("reason") or "")[:60])
    finally:
        worker.available_memory_bytes = real


def check_parallel(app, cat) -> None:
    section("2. 多文件 × 多模型并行批处理")

    with tempfile.TemporaryDirectory(prefix="ae_batch_") as tmp:
        paths = []
        for name in ("会议 A.wav", "会议 B.wav", "a-really-long-recording-name-for-stem-test.wav"):
            p = os.path.join(tmp, name)
            shutil.copy2(SAMPLE, p)
            paths.append(p)

        jobs = make_jobs(paths, ["dsp_baseline", "gtcrn"])
        check("作业数 = 文件数 × 模型数", len(jobs) == 6, f"{len(jobs)}")

        proc = worker.BatchProcessor(jobs, cat, max_workers=2)
        logs: list[str] = []
        proc.log.connect(logs.append)
        box = run_batch(app, proc)
        summary = box.get("summary", {})
        print(f"  汇总：{summary}")

        check("全部成功", summary.get("done") == 6 and summary.get("failed") == 0,
              f"done={summary.get('done')} failed={summary.get('failed')}")
        check("实际并发为 2", summary.get("workers") == 2, str(summary.get("workers")))

        # 输出位置与命名。三个输入都在同一个目录里，所以只应有 1 个输出子目录。
        out_dirs = {os.path.dirname(r.output_path) for r in proc.results.values()}
        expect_dir = os.path.join(tmp, outputs.OUTPUT_DIRNAME)
        check("输出集中在输入目录的输出子文件夹里",
              out_dirs == {expect_dir}, f"{[os.path.relpath(d, tmp) for d in out_dirs]}")

        # 关键：每个结果声称的路径必须**真的存在且落在预期目录**。
        # 之前正是因为"声称写出了、实际回退到了别处"，才漏掉一个文件。
        missing = [r.output_path for r in proc.results.values() if not os.path.isfile(r.output_path)]
        check("所有结果文件确实存在", not missing, f"缺失 {len(missing)} 个")
        check("没有走输出目录回退",
              not any("已回退" in line for line in logs), 
              next((l for l in logs if "已回退" in l), "")[:60])
        check("实际文件数与结果数一致",
              len([n for n in os.listdir(expect_dir) if n.endswith(".wav")]) == 6,
              f"{len([n for n in os.listdir(expect_dir) if n.endswith('.wav')])} 个文件")

        # 每个输入应各有 2 个变体（两个模型）
        for p in paths:
            vs = outputs.scan_variants(expect_dir, p)
            check(f"{os.path.basename(p)} 有 2 个变体", len(vs) == 2, f"{len(vs)} 个")

        # 并行真的并行了吗：各任务耗时之和 / 总墙钟时间 ≈ 加速比
        serial_sum = sum(r.elapsed for r in proc.results.values())
        wall = max(1e-6, summary.get("elapsed_s", 1e-6))
        speedup = serial_sum / wall
        check("并行确实重叠执行（加速比 > 1.2）", speedup > 1.2,
              f"任务耗时合计 {serial_sum:.2f}s / 墙钟 {wall:.2f}s = {speedup:.2f}×")

        # 第二次运行：全部复用，且 mtime 不变
        mtimes = {r.output_path: os.path.getmtime(r.output_path) for r in proc.results.values()}
        time.sleep(1.1)
        jobs2 = make_jobs(paths, ["dsp_baseline", "gtcrn"])
        proc2 = worker.BatchProcessor(jobs2, cat, max_workers=2)
        box2 = run_batch(app, proc2)
        s2 = box2.get("summary", {})
        check("第二次全部复用", s2.get("reused") == 6, f"reused={s2.get('reused')}")
        check("复用不重写文件（mtime 全未变）",
              all(os.path.getmtime(p) == mtimes[p] for p in mtimes))
        # 复用应当明显更快
        check("复用比首次计算快得多", s2.get("elapsed_s", 1e9) < summary.get("elapsed_s", 0),
              f"{s2.get('elapsed_s')}s vs {summary.get('elapsed_s')}s")


def check_failure_isolation(app, cat) -> None:
    section("3. 单个文件失败不带崩整批")

    with tempfile.TemporaryDirectory(prefix="ae_fail_") as tmp:
        good1 = os.path.join(tmp, "good1.wav")
        good2 = os.path.join(tmp, "good2.wav")
        shutil.copy2(SAMPLE, good1)
        shutil.copy2(SAMPLE, good2)

        bogus = os.path.join(tmp, "不存在.wav")
        broken = os.path.join(tmp, "损坏.wav")
        with open(broken, "wb") as fh:
            fh.write(b"RIFFxxxxWAVEjunkjunkjunk")

        jobs = [
            worker.BatchJob(index=0, media_path=good1, options=pipeline.PipelineOptions(engine_id="dsp_baseline")),
            worker.BatchJob(index=1, media_path=bogus, options=pipeline.PipelineOptions(engine_id="dsp_baseline")),
            worker.BatchJob(index=2, media_path=broken, options=pipeline.PipelineOptions(engine_id="dsp_baseline")),
            worker.BatchJob(index=3, media_path=good2, options=pipeline.PipelineOptions(engine_id="dsp_baseline")),
        ]
        proc = worker.BatchProcessor(jobs, cat, max_workers=2)
        box = run_batch(app, proc)
        s = box.get("summary", {})
        check("坏文件被跳过、好文件照常完成", s.get("done") == 2 and s.get("failed") == 2,
              f"done={s.get('done')} failed={s.get('failed')}")
        check("失败原因被记录且可读", all(proc.errors.get(i) for i in (1, 2)),
              str(proc.errors.get(1, ""))[:50])


def check_no_cross_dir_collision(app, cat) -> None:
    section("5. 同名同大小的不同文件不得互相串用（实测踩过的坑）")

    with tempfile.TemporaryDirectory(prefix="ae_collide_") as tmp:
        dir_a = os.path.join(tmp, "A")
        dir_b = os.path.join(tmp, "B")
        os.makedirs(dir_a); os.makedirs(dir_b)
        fa = os.path.join(dir_a, "interview.wav")
        fb = os.path.join(dir_b, "interview.wav")
        for f in (fa, fb):
            shutil.copy2(SAMPLE, f)          # 同名、同大小、同内容

        o1 = pipeline.PipelineOptions(engine_id="dsp_baseline")
        o2 = pipeline.PipelineOptions(engine_id="dsp_baseline")
        fp_a = outputs.fingerprint(fa, o1, None, "dsp")
        fp_b = outputs.fingerprint(fb, o2, None, "dsp")
        check("同参数下两份文件指纹不同（身份含绝对路径）", fp_a != fp_b,
              f"{fp_a} / {fp_b}")

        r1 = pipeline.run(fa, cat, o1)
        r2 = pipeline.run(fb, cat, o2)
        check("各自写出到自己的目录",
              os.path.dirname(r1.output_path) != os.path.dirname(r2.output_path))
        check("两份结果都是真算出来的（没有互相复用）",
              not r1.reused and not r2.reused)
        check("两份文件同时存在", os.path.isfile(r1.output_path) and os.path.isfile(r2.output_path))
        check("B 不会因为 A 已生成而被判为可复用",
              pipeline.run(fb, cat, pipeline.PipelineOptions(
                  engine_id="dsp_baseline", output_dir=os.path.dirname(r2.output_path))).reused is True)


def check_cancel(app, cat) -> None:
    section("4. 取消")

    with tempfile.TemporaryDirectory(prefix="ae_cancel_") as tmp:
        paths = []
        for i in range(6):
            p = os.path.join(tmp, f"c{i}.wav")
            shutil.copy2(SAMPLE, p)
            paths.append(p)

        jobs = make_jobs(paths, ["dsp_baseline"])
        proc = worker.BatchProcessor(jobs, cat, max_workers=1)
        # 立刻取消：第一个任务刚起步
        def cancel_soon():
            time.sleep(0.05)
            proc.cancel()
        threading.Thread(target=cancel_soon, daemon=True).start()

        box = run_batch(app, proc)
        s = box.get("summary", {})
        check("汇总标记为已取消", s.get("cancelled") is True, str(s.get("cancelled")))
        check("取消后未全部完成", (s.get("done", 0) + s.get("failed", 0)) <= 6,
              f"done={s.get('done')} failed={s.get('failed')}")
        print(f"  取消时的状态：done={s.get('done')} failed={s.get('failed')} 用时 {s.get('elapsed_s')}s")


def check_thread_local_engines(cat) -> None:
    section("5. 每个线程独立引擎实例（sherpa 对象不可跨线程共享）")

    spec_gtcrn = cat.engine("gtcrn")
    # 注意：**不能用 id() 比较**。第一个线程的对象在 release 之后会被回收，
    # 分配器会把同一地址给第二个线程的新对象，于是 id 相同但其实是两个实例。
    # 持有对象引用（而不是 id）才能正确比较。
    seen: dict[int, tuple] = {}

    def grab():
        a = engines.engine_for(spec_gtcrn, cat)
        b = engines.engine_for(spec_gtcrn, cat)
        seen[threading.get_ident()] = (a, b, a is b)   # 持引用，阻止回收

    t1 = threading.Thread(target=grab); t1.start(); t1.join()
    t2 = threading.Thread(target=grab); t2.start(); t2.join()

    vals = list(seen.values())
    check("同一线程内复用同一实例", all(v[2] for v in vals))
    check("不同线程拿到各自独立的实例", vals[0][0] is not vals[1][0])
    check("实例数量 = 线程数（各持一份）", len({id(v[0]) for v in vals}) == 2)
    engines.release_thread_engines()

    # LRU：每线程最多驻留 2 个，超出淘汰最久未用的
    for spec in (cat.engine("dsp_baseline"), spec_gtcrn,
                 cat.engine("dpdfnet2"), cat.engine("dpdfnet4")):
        engines.engine_for(spec, cat)
    cache = getattr(engines._local, "cache", {})
    check("线程内引擎数被 LRU 限制在 2 个", len(cache) == 2, f"{sorted(cache)}")
    check("保留的正是最后用到的两个",
          set(cache) == {"dpdfnet2", "dpdfnet4"}, f"{sorted(cache)}")
    engines.release_thread_engines()
    check("释放后缓存清空",
          not getattr(engines._local, "cache", {}))


def main() -> int:
    if not os.path.isfile(SAMPLE):
        print(f"[FAIL] 找不到测试音频：{SAMPLE}")
        return 2

    app = QApplication.instance() or QApplication([])
    cat = modelstore.load_catalog(ROOT)
    print(f"项目根：{ROOT}")

    check_memory_guard(app)
    check_parallel(app, cat)
    check_failure_isolation(app, cat)
    check_no_cross_dir_collision(app, cat)
    check_cancel(app, cat)
    check_thread_local_engines(cat)

    print()
    if failures:
        print(f"批处理验证结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("批处理验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
