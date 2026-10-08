"""分阶段耗时测量 —— 「先量后改」。

回答的问题是：**瓶颈在推理、还是在解码 / 重采样 / 被内存压到单并发？**
在补分阶段耗时之前，流水线只有总耗时，这个问题无法回答；而"要不要为 GPU 换
sherpa 的 GPU wheel（约 200 MB、还要解决 cuDNN 可见性）"恰恰要靠这个数字决定。

预设与界面一致（见 ui/main_window.py 的 `_apply_preset`）：

    faithful  识别用（保真）  ：纯降噪 + 高通 80 Hz + 归一化
    demud     去闷（保真档）  ：faithful + 搁架 +6 dB @ 3000 + 去闷 -5 dB @ 350
    listen    试听用（增强）  ：faithful + 搁架 +3 dB @ 3000 + 温和压缩 + 带宽扩展
    denoise   只跑降噪        ：量推理本身的上限，不叠任何后处理

用法（项目根目录）：

    .venv\\Scripts\\python.exe tools\\measure_stages.py <媒体文件> --threads 2,4
    .venv\\Scripts\\python.exe tools\\measure_stages.py <媒体文件> --preset demud --no-write
    .venv\\Scripts\\python.exe tools\\measure_stages.py <媒体文件> --max-seconds 300

每个配置打印一张分阶段表，并额外输出一行以 `JSON ` 开头的机器可读汇总。
写盘落在临时目录、收尾删除；**不碰输入文件，也不碰用户输出目录**。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import media_io, modelstore, pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 ui/main_window.py 的预设常量保持一致（那边是唯一真源，这里只做镜像）
SHELF_DB_LISTEN = 3.0
SHELF_DB_DEMUD = 6.0
MUD_CUT_DB_DEMUD = -5.0
HIGH_PASS_HZ = 80.0


def peak_rss_mb() -> float | None:
    """本进程的峰值驻留内存（MB）。自进程启动起累计，用于判断是否触到内存墙。"""
    try:
        import ctypes
        from ctypes import wintypes

        class _Pmc(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _Pmc()
        pmc.cb = ctypes.sizeof(_Pmc)
        # 必须显式声明 argtypes：GetCurrentProcess() 返回 −1 伪句柄，按默认的
        # c_int 形参传回去会 OverflowError（实测踩过）。
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
        )
        if not ok:
            return None
        return pmc.PeakWorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        return None


def build_options(preset: str, engine_id: str, threads: int,
                  bwe: str | None, write: bool, out_dir: str,
                  parallel: str = "auto") -> pipeline.PipelineOptions:
    o = pipeline.PipelineOptions(
        engine_id=engine_id,
        engine=pipeline.EngineParams(num_threads=threads),
        trim_silence=True,
        highpass_hz=HIGH_PASS_HZ,
        normalize=True,
        normalize_target_dbfs=-3.0,
        normalize_mode="peak",
        keep_channels=True,
        write_output=write,
        reuse_existing=False,
        output_dir=out_dir,
        parallel_channels=parallel,
    )
    if preset == "denoise":
        o.trim_silence = False
        o.normalize = False
        o.highpass_hz = 0.0
        return o

    if preset in ("demud", "listen"):
        o.shelf_enabled = True
        o.shelf_hz = 3000.0
        o.shelf_db = SHELF_DB_DEMUD if preset == "demud" else SHELF_DB_LISTEN
    if preset == "demud":
        o.mud_cut_db = MUD_CUT_DB_DEMUD
    if preset == "listen":
        o.compressor_enabled = True
        o.bwe_id = bwe or "novasr"
    return o


def run_one(media: str, catalog, preset: str, engine_id: str, threads: int,
            bwe: str | None, write: bool, parallel: str = "auto") -> dict:
    out_dir = tempfile.mkdtemp(prefix="ae_meas_") if write else ""
    label = f"{preset}/线程{threads}" + (f"/{bwe}" if bwe else "") + f"/声道{parallel}"
    print(f"\n{'=' * 72}\n配置 {label}\n{'=' * 72}", flush=True)
    t0 = time.perf_counter()
    try:
        options = build_options(preset, engine_id, threads, bwe, write, out_dir, parallel)
        result = pipeline.run(media, catalog, options)
    finally:
        wall = time.perf_counter() - t0
        if out_dir:
            shutil.rmtree(out_dir, ignore_errors=True)

    dur = result.source_stats.duration
    for name, secs in result.stage_times.items():
        print(f"  {pipeline._STAGE_LABELS.get(name, name):<12}{secs:>9.2f}s"
              f"  {secs / wall * 100 if wall else 0:>5.1f}%", flush=True)

    rss = peak_rss_mb()
    print(f"  {'-' * 40}")
    print(f"  总墙钟      {wall:>9.2f}s   （含写出 {result.stage_times.get('write', 0.0):.2f}s）")
    print(f"  elapsed/RTF 基准 {result.elapsed:.2f}s / RTF {result.rtf:.3f}（不含写出，沿用旧口径）")
    print(f"  素材时长    {dur:>9.2f}s   输出 {result.enhanced.sample_rate} Hz")
    if rss:
        print(f"  进程峰值驻留 {rss:>8.0f} MB（自启动累计）")

    return {
        "label": label,
        "preset": preset,
        "engine": engine_id,
        "threads": threads,
        "bwe": next((s.split(":", 1)[1] for s in result.stages if s.startswith("bwe:")), None),
        "parallel": parallel,
        "duration_s": round(dur, 2),
        "wall_s": round(wall, 2),
        "elapsed_s": round(result.elapsed, 2),
        "rtf": round(result.rtf, 4),
        "stages_s": {k: round(v, 2) for k, v in result.stage_times.items()},
        "peak_rss_mb": round(rss, 0) if rss else None,
        "out_rate": result.enhanced.sample_rate,
    }


def make_excerpt(media: str, seconds: float, start_s: float = 0.0) -> tuple[str, str]:
    """把素材裁成短片段写到临时 wav，返回 (片段路径, 待清理的目录)。

    长素材上比较**不同带宽扩展引擎**时必需：lavasr 在 18 分钟素材上是分钟级到
    小时级，无法为了选型反复整段跑。代价是解码器从容器换成 wav（片段本身是
    无损 wav，所以只影响"解码"那一格的数字，不影响后面的推理占比）。
    """
    tmp = tempfile.mkdtemp(prefix="ae_excerpt_")
    info = media_io.probe(media)
    idx = info.default_audio_index if info.is_video else None
    data = media_io.decode(media, stream_index=idx)
    a = int(max(0.0, start_s) * data.sample_rate)
    b = int((max(0.0, start_s) + seconds) * data.sample_rate) if seconds else len(data.samples)
    chunk = data.samples[a:min(b, len(data.samples))]
    path = os.path.join(tmp, "excerpt.wav")
    media_io.write_audio(path, chunk, data.sample_rate)
    print(f"片段：{path}（{len(chunk) / data.sample_rate:.1f}s @ {data.sample_rate} Hz，"
          f"起点 {start_s:.1f}s）")
    return path, tmp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--preset", default="listen",
                    choices=["listen", "demud", "faithful", "denoise"])
    ap.add_argument("--engine", default="dpdfnet2")
    ap.add_argument("--threads", default="2",
                    help="逗号分隔的线程数列表，例如 2,4（同一进程内依次跑）")
    ap.add_argument("--bwe", default=None, help="覆盖带宽扩展引擎 id（默认 novasr）")
    ap.add_argument("--parallel", default="auto", choices=["auto", "on", "off"],
                    help="声道并行策略（默认 auto：按空闲内存决定；off 用于隔离测量）")
    ap.add_argument("--no-write", action="store_true", help="不写盘（省去写出与磁盘占用）")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="只取前 N 秒（0 = 整段）；用于长素材上的引擎选型")
    ap.add_argument("--start-seconds", type=float, default=0.0,
                    help="片段起点（配合 --max-seconds）")
    args = ap.parse_args()

    media = args.media if os.path.isabs(args.media) else os.path.abspath(args.media)
    if not os.path.isfile(media):
        print(f"找不到媒体文件：{media}")
        return 2

    threads_list = [int(x) for x in str(args.threads).split(",") if x.strip()]

    info = media_io.probe(media)
    print(f"输入：{media}")
    print(f"  容器 {info.container}，时长 {info.duration:.2f}s，"
          f"{'视频（只取音轨）' if info.is_video else '音频'}")

    excerpt_dir = ""
    if args.max_seconds:
        media, excerpt_dir = make_excerpt(media, args.max_seconds, args.start_seconds)

    catalog = modelstore.load_catalog(ROOT)
    summaries = []
    try:
        for th in threads_list:
            summaries.append(
                run_one(media, catalog, args.preset, args.engine, th, args.bwe,
                        write=not args.no_write, parallel=args.parallel)
            )
            gc.collect()
    finally:
        if excerpt_dir:
            shutil.rmtree(excerpt_dir, ignore_errors=True)

    print(f"\n{'=' * 72}\n汇总\n{'=' * 72}")
    for s in summaries:
        print(f"  {s['label']:<24} 墙钟 {s['wall_s']:>8.2f}s  RTF {s['rtf']:.3f}"
              f"  峰值驻留 {s['peak_rss_mb'] or 0:.0f} MB")
    print("\n机器可读：")
    for s in summaries:
        print("JSON " + json.dumps(s, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
