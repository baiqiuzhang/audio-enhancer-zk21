"""批量下载模型权重（镜像优先，失败自动降级到下一个来源）。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\download_models.py              # 默认引擎 + GTCRN + 示例
    .venv\\Scripts\\python.exe tests\\download_models.py --all        # 全部（含 25.7 MB 的 DFN3）
    .venv\\Scripts\\python.exe tests\\download_models.py --engine dpdfnet8
    .venv\\Scripts\\python.exe tests\\download_models.py --check      # 只探测来源可达性，不下载
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import modelstore  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SET = ["gtcrn", "dpdfnet2"]


def human_bytes(n: int) -> str:
    return modelstore.human_size(n)


def collect(cat: modelstore.Catalog, engine_ids: list[str]) -> list[tuple[str, list]]:
    """返回 [(描述, [FileSpec, ...])]，已过滤掉本地就绪的。"""
    jobs: list[tuple[str, list]] = []
    for eid in engine_ids:
        spec = cat.engine(eid)
        if spec is None:
            print(f"  [跳过] 未知引擎 id：{eid}")
            continue
        if spec.kind == "dsp":
            print(f"  [跳过] {spec.name}：纯 numpy 实现，无需下载")
            continue
        missing = modelstore.missing_files(spec.files, cat)
        if not missing:
            print(f"  [已就绪] {spec.name}")
            continue
        need = sum((f.sources[0].size or 0) for f in missing)
        print(f"  [待下载] {spec.name}  {human_bytes(need)}")
        jobs.append((spec.name, missing))

    extra = cat.extra("samples")
    if extra:
        missing = modelstore.missing_files(extra.files, cat)
        if missing:
            print(f"  [待下载] {extra.name}")
            jobs.append((extra.name, missing))
        else:
            print(f"  [已就绪] {extra.name}")
    return jobs


def probe_only(cat: modelstore.Catalog, engine_ids: list[str]) -> int:
    print("探测各来源可达性（HEAD，超时 15s）…")
    print()
    bad = 0
    for eid in engine_ids:
        spec = cat.engine(eid)
        if spec is None or spec.kind == "dsp":
            continue
        if modelstore.engine_ready(spec, cat):
            print(f"{spec.name}：本地已就绪，跳过")
            continue
        print(spec.name)
        for fs in spec.files:
            for host, status in modelstore.probe_sources(fs):
                mark = "OK  " if status.startswith("HTTP 2") else "!!  "
                print(f"  {mark}{host:<26}{status}")
                if not status.startswith("HTTP 2"):
                    bad += 1
    print()
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="下载全部引擎（含 DFN3，约 80 MB）")
    ap.add_argument("--engine", action="append", default=None, help="指定引擎 id（可重复）")
    ap.add_argument("--check", action="store_true", help="只探测来源，不下载")
    args = ap.parse_args()

    cat = modelstore.load_catalog(ROOT)

    if args.engine:
        ids = args.engine
    elif args.all:
        ids = [e.id for e in cat.engines]
    else:
        ids = DEFAULT_SET

    print(f"模型目录：{cat.models_root}")
    print(f"目标：{', '.join(ids)}" )
    print()

    if args.check:
        return probe_only(cat, ids)

    jobs = collect(cat, ids)
    if not jobs:
        print("\n无需下载，全部已就绪。")
        return 0

    print()
    failed: list[str] = []
    for name, files in jobs:
        print(f"—— {name} ——")
        try:
            for fs in files:
                last = {"n": 0}
                t0 = time.perf_counter()

                def progress(done, total, _last=last, _t0=[t0], _name=os.path.basename(fs.path)):
                    pct = int(done * 100 / total) if total else 0
                    if pct >= _last["n"] + 10 or done == total:
                        _last["n"] = pct
                        print(f"    {_name}: {pct}%  ({human_bytes(done)}"
                              f"{' / ' + human_bytes(total) if total else ''})")

                modelstore.download_file(
                    fs, cat,
                    progress=progress,
                    log=lambda m: print(f"    {m}"),
                )
        except modelstore.DownloadError as exc:
            print(f"    [失败] {exc}")
            failed.append(name)
        print()

    if failed:
        print(f"以下项目下载失败：{', '.join(failed)}")
        print("提示：本机对 github.com 时通时断；下载器已优先走 hf-mirror 镜像。")
        print("可稍后重试，或在应用界面里点「下载所选模型」。")
        return 1
    print("全部下载完成并通过校验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
