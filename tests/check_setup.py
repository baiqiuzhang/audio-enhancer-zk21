"""安装自检：模型清单、文件校验、来源可达性、媒体 I/O 往返。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_setup.py            # 只查本地
    .venv\\Scripts\\python.exe tests\\check_setup.py --net      # 额外探测各来源 URL
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from core import media_io, modelstore  # noqa: E402


def section(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def check_catalog(probe_net: bool) -> int:
    section("1. 模型清单")
    cat = modelstore.load_catalog(ROOT)
    print(f"清单：models.json   引擎 {len(cat.engines)} 个   附加项 {len(cat.extras)} 个")
    print()

    header = f"{'引擎':<28}{'类型':<10}{'状态':<8}{'体积':>11}  文件"
    print(header)
    print("-" * (len(header) + 20))
    problems = 0
    for e in cat.engines:
        ready = modelstore.engine_ready(e, cat)
        if e.kind == "dsp":
            print(f"{e.name:<28}{e.kind:<10}{'就绪':<8}{'—':>11}  (无需下载)")
            continue
        sizes = []
        for f in e.files:
            p = cat.resolve(f.path)
            sizes.append(os.path.getsize(p) if os.path.isfile(p) else 0)
        have = sum(sizes)
        want = e.size_hint
        state = "就绪" if ready else "缺失"
        if not ready:
            problems += 1
        print(f"{e.name:<28}{e.kind:<10}{state:<8}{modelstore.human_size(have):>11}"
              f"  ({have * 100 // want if want else 0}% / {modelstore.human_size(want)})")
        for f in e.files:
            mark = "OK " if modelstore.file_ready(f, cat) else "!! "
            p = cat.resolve(f.path)
            print(f"{'':<28}  {mark}{os.path.basename(f.path):<34}{modelstore.human_size(os.path.getsize(p)) if os.path.isfile(p) else '缺失':>11}")

    for x in cat.extras:
        ready = modelstore.unit_ready(x.files, cat)
        print(f"{x.name:<28}{'extra':<10}{'就绪' if ready else '缺失':<8}{x.size_hint_text:>11}  用于：{x.required_by}")
        if not ready:
            problems += 1

    if probe_net:
        section("2. 来源可达性（HEAD）")
        for e in cat.engines + [modelstore.ExtraSpec(id=x.id, name=x.name, files=x.files) for x in cat.extras]:
            if e.kind == "dsp":
                continue
            for f in e.files:
                if modelstore.file_ready(f, cat):
                    continue  # 已在本地，无需联网
                for host, status in modelstore.probe_sources(f):
                    print(f"  {os.path.basename(f.path):<28}{host:<24}{status}")
    else:
        print("\n（未探测网络；加 --net 可探测缺失文件的来源 URL）")

    return problems


def check_media_io() -> int:
    section("3. 媒体 I/O 往返")
    src = os.path.join(ROOT, "samples", "speech_with_noise.wav")
    if not os.path.isfile(src):
        print(f"[跳过] 找不到 {src}")
        return 0

    failures = 0

    info = media_io.probe(src)
    print(f"probe：kind={info.kind}  容器={info.container}  时长={info.duration:.2f}s  "
          f"音轨={len(info.audio_streams)}  解码器={info.decoder}")
    print(f"       {info.audio_streams[0].label()}")
    if info.kind != "audio":
        print("  [FAIL] 应判定为 audio")
        failures += 1

    data = media_io.decode(src)
    print(f"decode：{data.frames} 帧 @ {data.sample_rate} Hz  {data.channels} 声道  "
          f"shape={data.samples.shape}  dtype={data.samples.dtype}")
    if data.sample_rate != 16000 or data.frames == 0:
        print("  [FAIL] 解码结果异常")
        failures += 1

    mono = media_io.to_mono(data.samples)
    up = media_io.resample(mono, 16000, 48000)
    back = media_io.resample(up, 48000, 16000)
    print(f"重采样：16k({len(mono)}) → 48k({len(up)}) → 16k({len(back)})  "
          f"长度差 {abs(len(back) - len(mono))} 帧")

    with tempfile.TemporaryDirectory(prefix="ae_io_") as tmp:
        for ext in (".wav", ".flac", ".mp3"):
            out = os.path.join(tmp, "roundtrip" + ext)
            try:
                media_io.write_audio(out, mono, 16000)
                rt = media_io.decode(out)
                ok = rt.frames > 0 and rt.sample_rate == 16000
                print(f"写出 {ext:<6} {os.path.getsize(out):>9,} B  → 回读 {rt.frames} 帧  "
                      f"{'OK' if ok else 'FAIL'}")
                if not ok:
                    failures += 1
            except Exception as exc:
                print(f"写出 {ext:<6} [FAIL] {exc}")
                failures += 1

    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", action="store_true", help="额外探测各来源 URL 可达性")
    args = ap.parse_args()

    print(f"项目根：{ROOT}")
    print(f"媒体模块：soundfile / PyAV 已就绪")
    try:
        import av  # noqa: PLC0415
        print(f"PyAV {av.__version__}")
    except Exception as exc:
        print(f"[警告] PyAV 不可用：{exc}（视频导入将不可用）")

    problems = check_catalog(args.net)
    problems += check_media_io()

    print()
    if problems:
        print(f"自检结束：{problems} 项待处理（见上方 !! 标记）")
        return 1
    print("自检结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
