"""媒体格式与视频链路验证。

覆盖：
  1. 格式矩阵：wav / flac / mp3 / ogg-opus / m4a(aac) 的解码
  2. 视频导入：mp4 / mkv（视频+音轨）→ 判定为 video → 提取音轨 → 处理 → 输出音频
  3. 多音轨：mkv 两条音轨，验证可选择且选出的内容确实不同
  4. 无音轨视频：必须给出明确报错，而不是静默失败

测试素材用 PyAV 现场合成，不依赖外部下载（除官方测试音频外）。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_media_formats.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import av  # noqa: E402

from core import media_io, modelstore, pipeline  # noqa: E402

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


# ---------------------------------------------------------------------------
# 合成媒体
# ---------------------------------------------------------------------------

def _audio_frame(samples: np.ndarray, rate: int) -> av.AudioFrame:
    data = np.ascontiguousarray(samples, dtype=np.float32)
    frame = av.AudioFrame(format="fltp", layout="mono", samples=len(data))
    frame.sample_rate = rate
    frame.planes[0].update(data)
    return frame


def _video_frame(w: int, h: int, i: int) -> av.VideoFrame:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = (i * 24) % 255
    arr[:, :, 1] = 60
    frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
    frame.pts = i
    return frame


def make_media(
    path: str,
    *,
    container: str,
    audio_tracks: list[np.ndarray] | None = None,
    rate: int = 16000,
    with_video: bool = False,
    seconds: float = 2.0,
) -> str:
    """合成一个媒体文件：可选视频流 + 若干音轨。"""
    out = av.open(path, "w", format=container)

    vstream = None
    if with_video:
        vstream = out.add_stream("mpeg4", rate=10)
        vstream.width, vstream.height = 160, 120
        vstream.pix_fmt = "yuv420p"

    astreams = []
    for _ in (audio_tracks or []):
        st = out.add_stream("aac", rate=rate)
        st.layout = "mono"
        astreams.append(st)

    if vstream is not None:
        for i in range(int(10 * seconds)):
            for pkt in vstream.encode(_video_frame(160, 120, i)):
                out.mux(pkt)
        for pkt in vstream.encode(None):
            out.mux(pkt)

    for st, samples in zip(astreams, audio_tracks or []):
        step = rate  # 1 秒一块
        for i in range(0, len(samples), step):
            for pkt in st.encode(_audio_frame(samples[i:i + step], rate)):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)

    out.close()
    return path


def tone(freq: float, seconds: float, rate: int = 16000, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. 格式矩阵
# ---------------------------------------------------------------------------

def check_formats(tmp: str) -> None:
    section("1. 格式矩阵（解码）")
    src, sr_in = sf.read(SAMPLE, dtype="float32")

    cases = [(".wav", "PCM_16"), (".flac", "PCM_16"), (".mp3", None), (".ogg", None)]
    for ext, subtype in cases:
        path = os.path.join(tmp, f"probe{ext}")
        try:
            kwargs = {"subtype": subtype} if subtype else {}
            sf.write(path, src, sr_in, **kwargs)
            info = media_io.probe(path)
            data = media_io.decode(path)
            ok = data.frames > 0 and info.kind == "audio"
            check(f"{ext:<6} 探测+解码", ok,
                  f"{info.container} {data.frames} 帧 @ {data.sample_rate} Hz")
        except Exception as exc:  # noqa: BLE001
            check(f"{ext:<6} 探测+解码", False, f"{type(exc).__name__}: {exc}")

    # m4a（AAC）—— libsndfile 不支持，必须靠 PyAV
    m4a = os.path.join(tmp, "probe.m4a")
    try:
        make_media(m4a, container="mp4", audio_tracks=[src], rate=sr_in)
        info = media_io.probe(m4a)
        data = media_io.decode(m4a)
        check("m4a  探测+解码（走 PyAV）", data.frames > 0,
              f"decoder={info.decoder} codec={info.audio_streams[0].codec} "
              f"{data.frames} 帧 @ {data.sample_rate} Hz")
    except Exception as exc:  # noqa: BLE001
        check("m4a  探测+解码（走 PyAV）", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 2/3/4. 视频链路
# ---------------------------------------------------------------------------

def check_video_pipeline(tmp: str, catalog: modelstore.Catalog) -> None:
    section("2. 视频导入 → 提取音轨 → 处理 → 输出音频")
    src, sr_in = sf.read(SAMPLE, dtype="float32")

    for container, ext in (("mp4", ".mp4"), ("matroska", ".mkv")):
        path = os.path.join(tmp, f"video{ext}")
        try:
            make_media(path, container=container, audio_tracks=[src], rate=sr_in,
                       with_video=True, seconds=2.0)
        except Exception as exc:  # noqa: BLE001
            check(f"{ext} 合成", False, f"{type(exc).__name__}: {exc}")
            continue

        try:
            info = media_io.probe(path)
            check(f"{ext} 判定为视频", info.kind == "video",
                  f"kind={info.kind} 视频流={len(info.video_streams)} 音轨={len(info.audio_streams)}")
            if info.video_streams:
                v = info.video_streams[0]
                check(f"{ext} 读到画面信息（但不处理）", v.width > 0 and v.height > 0,
                      f"{v.codec} {v.width}×{v.height}")

            data = media_io.decode(path)
            check(f"{ext} 提取音轨成功", data.frames > 0,
                  f"{data.frames} 帧 @ {data.sample_rate} Hz，{data.channels} 声道")

            opts = pipeline.PipelineOptions(
                engine_id="dpdfnet2",
                engine=pipeline.EngineParams(num_threads=2),
                output_ext=".wav",
                output_path=os.path.join(tmp, f"video_out{ext}.wav"),
            )
            r = pipeline.run(path, catalog, opts)
            out_is_audio = bool(r.output_path) and os.path.isfile(r.output_path)
            back = media_io.probe(r.output_path) if out_is_audio else None
            check(f"{ext} 处理并输出音频", out_is_audio and back.kind == "audio",
                  f"{back.container} {r.enhanced_stats.duration:.2f}s" if back else "无输出")
        except Exception as exc:  # noqa: BLE001
            check(f"{ext} 视频链路", False, f"{type(exc).__name__}: {exc}")

    section("3. 多音轨视频（mkv，两条音轨）")
    multi = os.path.join(tmp, "multi.mkv")
    try:
        make_media(multi, container="matroska",
                   audio_tracks=[tone(220, 2.0), tone(880, 2.0)],
                   with_video=True)
        info = media_io.probe(multi)
        check("探测到 2 条音轨", len(info.audio_streams) == 2,
              f"{[ (s.index, s.codec) for s in info.audio_streams ]}")

        idx0, idx1 = info.audio_streams[0].index, info.audio_streams[1].index
        a = media_io.to_mono(media_io.decode(multi, stream_index=idx0).samples)
        b = media_io.to_mono(media_io.decode(multi, stream_index=idx1).samples)
        n = min(len(a), len(b))
        # 两条音轨应显著不同；用相关系数判断
        if n > 0:
            corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
        else:
            corr = 1.0
        check("两条音轨内容确实不同", abs(corr) < 0.5, f"相关系数 {corr:.3f}")

        # 用第二条音轨跑完整流水线
        opts = pipeline.PipelineOptions(
            engine_id="gtcrn", engine=pipeline.EngineParams(num_threads=2)
        )
        r = pipeline.run(multi, catalog, opts, stream_index=idx1)
        check("指定音轨可走完整流水线", r.audio_stream_index == idx1 and r.enhanced.frames > 0,
              f"使用音轨 {r.audio_stream_index}")
    except Exception as exc:  # noqa: BLE001
        check("多音轨处理", False, f"{type(exc).__name__}: {exc}")

    section("4. 无音轨视频（必须明确报错）")
    silent = os.path.join(tmp, "noaudio.mp4")
    try:
        make_media(silent, container="mp4", audio_tracks=[], with_video=True)
        try:
            media_io.probe(silent)
            check("无音轨视频应报错", False, "却没有抛异常")
        except media_io.MediaError as exc:
            check("无音轨视频给出明确报错", "不包含音轨" in str(exc),
                  str(exc).splitlines()[0][:70])
        except Exception as exc:  # noqa: BLE001
            check("无音轨视频给出明确报错", False, f"抛了非 MediaError：{type(exc).__name__}")
    except Exception as exc:  # noqa: BLE001
        check("合成无音轨视频", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    if not os.path.isfile(SAMPLE):
        print(f"[FAIL] 找不到测试音频：{SAMPLE}")
        return 2

    print(f"PyAV {av.__version__}")
    catalog = modelstore.load_catalog(ROOT)

    with tempfile.TemporaryDirectory(prefix="ae_media_") as tmp:
        check_formats(tmp)
        check_video_pipeline(tmp, catalog)

    print()
    if failures:
        print(f"媒体格式验证结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("媒体格式验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
