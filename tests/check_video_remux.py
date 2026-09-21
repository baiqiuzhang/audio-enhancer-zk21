"""视频音轨回写（remux）的验证。

判据要硬，因为这个功能是**唯一可能不可逆**的操作：它读一个视频、写一个新视频。
所以本套件的核心断言是「源文件字节与 mtime 完全未变」，外加视频流必须原样复制、
音轨确实被替换、而其它流（含语言标签）不被静默丢掉。

全部素材与产物都在临时目录里，不碰 samples/ 与项目 out/。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_video_remux.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import av  # noqa: E402

from core import media_io, modelstore, pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATE = 16000

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
# 合成素材与只读探测
# ---------------------------------------------------------------------------

def tone(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(RATE * seconds), dtype=np.float32) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(RATE * seconds), dtype=np.float32)


def _video_frame(w: int, h: int, i: int) -> av.VideoFrame:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = (i * 24) % 255
    arr[:, :, 1] = 60
    f = av.VideoFrame.from_ndarray(arr, format="rgb24")
    f.pts = i
    return f


def _audio_frame(samples: np.ndarray) -> av.AudioFrame:
    f = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(samples[None, :], dtype=np.float32),
        format="fltp", layout="mono",
    )
    f.sample_rate = RATE
    return f


def make_video(path: str, tracks, *, lang: str | None = "eng",
               default_first: bool = True, seconds: float = 2.0) -> str:
    """合成一个带视频流 + 若干音轨的 mp4；音轨带语言标签，便于验证标签是否被保留。"""
    out = av.open(path, "w", format="mp4")
    vs = out.add_stream("mpeg4", rate=10)
    vs.width, vs.height = 160, 120
    vs.pix_fmt = "yuv420p"
    streams = []
    for i, samples in enumerate(tracks):
        st = out.add_stream("aac", rate=RATE)
        st.layout = "mono"
        if lang:
            st.metadata.update({"language": lang})
        if i == 0 and default_first:
            st.disposition = av.stream.Disposition.default
        streams.append((st, samples))

    for i in range(int(10 * seconds)):
        for pkt in vs.encode(_video_frame(160, 120, i)):
            out.mux(pkt)
    for pkt in vs.encode(None):
        out.mux(pkt)
    for st, samples in streams:
        for i in range(0, len(samples), RATE):
            for pkt in st.encode(_audio_frame(samples[i:i + RATE])):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    out.close()
    return path


def has_default(st) -> bool:
    """disposition 是 IntFlag，str() 返回的是数字而不是名字，所以必须按位判断
    （`'default' in str(st.disposition)` 恒为 False —— 这个坑让本套件误报过一次）。"""
    return bool(int(st.disposition) & int(av.stream.Disposition.default))


def stream_facts(path: str) -> list[dict]:
    with av.open(path) as c:
        out = []
        for st in c.streams:
            out.append({
                "index": st.index, "type": st.type,
                "codec": st.codec_context.name,
                "lang": st.metadata.get("language"),
                "default": has_default(st),
                "frames": st.frames,
                "wh": (st.codec_context.width, st.codec_context.height)
                if st.type == "video" else None,
                "dur": float(st.duration * st.time_base) if st.duration else None,
            })
        return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def snapshot(path: str) -> tuple[int, float, str]:
    return os.path.getsize(path), os.path.getmtime(path), sha256(path)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


# ---------------------------------------------------------------------------

def check_capability_table() -> None:
    section("1. 容器能力表自洽")
    for ext in (".mp4", ".mkv", ".webm", ".avi", ".flv", ".ts", ".wmv"):
        check(f"{ext} 支持回写且编解码器齐全",
              media_io.remux_supported_ext(ext)
              and ext in media_io.CONTAINER_AUDIO_CODECS
              and ext in media_io.MUXER_FOR_EXT,
              f"codec={media_io.CONTAINER_AUDIO_CODECS.get(ext)}")
    for ext in media_io.REFUSE_REMUX_EXTS:
        check(f"{ext} 被明确拒绝且给出原因",
              not media_io.remux_supported_ext(ext)
              and bool(media_io.remux_refusal_reason(ext)),
              (media_io.remux_refusal_reason(ext) or "")[:40])
    check("音频扩展名不会被当成可回写容器",
          not media_io.remux_supported_ext(".wav")
          and not media_io.remux_supported_ext(".mp3"))


def check_remux(tmp: str) -> None:
    section("2. 回写：视频原样复制、音轨被替换、其它流不丢")
    src = make_video(
        os.path.join(tmp, "two_tracks.mp4"),
        [tone(300, 2.0), tone(500, 2.0)],
    )
    before = snapshot(src)
    src_f = stream_facts(src)
    check("合成素材有 1 条视频 + 2 条音轨",
          len([s for s in src_f if s["type"] == "video"]) == 1
          and len([s for s in src_f if s["type"] == "audio"]) == 2,
          f"{[s['type'] for s in src_f]}")
    check("合成素材的音轨带语言标签",
          all(s["lang"] == "eng" for s in src_f if s["type"] == "audio"),
          str([s["lang"] for s in src_f if s["type"] == "audio"]))

    # 替换第一条音轨，新音频是 0.5 倍电平的另一个音
    new_audio = tone(800, 2.0, amp=0.125)
    dst = os.path.join(tmp, "remuxed.mp4")
    out = media_io.write_video_with_audio(src, dst, new_audio, RATE,
                                          replace_stream_index=None)
    check("产出文件存在", os.path.isfile(out), f"{os.path.getsize(out):,} B")

    after = snapshot(src)
    check("源文件大小未变", after[0] == before[0])
    check("源文件 mtime 未变", after[1] == before[1])
    check("源文件 sha256 未变（内容逐字节一致）", after[2] == before[2])

    dst_f = stream_facts(dst)
    sv = next(s for s in src_f if s["type"] == "video")
    dv = next((s for s in dst_f if s["type"] == "video"), None)
    check("新文件有视频流", dv is not None)
    if dv:
        check("视频 codec 一致", sv["codec"] == dv["codec"], f"{sv['codec']}→{dv['codec']}")
        check("分辨率一致", sv["wh"] == dv["wh"], f"{sv['wh']}→{dv['wh']}")
        check("帧数一致（真 stream copy）", sv["frames"] == dv["frames"],
              f"{sv['frames']}→{dv['frames']}")
        if sv["dur"] and dv["dur"]:
            check("视频时长一致（±0.1s）", abs(sv["dur"] - dv["dur"]) < 0.1,
                  f"{sv['dur']:.2f}→{dv['dur']:.2f}")
        check("视频流语言标签保留", sv["lang"] == dv["lang"],
              f"{sv['lang']}→{dv['lang']}")

    auds = [s for s in dst_f if s["type"] == "audio"]
    check("音轨数量未变（1 条被换掉、1 条保留）", len(auds) == 2, f"{len(auds)} 条")
    # **顺序即契约**：新音轨必须位于被替换音轨原来的位置，也就是第一条音轨。
    # 否则 default_audio_index（及播放器的默认选择）会指向没被处理的那条旧音轨。
    new_aud, kept_aud = auds[0], auds[1]
    check("新音轨排在第一条（否则播放器会播旧音轨）",
          media_io.probe(dst).default_audio_index == new_aud["index"],
          f"default_audio_index={media_io.probe(dst).default_audio_index} 新音轨 index={new_aud['index']}")
    check("保留的那条音轨语言标签未丢", kept_aud["lang"] == "eng", str(kept_aud["lang"]))
    check("新音轨继承了被替换音轨的语言标签", new_aud["lang"] == "eng", str(new_aud["lang"]))
    check("新音轨继承了 default 标记", new_aud["default"] is True, str(new_aud["default"]))

    back = media_io.decode(dst, media_io.probe(dst).default_audio_index)
    check("默认音轨解出来的就是新换上去的那条", len(back.samples) > 0,
          f"{len(back.samples)} 帧")
    check("换上去的确实是给的那一版（电平约为原始的一半）",
          0.3 < rms(back.samples) / max(rms(tone(800, 2.0)), 1e-9) < 0.7,
          f"RMS 比 {rms(back.samples) / max(rms(tone(800, 2.0)), 1e-9):.3f}")

    # --- 「回写之后再处理，读的是哪条音轨？」 -------------------------------
    # 用户会关心这一点：回写产生了一个音轨已被替换的视频，那么**下一次处理同一个源文件**
    # 时，读到的必须是**最初的原始音轨**，而不是上一次写回去的那条——否则每跑一次都会在
    # 上一次的结果上继续叠加，越跑越糟。
    section("2b. 回写之后，再次处理读的是原始音轨（不是写回去的那条）")
    pristine = media_io.decode(src, media_io.probe(src).default_audio_index)
    pre_rms = rms(pristine.samples)
    check("原始音轨电平就是 300 Hz 正弦那一档（0.25 幅度）",
          0.15 < pre_rms < 0.22, f"RMS {pre_rms:.4f}")

    run_src = pipeline_run_source(src, tmp)
    if run_src is None:
        check("二次处理能拿到「原音」", False, "流水线未返回结果")
    else:
        same_len = len(run_src.samples) == len(pristine.samples)
        identical = same_len and bool(
            np.array_equal(np.asarray(run_src.samples, dtype=np.float32),
                           np.asarray(pristine.samples, dtype=np.float32))
        )
        check("二次处理读到的「原音」与最初的原始音轨**逐样本一致**",
              identical,
              f"{len(run_src.samples)} vs {len(pristine.samples)} 帧"
              + ("" if same_len else "（长度都不同）"))
        check("二次处理读到的不是回写进去的那条（电平不是 0.5 倍那一档）",
              rms(run_src.samples) > 1.5 * rms(media_io.decode(dst, media_io.probe(dst).default_audio_index).samples),
              f"原音 RMS {rms(run_src.samples):.4f} vs 回写音轨 "
              f"{rms(media_io.decode(dst, media_io.probe(dst).default_audio_index).samples):.4f}")

    # 另一条容易混淆的路径：把**回写产物本身**当新输入导入 → 读到的当然是它自己的音轨
    # （就是增强后的那条）。这是预期行为，明确记下来以免被误认为缺陷。
    run_back = pipeline_run_source(dst, tmp)
    if run_back is not None:
        check("把回写产物当新输入时，读到的就是它自己的音轨（增强后的那条）",
              0.3 < rms(run_back.samples) / max(rms(tone(800, 2.0)), 1e-9) < 0.7,
              f"RMS {rms(run_back.samples):.4f}")


def pipeline_run_source(path: str, tmp: str):
    """跑一遍真流水线（不写盘），返回它解出来的「原音」。"""
    cat = modelstore.load_catalog(ROOT)
    opts = pipeline.PipelineOptions(
        engine_id="dsp_baseline",
        engine=pipeline.EngineParams(num_threads=1),
        trim_silence=False, highpass_hz=0.0, normalize=False,
        keep_channels=False, write_output=False, reuse_existing=False,
    )
    try:
        return pipeline.run(path, cat, opts).source
    except Exception as exc:                                    # noqa: BLE001
        print(f"    [注意] 流水线运行失败：{type(exc).__name__}: {exc}")
        return None


def check_refusals(tmp: str) -> None:
    section("3. 拒绝条件（宁可明确报错，不静默降级）")
    src = make_video(os.path.join(tmp, "refuse_src.mp4"), [tone(300, 1.0)], seconds=1.0)
    audio = tone(400, 1.0)
    cases = [
        ("dst 与 src 同一文件",
         lambda: media_io.write_video_with_audio(src, src, audio, RATE)),
        (".vob 容器",
         lambda: media_io.write_video_with_audio(src, os.path.join(tmp, "x.vob"), audio, RATE)),
        (".rmvb 容器",
         lambda: media_io.write_video_with_audio(src, os.path.join(tmp, "x.rmvb"), audio, RATE)),
        (".m4v 容器",
         lambda: media_io.write_video_with_audio(src, os.path.join(tmp, "x.m4v"), audio, RATE)),
        ("未知容器",
         lambda: media_io.write_video_with_audio(src, os.path.join(tmp, "x.zzz"), audio, RATE)),
        ("音频容器",
         lambda: media_io.write_video_with_audio(src, os.path.join(tmp, "x.wav"), audio, RATE)),
    ]
    for label, fn in cases:
        try:
            fn()
            check(f"拒绝：{label}", False, "本应抛错却成功了")
        except media_io.MediaError as exc:
            check(f"拒绝：{label}", True, str(exc)[:52])
        except Exception as exc:  # noqa: BLE001
            check(f"拒绝：{label}", False, f"抛了非 MediaError：{type(exc).__name__}")

    check("失败后没有留下半截临时文件",
          not [n for n in os.listdir(tmp) if ".part" in n],
          str([n for n in os.listdir(tmp) if ".part" in n]))


def check_pipeline_end_to_end(tmp: str, cat) -> None:
    section("4. 流水线端到端：video_out=True 产出新视频")
    src = make_video(os.path.join(tmp, "e2e.mp4"), [tone(300, 2.0)])
    before = snapshot(src)
    opts = pipeline.PipelineOptions(
        engine_id="dsp_baseline", output_ext=".mp4", video_out=True,
        write_output=True, highpass_hz=0.0,
    )
    res = pipeline.run(src, cat, opts)
    check("流水线运行成功", res is not None)
    check("产出的扩展名是源容器 .mp4",
          (res.output_path or "").lower().endswith(".mp4"), str(res.output_path))
    check("产出文件存在", bool(res.output_path) and os.path.isfile(res.output_path),
          str(res.output_path))
    check("源文件完全未被改动（字节+mtime）", snapshot(src) == before)
    if res.output_path and os.path.isfile(res.output_path):
        of = stream_facts(res.output_path)
        check("产出里有视频流与音轨",
              any(s["type"] == "video" for s in of)
              and any(s["type"] == "audio" for s in of),
              f"{[s['type'] for s in of]}")
        check("变体标签标出「写回视频」", "写回视频" in (res.variant_label or ""),
              str(res.variant_label))

    section("5. 视频模式与音频模式的指纹必须不同")
    a = pipeline.PipelineOptions(engine_id="dsp_baseline", output_ext=".mp4",
                                 video_out=False, write_output=True, highpass_hz=0.0)
    b = pipeline.PipelineOptions(engine_id="dsp_baseline", output_ext=".mp4",
                                 video_out=True, write_output=True, highpass_hz=0.0)
    from core import outputs  # noqa: PLC0415

    fa = outputs.fingerprint(src, a, None, "dsp")
    fb = outputs.fingerprint(src, b, None, "dsp")
    check("音频模式与视频模式指纹不同", fa != fb, f"{fa} vs {fb}")

    section("6. 流水线层面的拒绝条件")
    # 非视频输入
    wav = os.path.join(tmp, "not_video.wav")
    media_io.write_audio(wav, tone(300, 1.0), RATE)
    opts = pipeline.PipelineOptions(
        engine_id="dsp_baseline", output_ext=".wav", video_out=True,
        write_output=True, highpass_hz=0.0,
    )
    try:
        pipeline.run(wav, cat, opts)
        check("拒绝：非视频输入 + video_out", False, "本应抛错却成功了")
    except media_io.MediaError as exc:
        check("拒绝：非视频输入 + video_out", True, str(exc)[:52])

    # 裁剪真的生效
    padded = np.concatenate([silence(0.6), tone(300, 1.0), silence(0.6)])
    pad_src = make_video(os.path.join(tmp, "padded.mp4"), [padded], seconds=2.2)
    opts = pipeline.PipelineOptions(
        engine_id="dsp_baseline", output_ext=".mp4", video_out=True,
        write_output=True, trim_silence=True, highpass_hz=0.0,
    )
    try:
        pipeline.run(pad_src, cat, opts)
        check("拒绝：裁剪生效时写回视频", False, "本应抛错却成功了")
    except media_io.MediaError as exc:
        check("拒绝：裁剪生效时写回视频", True, str(exc)[:52])


def main() -> int:
    print(f"项目根：{ROOT}")
    cat = modelstore.load_catalog(ROOT)
    tmp = tempfile.mkdtemp(prefix="ae_remux_")
    try:
        print(f"临时目录：{tmp}")
        check_capability_table()
        check_remux(tmp)
        check_refusals(tmp)
        check_pipeline_end_to_end(tmp, cat)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"视频回写验证结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("视频回写验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
