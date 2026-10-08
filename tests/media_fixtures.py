"""测试共用的合成媒体素材（不依赖任何既有套件，避免互相牵动）。

为什么单独成文件：`check_video_remux.py` 里已有一份同样的 `make_video()`，但它那 48 项
断言是绿着的，**刻意不去改动它**（本项目对"已被验证的套件"的处置是先不动）。而新的
界面级测试（`smoke_ui.py` 的「输出视频」一节）也需要一个真实可探测的视频输入，所以在
这里提供一份最小实现，两边各自持有自己的素材构造代码。

用法（从 tests/ 目录下的脚本里）：
    from media_fixtures import make_video, tone, silence, RATE   # 需要 tests/ 在 sys.path 上
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import av  # noqa: E402

RATE = 16000


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
    """合成一个「视频流 + 若干音轨」的 mp4。音轨带语言标签。"""
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
