"""媒体探测 / 解码 / 编码 / 重采样。

音频文件与视频容器在这里被统一成同一件事：一段 float32 的 PCM。

- `probe()`  只读探测，不解码整段：判定 kind、枚举音轨、读取参数。
- `decode()` 解码为 numpy float32 数组。

视频只取音轨，画面一律不碰；输出始终是音频。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# 支持的扩展名（仅用于"先试哪条路"的启发式判断，kind 最终由流类型决定）
# ---------------------------------------------------------------------------

AUDIO_EXTS = {
    ".wav", ".wave", ".flac", ".mp3", ".ogg", ".oga", ".opus",
    ".m4a", ".aac", ".wma", ".aiff", ".aif", ".caf", ".au", ".w64",
}

VIDEO_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".m4v",
    ".wmv", ".mpg", ".mpeg", ".m2ts", ".vob", ".3gp", ".rmvb", ".asf",
}

ALL_EXTS = AUDIO_EXTS | VIDEO_EXTS

# soundfile 能直接读的容器（其余一律走 PyAV）
_SF_EXTS = {
    ".wav", ".wave", ".flac", ".mp3", ".ogg", ".oga", ".opus",
    ".aiff", ".aif", ".caf", ".au", ".w64",
}

# 输出格式 -> soundfile subtype
OUTPUT_SUBTYPES = {
    ".wav": "PCM_24",
    ".flac": "PCM_24",
    ".mp3": None,  # mp3 由 libsndfile 自行决定
}

OUTPUT_FORMATS = [
    ("WAV 24-bit（推荐）", ".wav"),
    ("FLAC 24-bit（无损压缩）", ".flac"),
    ("MP3（有损，体积小）", ".mp3"),
]


class MediaError(RuntimeError):
    """无法探测或解码该媒体文件。"""


@dataclass
class AudioStreamInfo:
    index: int          # 容器内的流序号（PyAV 路径下用于选择音轨）
    codec: str
    sample_rate: int
    channels: int
    bit_rate: int | None = None
    language: str | None = None
    duration: float | None = None

    def label(self) -> str:
        bits = f"{self.bit_rate // 1000} kbps" if self.bit_rate else "—"
        lang = f" · {self.language}" if self.language else ""
        return (
            f"音轨 {self.index}：{self.codec} · {self.sample_rate} Hz · "
            f"{self.channels} 声道 · {bits}{lang}"
        )


@dataclass
class VideoStreamInfo:
    index: int
    codec: str
    width: int
    height: int
    fps: float | None = None


@dataclass
class MediaInfo:
    path: str
    kind: str                       # "audio" | "video"
    container: str
    duration: float
    audio_streams: list[AudioStreamInfo] = field(default_factory=list)
    video_streams: list[VideoStreamInfo] = field(default_factory=list)
    decoder: str = ""               # "soundfile" | "pyav"
    notes: list[str] = field(default_factory=list)

    @property
    def is_video(self) -> bool:
        return self.kind == "video"

    @property
    def default_audio_index(self) -> int:
        """默认音轨：取容器里第一条音频流。"""
        if not self.audio_streams:
            raise MediaError("该文件不包含任何音轨。")
        return self.audio_streams[0].index

    def summary(self) -> str:
        kind = "视频（已提取音轨）" if self.is_video else "音频"
        return f"{kind} · {self.container} · {_fmt_time(self.duration)}"


@dataclass
class AudioData:
    """解码结果。samples 为 float32：单声道 (frames,)，多声道 (frames, channels)。"""

    samples: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else self.samples.shape[1]

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        return self.frames / float(self.sample_rate)


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:04.1f}"
    return f"{m}:{s:04.1f}"


def _av():
    """延迟导入 PyAV：即使它不可用，纯音频功能仍可工作。"""
    try:
        import av  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        raise MediaError(
            f"视频/容器解码不可用（PyAV 导入失败：{exc}）。"
            "请确认 .venv 已按 requirements.txt 安装。"
        ) from exc
    return av


def _norm_channels(n: int) -> str:
    """统一声道布局：单声道保持单声道，其余一律降到立体声。

    多声道素材（5.1 等）在增强流程里没有意义——模型只吃单声道，
    A/B 播放立体声也足够，因此在这里一次性收敛。
    """
    return "mono" if n <= 1 else "stereo"


# ---------------------------------------------------------------------------
# 探测
# ---------------------------------------------------------------------------

def probe(path: str) -> MediaInfo:
    """只读探测媒体文件，不解码整段音频。"""
    if not os.path.isfile(path):
        raise MediaError(f"文件不存在：{path}")

    ext = os.path.splitext(path)[1].lower()

    # 快路径：纯音频扩展名且 soundfile 能打开，就不必惊动 PyAV
    if ext in _SF_EXTS:
        try:
            info = sf.info(path)
            if info.frames > 0:
                return MediaInfo(
                    path=path,
                    kind="audio",
                    container=info.format or ext.lstrip("."),
                    duration=float(info.duration),
                    audio_streams=[
                        AudioStreamInfo(
                            index=0,
                            codec=(info.subtype or info.format or ext.lstrip(".")),
                            sample_rate=int(info.samplerate),
                            channels=int(info.channels),
                            duration=float(info.duration),
                        )
                    ],
                    decoder="soundfile",
                )
        except Exception:
            pass  # 落到 PyAV 再试一次

    return _probe_pyav(path, ext)


def _probe_pyav(path: str, ext: str) -> MediaInfo:
    av = _av()
    try:
        with av.open(path) as container:
            container_name = (container.format.name if container.format else "") or ext.lstrip(".")

            audio_streams: list[AudioStreamInfo] = []
            video_streams: list[VideoStreamInfo] = []

            for st in container.streams:
                if st.type == "audio":
                    cc = st.codec_context
                    audio_streams.append(
                        AudioStreamInfo(
                            index=st.index,
                            codec=(cc.codec.name if cc and cc.codec else "unknown"),
                            sample_rate=int(cc.sample_rate or 0),
                            channels=int(cc.channels or 0),
                            bit_rate=int(cc.bit_rate) if cc and cc.bit_rate else None,
                            language=(st.metadata or {}).get("language"),
                            duration=_stream_duration(st),
                        )
                    )
                elif st.type == "video":
                    cc = st.codec_context
                    video_streams.append(
                        VideoStreamInfo(
                            index=st.index,
                            codec=(cc.codec.name if cc and cc.codec else "unknown"),
                            width=int(cc.width or 0) if cc else 0,
                            height=int(cc.height or 0) if cc else 0,
                            fps=(float(st.average_rate) if st.average_rate else None),
                        )
                    )

            if not audio_streams:
                if video_streams:
                    raise MediaError(
                        f"该视频不包含音轨，无法提取音频"
                        f"（视频编码：{video_streams[0].codec}，"
                        f"{video_streams[0].width}x{video_streams[0].height}）。"
                    )
                raise MediaError(f"未在文件中找到任何音频流：{os.path.basename(path)}")

            # 时长优先容器级，其次最长音轨
            duration = float(container.duration or 0) / 1_000_000.0
            if duration <= 0:
                duration = max((a.duration or 0.0) for a in audio_streams)

            notes: list[str] = []
            for a in audio_streams:
                if a.channels > 2:
                    notes.append(
                        f"音轨 {a.index} 为 {a.channels} 声道，已降混为立体声播放/输出。"
                    )

            return MediaInfo(
                path=path,
                kind="video" if video_streams else "audio",
                container=container_name,
                duration=duration,
                audio_streams=audio_streams,
                video_streams=video_streams,
                decoder="pyav",
                notes=notes,
            )
    except MediaError:
        raise
    except Exception as exc:
        raise MediaError(
            f"无法解析该文件，可能是不支持的编码或已损坏。\n"
            f"文件：{os.path.basename(path)}\n底层错误：{exc}"
        ) from exc


def _stream_duration(st) -> float | None:
    try:
        if st.duration is not None and st.time_base is not None:
            return float(st.duration * st.time_base)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 解码
# ---------------------------------------------------------------------------

def decode(path: str, stream_index: int | None = None, mono: bool = False) -> AudioData:
    """把媒体文件解码为 float32。

    - 视频文件：提取指定音轨（默认第一条），画面被忽略。
    - 多声道（>2）统一降混为立体声；`mono=True` 时再降为单声道。
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in _SF_EXTS and stream_index is None:
        try:
            data = _decode_soundfile(path, mono=mono)
            if data is not None:
                return data
        except Exception:
            pass  # 落到 PyAV

    return _decode_pyav(path, stream_index=stream_index, mono=mono)


def _decode_soundfile(path: str, mono: bool) -> AudioData | None:
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.size == 0:
        return None
    if mono:
        samples = samples.mean(axis=1).astype(np.float32, copy=False)
    elif samples.shape[1] == 1:
        samples = samples[:, 0]
    return AudioData(samples=np.ascontiguousarray(samples, dtype=np.float32), sample_rate=int(rate))


def _decode_pyav(path: str, stream_index: int | None, mono: bool) -> AudioData:
    av = _av()
    try:
        with av.open(path) as container:
            streams = list(container.streams.audio)
            if not streams:
                raise MediaError(f"未在文件中找到任何音频流：{os.path.basename(path)}")

            if stream_index is None:
                st = streams[0]
            else:
                matches = [s for s in streams if s.index == stream_index]
                if not matches:
                    raise MediaError(f"文件里不存在音轨 {stream_index}。")
                st = matches[0]

            st.thread_type = "AUTO"
            cc = st.codec_context
            rate = int(cc.sample_rate or 0)
            layout = _norm_channels(int(cc.channels or 1))
            if mono:
                layout = "mono"
            if rate <= 0:
                raise MediaError("无法确定该音轨的采样率。")

            # 交给 FFmpeg 做格式转换（s16/fltp/… → 打包 float32），
            # 避免自己手写各采样格式的归一化。
            resampler = av.AudioResampler(format="flt", layout=layout, rate=rate)

            chunks: list[np.ndarray] = []
            for frame in container.decode(st):
                for converted in _resample_flush(resampler, frame):
                    chunks.append(_frame_to_2d(converted))
            for converted in _resample_flush(resampler, None):
                chunks.append(_frame_to_2d(converted))

            if not chunks:
                raise MediaError("该音轨没有可解码的音频数据。")

            samples = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
            if mono:
                samples = samples.mean(axis=1).astype(np.float32, copy=False)
            elif samples.shape[1] == 1:
                samples = samples[:, 0]
            return AudioData(
                samples=np.ascontiguousarray(samples, dtype=np.float32),
                sample_rate=rate,
            )
    except MediaError:
        raise
    except Exception as exc:
        codec = _guess_codec(path)
        raise MediaError(
            f"解码失败，可能是不支持的音频编码。\n"
            f"文件：{os.path.basename(path)}（编码：{codec}）\n底层错误：{exc}"
        ) from exc


def _guess_codec(path: str) -> str:
    try:
        av = _av()
        with av.open(path) as c:
            a = list(c.streams.audio)
            if a and a[0].codec_context and a[0].codec_context.codec:
                return a[0].codec_context.codec.name
    except Exception:
        pass
    return "unknown"


def _resample_flush(resampler, frame):
    """吞掉 PyAV 各版本 `AudioResampler.resample` 返回 list / 单帧 / None 的差异。"""
    out = resampler.resample(frame)
    if out is None:
        return []
    if isinstance(out, list):
        return [f for f in out if f is not None]
    return [out]


def _frame_to_2d(frame) -> np.ndarray:
    """PyAV 音频帧 → (frames, channels) float32。

    必须处理三种形态，否则不同帧的列数会不一致、拼接时炸掉：
      - 打包格式（我们强制请求的 'flt'）：(1, frames*channels)
      - 平面格式（'fltp' 等）：(channels, frames)
      - 单声道时 PyAV 可能直接给 (frames,) 或 (1, frames)
    """
    arr = np.asarray(frame.to_ndarray())
    ch = max(1, _frame_channels(frame))

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.shape[0] == 1:
        # 打包（含单声道）：(1, n*ch)
        arr = arr.reshape(-1, ch) if arr.size % ch == 0 else arr.reshape(-1, 1)
    elif arr.shape[0] == ch and arr.shape[1] != ch:
        # 平面：(ch, n)
        arr = arr.T
    else:
        arr = arr.reshape(arr.shape[0], -1)

    return np.ascontiguousarray(arr, dtype=np.float32)


def _frame_channels(frame) -> int:
    try:
        return int(frame.layout.nb_channels)
    except Exception:
        try:
            return len(frame.layout.channels)
        except Exception:
            return 1


# ---------------------------------------------------------------------------
# 通道 / 重采样
# ---------------------------------------------------------------------------

def to_mono(samples: np.ndarray) -> np.ndarray:
    """降到单声道。模型只吃单声道，所有引擎调用前都要过这里。"""
    if samples.ndim == 1:
        return samples
    if samples.shape[1] == 1:
        return np.ascontiguousarray(samples[:, 0], dtype=np.float32)
    return np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32)


def to_stereo(samples: np.ndarray) -> np.ndarray:
    """单声道复制成双声道，用于与立体声原音做 A/B 播放。"""
    if samples.ndim == 2:
        return samples
    return np.ascontiguousarray(np.stack([samples, samples], axis=1), dtype=np.float32)


def resample(samples: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """重采样。in_rate == out_rate 时原样返回，绝不重复过一遍重采样器。"""
    if in_rate == out_rate or samples.size == 0:
        return samples
    import soxr  # noqa: PLC0415

    out = soxr.resample(samples, in_rate, out_rate, quality="HQ")
    return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# 编码
# ---------------------------------------------------------------------------

def write_audio(
    path: str,
    samples: np.ndarray,
    sample_rate: int,
    subtype: str | None = None,
) -> str:
    """写出音频。格式由扩展名决定，默认 WAV PCM_24。

    `subtype` 可覆盖默认值——给 DeepFilterNet 传临时文件时必须用 "FLOAT"：
    它的 Rust 侧只支持到 16-bit，喂 PCM_24 会直接 panic（已实测，TooWide）。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in OUTPUT_SUBTYPES:
        raise MediaError(f"不支持的输出格式：{ext}（支持 {', '.join(OUTPUT_SUBTYPES)}）")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # 模型可能输出超过 ±1.0 的峰值，写整数 PCM 前必须夹紧
    data = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)

    kwargs = {}
    effective = subtype if subtype is not None else OUTPUT_SUBTYPES[ext]
    if effective:
        kwargs["subtype"] = effective
    if ext == ".wav":
        kwargs["endian"] = "FILE"   # 参数名是 endian（不是 endianness）
    if ext == ".mp3":
        kwargs["bitrate_mode"] = "VBR"

    sf.write(path, data, int(sample_rate), **kwargs)
    return path
