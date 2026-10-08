"""A/B 播放器：处理前 / 处理后同位置无缝切换。

要点：
- 两路音频都常驻内存，播放回调从「当前激活的那一路」按 `_pos` 取样，
  所以切换是采样级对齐的——同一时刻、同一位置，比较的只是音质。
- **增益匹配**是监听侧补偿：降噪后 RMS 必然下降，若不补偿，
  「变安静」会被误听成「变差」。该补偿不写回文件。
- 播放格式按两条轨道里**较高的采样率**统一，所以带宽扩展补出的高频在 A/B 试听里听得见
  （早先按源采样率播，16 kHz 素材的增强轨会被降采样回 16 kHz，补出来的高频全被丢掉）。
- 同一文件内部切换 A/B **不重开音频流**；换了采样率/声道数（换文件、或处理后目标率不同）
  才重开——不重开会变速播放或让回调拿到不匹配的形状。
"""

from __future__ import annotations

import threading

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal as Signal

from core import dsp, media_io

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None

SOURCE = "source"
ENHANCED = "enhanced"


class ABPlayer(QObject):
    positionChanged = Signal(float, float)   # (当前位置秒, 总时长秒)
    stateChanged = Signal(bool)              # 是否正在播放
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks: dict[str, np.ndarray] = {}
        self._active = SOURCE
        self._rate = 48000
        self._channels = 2
        self._pos = 0
        self._playing = False
        self._finished = False
        self._stream = None
        self._lock = threading.Lock()

        # 监听侧增益（不写回文件）
        self._gain_match = True
        self._match_gains = {SOURCE: 1.0, ENHANCED: 1.0}
        self._volume_db = 0.0
        self._level_warned = False

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def duration(self) -> float:
        n = self._frames
        return n / self._rate if self._rate else 0.0

    @property
    def _frames(self) -> int:
        return max((len(t) for t in self._tracks.values()), default=0)

    @property
    def active(self) -> str:
        return self._active

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def has_audio(self) -> bool:
        return bool(self._tracks) and self._frames > 0

    @property
    def gain_match(self) -> bool:
        return self._gain_match

    @property
    def volume_db(self) -> float:
        return self._volume_db

    # ------------------------------------------------------------------
    # 装载
    # ------------------------------------------------------------------
    def load(self, source: media_io.AudioData, enhanced: media_io.AudioData) -> None:
        """装载两条轨道。把两者的采样率与声道数统一到**最高档**，便于无缝切换。

        **播放采样率取两者的较大值**（2026-10-01 修正）：早先固定按**源**采样率播，
        于是 16 kHz 素材 + 带宽扩展（输出 48 kHz）时会把增强轨**降采样回 16 kHz**——
        补出来的 8–16 kHz 正好在 A/B 试听里被丢掉了，"试听用"预设的效果因此听不出来，
        而且那一趟重采样还要额外一整段内存（69 分钟约 1.6 GB）。现在按较高的那一侧播，
        低的一侧升采样（源升采样是无损的，不会丢掉任何已有内容）。
        """
        self.stop()
        src_rate = int(source.sample_rate)
        enh_rate = int(enhanced.sample_rate)
        rate = max(src_rate, enh_rate)
        src = np.asarray(source.samples, dtype=np.float32)
        enh = np.asarray(enhanced.samples, dtype=np.float32)

        if src_rate != rate:
            src = media_io.resample(src, src_rate, rate)
        if enh_rate != rate:
            enh = media_io.resample(enh, enh_rate, rate)

        # 声道数取两者的较大值；单声道一方复制成双声道以保持流格式一致
        n_ch = max(1 if src.ndim == 1 else src.shape[1],
                   1 if enh.ndim == 1 else enh.shape[1])
        n_ch = min(n_ch, 2)
        src = self._fit(src, n_ch)
        enh = self._fit(enh, n_ch)

        # 长度对齐，便于同位置比较
        n = min(len(src), len(enh))
        if len(src) != len(enh):
            src, enh = src[:n], enh[:n]

        with self._lock:
            self._tracks = {SOURCE: src, ENHANCED: enh}
            self._rate = rate
            self._channels = n_ch
            self._pos = 0
            self._finished = False
            self._active = SOURCE
            self._recompute_match_gains()

        # 格式变了就必须重开音频流（2026-10-01 修正）：`_open_stream` 把 samplerate/channels
        # 固定在打开那一刻，换一个采样率或声道数不同的文件后复用旧流，会得到变速播放或
        # 回调里形状不匹配的异常（流中止、界面却还显示"正在播放"）。
        self._close_stream()

        self.positionChanged.emit(0.0, self.duration)
        self.stateChanged.emit(False)

    @staticmethod
    def _fit(x: np.ndarray, n_ch: int) -> np.ndarray:
        if x.ndim == 1:
            if n_ch == 1:
                return np.ascontiguousarray(x.reshape(-1, 1), dtype=np.float32)
            return np.ascontiguousarray(np.stack([x, x], axis=1), dtype=np.float32)
        if x.shape[1] == n_ch:
            return np.ascontiguousarray(x, dtype=np.float32)
        if n_ch == 1:
            return media_io.to_mono(x).reshape(-1, 1)
        return media_io.to_stereo(media_io.to_mono(x))

    def _recompute_match_gains(self) -> None:
        """把「处理后」的整体响度对齐到「原音」，仅用于监听。"""
        self._match_gains = {SOURCE: 1.0, ENHANCED: 1.0}
        if not self._gain_match:
            return
        src = self._tracks.get(SOURCE)
        enh = self._tracks.get(ENHANCED)
        if src is None or enh is None or src.size == 0 or enh.size == 0:
            return

        def rms(a: np.ndarray) -> float:
            # 分块 RMS：旧写法 `np.square(a.astype(np.float64))` 会把整段放大成 float64
            # 再复制一份平方结果（实测 1200 s 立体声的瞬时峰值 = 素材字节的 4 倍 ≈ 1758 MB，
            # 69 分钟约 6 GB），而这里是**在 Qt 槽里持锁执行**的——既卡界面又会撞内存墙。
            # 分块版实测同速、峰值 122 MB、数值一致（见 core/dsp.py 的实测表）。
            return dsp.rms_blockwise(a)

        r_src, r_enh = rms(src), rms(enh)
        if r_src <= 1e-12 or r_enh <= 1e-12:
            return
        # 上限 ±24 dB，避免把极端素材抬成噪声墙
        gain = float(np.clip(r_src / r_enh, 10 ** (-24 / 20), 10 ** (24 / 20)))
        self._match_gains[ENHANCED] = gain

    def clear(self) -> None:
        self.stop()
        with self._lock:
            self._tracks = {}
            self._pos = 0
        self.positionChanged.emit(0.0, 0.0)

    # ------------------------------------------------------------------
    # 控制
    # ------------------------------------------------------------------
    def set_active(self, which: str) -> None:
        if which not in (SOURCE, ENHANCED):
            return
        with self._lock:
            self._active = which

    def set_gain_match(self, on: bool) -> None:
        self._gain_match = bool(on)
        with self._lock:
            self._recompute_match_gains()

    def set_volume_db(self, db: float) -> None:
        self._volume_db = float(db)

    def seek(self, seconds: float) -> None:
        # `_pos` 由音频回调线程与 UI 线程同时读写（回调推进、这里跳转），旧代码没用锁：
        # 拖动定位可能被"上一块的结尾"覆盖回去（表现为偶发回跳）。这里与回调共用同一把锁。
        with self._lock:
            pos = int(np.clip(seconds, 0.0, max(0.0, self.duration)) * self._rate)
            self._pos = min(pos, max(0, self._frames))
            self._finished = False
        self.positionChanged.emit(self._pos / self._rate, self.duration)

    def play(self) -> None:
        if not self.has_audio:
            return
        if sd is None:
            self.error.emit("sounddevice 不可用，无法播放。")
            return
        if self._pos >= self._frames:
            self._pos = 0
        if self._stream is None and not self._open_stream():
            return
        self._finished = False
        self._playing = True
        self._stream.start()
        self.stateChanged.emit(True)

    def pause(self) -> None:
        if self._stream is not None:
            self._stream.stop()
        self._playing = False
        self.stateChanged.emit(False)

    def stop(self) -> None:
        was = self._playing
        if self._stream is not None:
            self._stream.stop()
        self._playing = False
        self._pos = 0
        if was:
            self.stateChanged.emit(False)
        self.positionChanged.emit(0.0, self.duration)

    def shutdown(self) -> None:
        self._playing = False
        self._close_stream()

    def _close_stream(self) -> None:
        """关掉输出流（换文件/换格式/退出时都要用）。关失败也要把引用清掉，
        否则 `play()` 会因为 `self._stream is not None` 而跳过重开，播放静默失效。"""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:                  # noqa: BLE001
                self.error.emit(f"关闭音频输出流失败（已忽略，下次会重开）：{exc}")

    # ------------------------------------------------------------------
    # 音频回调
    # ------------------------------------------------------------------
    def _open_stream(self) -> bool:
        try:
            self._stream = sd.OutputStream(
                samplerate=self._rate,
                channels=self._channels,
                dtype="float32",
                callback=self._callback,
                blocksize=0,
            )
            return True
        except Exception as exc:
            self.error.emit(f"无法打开音频输出（{self._rate} Hz / {self._channels} 声道）：{exc}")
            self._stream = None
            return False

    def _callback(self, outdata, frames, time_info, status) -> None:
        with self._lock:
            track = self._tracks.get(self._active)
            gain = self._match_gains.get(self._active, 1.0) * (10.0 ** (self._volume_db / 20.0))
            pos = self._pos

        if track is None:
            outdata.fill(0.0)
            with self._lock:
                self._finished = True
            return

        end = min(pos + frames, len(track))
        chunk = track[pos:end]
        outdata[: len(chunk)] = chunk * gain
        if len(chunk) < frames:
            outdata[len(chunk):] = 0.0
            # `_pos/_finished` 与 UI 线程共享（seek/poll 会读），写回时要持同一把锁
            with self._lock:
                self._pos = len(track)
                self._finished = True
        else:
            with self._lock:
                self._pos = end

    # ------------------------------------------------------------------
    # 供界面定时器调用
    # ------------------------------------------------------------------
    def poll(self) -> None:
        """由界面定时器驱动：推进进度显示，并在播放到末尾时自动停止。"""
        if not self.has_audio:
            return
        if self._playing and self._finished:
            self.pause()
            self._pos = 0
            self.positionChanged.emit(0.0, self.duration)
            return
        if self._playing:
            self.positionChanged.emit(self._pos / self._rate, self.duration)
