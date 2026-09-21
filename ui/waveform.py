"""波形对比视图：上下两条包络（原音 / 处理后）共享同一时间轴。

用 QPainter 自绘，不引入任何绘图依赖。包络按控件宽度预计算并缓存，
只在尺寸或数据变化时重算，因此重绘（拖动播放头）是廉价的。
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QRectF, Qt, pyqtSignal as Signal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from . import theme

BG = QColor(theme.SUNKEN)
GRID = QColor(theme.BORDER)
WAVE_SRC = QColor(theme.BLUE)      # 「原音」用次色蓝
WAVE_ENH = QColor(theme.ACCENT)    # 「处理后」用主色绿
PLAYHEAD = QColor("#e8b84b")
TEXT = QColor(theme.TEXT_DIM)


def _envelope(x: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """把信号压成每像素列的 (min, max) 包络。"""
    n = len(x)
    if width <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    if n == 0:
        return np.zeros(width, np.float32), np.zeros(width, np.float32)
    if n < width:
        idx = np.linspace(0, n - 1, width)
        y = np.interp(idx, np.arange(n), x).astype(np.float32)
        return y, y
    k = n // width
    usable = k * width
    block = x[:usable].reshape(width, k)
    return block.min(axis=1).astype(np.float32), block.max(axis=1).astype(np.float32)


class WaveformView(QWidget):
    seekRequested = Signal(float)   # 秒

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

        self._src = None
        self._enh = None
        self._rate = 48000
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cache_width = -1
        self._cache_series = None
        self._pos_s = 0.0
        self._duration = 0.0
        self._trim: tuple[float, float] | None = None

    # ------------------------------------------------------------------
    def set_data(self, source: np.ndarray | None, enhanced: np.ndarray | None, rate: int) -> None:
        self._src = self._mono(source)
        self._enh = self._mono(enhanced)
        self._rate = max(1, int(rate))
        self._duration = max(
            (len(v) for v in (self._src, self._enh) if v is not None), default=0
        ) / self._rate
        self._invalidate()
        self.update()

    def clear_data(self) -> None:
        self._src = self._enh = None
        self._duration = 0.0
        self._pos_s = 0.0
        self._invalidate()
        self.update()

    @staticmethod
    def _mono(x: np.ndarray | None) -> np.ndarray | None:
        if x is None:
            return None
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 2:
            a = a.mean(axis=1)
        return np.ascontiguousarray(a, dtype=np.float32)

    def _invalidate(self) -> None:
        self._cache.clear()
        self._cache_width = -1
        self._cache_series = None

    def set_position(self, seconds: float) -> None:
        self._pos_s = max(0.0, float(seconds))
        self.update()

    def set_trim(self, start_s: float | None, end_s: float | None) -> None:
        """标出会被裁掉的区间（两侧压暗），便于处理前确认。"""
        self._trim = None if start_s is None or end_s is None else (float(start_s), float(end_s))
        self.update()

    def set_duration(self, seconds: float) -> None:
        self._duration = max(0.0, float(seconds))
        self.update()

    def resizeEvent(self, event):  # noqa: N802 — Qt 命名
        self._invalidate()
        super().resizeEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        if self._duration <= 0:
            return
        frac = min(1.0, max(0.0, event.pos().x() / max(1.0, self.width())))
        self.seekRequested.emit(frac * self._duration)

    # ------------------------------------------------------------------
    def _envelopes(self) -> None:
        w = max(1, self.width())
        series = (id(self._src), id(self._enh))
        if w == self._cache_width and series == self._cache_series:
            return
        self._cache_width = w
        self._cache_series = series
        self._cache = {}
        if self._src is not None:
            self._cache["src"] = _envelope(self._src, w)
        if self._enh is not None:
            self._cache["enh"] = _envelope(self._enh, w)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        p.setRenderHint(QPainter.Antialiasing, False)

        w, h = self.width(), self.height()
        if w < 4 or h < 20:
            return

        rows = [("src", "原音", WAVE_SRC), ("enh", "处理后", WAVE_ENH)]
        pad_top, gap, pad_bottom = 16, 12, 14
        row_h = max(8, (h - pad_top - gap - pad_bottom) // 2)

        # 时间网格（每 1/10 宽一条）
        p.setPen(QPen(GRID, 1))
        for i in range(1, 10):
            x = int(w * i / 10)
            p.drawLine(x, pad_top, x, pad_top + row_h * 2 + gap)

        self._envelopes()

        for idx, (key, label, color) in enumerate(rows):
            top = pad_top + idx * (row_h + gap)
            mid = top + row_h / 2.0

            p.setPen(QPen(GRID, 1))
            p.drawLine(0, int(mid), w, int(mid))

            env = self._cache.get(key)
            p.setPen(QPen(color, 1))
            if env is None:
                p.setPen(QPen(TEXT, 1))
                p.setFont(QFont("Segoe UI", 8))
                p.drawText(QRectF(6, top, w - 12, row_h), Qt.AlignVCenter, f"{label}（无数据）")
                continue

            mn, mx = env
            span = float(max(np.max(np.abs(mx)), np.max(np.abs(mn)), 1e-6))
            half = row_h / 2.0 - 1.0
            for x in range(min(len(mx), w)):
                y0 = mid - float(mx[x]) / span * half
                y1 = mid - float(mn[x]) / span * half
                if y1 - y0 < 1.0:
                    y1 = y0 + 1.0
                p.drawLine(x, int(y0), x, int(y1))

            p.setPen(QPen(TEXT, 1))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(6, top + 11, f"{label}  峰值 {span:.3f}")

        # 播放头
        if self._duration > 0:
            # 会被裁掉的区间压暗
            if self._trim is not None:
                x0 = int(w * max(0.0, min(1.0, self._trim[0] / self._duration)))
                x1 = int(w * max(0.0, min(1.0, self._trim[1] / self._duration)))
                shade = QColor(0, 0, 0, 130)
                p.fillRect(0, pad_top, max(0, x0), row_h * 2 + gap, shade)
                p.fillRect(x1, pad_top, max(0, w - x1), row_h * 2 + gap, shade)

            x = int(w * min(1.0, self._pos_s / self._duration))
            p.setPen(QPen(PLAYHEAD, 1))
            p.drawLine(x, pad_top - 6, x, pad_top + row_h * 2 + gap + 4)

        # 圆角描边，与其它面板的观感一致
        p.setPen(QPen(GRID, 1))
        p.setBrush(Qt.NoBrush)
        border = QPainterPath()
        border.addRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 8.0, 8.0)
        p.drawPath(border)

        p.end()
