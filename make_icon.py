"""生成应用图标 icon.ico。

用 Qt 绘制再手工封装成 ICO：每个尺寸都单独按比例重绘（而不是缩小 256 的图），
这样 16×16 的工具栏尺寸也清晰。ICO 内使用 BMP(DIB) 条目——兼容性最好的形式。

用法（项目根目录）：
    .venv\\Scripts\\python.exe make_icon.py
"""

from __future__ import annotations

import os
import struct
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PyQt5.QtGui import (  # noqa: E402
    QBrush,
    QColor,
    QGuiApplication,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
SIZES = [16, 24, 32, 48, 64, 128, 256]

BG_TOP = QColor("#26282d")
BG_BOTTOM = QColor("#17181b")
BORDER = QColor("#3d4149")
BAR_SRC = QColor("#6c8ebf")      # 与界面波形视图的「原音」同色
BAR_ENH = QColor("#4fa97a")      # 与「处理后」同色

# 波形包络（0..1），大致对称、中间高，像一段语音的起落
ENVELOPE = [0.22, 0.44, 0.34, 0.68, 0.46, 0.92, 0.58, 0.74, 0.30, 0.52, 0.24]


def render(size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)

    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)

    # 圆角底板
    r = size * 0.20
    rect = QRectF(0.5, 0.5, size - 1.0, size - 1.0)
    grad = QLinearGradient(0.0, 0.0, 0.0, float(size))
    grad.setColorAt(0.0, BG_TOP)
    grad.setColorAt(1.0, BG_BOTTOM)
    path = QPainterPath()
    path.addRoundedRect(rect, r, r)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(grad))
    p.drawPath(path)

    if size >= 32:
        p.setPen(QPen(BORDER, max(1.0, size * 0.012)))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    # 波形竖条：从冷色渐变到暖色，暗示「处理前 → 处理后」
    n = len(ENVELOPE)
    slot = size * 0.78 / n
    bar_w = max(1.0, slot * 0.52)
    left = (size - slot * n) / 2.0
    mid = size / 2.0
    max_h = size * 0.60

    for i, amp in enumerate(ENVELOPE):
        h = max(bar_w, max_h * amp)
        x = left + slot * i + (slot - bar_w) / 2.0
        t = i / max(1, n - 1)
        color = QColor(
            int(BAR_SRC.red() + (BAR_ENH.red() - BAR_SRC.red()) * t),
            int(BAR_SRC.green() + (BAR_ENH.green() - BAR_SRC.green()) * t),
            int(BAR_SRC.blue() + (BAR_ENH.blue() - BAR_SRC.blue()) * t),
        )
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(
            QRectF(x, mid - h / 2.0, bar_w, h),
            min(bar_w / 2.0, size * 0.06),
            min(bar_w / 2.0, size * 0.06),
        )

    p.end()
    return img


def dib_bytes(img: QImage) -> bytes:
    """QImage → ICO 里的 BMP(DIB) 条目：BITMAPINFOHEADER + 自下而上的 BGRA + 全零 AND 掩码。"""
    s = img.width()
    assert img.height() == s

    # Format_ARGB32 在小端机器上内存布局即 B,G,R,A —— 正是 32bpp BI_RGB 要的顺序
    data = bytes(img.constBits().asstring(s * s * 4))
    rows = [data[y * s * 4:(y + 1) * s * 4] for y in range(s)]
    rows.reverse()                                     # BMP 自下而上
    pixels = b"".join(rows)

    mask_row = ((s + 31) // 32) * 4                    # 1bpp，行按 4 字节对齐
    mask = b"\x00" * (mask_row * s)                    # 全 0 = 用 alpha 通道

    header = struct.pack(
        "<IiiHHIIiiII",
        40,          # biSize
        s,           # biWidth
        s * 2,       # biHeight —— XOR 图 + AND 掩码，故为两倍
        1,           # biPlanes
        32,          # biBitCount
        0,           # biCompression = BI_RGB
        len(pixels) + len(mask),
        0, 0, 0, 0,
    )
    return header + pixels + mask


def build_ico(images: list[QImage]) -> bytes:
    count = len(images)
    out = struct.pack("<HHH", 0, 1, count)             # ICONDIR：reserved, type=1, count
    offset = 6 + 16 * count
    entries, blobs = [], []
    for img in images:
        blob = dib_bytes(img)
        blobs.append(blob)
        s = img.width()
        entries.append(struct.pack(
            "<BBBBHHII",
            0 if s >= 256 else s,      # 256 用 0 表示
            0 if s >= 256 else s,
            0,                         # 调色板数（真彩为 0）
            0,                         # reserved
            1,                         # planes
            32,                        # bitCount
            len(blob),
            offset,
        ))
        offset += len(blob)
    return out + b"".join(entries) + b"".join(blobs)


def main() -> int:
    app = QGuiApplication(sys.argv)   # noqa: F841 — QPainter/QImage 需要一个 QGuiApplication
    images = [render(s) for s in SIZES]
    ico = build_ico(images)

    path = os.path.join(ROOT, "icon.ico")
    with open(path, "wb") as fh:
        fh.write(ico)
    print(f"已写出 {path}  {len(ico):,} B  含 {len(SIZES)} 个尺寸：{SIZES}")

    # 回读校验：用 Qt 自己解析一遍，确认 ICO 结构没写错
    back = QImage(path)
    ok = not back.isNull()
    print(f"回读校验：{'通过' if ok else '失败'}  {back.width()}×{back.height()}  {back.format()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
