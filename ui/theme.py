"""深色主题：调色板 + 全局样式表。

两层叠加，而不是只用样式表：
1. 用 **Fusion** 样式打底并设置深色 QPalette —— 这样即使某些控件没被 QSS 覆盖，
   原生绘制的部分（下拉箭头、勾选标记、滚动条）也会按深色正确渲染，
   不会出现「箭头看不见」这类只能靠肉眼发现的毛病。
2. 再叠 QSS 调整具体的圆角、间距、强调色。

调色板与 icon.ico 同源：主色绿 #4fa97a（处理后），次色蓝 #6c8ebf（原音）。
"""

from __future__ import annotations

from PyQt5.QtGui import QColor, QFont, QPalette

# ---------------------------------------------------------------------------
# 调色板
# ---------------------------------------------------------------------------

BG = "#141518"          # 窗口底
PANEL = "#1b1d21"       # 面板/分组底
RAISED = "#232529"      # 输入框、列表底
RAISED_HI = "#2b2e33"   # 悬停
SUNKEN = "#191a1e"      # 按下/凹陷

BORDER = "#33363b"
BORDER_HI = "#454951"

TEXT = "#d9dce2"
TEXT_DIM = "#949ba6"
TEXT_FAINT = "#666d78"

ACCENT = "#4fa97a"      # 主色：处理后 / 主按钮
ACCENT_HI = "#5fbd8d"
ACCENT_DIM = "#3c7c5c"
ACCENT_TINT = "#22352b" # 选中项底

BLUE = "#6c8ebf"        # 次色：原音
WARN = "#d08a3a"
ERROR = "#c85a5a"

UI_FONT = "Microsoft YaHei UI"
MONO_FONT = "Consolas"


def palette() -> QPalette:
    p = QPalette()

    roles = {
        QPalette.Window: BG,
        QPalette.WindowText: TEXT,
        QPalette.Base: RAISED,
        QPalette.AlternateBase: PANEL,
        QPalette.Text: TEXT,
        QPalette.PlaceholderText: TEXT_FAINT,
        QPalette.Button: RAISED,
        QPalette.ButtonText: TEXT,
        QPalette.BrightText: "#ffffff",
        # Light/Midlight/Mid/Dark 是**原生绘制用的斜面与边框色**。它们必须与面板底色
        # (#1b1d21) 拉开差距，否则 Fusion 画出来的单选圆圈、分隔线之类会淡到看不见
        # ——实测把 Dark 设成 #191a1e（几乎等于底色）时，未选中单选的圆圈基本消失。
        QPalette.Light: "#5a616c",
        QPalette.Midlight: "#474d56",
        QPalette.Mid: "#3d4249",
        QPalette.Dark: "#2b2f36",
        QPalette.Shadow: "#0f1013",
        QPalette.Highlight: ACCENT_DIM,
        QPalette.HighlightedText: "#f2fbf6",
        QPalette.Link: BLUE,
        QPalette.LinkVisited: ACCENT,
        QPalette.ToolTipBase: RAISED_HI,
        QPalette.ToolTipText: TEXT,
    }

    # 关键：**Active 与 Inactive 两组都要设**。
    # QPalette.setColor(role, color) 默认只写 Active 组；窗口一旦失去焦点，
    # Qt 就改用 Inactive 组，那里没设过的角色会回落到系统默认的亮色调色板——
    # 表现为「切到别的窗口，本应用整个变白、按钮文字看不见」。
    # 这个问题只在失焦状态下才显形，光看一次截图很容易漏掉。
    for group in (QPalette.Active, QPalette.Inactive):
        for role, color in roles.items():
            p.setColor(group, role, QColor(color))

    # 禁用态：整组一起压暗，避免局部残留高对比
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(TEXT_FAINT))
    p.setColor(QPalette.Disabled, QPalette.Window, QColor(BG))
    p.setColor(QPalette.Disabled, QPalette.Base, QColor("#1c1e22"))
    p.setColor(QPalette.Disabled, QPalette.Button, QColor("#1c1e22"))
    p.setColor(QPalette.Disabled, QPalette.Highlight, QColor(BORDER))
    p.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(TEXT_FAINT))
    return p


def qss() -> str:
    return f"""
/* ---------- 基础 ---------- */
/* 注意：这里刻意不给 QWidget 设 background —— 那会让每个子控件都画出不透明
   矩形，在分组框里表现为一块块色斑。底色由 QPalette 统一负责。 */
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{
    background: {RAISED_HI};
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: 6px;
    padding: 6px 8px;
}}

/* ---------- 工具栏 ---------- */
QToolBar {{
    background: {PANEL};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 5px 8px;
    spacing: 4px;
}}
QToolBar::separator {{
    background: {BORDER};
    width: 1px;
    margin: 6px 8px;
}}
QToolButton {{
    background: transparent;
    color: {TEXT};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px 12px;
}}
QToolButton:hover {{ background: {RAISED_HI}; border-color: {BORDER_HI}; }}
QToolButton:pressed {{ background: {SUNKEN}; }}
QToolButton:disabled {{ color: {TEXT_FAINT}; }}

/* ---------- 分组框 ---------- */
QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 9px;
    margin-top: 15px;
    padding: 13px 12px 11px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 11px;
    padding: 0 5px;
    color: {TEXT_DIM};
}}

/* ---------- 按钮 ---------- */
QPushButton {{
    background: {RAISED};
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background: {RAISED_HI}; border-color: {ACCENT_DIM}; }}
QPushButton:pressed {{ background: {SUNKEN}; }}
QPushButton:disabled {{
    background: #1c1e22;
    color: {TEXT_FAINT};
    border-color: {BORDER};
}}
QPushButton[accent="true"] {{
    background: {ACCENT_DIM};
    color: #f2fbf6;
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{ background: {ACCENT}; border-color: {ACCENT_HI}; }}
QPushButton[accent="true"]:pressed {{ background: #336b4e; }}
QPushButton[accent="true"]:disabled {{
    background: #20262a; color: {TEXT_FAINT}; border-color: {BORDER};
}}

/* ---------- 列表 ---------- */
QListWidget, QListView {{
    background: {RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 3px;
    outline: none;
}}
QListWidget::item {{
    padding: 7px 9px;
    border-radius: 6px;
    margin: 1px 0;
}}
QListWidget::item:hover {{ background: {RAISED_HI}; }}
QListWidget::item:selected {{
    background: {ACCENT_TINT};
    border: 1px solid {ACCENT_DIM};
}}

/* ---------- 输入 ---------- */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {RAISED};
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: 6px;
    padding: 5px 8px;
    min-height: 18px;
    selection-background-color: {ACCENT_DIM};
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
    border-color: {ACCENT_DIM};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {RAISED};
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: 6px;
    padding: 3px;
    outline: none;
    selection-background-color: {ACCENT_TINT};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {RAISED_HI};
    border: none;
    width: 15px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {BORDER_HI};
}}

/* ---------- 勾选 / 单选 ---------- */
/* 这里只设 spacing 与文字颜色，**不要写 background，也不要提 ::indicator**。
   实测（隔离实验，以指示器区域非背景像素数计）：只要给 QRadioButton 加一句
   background: transparent，未选中单选的圆圈就会从 94 个像素掉到 2 个——圆圈
   基本消失；只写 ::indicator 的 width/height 也会把它切成"样式表驱动绘制"
   而不再画原生指示器。Qt 这块行为很敏感，统一交给 Fusion 原生绘制。 */
QCheckBox, QRadioButton {{ spacing: 8px; color: {TEXT}; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {TEXT_FAINT}; }}

/* ---------- 滑块 ---------- */
QSlider {{ background: transparent; }}
QSlider::groove:horizontal {{
    height: 4px;
    background: {SUNKEN};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    height: 4px;
    background: {ACCENT_DIM};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 13px;
    margin: -5px 0;
    border-radius: 6px;
    background: {TEXT_DIM};
    border: 1px solid {BORDER_HI};
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; border-color: {ACCENT_HI}; }}
QSlider::handle:horizontal:pressed {{ background: {ACCENT_HI}; }}
QSlider:disabled::sub-page:horizontal {{ background: {BORDER}; }}
QSlider:disabled::handle:horizontal {{ background: {BORDER_HI}; }}

/* ---------- 进度条 ---------- */
QProgressBar {{
    background: {SUNKEN};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 3px;
}}

/* ---------- 文本区 ---------- */
QPlainTextEdit, QTextEdit, QTextBrowser {{
    background: {SUNKEN};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px;
    selection-background-color: {ACCENT_DIM};
    selection-color: #f2fbf6;
}}
/* 说明文字框：无边框、融入背景 */
QPlainTextEdit#engineNote {{
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 2px 2px 0 2px;
    color: {TEXT_DIM};
}}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_HI};
    border-radius: 5px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {ACCENT_DIM}; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_HI};
    border-radius: 5px;
    min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT_DIM}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---------- 其它 ---------- */
QScrollArea {{ background: {BG}; border: none; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:hover {{ background: {ACCENT_DIM}; }}
QStatusBar {{
    background: {PANEL};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
}}
QStatusBar::item {{ border: none; }}
QLabel {{ background: transparent; color: {TEXT}; }}
QLabel#appTitle {{ font-size: 15px; font-weight: 600; color: {TEXT}; }}
QLabel#appSubtitle {{ color: {TEXT_FAINT}; }}
QLabel#statusLine {{ color: {TEXT_DIM}; }}
QLabel#sectionHint {{ color: {TEXT_FAINT}; font-size: 11px; }}
QLabel#fileLabel {{ font-size: 12px; font-weight: 600; color: {ACCENT_HI}; }}
QLabel#metricKey {{ color: {TEXT_FAINT}; }}
"""


def apply(app) -> None:
    """给 QApplication 装上深色主题与字体。"""
    app.setStyle("Fusion")
    app.setPalette(palette())
    app.setStyleSheet(qss())

    font = QFont(UI_FONT, 9)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)
