"""主窗口：导入（音频/视频）→ 挑选模型 → 处理 → A/B 播放 → 导出。"""

from __future__ import annotations

import os
import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,          # PyQt5 里 QAction 在 QtWidgets（PySide6 在 QtGui）
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core import media_io, modelstore, pipeline
from core.worker import AnalyzeWorker, DownloadWorker, PipelineWorker

from . import theme
from .player import ENHANCED, SOURCE, ABPlayer
from .waveform import WaveformView

READY_COLOR = QColor(theme.ACCENT)
MISSING_COLOR = QColor(theme.WARN)

FILTER_PARTS = " ".join(f"*{e}" for e in sorted(media_io.ALL_EXTS))
FILE_FILTER = (
    f"媒体文件（{FILTER_PARTS}）"
    f";;音频文件（{' '.join('*' + e for e in sorted(media_io.AUDIO_EXTS))}）"
    f";;视频文件（{' '.join('*' + e for e in sorted(media_io.VIDEO_EXTS))}）"
    f";;所有文件（*.*）"
)


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


class MainWindow(QMainWindow):
    def __init__(self, project_root: str, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        self.catalog = modelstore.load_catalog(project_root)

        self._media_path: str | None = None
        self._info: media_io.MediaInfo | None = None
        self._stream_index: int | None = None
        self._stats = None
        self._span = None
        self._result: pipeline.PipelineResult | None = None

        self._analyze_worker: AnalyzeWorker | None = None
        self._pipeline_worker: PipelineWorker | None = None
        self._download_worker: DownloadWorker | None = None

        self.player = ABPlayer(self)
        self.player.positionChanged.connect(self._on_position)
        self.player.stateChanged.connect(self._on_player_state)
        self.player.error.connect(lambda m: self._log(f"[播放] {m}"))

        # 主题在这里兜底应用，这样无论从 run.py 还是从测试脚本进来，界面都是一致的深色。
        # 入口若已先设置过样式表，这里就不再重复覆盖。
        app = QApplication.instance()
        if app is not None and not app.styleSheet():
            theme.apply(app)

        self.setWindowTitle("AudioEnhancer_zk21 — 本地语音增强")
        self.resize(1340, 900)
        self.setMinimumSize(1060, 640)
        self.setAcceptDrops(True)

        self._icon_path = os.path.join(project_root, "icon.ico")
        if os.path.isfile(self._icon_path):
            self.setWindowIcon(QIcon(self._icon_path))

        self._build_ui()
        self._populate_engines()
        self._install_shortcuts()
        self._sync_left_min_width()   # 列表填好之后再定宽度下限

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self.player.poll)
        self._timer.start()

        self._log(f"项目目录：{project_root}")
        self._log(f"模型目录：{self.catalog.models_root}")
        self._log("导入音频或视频文件即可开始（视频只提取音轨，输出为音频）。")
        self._refresh_engine_states()

    # ==================================================================
    # 界面搭建
    # ==================================================================
    def _build_ui(self) -> None:
        toolbar = self.addToolBar("主工具栏")
        toolbar.setMovable(False)

        act_open = QAction("导入文件…", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self._choose_file)
        toolbar.addAction(act_open)

        act_sample = QAction("载入示例", self)
        act_sample.triggered.connect(self._load_sample)
        toolbar.addAction(act_sample)

        toolbar.addSeparator()
        act_export = QAction("导出为…", self)
        act_export.setShortcut(QKeySequence.Save)
        act_export.triggered.connect(self._export_as)
        toolbar.addAction(act_export)
        self.act_export = act_export
        act_export.setEnabled(False)   # 处理完成后才可导出

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 900])
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("就绪")

    def _build_left_panel(self) -> QWidget:
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(12, 12, 8, 12)
        lay.setSpacing(12)

        lay.addWidget(self._build_header())
        lay.addWidget(self._build_info_box())
        lay.addWidget(self._build_model_box(), 1)   # 列表吃掉多余的竖向空间
        lay.addWidget(self._build_params_box())

        row_run = QHBoxLayout()
        row_run.setSpacing(8)
        self.btn_run = QPushButton("开始处理")
        self.btn_run.setProperty("accent", True)    # 主按钮用强调色
        self.btn_run.setMinimumHeight(38)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._start_process)
        row_run.addWidget(self.btn_run, 3)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setMinimumHeight(38)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_process)
        row_run.addWidget(self.btn_cancel, 1)
        lay.addLayout(row_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        lay.addWidget(self.progress)

        self.lbl_status = QLabel("就绪")
        self.lbl_status.setObjectName("statusLine")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        # 关键：整列放进滚动区。窗口变矮时出现滚动条，而不是把子控件压扁、压重叠——
        # 「模型说明文字遮挡模型列表」的根因就是竖向空间不足时 Qt 强行压缩子控件。
        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 横向滚动条是关掉的，所以左侧栏宽度**必须**容得下内容的真实最小宽度，
        # 否则右侧会被直接裁掉（写死宽度就会踩这个坑）。这里按实测最小宽度来定。
        need = content.minimumSizeHint().width() + 14
        scroll.setMinimumWidth(need)
        scroll.setMaximumWidth(max(620, need + 60))
        self._left_min_width = need
        self.left_scroll = scroll      # 供测试检查横向是否被裁切
        return scroll

    def _sync_left_min_width(self) -> None:
        """按内容实测最小宽度同步左侧栏的宽度下限。

        必须在引擎列表与说明文字**填好之后**再测一次：构造时列表还是空的，
        那时测出来的最小宽度偏小，分栏就会把它压到下限以下，右侧内容被直接裁掉
        （横向滚动条是关的，不会出现滚动条来兜底）。
        """
        scroll = getattr(self, "left_scroll", None)
        if scroll is None:
            return
        content = scroll.widget()
        if content is None:
            return
        layout = content.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        need = content.minimumSizeHint().width() + 14
        if need <= 0:
            return
        self._left_min_width = need
        scroll.setMinimumWidth(need)
        scroll.setMaximumWidth(max(620, need + 60))

    def _build_header(self) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(2, 0, 2, 0)
        h.setSpacing(11)

        logo = QLabel()
        logo.setFixedSize(40, 40)
        if os.path.isfile(self._icon_path):
            pm = QPixmap(self._icon_path)
            if not pm.isNull():
                logo.setPixmap(pm.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        h.addWidget(logo)

        col = QVBoxLayout()
        col.setSpacing(2)
        title = QLabel("AudioEnhancer_zk21")
        title.setObjectName("appTitle")
        sub = QLabel("本地语音增强 · 支持音频与视频，只输出音频")
        sub.setObjectName("appSubtitle")
        sub.setWordWrap(True)
        col.addWidget(title)
        col.addWidget(sub)
        h.addLayout(col, 1)
        return row

    def _build_info_box(self) -> QWidget:
        box = QGroupBox("媒体信息")
        g = QVBoxLayout(box)
        g.setSpacing(7)

        self.lbl_file = QLabel("尚未导入文件")
        self.lbl_file.setObjectName("fileLabel")
        self.lbl_file.setWordWrap(True)
        g.addWidget(self.lbl_file)

        self.lbl_media = QLabel("—")
        self.lbl_media.setWordWrap(True)
        g.addWidget(self.lbl_media)

        self.row_stream = QWidget()
        rs = QHBoxLayout(self.row_stream)
        rs.setContentsMargins(0, 0, 0, 0)
        rs.setSpacing(6)
        rs.addWidget(QLabel("音轨："))
        self.cmb_stream = QComboBox()
        self.cmb_stream.setEnabled(False)
        self.cmb_stream.currentIndexChanged.connect(self._on_stream_changed)
        rs.addWidget(self.cmb_stream, 1)
        g.addWidget(self.row_stream)
        self.row_stream.setVisible(False)

        self.lbl_stats = QLabel("—")
        self.lbl_stats.setWordWrap(True)
        g.addWidget(self.lbl_stats)
        return box

    def _build_model_box(self) -> QWidget:
        box = QGroupBox("模型")
        g = QVBoxLayout(box)
        g.setSpacing(8)

        self.list_engines = QListWidget()
        self.list_engines.setSelectionMode(QAbstractItemView.SingleSelection)
        # 按整项滚动，而不是按像素：按像素滚会把首条切成半截，看着像布局坏了
        self.list_engines.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
        # 关掉横向滚动条：条目过长时靠省略号截断，而不是冒出一条滚动条
        self.list_engines.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_engines.setTextElideMode(Qt.ElideRight)
        self.list_engines.setMinimumHeight(170)
        self.list_engines.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.list_engines.currentItemChanged.connect(self._on_engine_changed)
        g.addWidget(self.list_engines, 1)

        # 说明用**固定高度**的只读框，而不是 QLabel + wordWrap。
        # QLabel 的换行高度协商不可靠，文字一长就会撑动/压到相邻控件；
        # 这里高度恒定，内容多了自己滚动，还能选中复制。
        self.note_view = QPlainTextEdit()
        self.note_view.setObjectName("engineNote")
        self.note_view.setReadOnly(True)
        self.note_view.setFixedHeight(80)
        self.note_view.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.note_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.note_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.note_view.setFont(QFont(theme.UI_FONT, 8))
        g.addWidget(self.note_view)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_download = QPushButton("下载所选模型")
        self.btn_download.clicked.connect(self._download_selected)
        row.addWidget(self.btn_download, 1)
        self.btn_redownload = QPushButton("校验全部")
        self.btn_redownload.clicked.connect(self._refresh_engine_states)
        row.addWidget(self.btn_redownload)
        g.addLayout(row)
        return box

    def _build_params_box(self) -> QWidget:
        box = QGroupBox("参数")
        gp = QGridLayout(box)
        gp.setHorizontalSpacing(8)
        gp.setVerticalSpacing(7)
        r = 0

        self.lbl_atten = QLabel("降噪强度")
        gp.addWidget(self.lbl_atten, r, 0)
        row_a = QHBoxLayout()
        row_a.setSpacing(8)
        self.sld_atten = QSlider(Qt.Horizontal)
        self.sld_atten.setRange(0, 30)
        self.sld_atten.setValue(12)
        self.sld_atten.valueChanged.connect(self._on_atten_changed)
        self.lbl_atten_val = QLabel("12 dB")
        self.lbl_atten_val.setMinimumWidth(46)
        self.lbl_atten_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row_a.addWidget(self.sld_atten, 1)
        row_a.addWidget(self.lbl_atten_val)
        gp.addLayout(row_a, r, 1)
        r += 1

        # 文案纠正（实测）：attenuation_limit_db 是"允许的最大衰减量"。
        # 0 表示**不设上限**，也就是压得最狠；从 6 往上才是"数值越大压得越狠"。
        # 另一处坑：DeepFilterNet3 用相反约定（它按 0 = 不降噪 / 100 = 全降噪计）。
        self.lbl_atten_hint = QLabel(
            "0 = 不设上限（压得最狠）；6–12 保真度较好。\n"
            "注意 DeepFilterNet3 用相反约定（它按 0/100 计）。"
        )
        self.lbl_atten_hint.setObjectName("sectionHint")
        self.lbl_atten_hint.setWordWrap(True)   # 单行不换行会把整列的宽度下限顶大
        gp.addWidget(self.lbl_atten_hint, r, 1)
        r += 1

        self.chk_trim = QCheckBox("裁剪首尾静音")
        self.chk_trim.setToolTip("按自适应阈值裁掉开头/结尾的静音段，中间静音默认保留（不改变时间轴）")
        gp.addWidget(self.chk_trim, r, 0, 1, 2)
        r += 1

        self.chk_norm = QCheckBox("增益归一化")
        gp.addWidget(self.chk_norm, r, 0)
        row_n = QHBoxLayout()
        row_n.setSpacing(6)
        self.cmb_norm_mode = QComboBox()
        self.cmb_norm_mode.addItem("按峰值", "peak")
        self.cmb_norm_mode.addItem("按 RMS", "rms")
        self.spn_norm_target = QDoubleSpinBox()
        self.spn_norm_target.setRange(-40.0, 0.0)
        self.spn_norm_target.setValue(-3.0)
        self.spn_norm_target.setSuffix(" dBFS")
        self.spn_norm_target.setSingleStep(0.5)
        row_n.addWidget(self.cmb_norm_mode)
        row_n.addWidget(self.spn_norm_target)
        gp.addLayout(row_n, r, 1)
        r += 1

        self.chk_stereo = QCheckBox("保持立体声（逐声道处理）")
        self.chk_stereo.setChecked(True)
        gp.addWidget(self.chk_stereo, r, 0, 1, 2)
        r += 1

        gp.addWidget(QLabel("线程数"), r, 0)
        self.spn_threads = QSpinBox()
        self.spn_threads.setRange(1, 16)
        self.spn_threads.setValue(2)
        gp.addWidget(self.spn_threads, r, 1)
        r += 1

        gp.addWidget(QLabel("输出格式"), r, 0)
        self.cmb_format = QComboBox()
        for label, ext in media_io.OUTPUT_FORMATS:
            self.cmb_format.addItem(label, ext)
        gp.addWidget(self.cmb_format, r, 1)
        r += 1

        self.chk_autoexport = QCheckBox("处理后自动导出到 out/")
        self.chk_autoexport.setChecked(True)
        gp.addWidget(self.chk_autoexport, r, 0, 1, 2)
        r += 1

        gp.setColumnStretch(1, 1)
        return box

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 10, 10, 10)
        lay.setSpacing(8)

        self.waveform = WaveformView()
        self.waveform.seekRequested.connect(self._seek)
        lay.addWidget(self.waveform, 3)

        # --- A/B 播放器 ----------------------------------------------
        box_ab = QGroupBox("试听对比（A/B）")
        gb = QVBoxLayout(box_ab)

        row_top = QHBoxLayout()
        row_top.setSpacing(8)
        self.btn_src = QRadioButton("原音")
        self.btn_enh = QRadioButton("处理后")
        self.btn_src.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.btn_src)
        grp.addButton(self.btn_enh)
        self.btn_src.toggled.connect(
            lambda on: self.player.set_active(SOURCE if on else ENHANCED)
        )
        row_top.addWidget(self.btn_src)
        row_top.addWidget(self.btn_enh)
        row_top.addSpacing(14)

        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setMinimumSize(94, 32)
        self.btn_play.clicked.connect(self._toggle_play)
        row_top.addWidget(self.btn_play)
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setMinimumHeight(32)
        self.btn_stop.clicked.connect(self.player.stop)
        row_top.addWidget(self.btn_stop)
        row_top.addStretch(1)
        gb.addLayout(row_top)

        row_pos = QHBoxLayout()
        row_pos.setSpacing(10)
        self.sld_pos = QSlider(Qt.Horizontal)
        self.sld_pos.setRange(0, 1000)
        self.sld_pos.setValue(0)
        self.sld_pos.sliderMoved.connect(self._on_slider_moved)
        row_pos.addWidget(self.sld_pos, 1)
        self.lbl_time = QLabel("00:00.00 / 00:00.00")
        self.lbl_time.setFont(QFont(theme.MONO_FONT, 9))
        self.lbl_time.setMinimumWidth(140)
        self.lbl_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row_pos.addWidget(self.lbl_time)
        gb.addLayout(row_pos)

        row_vol = QHBoxLayout()
        row_vol.setSpacing(8)
        self.chk_gainmatch = QCheckBox("增益匹配")
        self.chk_gainmatch.setChecked(True)
        self.chk_gainmatch.setToolTip(
            "把「处理后」的整体响度对齐到「原音」再试听。\n"
            "否则降噪后自然变安静，容易被误听成「变差」。\n"
            "该补偿只在试听时生效，不会写进输出文件。"
        )
        self.chk_gainmatch.toggled.connect(self.player.set_gain_match)
        row_vol.addWidget(self.chk_gainmatch)

        row_vol.addWidget(QLabel("音量"))
        self.sld_vol = QSlider(Qt.Horizontal)
        self.sld_vol.setRange(-24, 12)
        self.sld_vol.setValue(0)
        self.sld_vol.setMaximumWidth(150)
        self.sld_vol.valueChanged.connect(
            lambda v: (self.player.set_volume_db(v), self.lbl_vol.setText(f"{v:+d} dB"))
        )
        self.lbl_vol = QLabel("+0 dB")
        self.lbl_vol.setMinimumWidth(52)
        row_vol.addWidget(self.sld_vol)
        row_vol.addWidget(self.lbl_vol)
        row_vol.addStretch(1)

        hint = QLabel("空格 播放/暂停 · A 原音 · B 处理后")
        hint.setObjectName("sectionHint")
        row_vol.addWidget(hint)
        gb.addLayout(row_vol)

        lay.addWidget(box_ab)

        # --- 日志 -----------------------------------------------------
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(3000)
        self.txt_log.setFont(QFont(theme.MONO_FONT, 9))
        self.txt_log.setPlaceholderText("处理日志会显示在这里")
        lay.addWidget(self.txt_log, 2)

        for w in (self.btn_play, self.btn_stop, self.btn_src, self.btn_enh, self.sld_pos):
            w.setEnabled(False)
        return panel

    def _install_shortcuts(self) -> None:
        act_play = QAction(self)
        act_play.setShortcut(Qt.Key_Space)
        act_play.triggered.connect(self._toggle_play)
        self.addAction(act_play)

        act_a = QAction(self)
        act_a.setShortcut(Qt.Key_A)
        act_a.triggered.connect(lambda: self.btn_src.setChecked(True))
        self.addAction(act_a)

        act_b = QAction(self)
        act_b.setShortcut(Qt.Key_B)
        act_b.triggered.connect(lambda: self.btn_enh.setChecked(True))
        self.addAction(act_b)

    # ==================================================================
    # 模型列表
    # ==================================================================
    def _populate_engines(self) -> None:
        self.list_engines.clear()
        for spec in self.catalog.engines:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, spec.id)
            self.list_engines.addItem(item)

        default = self.catalog.default_engine
        for row in range(self.list_engines.count()):
            if self.list_engines.item(row).data(Qt.UserRole) == default.id:
                self.list_engines.setCurrentRow(row)
                break
        if self.list_engines.currentRow() < 0 and self.list_engines.count():
            self.list_engines.setCurrentRow(0)
        self._fit_engine_list_height()

    def _fit_engine_list_height(self) -> None:
        """把列表高度调到能一次显示全部模型。

        否则列表内部会出现滚动条，而 setCurrentRow 的自动滚动会把首条切掉一半，
        看起来像是布局出了问题。左侧整列本来就在滚动区里，列表可以放心长高：
        空间不够时由外层滚动，而不是让列表自己滚。
        """
        n = self.list_engines.count()
        if not n:
            return
        cap = 460

        # 第一版估值：QSS 给 item 加的 padding/margin **不会**体现在 sizeHintForRow 里，
        # 所以这一版一定偏小，只用来给布局一个起点。
        total = sum(self.list_engines.sizeHintForRow(i) for i in range(n))
        total += 2 * self.list_engines.frameWidth() + 10
        self.list_engines.setMinimumHeight(min(total, cap))

        # 校正：布局跑过一次之后，实际内容高度 = 滚动范围 + 视口高度。
        sb = self.list_engines.verticalScrollBar()
        if sb.maximum() > 0:
            actual = sb.maximum() + self.list_engines.viewport().height()
            self.list_engines.setMinimumHeight(min(actual + 2, cap))

    def showEvent(self, event):  # noqa: N802 — Qt 命名
        super().showEvent(event)
        if not getattr(self, "_height_fitted", False):
            # 此时才第一次有真实布局与样式打磨，用 0 延时把两项校正跑在布局之后。
            # 顺序要紧：先定列表高度，再按新的内容最小宽度同步左侧栏宽度。
            self._height_fitted = True

            def _after_show() -> None:
                self._fit_engine_list_height()
                self._sync_left_min_width()

            QTimer.singleShot(0, _after_show)

    def _refresh_engine_states(self) -> None:
        for row in range(self.list_engines.count()):
            item = self.list_engines.item(row)
            spec = self.catalog.engine(item.data(Qt.UserRole))
            if spec is None:
                continue
            ready = modelstore.engine_ready(spec, self.catalog)
            if spec.kind == "dsp":
                status, color = "无需下载", None
            elif ready:
                status, color = "已就绪", None
            else:
                status, color = "需下载", MISSING_COLOR

            size = modelstore.human_size(spec.size_hint) if spec.size_hint else "内置"
            item.setText(f"{spec.name}    [{status}]\n{spec.speed} · {size}")
            # 只给「需要处理」的项上色（橙色），就绪项保持正文色。
            # 全部染绿会让 8 条一样重、反而分不出重点。
            item.setForeground(color if color is not None else QColor(theme.TEXT))
            item.setToolTip(spec.note)

        self._on_engine_changed(self.list_engines.currentItem())

    def _on_engine_changed(self, item) -> None:
        if item is None:
            return
        spec = self.catalog.engine(item.data(Qt.UserRole))
        if spec is None:
            return
        ready = modelstore.engine_ready(spec, self.catalog)
        self.note_view.setPlainText(spec.note)
        self.note_view.setToolTip(spec.note)
        self._sync_left_min_width()   # 说明文字换了，宽度下限可能要跟着变
        self.btn_download.setEnabled(not ready and spec.kind != "dsp")
        self.btn_download.setText("下载所选模型" if not ready else "模型已就绪")

        self.lbl_atten.setEnabled(spec.sherpa_type == "dpdfnet")
        self.sld_atten.setEnabled(spec.sherpa_type == "dpdfnet")
        self.lbl_atten_val.setEnabled(spec.sherpa_type == "dpdfnet")
        self.lbl_atten_hint.setEnabled(spec.sherpa_type == "dpdfnet")
        self._update_run_enabled()

    def _on_atten_changed(self, v: int) -> None:
        # 0 是"不设上限"，显示成 dB 会误导（实测：0 时噪声底压得最低）
        self.lbl_atten_val.setText("不限（最狠）" if v == 0 else f"{v} dB")

    def _selected_engine(self):
        item = self.list_engines.currentItem()
        if item is None:
            return None
        return self.catalog.engine(item.data(Qt.UserRole))

    # ==================================================================
    # 导入
    # ==================================================================
    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    ext = os.path.splitext(url.toLocalFile())[1].lower()
                    if ext in media_io.ALL_EXTS:
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event):  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if os.path.splitext(path)[1].lower() in media_io.ALL_EXTS:
                    self._import_file(path)
                    event.acceptProposedAction()
                    return
        event.ignore()

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入音频或视频", "", FILE_FILTER)
        if path:
            self._import_file(path)

    def _load_sample(self) -> None:
        sample = os.path.join(self.project_root, "samples", "speech_with_noise.wav")
        if not os.path.isfile(sample):
            self._log("[提示] 示例音频不存在，正在尝试补取…")
            extra = self.catalog.extra("samples")
            if extra:
                self._download_files(extra.files, "示例音频")
            return
        self._import_file(sample)

    def _import_file(self, path: str) -> None:
        if self._pipeline_worker and self._pipeline_worker.isRunning():
            QMessageBox.information(self, "正在处理", "请等待当前处理完成或取消后再导入。")
            return

        self.player.stop()
        self.player.clear()
        self.waveform.clear_data()
        self._result = None
        self._stats = None
        self._span = None
        self._stream_index = None
        self._media_path = path
        self.act_export.setEnabled(False)

        self.lbl_file.setText(os.path.basename(path))
        self.lbl_file.setToolTip(path)
        self.lbl_media.setText("正在探测…")
        self.lbl_stats.setText("—")
        self.waveform.set_trim(None, None)
        self._log(f"导入：{path}")

        if self._analyze_worker and self._analyze_worker.isRunning():
            self._analyze_worker.terminate()
            self._analyze_worker.wait(2000)

        self._analyze_worker = AnalyzeWorker(path, None, self)
        self._analyze_worker.progressed.connect(self.lbl_status.setText)
        self._analyze_worker.analyzed.connect(self._on_analyzed)
        self._analyze_worker.failed.connect(self._on_analyze_failed)
        self._analyze_worker.start()

    def _on_analyze_failed(self, message: str) -> None:
        self.lbl_media.setText("无法读取")
        self.lbl_stats.setText("—")
        self._log(f"[错误] {message}")
        QMessageBox.warning(self, "导入失败", message)
        self._update_run_enabled()

    def _on_analyzed(self, info, data, stats, span) -> None:
        self._info = info
        self._stats = stats
        self._span = span
        if self._stream_index is None:
            self._stream_index = info.default_audio_index

        kind = "视频（只取音轨，输出音频）" if info.is_video else "音频"
        lines = [
            f"类型：{kind}    容器：{info.container}    时长：{_fmt_time(info.duration)}",
            f"解码：{info.decoder}    采样率：{data.sample_rate} Hz    声道：{data.channels}",
        ]
        if info.is_video and info.video_streams:
            v = info.video_streams[0]
            fps = f"{v.fps:.2f} fps" if v.fps else "帧率未知"
            lines.append(f"画面（不处理）：{v.codec} {v.width}×{v.height} {fps}")
        for note in info.notes:
            lines.append(f"· {note}")
        self.lbl_media.setText("\n".join(lines))

        self.lbl_stats.setText(
            f"电平：{stats.describe()}\n"
            f"帧电平 p10/p50/p90：{stats.frame_db_p10:.1f} / {stats.frame_db_p50:.1f} / "
            f"{stats.frame_db_p90:.1f} dBFS\n"
            f"判断：{stats.verdict()}"
        )

        self.cmb_stream.blockSignals(True)
        self.cmb_stream.clear()
        for s in info.audio_streams:
            self.cmb_stream.addItem(s.label(), s.index)
        idx = self.cmb_stream.findData(self._stream_index)
        if idx >= 0:
            self.cmb_stream.setCurrentIndex(idx)
        self.cmb_stream.blockSignals(False)
        self.row_stream.setVisible(len(info.audio_streams) > 1)
        if len(info.audio_streams) > 1:
            self._log(f"检测到 {len(info.audio_streams)} 条音轨，可在「音轨」下拉中选择。")

        if span is not None and (span.start_frame > 0 or span.end_frame < len(data.samples)):
            trimmed = span.total_s - span.active_s
            self.lbl_stats.setText(
                self.lbl_stats.text()
                + f"\n静音检测：可用内容 {span.start_s:.2f}s–{span.end_s:.2f}s"
                  f"（可裁掉 {trimmed:.2f}s）"
            )
            self.waveform.set_trim(span.start_s, span.end_s)
        else:
            self.waveform.set_trim(None, None)

        # 试听「原音」：处理前也能听
        self.player.load(data, data)
        self._set_transport_enabled(True)
        self.waveform.set_data(data.samples, None, data.sample_rate)
        self.waveform.set_duration(data.duration)
        self._log(
            f"就绪：{_fmt_time(data.duration)}，{data.sample_rate} Hz，{data.channels} 声道。"
        )
        self._update_run_enabled()

    def _on_stream_changed(self, _index: int) -> None:
        value = self.cmb_stream.currentData()
        if value is None or self._media_path is None:
            return
        if value == self._stream_index:
            return
        self._stream_index = int(value)
        self._log(f"切换到音轨 {value}，重新读取…")
        path = self._media_path
        self._import_file(path)

    # ==================================================================
    # 处理
    # ==================================================================
    def _update_run_enabled(self) -> None:
        spec = self._selected_engine()
        busy = bool(self._pipeline_worker and self._pipeline_worker.isRunning())
        can = (
            self._media_path is not None
            and spec is not None
            and modelstore.engine_ready(spec, self.catalog)
            and not busy
        )
        self.btn_run.setEnabled(can)
        if spec is not None and not modelstore.engine_ready(spec, self.catalog) and self._media_path:
            self.lbl_status.setText(f"模型「{spec.name}」尚未下载")
        elif can:
            self.lbl_status.setText("就绪")

    def _collect_options(self) -> pipeline.PipelineOptions:
        spec = self._selected_engine()
        ext = self.cmb_format.currentData() or ".wav"
        out_path = None
        if self.chk_autoexport.isChecked() and self._media_path:
            out_path = pipeline.default_output_path(
                self._media_path, spec.id, self.project_root, ext
            )
        return pipeline.PipelineOptions(
            engine_id=spec.id,
            engine=pipeline.EngineParams(
                num_threads=self.spn_threads.value(),
                attenuation_limit_db=float(self.sld_atten.value()),
            ),
            trim_silence=self.chk_trim.isChecked(),
            normalize=self.chk_norm.isChecked(),
            normalize_target_dbfs=float(self.spn_norm_target.value()),
            normalize_mode=self.cmb_norm_mode.currentData() or "peak",
            keep_channels=self.chk_stereo.isChecked(),
            output_rate=None,
            output_ext=ext,
            output_path=out_path,
        )

    def _start_process(self) -> None:
        if not self._media_path:
            return
        spec = self._selected_engine()
        if spec is None:
            return
        self.player.pause()
        options = self._collect_options()

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self._log("—" * 30)
        self._log(f"开始处理：{spec.name}")

        self._pipeline_worker = PipelineWorker(
            self._media_path, self.catalog, options, self._stream_index, self
        )
        self._pipeline_worker.progressed.connect(self._on_progress)
        self._pipeline_worker.succeeded.connect(self._on_processed)
        self._pipeline_worker.failed.connect(self._on_failed)
        self._pipeline_worker.cancelled.connect(self._on_cancelled)
        self._pipeline_worker.start()

    def _cancel_process(self) -> None:
        if self._pipeline_worker and self._pipeline_worker.isRunning():
            self._log("正在取消…")
            self._pipeline_worker.cancel()
            self.btn_cancel.setEnabled(False)

    def _on_progress(self, frac: float, message: str) -> None:
        self.progress.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        if message:
            self.lbl_status.setText(message)

    def _on_cancelled(self) -> None:
        self._log("已取消。")
        self.lbl_status.setText("已取消")
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(0)
        self._update_run_enabled()

    def _on_failed(self, message: str) -> None:
        self._log(f"[错误] {message}")
        self.lbl_status.setText("处理失败")
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(0)
        self._update_run_enabled()
        QMessageBox.critical(self, "处理失败", message)

    def _on_processed(self, result: pipeline.PipelineResult) -> None:
        self._result = result
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(1000)
        self.lbl_status.setText("处理完成")

        self._log(
            f"完成：{result.engine_name}  RTF {result.rtf:.3f}  耗时 {result.elapsed:.2f}s"
        )
        self._log(f"  原音：{result.source_stats.describe()}")
        self._log(f"  处理后：{result.enhanced_stats.describe()}")
        if result.output_path and os.path.isfile(result.output_path):
            self._log(f"  已导出：{result.output_path}")

        self.player.load(result.source, result.enhanced)
        self.waveform.set_data(result.source.samples, result.enhanced.samples,
                               result.source.sample_rate)
        self.waveform.set_duration(self.player.duration)
        self._set_transport_enabled(True)
        self.act_export.setEnabled(True)
        self.btn_enh.setChecked(True)      # 默认听「处理后」
        self._update_run_enabled()

    def _set_transport_enabled(self, on: bool) -> None:
        for w in (self.btn_play, self.btn_stop, self.btn_src, self.btn_enh, self.sld_pos):
            w.setEnabled(on)

    # ==================================================================
    # 播放控制
    # ==================================================================
    def _toggle_play(self) -> None:
        if not self.player.has_audio:
            return
        if self.player.playing:
            self.player.pause()
        else:
            self.player.play()

    def _seek(self, seconds: float) -> None:
        self.player.seek(seconds)

    def _on_slider_moved(self, value: int) -> None:
        if not self.player.has_audio:
            return
        seconds = self.player.duration * value / 1000.0
        self.player.seek(seconds)

    def _on_position(self, pos: float, duration: float) -> None:
        self.lbl_time.setText(f"{_fmt_time(pos)} / {_fmt_time(duration)}")
        self.waveform.set_position(pos)
        self.waveform.set_duration(duration)
        if not self.sld_pos.isSliderDown() and duration > 0:
            self.sld_pos.blockSignals(True)
            self.sld_pos.setValue(int(min(1000, pos / duration * 1000)))
            self.sld_pos.blockSignals(False)

    def _on_player_state(self, playing: bool) -> None:
        self.btn_play.setText("⏸ 暂停" if playing else "▶ 播放")

    # ==================================================================
    # 下载与导出
    # ==================================================================
    def _download_selected(self) -> None:
        spec = self._selected_engine()
        if spec is None or not spec.files:
            return
        self._download_files(spec.files, spec.name)

    def _download_files(self, files, label: str) -> None:
        if self._download_worker and self._download_worker.isRunning():
            self._log("[提示] 已有下载任务在执行。")
            return
        self._log(f"开始下载：{label}")
        self.btn_download.setEnabled(False)
        self._download_worker = DownloadWorker(files, self.catalog, self)
        self._download_worker.progressed.connect(self._on_progress)
        self._download_worker.log.connect(self._log)
        self._download_worker.finished_all.connect(self._on_download_done)
        self._download_worker.start()

    def _on_download_done(self, ok: bool, message: str) -> None:
        if ok:
            self._log("下载完成。")
            self.lbl_status.setText("下载完成")
        else:
            self._log(f"[下载失败] {message}")
            self.lbl_status.setText("下载失败")
            QMessageBox.warning(
                self,
                "下载失败",
                message
                + "\n\n如果 github.com 连不通，可稍后重试——本机网络对 GitHub 时通时断，"
                  "下载器会优先走 hf-mirror 镜像。",
            )
        self.progress.setValue(0)
        self._refresh_engine_states()

    def _export_as(self) -> None:
        if self._result is None:
            return
        stem = os.path.splitext(os.path.basename(self._result.source_path))[0]
        default = os.path.join(
            self.project_root, "out", f"{stem}_{self._result.engine_id}.wav"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "导出增强后的音频", default,
            "WAV 24-bit (*.wav);;FLAC (*.flac);;MP3 (*.mp3)",
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in media_io.OUTPUT_SUBTYPES:
            path += ".wav"
        try:
            media_io.write_audio(path, self._result.enhanced.samples,
                                 self._result.enhanced.sample_rate)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", f"{type(exc).__name__}: {exc}")
            return
        self._log(f"已导出：{path}")
        self.lbl_status.setText("已导出")

    # ==================================================================
    # 杂项
    # ==================================================================
    def _log(self, message: str) -> None:
        self.txt_log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message}")

    def closeEvent(self, event):  # noqa: N802
        self.player.shutdown()
        for worker in (self._analyze_worker, self._pipeline_worker, self._download_worker):
            if worker is not None and worker.isRunning():
                if hasattr(worker, "cancel"):
                    worker.cancel()
                worker.wait(3000)
        super().closeEvent(event)
