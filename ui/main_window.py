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

from core import media_io, modelstore, outputs, pipeline
from core.worker import AnalyzeWorker, BatchJob, BatchProcessor, DownloadWorker, PipelineWorker

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

# 试听用预设的搁架增益与高通截止。控件文案与 PipelineOptions 都取这里，避免两处漂移。
SHELF_DB = 3.0
HIGH_PASS_HZ = 80

# 「去闷（保真档）」的实测标定值。依据：真实素材（18 分钟 mp4，300s 起 90s 片段）
# 上 200–500 Hz 比中频高 +2.8~+4.0 dB、2–5 kHz 比中频低 −11~−12 dB。
# 该组合实测把发闷指数从 +16.4 降到 +9.2 dB（改善 7.2 dB），且峰值未削顶。
# 搁架中心 3000 Hz 优于 4000 Hz（改善 +7.21 vs +5.54 dB）——亏空在 2–5 kHz。
MUD_CUT_DB_PRESET = -5.0
SHELF_DB_PRESET = 6.0


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

        # 待处理文件列表（批处理的对象）。当前的「正在看的文件」仍是标量 _media_path 那一组，
        # 列表只是多了一层：选中哪一行，哪一行就回填成当前文件，因此既有的单文件链路
        # （播放器 / 波形 / 导出 / _start_process）完全不用改。
        self._files: list[str] = []
        self._batch: BatchProcessor | None = None
        self._batch_paths: list[str] = []

        # 当前文件解码后的原音（试听变体时要用它做 A/B 的原音轨）
        self._source_audio: media_io.AudioData | None = None
        # 当前文件已有的结果（变体），以及它们所在目录
        self._variants: list = []
        self._variant_out_dir: str = ""
        # 正在试听的变体。它不是"处理结果"，所以单独记，导出时优先用它
        self._audition: media_io.AudioData | None = None

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
        lay.addWidget(self._build_files_box())
        lay.addWidget(self._build_variants_box())
        lay.addWidget(self._build_model_box(), 1)   # 列表吃掉多余的竖向空间
        lay.addWidget(self._build_preset_box())
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

    def _build_files_box(self) -> QWidget:
        box = QGroupBox("待处理文件")
        g = QVBoxLayout(box)
        g.setSpacing(7)

        self.list_files = QListWidget()
        # 多选：批量处理操作的是一批文件，而不是"当前这一个"
        self.list_files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_files.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
        self.list_files.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_files.setTextElideMode(Qt.ElideRight)
        self.list_files.setMinimumHeight(84)
        # 选中哪一行，哪一行就成为"当前文件"（走既有的 _import_file 链路）
        self.list_files.currentItemChanged.connect(self._on_file_row_changed)
        self.list_files.itemSelectionChanged.connect(self._update_run_enabled)
        g.addWidget(self.list_files, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_add_files = QPushButton("添加…")
        self.btn_add_files.clicked.connect(self._choose_files)
        row.addWidget(self.btn_add_files, 2)
        self.btn_remove_files = QPushButton("移除")
        self.btn_remove_files.clicked.connect(self._remove_selected_files)
        row.addWidget(self.btn_remove_files, 1)
        self.btn_clear_files = QPushButton("清空")
        self.btn_clear_files.clicked.connect(self._clear_files)
        row.addWidget(self.btn_clear_files, 1)
        g.addLayout(row)

        # 批处理与单文件是两个入口，**刻意不合并**：_start_process 的同步时序是
        # smoke_ui / render_ui 的测试契约（调用后立刻断言 btn_cancel 可用，并等 _result），
        # 把它塞进异步的 BatchProcessor 会引入"内存不足时空转超时"等新失败面。
        self.btn_batch = QPushButton("批量处理选中文件")
        self.btn_batch.setMinimumHeight(30)
        self.btn_batch.setEnabled(False)
        self.btn_batch.clicked.connect(self._start_batch)
        g.addWidget(self.btn_batch)
        return box

    def _build_variants_box(self) -> QWidget:
        box = QGroupBox("已有结果（变体）")
        g = QVBoxLayout(box)
        g.setSpacing(7)

        self.list_variants = QListWidget()
        self.list_variants.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_variants.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
        self.list_variants.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_variants.setTextElideMode(Qt.ElideRight)
        self.list_variants.setMinimumHeight(72)
        self.list_variants.itemSelectionChanged.connect(self._on_variant_selected)
        g.addWidget(self.list_variants, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_scan_variants = QPushButton("重新扫描")
        self.btn_scan_variants.clicked.connect(self._refresh_variants)
        row.addWidget(self.btn_scan_variants, 1)
        self.lbl_variant_hint = QLabel("选中即试听（会重置播放位置）")
        self.lbl_variant_hint.setObjectName("sectionHint")
        self.lbl_variant_hint.setWordWrap(True)
        self.lbl_variant_hint.setToolTip(
            "变体就是「输入文件所在目录下 AudioEnhancer_zk21/ 里已有的结果」。\n"
            "选中后与原音做 A/B 对比：原音会按该变体记录的裁剪区间切成同一段，\n"
            "避免裁剪造成整段错位。切换变体会把播放位置归零。"
        )
        row.addWidget(self.lbl_variant_hint, 2)
        g.addLayout(row)
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

        self.chk_autoexport = QCheckBox("处理完自动写出结果（输入目录 AudioEnhancer_zk21/）")
        self.chk_autoexport.setToolTip(
            "勾选后按命名规范写到输入文件所在目录的子文件夹里，并按参数指纹复用已有结果。\n"
            "取消勾选则只在内存里出结果，需要落盘时用工具栏的「导出为…」。"
        )
        self.chk_autoexport.setChecked(True)
        gp.addWidget(self.chk_autoexport, r, 0, 1, 2)
        r += 1

        gp.addWidget(QLabel("高通（0 = 关闭）"), r, 0)
        self.spn_highpass = QSpinBox()
        self.spn_highpass.setRange(0, 300)
        self.spn_highpass.setValue(HIGH_PASS_HZ)
        self.spn_highpass.setSuffix(" Hz")
        self.spn_highpass.setToolTip("所有引擎共用的一个前处理阶段；设 0 即关闭。")
        gp.addWidget(self.spn_highpass, r, 1)
        r += 1

        # 听感增强是「修饰」（只改频响与动态，不编造内容）。带宽扩展是「生成」，
        # 性质不同，所以它放在预设区、并且默认关闭。
        self.chk_shelf = QCheckBox("听感增强：高频搁架")
        self.chk_shelf.setToolTip(
            "搁架提升把 2–5 kHz 的「清晰度」区抬起来。只改频响，不造内容。\n"
            "中心频率固定 3000 Hz（实测比 4000 Hz 更能补到 2–3 kHz）。"
        )
        gp.addWidget(self.chk_shelf, r, 0)
        self.spn_shelf_db = QDoubleSpinBox()
        self.spn_shelf_db.setRange(0.0, 12.0)
        self.spn_shelf_db.setValue(SHELF_DB)
        self.spn_shelf_db.setSingleStep(1.0)
        self.spn_shelf_db.setSuffix(" dB")
        gp.addWidget(self.spn_shelf_db, r, 1)
        r += 1

        # 去闷：压掉 200–500 Hz 的「发闷」堆积。实测这是「浑浊」的主因。
        gp.addWidget(QLabel("去闷（200–500 Hz）"), r, 0)
        self.spn_mudcut = QDoubleSpinBox()
        self.spn_mudcut.setRange(-12.0, 0.0)
        self.spn_mudcut.setValue(0.0)
        self.spn_mudcut.setSingleStep(1.0)
        self.spn_mudcut.setSuffix(" dB")
        self.spn_mudcut.setToolTip(
            "在 350 Hz 附近做**钟形**衰减，压掉「发闷」的堆积（0 = 关闭）。\n"
            "用钟形而不是搁架，是为了不动 100 Hz 以下的基频——"
            "否则声音会变薄。实测素材该区比中频高 +2.8~+4.0 dB。"
        )
        gp.addWidget(self.spn_mudcut, r, 1)
        r += 1

        self.chk_comp = QCheckBox("听感增强：温和压缩（AGC）")
        self.chk_comp.setToolTip("把语音段的动态范围压小，小声的部分听得更清。")
        gp.addWidget(self.chk_comp, r, 0, 1, 2)
        r += 1

        gp.setColumnStretch(1, 1)
        return box

    def _build_preset_box(self) -> QWidget:
        box = QGroupBox("预设")
        g = QVBoxLayout(box)
        g.setSpacing(7)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_preset_faithful = QPushButton("识别用（保真）")
        self.btn_preset_demud = QPushButton("去闷（保真档）")
        self.btn_preset_listen = QPushButton("试听用（增强）")
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        for btn in (self.btn_preset_faithful, self.btn_preset_demud,
                    self.btn_preset_listen):
            btn.setCheckable(True)
            btn.setMinimumHeight(30)
            grp.addButton(btn)
            row.addWidget(btn, 1)
        # 刻意不预选任何一档：初始状态等于「自定义」，不改动既有默认行为
        self.btn_preset_faithful.setToolTip(
            "只做降噪 + 高通 + 裁剪 + 归一化，不做带宽扩展、不做听感增强。\n"
            "识别 / 取证用这一档：带宽扩展是生成式的，会编造高频内容。"
        )
        self.btn_preset_demud.setToolTip(
            f"按实测标定压掉「发闷」：去闷 {MUD_CUT_DB_PRESET:+g} dB @ 350 Hz（钟形）"
            f" + 搁架 {SHELF_DB_PRESET:+g} dB @ 3000 Hz。\n"
            "只改频响、不重合成、不编造内容，所以仍然可以用于识别——"
            "但它会改变音色（低频变少、高频变多）。\n"
            "实测把发闷指数从 +16.4 降到 +9.2 dB。不开压缩：压缩改动态，与保真目标冲突。"
        )
        self.btn_preset_listen.setToolTip(
            "在识别用的基础上加搁架 EQ 与温和压缩，并可按需开启带宽扩展。\n"
            "只用于试听。"
        )
        self.btn_preset_faithful.clicked.connect(lambda: self._apply_preset("faithful"))
        self.btn_preset_demud.clicked.connect(lambda: self._apply_preset("demud"))
        self.btn_preset_listen.clicked.connect(lambda: self._apply_preset("listen"))
        g.addLayout(row)

        bwe_row = QHBoxLayout()
        bwe_row.setSpacing(8)
        bwe_row.addWidget(QLabel("带宽扩展"))
        self.cmb_bwe = QComboBox()
        self.cmb_bwe.addItem("关闭", None)
        self._bwe_default_index = 0
        for spec in self.catalog.engines:
            if spec.kind == "bwe":
                self.cmb_bwe.addItem(spec.name, spec.id)
                if self._bwe_default_index == 0:
                    self._bwe_default_index = self.cmb_bwe.count() - 1
                if spec.bwe_default:          # models.json 标了默认档就用它
                    self._bwe_default_index = self.cmb_bwe.count() - 1
        bwe_row.addWidget(self.cmb_bwe, 1)
        g.addLayout(bwe_row)

        self.lbl_bwe_hint = QLabel(
            "带宽扩展是生成式的：会编造高频内容，只用于试听，不要用于识别或取证。"
        )
        self.lbl_bwe_hint.setObjectName("sectionHint")
        self.lbl_bwe_hint.setWordWrap(True)
        g.addWidget(self.lbl_bwe_hint)
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

    def _apply_preset(self, name: str) -> None:
        """预设只负责设置控件状态，**不绕过** _collect_options —— 参数只有一个来源。

        「识别用 / 试听用」这一对语义来自 tests/check_enhance.py 里的三条对照用例：
        识别用 = 纯降噪；试听用 = 再加搁架 EQ + 压缩（+ 可选带宽扩展）。
        """
        self.chk_trim.setChecked(True)
        self.chk_norm.setChecked(True)
        self.spn_highpass.setValue(HIGH_PASS_HZ)
        if name == "faithful":
            self.chk_shelf.setChecked(False)
            self.spn_shelf_db.setValue(SHELF_DB)
            self.spn_mudcut.setValue(0.0)
            self.chk_comp.setChecked(False)
            self.cmb_bwe.setCurrentIndex(0)          # 关闭带宽扩展
            self._log("已套用预设：识别用（保真）—— 不编造高频内容")
        elif name == "demud":
            self.chk_shelf.setChecked(True)
            self.spn_shelf_db.setValue(SHELF_DB_PRESET)
            self.spn_mudcut.setValue(MUD_CUT_DB_PRESET)
            # 刻意不开压缩：压缩改动态，与"保真"目标冲突
            self.chk_comp.setChecked(False)
            self.cmb_bwe.setCurrentIndex(0)
            self._log(
                f"已套用预设：去闷（保真档）—— {MUD_CUT_DB_PRESET:+g} dB @ 350 Hz（钟形）"
                f" + 搁架 {SHELF_DB_PRESET:+g} dB @ 3000 Hz；只改频响，不造内容"
            )
        else:
            self.chk_shelf.setChecked(True)
            self.spn_shelf_db.setValue(SHELF_DB)
            self.spn_mudcut.setValue(0.0)
            self.chk_comp.setChecked(True)
            self.cmb_bwe.setCurrentIndex(self._bwe_default_index)
            self._log(
                f"已套用预设：试听用（增强）—— 带宽扩展「{self.cmb_bwe.currentText()}」"
                "（生成式，会编造高频，仅用于试听）"
            )

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
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if os.path.splitext(path)[1].lower() in media_io.ALL_EXTS:
                    paths.append(path)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self._add_files(paths)       # 拖入多个文件时全部收下，而不是只取第一个

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "导入音频或视频（可多选）", "", FILE_FILTER
        )
        if paths:
            self._add_files(paths)

    def _choose_file(self) -> None:
        """工具栏「导入文件…」的入口。保留这个名字是为了不动既有接线。"""
        self._choose_files()

    # ------------------------------------------------------------------
    # 文件列表
    # ------------------------------------------------------------------
    def _file_row(self, path: str) -> int:
        target = os.path.abspath(path)
        for row in range(self.list_files.count()):
            p = self.list_files.item(row).data(Qt.UserRole)
            if p and os.path.abspath(p) == target:
                return row
        return -1

    def _add_files(self, paths) -> None:
        added = 0
        for raw in paths:
            path = os.path.abspath(raw)
            if os.path.splitext(path)[1].lower() not in media_io.ALL_EXTS:
                continue
            if self._file_row(path) >= 0:
                continue
            self._files.append(path)
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            self.list_files.addItem(item)
            added += 1
        if added:
            self._log(f"加入 {added} 个文件，共 {len(self._files)} 个待处理。")
        # 还没有选中项时自动选中第一个新加入的文件（会触发导入）
        if added and self.list_files.currentRow() < 0:
            self.list_files.setCurrentRow(len(self._files) - added)
        self._update_run_enabled()

    def _sync_file_row(self, path: str) -> None:
        """把列表选中项对齐到给定文件；不在列表里就补进去（导入来源不止「添加…」）。"""
        row = self._file_row(path)
        if row < 0:
            abs_path = os.path.abspath(path)
            self._files.append(abs_path)
            item = QListWidgetItem(os.path.basename(abs_path))
            item.setData(Qt.UserRole, abs_path)
            item.setToolTip(abs_path)
            self.list_files.addItem(item)
            row = self.list_files.count() - 1
        if self.list_files.currentRow() != row:
            # 屏蔽信号：这里只是让选区跟随目标文件，不该反过来再触发一次导入
            self.list_files.blockSignals(True)
            self.list_files.setCurrentRow(row)
            self.list_files.blockSignals(False)

    def _on_file_row_changed(self, current, _previous=None) -> None:
        if current is None:
            return
        path = current.data(Qt.UserRole)
        if not path:
            return
        if self._media_path and os.path.abspath(path) == os.path.abspath(self._media_path):
            return
        self._import_file(path)

    def _selected_files(self) -> list[str]:
        items = self.list_files.selectedItems()
        items.sort(key=self.list_files.row)
        return [p for p in (it.data(Qt.UserRole) for it in items) if p]

    def _remove_selected_files(self) -> None:
        rows = sorted(
            (self.list_files.row(it) for it in self.list_files.selectedItems()), reverse=True
        )
        if not rows:
            return
        removed = set()
        for row in rows:
            item = self.list_files.takeItem(row)
            if item is not None:
                p = item.data(Qt.UserRole)
                if p:
                    removed.add(os.path.abspath(p))
        self._files = [p for p in self._files if os.path.abspath(p) not in removed]
        self._log(f"移除 {len(rows)} 个文件，剩 {len(self._files)} 个。")
        if self._media_path and os.path.abspath(self._media_path) in removed:
            self._reset_current()
        self._update_run_enabled()

    def _clear_files(self) -> None:
        self.list_files.blockSignals(True)
        self.list_files.clear()
        self.list_files.blockSignals(False)
        self._files = []
        self._reset_current()
        self._log("已清空待处理文件。")
        self._update_run_enabled()

    def _reset_current(self) -> None:
        """回到「没有当前文件」的状态（移除 / 清空列表时用）。"""
        self.player.stop()
        self.player.clear()
        self.waveform.clear_data()
        self._result = None
        self._stats = None
        self._span = None
        self._info = None
        self._stream_index = None
        self._media_path = None
        self._source_audio = None
        self._audition = None
        self.act_export.setEnabled(False)
        self.lbl_file.setText("尚未导入文件")
        self.lbl_file.setToolTip("")
        self.lbl_media.setText("—")
        self.lbl_stats.setText("—")
        self.cmb_stream.blockSignals(True)
        self.cmb_stream.clear()
        self.cmb_stream.blockSignals(False)
        self.row_stream.setVisible(False)
        self._set_transport_enabled(False)
        self._refresh_variants()

    # ------------------------------------------------------------------
    # 变体列表（已有结果）— 只做试听，不重新处理
    # ------------------------------------------------------------------
    def _refresh_variants(self) -> None:
        """重新扫描当前文件所在输出目录里的全部已有结果。"""
        self.list_variants.blockSignals(True)
        self.list_variants.clear()
        self._variants = []
        self._variant_out_dir = ""

        path = self._media_path
        if path:
            out_dir, fallback = outputs.output_dir_for(path, self.project_root)
            if fallback:
                self._log(f"[注意] {fallback}")
            self._variant_out_dir = out_dir
            try:
                entries = outputs.scan_variants(out_dir, path)
            except Exception as exc:      # noqa: BLE001 — 扫描失败不该影响主流程
                self._log(f"[注意] 变体扫描失败：{type(exc).__name__}: {exc}")
                entries = []
            self._variants = entries
            for entry in entries:
                item = QListWidgetItem(self._variant_text(entry))
                item.setData(Qt.UserRole, entry)
                item.setToolTip(entry.path_in(out_dir))
                self.list_variants.addItem(item)
            if entries:
                self._log(f"发现 {len(entries)} 个已有结果（{os.path.basename(out_dir)}/）。")
        self.list_variants.blockSignals(False)

    @staticmethod
    def _variant_text(entry) -> str:
        """变体显示文本。

        不用 VariantEntry.describe()：它内部算出的采样率/RTF 从未被使用（只返回 chain_label），
        这里按其原意把指标拼出来。
        """
        label = entry.chain_label or os.path.basename(entry.output)
        stats = entry.stats or {}
        bits = []
        sr = stats.get("sample_rate")
        if sr:
            bits.append(f"{int(sr) // 1000} kHz")
        rtf = stats.get("rtf")
        if rtf:
            bits.append(f"RTF {rtf:.2f}")
        if not entry.from_manifest:
            bits.append("无清单信息")
        return f"{label}    [{' · '.join(bits)}]" if bits else label

    def _selected_variant(self):
        items = self.list_variants.selectedItems()
        return items[0].data(Qt.UserRole) if items else None

    def _source_for_span(self, span) -> media_io.AudioData | None:
        """取与某个变体时间轴对齐的原音。

        变体是**裁剪后**的音频（从裁剪点开始），而缓存的原始音频从 0 秒开始；
        直接对放会整段错位（裁掉 2 秒前导静音就错位 2 秒）。所以按变体记录的
        裁剪区间把原音切成同一段。没有区间信息时退回整段原音。
        """
        data = self._source_audio
        if data is None:
            return None
        if not span:
            return data
        rate = int(data.sample_rate)
        start = max(0, int(float(span[0]) * rate))
        end = min(len(data.samples), int(float(span[1]) * rate))
        if end <= start or start == 0 and end >= len(data.samples):
            return data
        return media_io.AudioData(samples=data.samples[start:end], sample_rate=rate)

    def _on_variant_selected(self) -> None:
        entry = self._selected_variant()
        if entry is None:
            return
        path = entry.path_in(self._variant_out_dir)
        if not os.path.isfile(path):
            self._log(f"[提示] 变体文件已不存在：{os.path.basename(path)}")
            self._refresh_variants()
            return
        source = self._source_for_span(entry.trim_span)
        if source is None:
            self._log("[提示] 原音尚未载入，先导入文件再试听变体。")
            return
        try:
            variant = media_io.decode(path)
        except Exception as exc:      # noqa: BLE001
            self._log(f"[错误] 读取变体失败：{type(exc).__name__}: {exc}")
            return

        self.player.pause()
        # 复用现有双轨播放器：把变体当作「处理后」轨载入。播放器会把两条轨截到较短者，
        # 因此长度不一致也不会出错。
        self.player.load(source, variant)
        self.waveform.set_data(source.samples, variant.samples, source.sample_rate)
        self.waveform.set_duration(self.player.duration)
        self._set_transport_enabled(True)
        self.btn_enh.setChecked(True)
        self._audition = variant
        self.act_export.setEnabled(True)     # 允许把这个变体导出到别处
        self._log(f"试听变体：{os.path.basename(path)}")

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
        self._source_audio = None
        self._audition = None
        self._sync_file_row(path)      # 列表选区跟随（来源可能是拖放 / 示例 / 音轨切换）
        self._refresh_variants()       # 立刻显示这个文件已有的结果

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
        self._source_audio = data          # 试听变体时要用它当 A/B 的原音轨
        self._audition = None
        self.player.load(data, data)
        self._set_transport_enabled(True)
        self.waveform.set_data(data.samples, None, data.sample_rate)
        self.waveform.set_duration(data.duration)
        self._refresh_variants()
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
        busy = self._batch is not None or bool(
            self._pipeline_worker and self._pipeline_worker.isRunning()
        )
        ready = spec is not None and modelstore.engine_ready(spec, self.catalog)
        can = self._media_path is not None and ready and not busy
        self.btn_run.setEnabled(can)

        # 批量处理的对象：列表里勾选了就用勾选的那些，否则用整张列表
        targets = self._selected_files() or list(self._files)
        self.btn_batch.setEnabled(bool(targets) and ready and not busy)
        self.btn_batch.setText(
            f"批量处理 {len(targets)} 个文件" if targets else "批量处理选中文件"
        )

        if spec is not None and not modelstore.engine_ready(spec, self.catalog) and self._media_path:
            self.lbl_status.setText(f"模型「{spec.name}」尚未下载")
        elif can:
            self.lbl_status.setText("就绪")

    def _collect_options(self) -> pipeline.PipelineOptions:
        spec = self._selected_engine()
        ext = self.cmb_format.currentData() or ".wav"
        # 刻意**不设** output_path：那是"导出语义"，会让流水线走显式路径分支，从而绕开
        # 命名规范、参数指纹与同参数复用（见 core/pipeline.py 里 output_path is None 那段）。
        # 自动写出改由 write_output 控制，落点是输入文件所在目录的 AudioEnhancer_zk21/ 子文件夹。
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
            write_output=self.chk_autoexport.isChecked(),
            highpass_hz=float(self.spn_highpass.value()),
            bwe_id=self.cmb_bwe.currentData(),
            shelf_enabled=self.chk_shelf.isChecked(),
            shelf_db=float(self.spn_shelf_db.value()),
            mud_cut_db=float(self.spn_mudcut.value()),
            compressor_enabled=self.chk_comp.isChecked(),
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

    def _start_batch(self) -> None:
        """批量处理列表里勾选的文件（没勾选就是整张列表）。"""
        spec = self._selected_engine()
        if spec is None:
            return
        paths = self._selected_files() or list(self._files)
        if not paths:
            return
        self.player.pause()

        # stream_index 留空：批处理里各文件的音轨布局未必相同，交给流水线各自解析默认音轨。
        # 每个 job 单独取一份 options，避免同一个 dataclass 被多个 worker 线程共享。
        jobs = [
            BatchJob(index=i, media_path=p, options=self._collect_options(), stream_index=None)
            for i, p in enumerate(paths)
        ]
        self._batch_paths = list(paths)

        self._log("—" * 30)
        self._log(f"开始批量处理：{len(paths)} 个文件 · 引擎 {spec.name}")
        self._log("  输出目录：各输入文件所在目录下的 AudioEnhancer_zk21/")

        self.btn_run.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)

        self._batch = BatchProcessor(jobs, self.catalog, 2, self)
        self._batch.job_progress.connect(self._on_batch_progress)
        self._batch.job_done.connect(self._on_batch_done)
        self._batch.job_failed.connect(self._on_batch_failed)
        self._batch.overall.connect(self._on_batch_overall)
        self._batch.log.connect(self._log)
        self._batch.finished_all.connect(self._on_batch_finished)
        # start() 在内存不足时会**同步**发出 finished_all（一个作业都不跑），所以信号必须
        # 先接好；_on_batch_finished 因此只做状态复位，不假设批次真的跑起来了。
        self._batch.start()

    def _on_batch_progress(self, index: int, frac: float, _message: str) -> None:
        self.progress.setValue(int(max(0.0, min(1.0, float(frac))) * 1000))
        total = len(self._batch_paths)
        name = os.path.basename(self._batch_paths[index]) if 0 <= index < total else "?"
        self.lbl_status.setText(f"批处理 {index + 1}/{total}：{name}")

    def _on_batch_overall(self, done: int, total: int) -> None:
        self.lbl_status.setText(f"批处理进度 {done}/{total}")

    def _on_batch_done(self, index: int, result) -> None:
        total = len(self._batch_paths)
        path = self._batch_paths[index] if 0 <= index < total else ""
        name = os.path.basename(path) if path else "?"
        reuse = "（复用已有结果）" if getattr(result, "reused", False) else ""
        self._log(
            f"[{index + 1}/{total}] 完成 {name}{reuse}  "
            f"RTF {result.rtf:.3f}  耗时 {result.elapsed:.2f}s"
        )
        if result.output_path:
            self._log(f"     → {result.output_path}")
        # 只有完成的正好是当前在看的文件才驱动播放器/波形，
        # 否则批量处理 10 个文件就会把播放器重载 10 次
        if path and self._media_path and os.path.abspath(path) == os.path.abspath(self._media_path):
            self._apply_result_to_view(result)

    def _on_batch_failed(self, index: int, message: str) -> None:
        total = len(self._batch_paths)
        path = self._batch_paths[index] if 0 <= index < total else ""
        name = os.path.basename(path) if path else "?"
        # 失败被隔离在单个文件上：只记日志，不弹模态框（模态框会让自动化测试挂住）
        self._log(f"[{index + 1}/{total}] 失败 {name}：{message}")

    def _on_batch_finished(self, cancelled: bool, summary) -> None:
        self._batch = None
        self._batch_paths = []
        self.btn_cancel.setEnabled(False)

        info = summary if isinstance(summary, dict) else {}
        if info.get("refused"):
            # 内存守卫拒绝启动：只报告占用，不弹模态框
            self._log(f"[批处理未启动] {info.get('reason', '资源不足')}")
            self.lbl_status.setText("批处理未启动（内存不足）")
            self.progress.setValue(0)
        elif cancelled or info.get("cancelled"):
            self._log(
                f"批处理已取消：成功 {info.get('done', 0)}，失败 {info.get('failed', 0)}。"
            )
            self.lbl_status.setText("批处理已取消")
            self.progress.setValue(0)
        else:
            self._log(
                f"批处理完成：成功 {info.get('done', 0)}，失败 {info.get('failed', 0)}，"
                f"复用 {info.get('reused', 0)}，并发 {info.get('workers', 0)}，"
                f"耗时 {info.get('elapsed_s', 0)}s。"
            )
            self.lbl_status.setText("批处理完成")
            self.progress.setValue(1000)
        self._refresh_variants()      # 批处理可能给当前文件新增了结果
        self._update_run_enabled()

    def _cancel_process(self) -> None:
        if self._batch is not None:
            self._log("正在取消批处理…（已在处理的文件会尽快收尾）")
            self._batch.cancel()
            self.btn_cancel.setEnabled(False)
            return
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
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(1000)
        self.lbl_status.setText("处理完成")

        self._log(
            f"完成：{result.engine_name}  RTF {result.rtf:.3f}  耗时 {result.elapsed:.2f}s"
        )
        self._log(f"  原音：{result.source_stats.describe()}")
        self._log(f"  处理后：{result.enhanced_stats.describe()}")
        if result.output_path and os.path.isfile(result.output_path):
            self._log(f"  已写出：{result.output_path}")

        self._apply_result_to_view(result)
        self._refresh_variants()      # 刚写出的结果应立刻出现在变体列表里
        self._update_run_enabled()

    def _apply_result_to_view(self, result: pipeline.PipelineResult) -> None:
        """把结果接到当前文件的播放器 / 波形 / 导出上（单文件与批处理共用）。"""
        self._result = result
        self._audition = None      # 新的处理结果取代之前试听的变体
        self.player.load(result.source, result.enhanced)
        self.waveform.set_data(result.source.samples, result.enhanced.samples,
                               result.source.sample_rate)
        self.waveform.set_duration(self.player.duration)
        self._set_transport_enabled(True)
        self.act_export.setEnabled(True)
        self.btn_enh.setChecked(True)      # 默认听「处理后」

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
        # 导出对象：优先当前正在试听的变体（用户正听着它），否则最近一次处理的结果。
        # 没有试听变体时行为与以前完全一致。
        if self._audition is not None:
            payload = self._audition
            engine_id = "variant"
            stem = os.path.splitext(os.path.basename(self._media_path or "output"))[0]
        elif self._result is not None:
            payload = self._result.enhanced
            engine_id = self._result.engine_id
            stem = os.path.splitext(os.path.basename(self._result.source_path))[0]
        else:
            return
        default = os.path.join(self.project_root, "out", f"{stem}_{engine_id}.wav")
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
            media_io.write_audio(path, payload.samples, payload.sample_rate)
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
        if self._batch is not None:
            # BatchProcessor 内部是 daemon 线程 + ThreadPoolExecutor，没有 isRunning()，
            # 发完取消信号就放手（daemon 线程不会阻塞进程退出）
            self._batch.cancel()
            self._batch = None
        for worker in (self._analyze_worker, self._pipeline_worker, self._download_worker):
            if worker is not None and worker.isRunning():
                if hasattr(worker, "cancel"):
                    worker.cancel()
                worker.wait(3000)
        super().closeEvent(event)
