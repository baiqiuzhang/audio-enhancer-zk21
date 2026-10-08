"""界面渲染与布局检查。

做两件事：
1. 截取窗口图像（供目视确认深色主题真的渲染出来了）；
2. **程序化检测兄弟控件重叠与横向裁切** —— 前者正是「模型说明文字遮挡模型列表」
   的判据；后者是「左侧内容比视口宽、而横向滚动条是关的」导致的右侧被切掉。

关键：**不能用 offscreen 平台**。实测该平台在本机字体库为 0 个字体族，
所有文字都渲染不出来，截图无法反映真实观感。这里改用真实的 windows 平台，
并给窗口加 WA_DontShowOnScreen —— 布局与字体都是真的，但窗口不会出现在屏幕上。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\render_ui.py
"""

from __future__ import annotations

import os
import sys
import time

# Windows 中文控制台的默认编码是 GBK，断言名里的 ✓ 会让 print 抛
# UnicodeEncodeError 并让整个脚本以非 0 状态退出（测试其实全过了）。
# 这里尽量把两个流切到 UTF-8；切不动也不影响测试逻辑。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtCore import QRect, Qt  # noqa: E402
from PyQt5.QtGui import QColor  # noqa: E402
from PyQt5.QtWidgets import QApplication, QGroupBox, QScrollArea, QWidget  # noqa: E402

from ui.main_window import MainWindow  # noqa: E402
from ui.player import ENHANCED  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "speech_with_noise.wav")
OUT = os.path.join(ROOT, "out")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def wait_until(app, cond, timeout_s: float, label: str) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    print(f"  [超时] {label}")
    return False


def visible_children(container: QWidget) -> list[QWidget]:
    """直接子控件里可见、且不是滚动区内部实现的那些。"""
    kids = []
    for child in container.children():
        if not isinstance(child, QWidget):
            continue
        if not child.isVisible():
            continue
        if child.objectName().startswith("qt_"):
            continue
        kids.append(child)
    return kids


def find_overlaps(widget: QWidget, path: str = "") -> list[str]:
    """递归检查同一父控件下的兄弟控件是否两两重叠。"""
    problems: list[str] = []
    kids = visible_children(widget)

    for i in range(len(kids)):
        for j in range(i + 1, len(kids)):
            a, b = kids[i], kids[j]
            if a.objectName() == "engineNote" or b.objectName() == "engineNote":
                pass
            ra, rb = a.geometry(), b.geometry()
            if ra.isValid() and rb.isValid() and ra.intersects(rb):
                # 允许"子内容型"重叠：例如滚动区与其 viewport 的关系由 Qt 管理
                problems.append(
                    f"{path}/{type(widget).__name__}: "
                    f"{type(a).__name__}({a.objectName() or '-'}) 与 "
                    f"{type(b).__name__}({b.objectName() or '-'}) 重叠  "
                    f"{ra.getRect()} ∩ {rb.getRect()}"
                )

    for child in kids:
        if isinstance(child, QScrollArea):
            vp = child.widget()
            if vp is not None:
                problems += find_overlaps(vp, f"{path}/ScrollArea")
        elif isinstance(child, QGroupBox):
            problems += find_overlaps(child, f"{path}/{child.title()}")
        else:
            problems += find_overlaps(child, path)
    return problems


def save_zooms(win, pixmap, out_dir: str) -> None:
    """把关键区域放大另存，便于逐像素核对细节。

    整窗截图缩小看会漏掉「单选被画成方块」「条目被切一半」这类问题——
    它们在 1340px 宽的图上几乎看不出来。裁剪区域按控件的实际位置算，
    而不是写死像素坐标（写死的话界面一改就框不准了）。
    """
    from PyQt5.QtCore import QPoint, Qt  # noqa: PLC0415

    img = pixmap.toImage()

    def around(widget, pad_x=14, pad_y=12):
        """控件在窗口坐标下的外扩矩形。"""
        top_left = widget.mapTo(win, QPoint(0, 0))
        return (
            max(0, top_left.x() - pad_x),
            max(0, top_left.y() - pad_y),
            min(img.width(), top_left.x() + widget.width() + pad_x),
            min(img.height(), top_left.y() + widget.height() + pad_y),
        )

    crops = [
        # 单选/复选与滑块：确认指示器形状与勾选标记
        ("ab_controls", around(win.btn_src, pad_x=12, pad_y=46), 3.0),
        # 模型列表：确认状态徽标、选中态、说明框不再压住列表
        ("list", around(win.list_engines, pad_x=8, pad_y=8), 2.0),
        # 状态栏
        ("status", (0, max(0, img.height() - 30), min(760, img.width()), img.height()), 3.0),
    ]
    for name, (x0, y0, x1, y1), scale in crops:
        x1, y1 = min(img.width(), x1), min(img.height(), y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        sub = img.copy(x0, y0, x1 - x0, y1 - y0)
        big = sub.scaled(int(sub.width() * scale), int(sub.height() * scale),
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
        path = os.path.join(out_dir, f"_zoom_{name}.png")
        big.save(path)
        print(f"  已保存 out/_zoom_{name}.png  {big.width()}×{big.height()}")


def indicator_rect(widget):
    """用 QStyle 取指示器的真实矩形。

    别自己猜像素位置：实测指示器落在 x≈1..11，而按"控件左 24px 的中点"去采样
    会落在 x=12（空处），于是把画得好好的圆环误判成"没画出来"。
    """
    from PyQt5.QtWidgets import QRadioButton, QStyle, QStyleOptionButton  # noqa: PLC0415

    opt = QStyleOptionButton()
    if hasattr(widget, "initStyleOption"):
        widget.initStyleOption(opt)
    else:
        opt.initFrom(widget)
    element = (QStyle.SE_RadioButtonIndicator if isinstance(widget, QRadioButton)
               else QStyle.SE_CheckBoxIndicator)
    return widget.style().subElementRect(element, opt, widget)


def _bg_color(img) -> QColor:
    """取整张抓图里出现最多的颜色作为背景。

    不要用"控件最右侧那个像素"当背景参照：那里的文字笔画会让参照色变成浅色，
    于是深色的指示器方框整个被判成"墨水"。用众数颜色才稳。
    """
    counts: dict[str, int] = {}
    for y in range(0, img.height(), 1):
        for x in range(0, img.width(), 1):
            name = img.pixelColor(x, y).name()
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return QColor("#000000")
    return QColor(max(counts.items(), key=lambda kv: kv[1])[0])


def _rect(img, widget) -> QRect:
    r = indicator_rect(widget)
    if r.isNull() or r.width() <= 0:
        r = QRect(0, 0, min(18, img.width()), img.height())
    return QRect(max(0, r.left()), max(0, r.top()),
                 min(r.width(), img.width() - max(0, r.left())),
                 min(r.height(), img.height() - max(0, r.top())))


def indicator_ink(widget) -> tuple[int, str]:
    """统计指示器矩形里"明显不是背景色"的像素数。

    适合判断**圆环是否被画出来**（圆环在控件背景之上）。注意它无法判断复选框
    的勾：Fusion 给复选框画的是填充底色的方框，整框都算"非背景"，勾与不勾同值。
    要判断勾选标记请用 indicator_diff。
    """
    img = widget.grab().toImage()
    if img.isNull() or img.width() < 4:
        return 0, "#000000"
    bg = _bg_color(img)
    r = _rect(img, widget)
    n = 0
    for y in range(r.top(), r.bottom() + 1):
        for x in range(r.left(), r.right() + 1):
            c = img.pixelColor(x, y)
            if (abs(c.red() - bg.red()) + abs(c.green() - bg.green())
                    + abs(c.blue() - bg.blue())) > 40:
                n += 1
    return n, bg.name()


def indicator_center(widget) -> str:
    """指示器矩形中心点的颜色。

    「选中/未选中是否看得**出来**」的判据之一：选中时中心被圆点填上，
    未选中时中心应仍是背景色。注意它只适用于单选（圆环+圆点）；
    复选框的 ✓ 是角上的斜线笔画，不在中心，请用 indicator_diff。
    """
    img = widget.grab().toImage()
    if img.isNull() or img.width() < 4:
        return "#000000"
    r = _rect(img, widget)
    cx = min(img.width() - 1, max(0, r.center().x()))
    cy = min(img.height() - 1, max(0, r.center().y()))
    return img.pixelColor(cx, cy).name()


def indicator_diff(a, b) -> tuple[int, int]:
    """逐像素比较两个控件指示器区域的差异。

    这是判断「勾上到底看不看得出来」最稳的度量，因为它**不含任何假设**：
    只要勾选标记存在，已勾选与未勾选的指示器图像就必然不同；若标记被抹掉，
    两者会完全一致。前面试过的"中心点颜色""接近前景色的像素数"都不稳——
    Fusion 的 ✓ 是角上的斜线笔画、不在中心，而且它的方框是填充底色绘制的。
    """
    ia, ib = a.grab().toImage(), b.grab().toImage()
    if ia.isNull() or ib.isNull():
        return 0, 0
    ra, rb = _rect(ia, a), _rect(ib, b)
    w, h = min(ra.width(), rb.width()), min(ra.height(), rb.height())
    diff = 0
    for y in range(h):
        for x in range(w):
            ca = ia.pixelColor(ra.left() + x, ra.top() + y)
            cb = ib.pixelColor(rb.left() + x, rb.top() + y)
            if (abs(ca.red() - cb.red()) + abs(ca.green() - cb.green())
                    + abs(ca.blue() - cb.blue())) > 40:
                diff += 1
    return diff, w * h


def main() -> int:
    if not os.path.isfile(SAMPLE):
        print(f"[FAIL] 找不到测试音频：{SAMPLE}")
        return 2
    os.makedirs(OUT, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(ROOT)
    # 布局与字体都按真实平台走，但窗口不显示到屏幕上
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.show()
    app.processEvents()

    from PyQt5.QtGui import QFontDatabase, QPalette  # noqa: PLC0415
    from PyQt5.QtWidgets import QStyleFactory  # noqa: PLC0415

    print("1. 主题与字体是否生效")
    n_fonts = len(QFontDatabase().families())
    check("平台字体库可用（否则截图会没有文字）", n_fonts > 0, f"{n_fonts} 个字体族")
    check("已设置样式表", bool(app.styleSheet()), f"{len(app.styleSheet())} 字符")

    # 设了样式表之后 app.style() 会被包成 QStyleSheetStyle，无法直接断言底层是 Fusion；
    # 改为断言 Fusion「可用」，再直接检查调色板是否真的是深色（那才是用户看得见的效果）。
    keys = [k.lower() for k in QStyleFactory.keys()]
    check("Fusion 样式可用", "fusion" in keys, ", ".join(QStyleFactory.keys()))

    pal = app.palette()
    for group, gname in ((QPalette.Active, "Active"), (QPalette.Inactive, "Inactive")):
        w = pal.color(group, QPalette.Window)
        check(f"调色板 {gname} 组为深色", w.lightness() < 60,
              f"Window={w.name()} 亮度={w.lightness()}")
    for group, gname in ((QPalette.Active, "Active"), (QPalette.Inactive, "Inactive")):
        t = pal.color(group, QPalette.WindowText)
        check(f"调色板 {gname} 组文字为浅色", t.lightness() > 140,
              f"WindowText={t.name()} 亮度={t.lightness()}")

    check("窗口图标已加载", not win.windowIcon().isNull())
    check("说明框为固定高度", win.note_view.height() == 80, f"{win.note_view.height()} px")

    from PyQt5.QtWidgets import QAbstractItemView  # noqa: PLC0415

    check("模型列表按整项滚动（不会把条目切成半截）",
          win.list_engines.verticalScrollMode() == QAbstractItemView.ScrollPerItem,
          f"mode={win.list_engines.verticalScrollMode()}")
    sel = win.list_engines.visualItemRect(win.list_engines.currentItem())
    vp = win.list_engines.viewport().rect()
    check("选中项完整可见", vp.contains(sel) or sel.height() == 0,
          f"选中项 {sel.getRect()} / 视口 {vp.getRect()}")
    check("无需横向滚动（条目不会被左右切）",
          win.list_engines.horizontalScrollBar().maximum() == 0,
          f"横向范围 0..{win.list_engines.horizontalScrollBar().maximum()}")
    print(f"  左侧栏宽度下限（按内容实测）：{win._left_min_width} px")

    print("\n2. 导入素材（让界面进入有内容的状态）")
    win._import_file(SAMPLE)
    ok = wait_until(app, lambda: win._stats is not None, timeout_s=60, label="媒体分析")
    check("分析完成", ok)
    app.processEvents()
    win.grab().save(os.path.join(OUT, "ui_01_ready.png"))
    print("  已保存 out/ui_01_ready.png")

    print("\n3. 各尺寸下的布局重叠检测")
    sizes = [(1340, 900), (1180, 720), (1060, 640), (1000, 560)]
    for w, h in sizes:
        win.resize(w, h)
        app.processEvents()
        time.sleep(0.05)
        app.processEvents()

        problems = find_overlaps(win)
        check(f"{w}×{h} 无兄弟控件重叠", not problems,
              f"{len(problems)} 处" if problems else
              f"列表高 {win.list_engines.height()}px（下限 {win.list_engines.minimumHeight()}px）")
        for p in problems[:3]:
            print(f"        · {p}")

        check(f"{w}×{h} 模型列表未被压到下限以下",
              win.list_engines.height() >= win.list_engines.minimumHeight(),
              f"{win.list_engines.height()} px")

        sa = win.left_scroll
        cw, vw = sa.widget().width(), sa.viewport().width()
        check(f"{w}×{h} 左侧内容未被横向裁切", cw <= vw + 1, f"内容 {cw} / 视口 {vw}")

    win.resize(1000, 560)
    app.processEvents()
    win.grab().save(os.path.join(OUT, "ui_02_small.png"))
    print("  已保存 out/ui_02_small.png（最小尺寸下的滚动效果）")

    print("\n4. 处理后再截一张（含波形、A/B 与日志）")
    win.resize(1340, 900)
    app.processEvents()
    win.chk_trim.setChecked(True)
    win.chk_norm.setChecked(True)
    win._start_process()
    ok = wait_until(app, lambda: win._result is not None, timeout_s=300, label="处理完成")
    check("处理完成", ok)
    if ok:
        win.btn_enh.setChecked(True)
        win._seek(1.0)
        app.processEvents()
        time.sleep(0.05)
        app.processEvents()
        check("播放头已移动", win.player.active == ENHANCED)

        # 指示器是否真的被画出来了（未选中的原音 / 选中的处理后）
        in_src, bg_src = indicator_ink(win.btn_src)
        in_enh, _ = indicator_ink(win.btn_enh)
        check("未选中单选的圆圈已绘制", in_src > 25, f"{in_src} 像素（背景 {bg_src}）")
        check("选中单选的圆点已绘制", in_enh > 25, f"{in_enh} 像素")

        c_src, c_enh = indicator_center(win.btn_src), indicator_center(win.btn_enh)
        check("指示器中心可区分选中态", c_src != c_enh,
              f"未选中中心 {c_src} / 选中中心 {c_enh}")

        # 对照组要用**真的未勾选**的复选框：此时 chk_trim / chk_norm 已被上面
        # 勾上，拿它们对比会得到"逐像素相同"的假结论。这里临时取消 chk_stereo。
        win.chk_stereo.setChecked(False)
        app.processEvents()
        cdiff, ctot = indicator_diff(win.chk_gainmatch, win.chk_stereo)
        win.chk_stereo.setChecked(True)
        app.processEvents()
        check("复选框勾选态与未勾选态外观不同（✓ 确实画出来了）",
              cdiff > 5, f"{cdiff}/{ctot} 像素不同")

        problems = find_overlaps(win)
        check("处理完成态无布局重叠", not problems,
              f"{len(problems)} 处" if problems else "")
        pm = win.grab()
        pm.save(os.path.join(OUT, "ui_03_processed.png"))
        print("  已保存 out/ui_03_processed.png")
        save_zooms(win, pm, OUT)

    win.close()
    app.processEvents()

    print()
    if failures:
        print(f"界面检查结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("界面检查结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
