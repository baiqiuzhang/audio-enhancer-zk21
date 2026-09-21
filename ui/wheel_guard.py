"""让「可滚动页面上的选择类控件」不再被滚轮误改。

问题：Qt 的默认行为是——只要鼠标停在 `QComboBox` / `QAbstractSpinBox`（数字框）/ `QSlider`
上，**滚轮就会改它的值**，哪怕这个控件从来没被点过。而左侧调参栏本身放在滚动区里，
于是"往下滚页面"经常变成"改了某个参数"，而且改完还没有任何提示。

规则（本项目采用）：
- **没有焦点**（最常见）：滚轮交给外层滚动区，页面照常滚动，控件值不变；
- **有焦点**（用户刚点过它，意图明确）：保持 Qt 原生行为，可用滚轮微调；
- 找不到外层滚动区时：**吞掉**滚轮事件而不改值——宁可这一下不滚动，也不要误改参数。

用法（主窗口构造末尾调用一次即可）：

    from .wheel_guard import install_wheel_guard
    install_wheel_guard(self)
"""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QWidget,
)

# 会被滚轮误改的控件类型
GUARDED = (QComboBox, QAbstractSpinBox, QAbstractSlider)


def _enclosing_scroll_area(w: QWidget) -> QAbstractScrollArea | None:
    p = w.parentWidget()
    while p is not None:
        if isinstance(p, QAbstractScrollArea):
            return p
        p = p.parentWidget()
    return None


class WheelGuard(QObject):
    """装在受护控件上的事件过滤器。见模块文档。"""

    def eventFilter(self, obj, event):  # noqa: N802 (Qt 命名)
        if event.type() != QEvent.Wheel:
            return False
        try:
            if obj.hasFocus():
                return False                      # 有焦点：交给 Qt 原生处理
            area = _enclosing_scroll_area(obj) if isinstance(obj, QWidget) else None
            if area is not None:
                self._scroll(area, event)
            return True                           # 一律吞掉，绝不落到控件自身的滚轮处理
        except Exception:
            return True                           # 出错也只吞掉，不影响应用

    @staticmethod
    def _scroll(area: QAbstractScrollArea, event) -> None:
        """把滚轮转成滚动区的滚动。

        **不要**把同一个事件对象再 `sendEvent` 给滚动区视口：实测那样会让进程在事件
        派发过程中直接崩溃（空的下拉框上必崩，非空的也崩；而摘掉护栏则一切正常，
        A/B 对照过）。这里改为直接驱动滚动条——效果等价（按滚轮的步进量滚动），
        但不涉及事件的二次派发。
        """
        sb = area.verticalScrollBar()
        if sb is None or sb.maximum() <= sb.minimum():
            return
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y() * 8    # 触摸板只给 pixelDelta 的情况
        if delta:
            sb.setValue(sb.value() - delta)


def install_wheel_guard(root: QWidget) -> WheelGuard:
    """给 `root` 下所有受护控件装上滚轮护栏（幂等）。返回过滤器本身（需保引用）。"""
    guard = getattr(root, "_wheel_guard", None)
    if guard is None:
        guard = WheelGuard(root)
        root._wheel_guard = guard          # 持有引用，否则被 GC 后过滤器失效
    for cls in GUARDED:
        for w in root.findChildren(cls):
            w.installEventFilter(guard)
    return guard
