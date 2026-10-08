"""跨模块共用的异常类型。

为什么单独一个模块：`Cancelled` 原本在 `core/dsp.py` 与 `core/engines.py` 里**各定义了一份**，
而 `pipeline`/`worker` 只捕获 `engines.Cancelled`。于是取消发生在 DSP/听感 EQ 阶段时，
`dsp.Cancelled` 会落到 `except Exception` 分支——用户点「取消」看到的是
「处理失败：Cancelled: 已取消」加一个错误弹窗，而不是干净的"已取消"（实测于 2026-10-01
的取消路径排查）。两处同名异常也让"谁能捕获谁"变得难推理。

放在这里还有一层技术原因：`engines.py` 已经在模块级导入 `media_io`，所以 `media_io`
不能反过来在模块级导入 `engines` 来拿这个异常（会成环）。本模块**不导入任何 core 模块**，
谁都可以安全地 import 它。
"""

from __future__ import annotations

__all__ = ["Cancelled"]


class Cancelled(RuntimeError):
    """调用方请求取消（用户点「取消」或批处理被取消）。"""
