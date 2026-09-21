"""UI 冒烟测试：无头模式下跑通「导入 → 处理 → A/B → 导出」全链路。

用 Qt 的 offscreen 平台，不弹窗口，可在命令行/CI 里跑。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\smoke_ui.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "speech_with_noise.wav")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from core import media_io  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from ui.player import ENHANCED, SOURCE  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def wait_until(app, cond, timeout_s: float = 180.0, label: str = "") -> bool:
    """转事件循环直到条件成立或超时。"""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    print(f"  [超时] 等待 {label or '条件'} 超过 {timeout_s:.0f}s")
    return False


def main() -> int:
    if not os.path.isfile(SAMPLE):
        print(f"[FAIL] 找不到测试音频：{SAMPLE}")
        return 2

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(ROOT)
    win.show()

    print("1. 启动与初始状态")
    check("窗口标题正确", "AudioEnhancer_zk21" in win.windowTitle())
    n_engines = win.list_engines.count()
    check("模型列表已填充", n_engines >= 8, f"{n_engines} 项")
    check("默认选中 DPDFNet2", win._selected_engine().id == "dpdfnet2",
          win._selected_engine().id)
    check("未导入文件时「开始处理」禁用", not win.btn_run.isEnabled())
    check("未处理时「导出」禁用", not win.act_export.isEnabled())

    print("\n2. 导入（走后台分析线程）")
    win._import_file(SAMPLE)
    ok = wait_until(app, lambda: win._stats is not None and win._info is not None,
                    timeout_s=60, label="媒体分析")
    check("分析完成", ok)
    if not ok:
        return _finish(app, win)

    check("判定为音频", win._info.kind == "audio", win._info.kind)
    check("采样率识别正确", win._info.audio_streams[0].sample_rate == 16000)
    check("电平统计有值", win._stats.rms > 0)
    check("「开始处理」已启用", win.btn_run.isEnabled())
    check("播放器已载入原音", win.player.has_audio)
    check("波形已载入数据", win.waveform._src is not None)

    print("\n3. 试听原音（offscreen 下无音频设备也应优雅降级）")
    win.btn_src.setChecked(True)
    win._toggle_play()
    app.processEvents()
    check("切换原音后 active == source", win.player.active == SOURCE)
    win._toggle_play()   # 暂停/结束

    print("\n4. 处理（dpdfnet2，开裁剪 + 归一化）")
    win.chk_trim.setChecked(True)
    win.chk_norm.setChecked(True)
    win._start_process()
    check("处理中「取消」可用", win.btn_cancel.isEnabled())

    ok = wait_until(app, lambda: win._result is not None, timeout_s=300, label="处理完成")
    check("处理完成且产出结果", ok)
    if not ok:
        return _finish(app, win)

    r = win._result
    check("结果含原音与处理后", r.source.frames > 0 and r.enhanced.frames > 0)
    check("长度对齐", len(r.source.samples) == len(r.enhanced.samples),
          f"{len(r.source.samples)} vs {len(r.enhanced.samples)}")
    check("音量合理未削顶", r.enhanced_stats.peak <= 1.0,
          f"峰值 {r.enhanced_stats.peak:.4f}")
    check("自动导出文件已生成", bool(r.output_path) and os.path.isfile(r.output_path),
          r.output_path or "无")

    print("\n5. A/B 与波形")
    check("播放器已载入两条轨道", set(win.player._tracks) == {SOURCE, ENHANCED})
    check("波形含两条包络", win.waveform._src is not None and win.waveform._enh is not None)
    check("默认切到「处理后」", win.player.active == ENHANCED, win.player.active)
    win.btn_src.setChecked(True)
    app.processEvents()
    check("切回原音生效", win.player.active == SOURCE)

    win._seek(1.0)
    app.processEvents()
    check("拖动定位生效", 0.9 < win.player._pos / win.player._rate < 1.1,
          f"{win.player._pos / win.player._rate:.3f}s")

    gm_before = win.player._match_gains[ENHANCED]
    check("增益匹配算出了补偿值", gm_before > 0)
    win.chk_gainmatch.setChecked(False)
    app.processEvents()
    check("关闭增益匹配后补偿为 1.0", win.player._match_gains[ENHANCED] == 1.0)
    win.chk_gainmatch.setChecked(True)
    app.processEvents()

    print("\n6. 导出（换格式）")
    win.act_export.setEnabled(True)
    from PyQt5.QtWidgets import QFileDialog  # noqa: PLC0415

    out = os.path.join(ROOT, "out", "_ui_smoke_export.flac")
    orig = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out, ""))
    try:
        win._export_as()
    finally:
        QFileDialog.getSaveFileName = orig
    app.processEvents()
    check("FLAC 导出成功", os.path.isfile(out),
          f"{os.path.getsize(out):,} B" if os.path.isfile(out) else "缺失")
    if os.path.isfile(out):
        info = media_io.probe(out)
        check("导出文件可回读", info.duration > 0, f"{info.duration:.2f}s")

    print("\n7. 模型切换与状态刷新")
    win._refresh_engine_states()
    check("GTCRN 显示为已就绪",
          "已就绪" in win.list_engines.item(1).text(), win.list_engines.item(1).text().split("\n")[0])
    for row in range(win.list_engines.count()):
        win.list_engines.setCurrentRow(row)
        app.processEvents()
        if win._selected_engine().id == "dsp_baseline":
            check("DSP 基线不需要下载", win.btn_download.isEnabled() is False)
    win._refresh_engine_states()

    print("\n8. 从界面下载模型（删掉最小那个再让它下回来）")
    import hashlib
    import shutil

    from core import modelstore
    cat = win.catalog
    spec = cat.engine("gtcrn")
    target = cat.resolve(spec.files[0].path)
    want = spec.files[0].sources[0].sha256
    backup = target + ".smokebak"
    shutil.move(target, backup)
    try:
        win._refresh_engine_states()
        row = next(r for r in range(win.list_engines.count())
                   if win.list_engines.item(r).data(0x0100) == "gtcrn")
        win.list_engines.setCurrentRow(row)
        app.processEvents()
        check("缺失时状态变为「需下载」",
              "需下载" in win.list_engines.item(row).text())
        check("下载按钮可用", win.btn_download.isEnabled())

        win._download_selected()
        ok = wait_until(
            app,
            lambda: os.path.isfile(target)
            and hashlib.sha256(open(target, "rb").read()).hexdigest() == want
            and win._download_worker is not None and not win._download_worker.isRunning(),
            timeout_s=180, label="模型下载",
        )
        check("界面下载完成且校验通过", ok)
        check("下载后状态回到「已就绪」",
              "已就绪" in win.list_engines.item(row).text())
    finally:
        if os.path.isfile(backup):
            if os.path.isfile(target):
                os.remove(target)
            shutil.move(backup, target)

    return _finish(app, win)


def _finish(app, win) -> int:
    win.close()
    app.processEvents()
    print()
    if failures:
        print(f"UI 冒烟测试结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("UI 冒烟测试结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
