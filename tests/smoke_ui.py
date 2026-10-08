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

from PyQt5.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QWheelEvent  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
)

from core import media_io  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from ui.player import ENHANCED, SOURCE  # noqa: E402

failures: list[str] = []


def make_wheel(delta: int = -120) -> QWheelEvent:
    """构造一个滚轮事件（PyQt5 用 QPointF 版的位置参数）。"""
    return QWheelEvent(QPointF(5.0, 5.0), QPointF(5.0, 5.0), QPoint(0, 0), QPoint(0, delta),
                       Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)


def widget_value(w):
    """控件的"值"（下拉框取 index，滑块取整数，数字框取浮点）。"""
    if isinstance(w, QComboBox):
        return w.currentIndex()
    if isinstance(w, QAbstractSlider):
        return int(w.value())
    return float(w.value())


def enclosing_scroll(w):
    p = w.parentWidget()
    while p is not None:
        if isinstance(p, QAbstractScrollArea):
            return p
        p = p.parentWidget()
    return None


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

    print("\n1b. 滚轮护栏：无焦点时滚轮不得改参数（应交给页面滚动）")
    win.list_engines.setFocus()
    app.processEvents()
    targets: list = []
    for cls in (QComboBox, QAbstractSpinBox, QAbstractSlider):
        targets.extend([w for w in win.findChildren(cls) if w.isEnabled()])
    check("确实找到了选择类控件（下拉框/数字框/滑块）", len(targets) >= 8,
          f"{len(targets)} 个")
    before = {id(w): widget_value(w) for w in targets}
    for w in targets:
        if w.hasFocus():
            continue                     # 有焦点时按设计允许微调，这里只测无焦点的情况
        QApplication.sendEvent(w, make_wheel(-120))
        QApplication.sendEvent(w, make_wheel(+120))
    app.processEvents()
    changed = [w for w in targets if widget_value(w) != before[id(w)]]
    check("无焦点时滚轮不改动任何选择类控件",
          not changed,
          f"被改动的：{[f'{type(w).__name__}' for w in changed][:6]}")

    # 反向确认：没有护栏时是会变的（否则上一条可能只是"滚轮事件没送到"）
    from ui.wheel_guard import WheelGuard  # noqa: PLC0415
    guard = win._wheel_guard
    probe = targets[0]
    probe.removeEventFilter(guard)
    v0 = widget_value(probe)
    QApplication.sendEvent(probe, make_wheel(-120))
    app.processEvents()
    check("（反向验证）摘掉护栏后同一个控件确实会被滚轮改动",
          widget_value(probe) != v0, f"{v0} → {widget_value(probe)}")
    probe.installEventFilter(guard)
    probe.setFocusPolicy(probe.focusPolicy())      # 保持原状

    # 有焦点时仍可微调（保住"用户点过它、想用滚轮微调"的用法）
    probe.setFocus()
    app.processEvents()
    if probe.hasFocus():
        v1 = widget_value(probe)
        QApplication.sendEvent(probe, make_wheel(-120))
        app.processEvents()
        check("有焦点时滚轮仍可微调（原生行为保留）",
              widget_value(probe) != v1, f"{v1} → {widget_value(probe)}")
    else:
        print("  [跳过]  该环境下拿不到焦点，「有焦点时可微调」未验证")
    win.list_engines.setFocus()
    app.processEvents()

    # 滚轮应交给外层滚动区（把窗口缩到最小尺寸，让左栏必然出现滚动范围）
    area = enclosing_scroll(probe)
    saved = win.size()
    win.resize(1000, 560)
    app.processEvents()
    if area is not None and area.verticalScrollBar().maximum() > 0:
        sb = area.verticalScrollBar()
        sb.setValue(0)
        app.processEvents()
        v_before = sb.value()
        QApplication.sendEvent(probe, make_wheel(-120))
        app.processEvents()
        check("滚轮被交给页面滚动区（页面确实滚动了）",
              sb.value() > v_before, f"滚动条 {v_before} → {sb.value()}")
    else:
        print("  [跳过]  该尺寸下左栏无需滚动，未验证「交给滚动区」这一条")
    win.resize(saved)
    app.processEvents()

    # 恢复现场：本节为了做"反向验证"和"有焦点"两次确认，**故意**改动过控件值。
    # 不恢复的话后续处理会用被改过的参数——实测踩过：把「带宽扩展」从"关闭"改成开启，
    # 于是后面的"长度对齐"断言失败。那是本测试的副作用，不是应用的问题。
    for w in targets:
        v = before[id(w)]
        if widget_value(w) != v:
            if isinstance(w, QComboBox):
                w.setCurrentIndex(int(v))
            else:
                w.setValue(v)
    app.processEvents()
    left = [w for w in targets if widget_value(w) != before[id(w)]]
    check("本节结束后所有控件已恢复原值（测试不留副作用）", not left,
          f"未恢复的：{[type(w).__name__ for w in left][:6]}")

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
        # 状态文字由**排队投递**的 finished_all 信号触发刷新，直接断言会偶发失败
        # （实测 4 连跑里 1 次：文件已就绪、列表文字还没刷）。这里等它刷出来。
        shot = wait_until(app, lambda: "已就绪" in win.list_engines.item(row).text(),
                          timeout_s=15, label="下载后状态刷新")
        check("下载后状态回到「已就绪」", shot,
              win.list_engines.item(row).text().split("\n")[0])
    finally:
        if os.path.isfile(backup):
            if os.path.isfile(target):
                os.remove(target)
            shutil.move(backup, target)

    print("\n9. 输出模式：逐文件复位 + 运行前可见（用户事故的根因）")
    # 事故：manifest 里 video_out=true、chain_label 含「写回视频」，用户却认为没勾。
    # 根因是复选框状态跨文件粘滞（_on_analyzed 对可回写视频只 setEnabled(True)，
    # 从不复位 checked），且运行前没有任何地方说明这一次输出音频还是视频。
    import tempfile  # noqa: PLC0415

    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from media_fixtures import make_video, tone  # noqa: PLC0415

    tmpdir = tempfile.mkdtemp(prefix="ae_ui_mode_")
    vid = make_video(os.path.join(tmpdir, "mode.mp4"), [tone(300, 1.0)])
    win._import_file(vid)
    ok = wait_until(app, lambda: win._info is not None and win._info.path == vid,
                    timeout_s=90, label="视频分析")
    check("视频导入并探测为视频", ok and win._info.is_video,
          f"{ok} / {getattr(win._info, 'kind', '?')}")
    check("导入视频后「输出视频」可用但**未勾选**",
          win.chk_video_out.isEnabled() and not win.chk_video_out.isChecked(),
          f"enabled={win.chk_video_out.isEnabled()} checked={win.chk_video_out.isChecked()}")
    check("运行前提示为「音频（.wav）」",
          "音频" in win.lbl_out_mode.text() and ".wav" in win.lbl_out_mode.text(),
          win.lbl_out_mode.text())
    o_a = win._collect_options(vid)
    check("未勾选时 options: video_out=False 且 ext=.wav",
          o_a.video_out is False and o_a.output_ext == ".wav",
          f"{o_a.video_out} / {o_a.output_ext}")

    win.chk_video_out.setChecked(True)
    app.processEvents()
    check("勾选后提示为「写回视频（.mp4）」",
          "写回视频" in win.lbl_out_mode.text() and ".mp4" in win.lbl_out_mode.text(),
          win.lbl_out_mode.text())
    o_b = win._collect_options(vid)
    check("勾选后 options: video_out=True 且扩名取源容器 .mp4",
          o_b.video_out is True and o_b.output_ext == ".mp4",
          f"{o_b.video_out} / {o_b.output_ext}")
    plan = win._describe_output_plan(vid, o_b)
    check("运行前日志已给出预测落点（含 .mp4）", ".mp4" in plan and "写回视频" in plan, plan)

    # 根因回归：换文件必须回到「输出音频」，绝不继承
    vid2 = make_video(os.path.join(tmpdir, "mode2.mp4"), [tone(220, 1.0)])
    win._import_file(vid2)
    ok = wait_until(app, lambda: win._info is not None and win._info.path == vid2,
                    timeout_s=90, label="第二个视频分析")
    check("换文件后「输出视频」回到未勾选（不继承上一个文件）",
          ok and not win.chk_video_out.isChecked(),
          f"ok={ok} checked={win.chk_video_out.isChecked()}")
    check("换文件后 options 回到音频模式",
          win._collect_options(vid2).video_out is False)
    check("换文件后提示回到「音频」", "音频" in win.lbl_out_mode.text(), win.lbl_out_mode.text())

    # 音频输入：禁用 + 未勾选
    win._import_file(SAMPLE)
    ok = wait_until(app, lambda: win._info is not None and win._info.path == SAMPLE,
                    timeout_s=60, label="音频重新导入")
    check("音频输入时「输出视频」禁用且未勾选",
          ok and not win.chk_video_out.isEnabled() and not win.chk_video_out.isChecked(),
          f"ok={ok} enabled={win.chk_video_out.isEnabled()}")

    # 批处理：每份 options 按**该文件自己**的容器判定，而不是"当前正在看的文件"
    win.chk_video_out.setChecked(True)      # 编程置位：音频输入下控件是禁用的，但语义要能测
    app.processEvents()
    o_mix_video = win._collect_options(vid)
    o_mix_audio = win._collect_options(SAMPLE)
    check("批处理（混合批次）：视频→写回、音频→音频",
          o_mix_video.video_out is True and o_mix_video.output_ext == ".mp4"
          and o_mix_audio.video_out is False and o_mix_audio.output_ext == ".wav",
          f"{o_mix_video.output_ext} / {o_mix_audio.output_ext}")
    from core.worker import BatchJob  # noqa: PLC0415

    win.txt_log.clear()
    win._log_batch_output_plan(
        [BatchJob(index=0, media_path=vid, options=o_mix_video),
         BatchJob(index=1, media_path=SAMPLE, options=o_mix_audio)],
        want_video=True,
    )
    text = win.txt_log.toPlainText()
    check("预检日志逐文件写明模式（视频=写回视频）", "写回视频（.mp4）" in text, text[-200:])
    check("预检日志写明音频文件为何不写回", "写不回去" in text, text[-200:])
    check("预检日志给出计数汇总", "1 个写回视频 / 1 个输出音频" in text, text[-200:])
    win.chk_video_out.setChecked(False)
    app.processEvents()

    print("\n10. 竞态与一致性（2026-10-01 审计新增）")
    from core import engines, outputs, pipeline  # noqa: PLC0415

    # 10a 迟到的分析结果必须被丢弃（代次校验）：旧 worker 可能已经把 analyzed 发出来了
    class _StaleInfo:
        """哨兵：只要它出现在 win._info 上，就说明过期结果被采纳了。"""

        path = "STALE"
        kind = "audio"
        is_video = False

    stale = _StaleInfo()
    before_info = win._info
    win._import_file(SAMPLE)                       # 让代次前进一步
    win._on_analyzed(0, stale, None, None, None)   # 代次 0 = 过期
    app.processEvents()
    check("过期分析结果被丢弃（_info 没有被哨兵覆盖）", win._info is not stale,
          f"_info 变成了哨兵：{win._info is stale}")
    check("未过期以外的状态也未被改动", win._info is before_info or win._info is None,
          f"{win._info is before_info}")
    check("日志里写明了忽略过期结果", "过期的分析结果" in win.txt_log.toPlainText())

    # 10b 处理中「批量处理」必须一起禁用（旧版只关「开始处理」→ 两个任务能同时跑）
    ok = wait_until(app, lambda: not win._analyzing, timeout_s=90, label="重新分析完成")
    check("分析完成", ok)
    from core.worker import PipelineWorker  # noqa: PLC0415

    check("分析期间/就绪后按钮状态自洽",
          (not win._analyzing) or (not win.btn_batch.isEnabled()),
          f"analyzing={win._analyzing} batch={win.btn_batch.isEnabled()}")
    win._start_process()
    app.processEvents()
    check("单文件处理中「批量处理」也被禁用", not win.btn_batch.isEnabled())
    check("单文件处理中「取消」可用", win.btn_cancel.isEnabled())
    win._cancel_process()
    wait_until(app, lambda: not (win._pipeline_worker and win._pipeline_worker.isRunning()),
               timeout_s=30, label="取消完成")
    app.processEvents()

    # 10c 预测的文件名必须与实际写出的一致（指纹里的 stream_index 曾传 None → 永远预测错）
    win._import_file(SAMPLE)
    ok = wait_until(app, lambda: win._info is not None and not win._analyzing,
                    timeout_s=90, label="示例音频分析")
    check("示例音频已就绪", ok and win._media_path == SAMPLE, str(win._media_path))
    opts = win._collect_options(SAMPLE)
    plan = win._describe_output_plan(SAMPLE, opts)
    out_dir, _fb = outputs.output_dir_for(SAMPLE, win.project_root)
    eng_kind = engines.engine_kind(win.catalog.engine(opts.engine_id))
    expected, _fp = outputs.expected_path(SAMPLE, opts, out_dir, win._stream_index, eng_kind)
    check("预测落点与 expected_path 完全一致",
          os.path.basename(expected) in plan, f"{os.path.basename(expected)} vs {plan}")
    res = pipeline.run(SAMPLE, win.catalog, opts)
    check("实际写出的文件名与预测一致",
          os.path.basename(res.output_path or "") == os.path.basename(expected),
          f"{os.path.basename(res.output_path or '')} vs {os.path.basename(expected)}")

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
