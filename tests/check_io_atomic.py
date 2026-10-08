"""写出与复用的完整性：原子写 + 半截产物不得被当成"已算好"。

背景（2026-10-01 实测缺陷）：pipeline 判断"算过没有"**只看文件名里的指纹**，
所以任何在**最终文件名**上留下的半截文件，下次都会被当成"参数与已有结果完全一致，
直接复用"。旧版 `write_audio` 是 `sf.write(最终路径)`，于是三类情况都会留下它：
磁盘写满、写出途中被强杀、断电。后果分两档——轻度截断被 `align_to` 用原音尾部
静默补齐（此后一直复用一份悄悄少了若干秒的产物）；重度截断抛
「这是上游缺陷，已拒绝静默裁掉」，而用户不知道该删哪个文件，**每次重试都卡在同一处**。

本套件钉住四条：

1. `write_audio` 失败时**最终路径下不存在文件**，且只留下（并清理）自己的 `.part`。
2. `.part` 文件不会被变体扫描认成本工具的结果。
3. 人为截断的产物**不会被静默复用**，而是重算。
4. 正常产物仍然照旧复用（别把复用一起修坏）。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_io_atomic.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import media_io, modelstore, outputs, pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def section(t: str) -> None:
    print()
    print(t)
    print("=" * len(t))


def tone(seconds: float = 2.0, freq: float = 300.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.05 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def options(out_dir: str) -> pipeline.PipelineOptions:
    return pipeline.PipelineOptions(
        engine_id="dsp_baseline", output_dir=out_dir, highpass_hz=0.0,
        reuse_existing=True,
    )


def main() -> int:
    print(f"项目根：{ROOT}")
    cat = modelstore.load_catalog(ROOT)
    tmp = tempfile.mkdtemp(prefix="ae_ioatomic_")
    out_dir = os.path.join(tmp, "out")
    src = os.path.join(tmp, "t.wav")
    media_io.write_audio(src, tone(), SR)

    try:
        section("1. 原子写：失败时最终路径下不留文件")
        target = os.path.join(tmp, "fail.wav")
        real_write = sf.write

        def boom(*_a, **_k):
            raise OSError("模拟磁盘写满")

        sf.write = boom
        raised = ""
        try:
            media_io.write_audio(target, tone(), SR)
        except OSError as exc:
            raised = str(exc)
        finally:
            sf.write = real_write
        check("写出失败会抛错", raised != "", raised)
        check("最终路径下不存在半截文件", not os.path.exists(target))
        check("不留自己的 .part 残片",
              not [n for n in os.listdir(tmp) if ".part" in n],
              str([n for n in os.listdir(tmp) if ".part" in n]))

        section("2. 正常写出仍是原子语义（最终文件存在、没有 .part 残留）")
        ok_path = os.path.join(tmp, "ok.wav")
        media_io.write_audio(ok_path, tone(), SR)
        check("文件已写出", os.path.isfile(ok_path) and os.path.getsize(ok_path) > 0)
        check("无 .part 残留", not [n for n in os.listdir(tmp) if ".part" in n])

        section("3. .part 文件不被变体扫描认成本工具的结果")
        info = media_io.probe(ok_path)
        stem = outputs.safe_stem(ok_path)
        fake = os.path.join(out_dir)
        os.makedirs(fake, exist_ok=True)
        good = outputs.VariantEntry(fp="abcdef123456", output=f"{stem}__dn-x__abcdef123456.wav")
        part = f"{stem}__dn-x__abcdef123456.part123_456.wav"
        media_io.write_audio(os.path.join(fake, part), tone(), SR)
        media_io.write_audio(os.path.join(fake, good.output), tone(), SR)
        found = {v.output for v in outputs.scan_variants(fake, ok_path)}
        check("只列出真正的产物", good.output in found and part not in found, str(sorted(found)))

        section("4. 截断的产物不得被静默复用（而是重算）")
        r1 = pipeline.run(src, cat, options(out_dir))
        check("首次运行是真算出来的", not r1.reused and r1.output_path is not None)
        r2 = pipeline.run(src, cat, options(out_dir))
        check("同参数第二次确实复用（别把复用修坏）", r2.reused, str(r2.reused))

        # 人为把产物截到 1/4 —— 旧版非原子写/被强杀会留下这种文件
        backup = r1.output_path + ".bak"
        os.replace(r1.output_path, backup)
        decoded = media_io.decode(backup)
        media_io.write_audio(r1.output_path, decoded.samples[: len(decoded.samples) // 4],
                             decoded.sample_rate)

        r3 = pipeline.run(src, cat, options(out_dir))
        want = len(r1.enhanced.samples) / r1.enhanced.sample_rate
        got = len(r3.enhanced.samples) / r3.enhanced.sample_rate
        check("截断产物未被复用", not r3.reused, f"reused={r3.reused}")
        check("重算后时长与期望一致（±1%）", abs(got - want) <= 0.01 * want,
              f"{got:.3f}s vs {want:.3f}s")

        section("5. 轻度截断（<1%）也必须被发现")
        # 只砍掉 0.5%（80 帧）：旧行为会用原音尾部静默补齐并一直复用
        os.replace(r1.output_path, backup)
        cut = int(len(decoded.samples) * 0.995)
        media_io.write_audio(r1.output_path, decoded.samples[:cut], decoded.sample_rate)
        r4 = pipeline.run(src, cat, options(out_dir))
        check("0.5% 截断同样不被复用", not r4.reused, f"reused={r4.reused}")

        section("6. 容差不能严过头：正常的 mp3 产物必须照旧复用")
        # mp3 有编码器延迟/补齐，若容差按无损容器定会被判成损坏而每次重算
        mp3_dir = os.path.join(tmp, "out_mp3")
        mp3_opts = pipeline.PipelineOptions(
            engine_id="dsp_baseline", output_dir=mp3_dir, highpass_hz=0.0,
            output_ext=".mp3", reuse_existing=True,
        )
        m1 = pipeline.run(src, cat, mp3_opts)
        m2 = pipeline.run(src, cat, mp3_opts)
        check("mp3 首次真算", not m1.reused, str(m1.output_path))
        check("mp3 同参数第二次复用（容差足够宽）", m2.reused, f"reused={m2.reused}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"I/O 完整性验证结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("I/O 完整性验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
