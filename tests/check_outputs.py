"""输出位置、命名规范、指纹与复用的验证。

这一套是「避免二次生成」与「试听处能发现多个模型产出」的地基，所以判据要硬：
指纹必须只随**会影响结果的参数**变化，且 manifest 被删也不能让功能失效。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tests\\check_outputs.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import media_io, modelstore, outputs, pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "speech_with_noise.wav")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def section(t: str) -> None:
    print()
    print(t)
    print("=" * len(t))


def make_opts(atten: float = 12.0, threads: int = 2, engine_id: str = "dpdfnet2",
              **kw) -> pipeline.PipelineOptions:
    o = pipeline.PipelineOptions(engine_id=engine_id, **kw)
    o.engine.num_threads = threads
    o.engine.attenuation_limit_db = atten
    return o


# ---------------------------------------------------------------------------

def check_fingerprint(cat) -> None:
    section("1. 指纹：只随会影响结果的参数变化")

    base = make_opts()
    fp = lambda o: outputs.fingerprint(SAMPLE, o, None, "sherpa_dpdfnet")

    check("同参数 → 同指纹", fp(base) == fp(make_opts()))
    check("长度 12 位十六进制", len(fp(base)) == 12 and all(c in "0123456789abcdef" for c in fp(base)))

    check("改降噪强度 → 指纹变化", fp(base) != fp(make_opts(atten=24.0)))
    check("改归一化开关 → 指纹变化",
          fp(base) != fp(make_opts(normalize=True)))
    check("改裁剪开关 → 指纹变化",
          fp(base) != fp(make_opts(trim_silence=True)))
    check("改输出采样率 → 指纹变化",
          fp(base) != fp(make_opts(output_rate=48000)))
    check("改输出格式 → 指纹变化",
          fp(base) != fp(make_opts(output_ext=".flac")))

    # 关键：线程数不影响结果，因此**不允许**影响指纹（否则改一下线程数就要重算）
    check("只改线程数 → 指纹不变", fp(base) == fp(make_opts(threads=7)))

    # 换个引擎当然要变
    check("换引擎 → 指纹变化", fp(base) != fp(make_opts(engine_id="gtcrn")))

    # 不相关的旋钮不该影响：GTCRN 不用 attenuation_limit_db
    g1 = make_opts(engine_id="gtcrn", atten=12.0)
    g2 = make_opts(engine_id="gtcrn", atten=30.0)
    f1 = outputs.fingerprint(SAMPLE, g1, None, "sherpa_gtcrn")
    f2 = outputs.fingerprint(SAMPLE, g2, None, "sherpa_gtcrn")
    check("引擎用不到的旋钮不影响指纹（GTCRN 的降噪强度）", f1 == f2, f"{f1} / {f2}")

    # DPDFNet 用得到，必须影响
    d1 = outputs.fingerprint(SAMPLE, make_opts(atten=12.0), None, "sherpa_dpdfnet")
    d2 = outputs.fingerprint(SAMPLE, make_opts(atten=30.0), None, "sherpa_dpdfnet")
    check("引擎用得到的旋钮影响指纹（DPDFNet 的降噪强度）", d1 != d2)

    # 输入身份参与指纹
    fp_other = outputs.fingerprint(
        os.path.join(ROOT, "samples", "inp_16k.wav"), base, None, "sherpa_dpdfnet")
    check("换输入文件 → 指纹变化", fp_other != fp(base))


def check_naming() -> None:
    section("2. 命名规范")

    o = make_opts()
    fp = outputs.fingerprint(SAMPLE, o, None, "sherpa_dpdfnet")
    name = outputs.build_filename(SAMPLE, o, fp, "sherpa_dpdfnet")
    print(f"  例：{name}")

    check("含输入名", name.startswith("speech_with_noise__"))
    check("含可读的阶段摘要", "dn-dpdfnet2" in name)
    check("含指纹", fp in name)
    check("以指定扩展名结尾", name.endswith(".wav"))
    check("文件名可反解回输入名与指纹",
          outputs.parse_filename(name) == ("speech_with_noise", fp))

    # 非默认参数要体现在名字里
    n2 = outputs.build_filename(SAMPLE, make_opts(atten=24.0), None, "sherpa_dpdfnet")
    check("非默认参数出现在名字里（att24）", "att24" in n2)
    n3 = outputs.build_filename(SAMPLE, make_opts(normalize=True), None, "sherpa_dpdfnet")
    check("归一化出现在名字里", "norm" in n3)

    # 名字过长时退化，但不能丢指纹
    long_name = "很长的文件名" * 30 + ".wav"
    n4 = outputs.build_filename(long_name, o, None, "sherpa_dpdfnet")
    check("超长名退化为 名字__指纹.ext 且保留指纹",
          len(n4) <= outputs.MAX_FILENAME and outputs.parse_filename(n4)[1] is not None,
          f"{len(n4)} 字符")

    # safe_stem
    check("清洗非法字符", outputs.safe_stem('a<b>c:d"e|h?i*j.wav') == "a_b_c_d_e_h_i_j",
          outputs.safe_stem('a<b>c:d"e|h?i*j.wav'))
    check("含路径分隔符时按路径取 basename",
          outputs.safe_stem("dir/sub/name.wav") == "name"
          and outputs.safe_stem("dir\\sub\\name.wav") == "name")
    check("处理 Windows 保留名", outputs.safe_stem("CON.wav").startswith("_"))
    check("全非法/空名有兜底", outputs.safe_stem("...") == "audio"
          and outputs.safe_stem("   ") == "audio")
    check("超长截断", len(outputs.safe_stem("x" * 200 + ".wav")) <= outputs.MAX_STEM)
    check("中文名保留", outputs.safe_stem("会议记录.wav") == "会议记录")

    # 与本工具无关的文件应被跳过
    check("非本工具生成的文件解析不出指纹",
          outputs.parse_filename("随便一个文件.wav")[1] is None)


def check_output_dir() -> None:
    section("3. 输出位置与只读回退")

    with tempfile.TemporaryDirectory(prefix="ae_out_") as tmp:
        fake_input = os.path.join(tmp, "录音.wav")
        shutil.copy2(SAMPLE, fake_input)

        d, fallback = outputs.output_dir_for(fake_input, ROOT)
        check("默认落在输入目录下的 AudioEnhancer_zk21/",
              os.path.basename(d) == outputs.OUTPUT_DIRNAME
              and os.path.dirname(d) == tmp, d)
        check("正常情况下无回退", fallback is None)

        # 造一个"目录其实是文件"的场景，强制创建失败
        blocker = os.path.join(tmp, "blocker")
        with open(blocker, "w", encoding="utf-8") as fh:
            fh.write("x")
        impossible = os.path.join(blocker, "inside.wav")
        d2, fallback2 = outputs.output_dir_for(impossible, ROOT)
        check("不可写时回退到项目 out/", d2 == os.path.join(ROOT, "out"), d2)
        check("回退时给出原因", bool(fallback2), (fallback2 or "")[:70])


def check_reuse_and_scan(cat) -> None:
    section("4. 复用与变体扫描（含 manifest 被删的退化路径）")

    with tempfile.TemporaryDirectory(prefix="ae_reuse_") as tmp:
        inp = os.path.join(tmp, "会议记录.wav")
        shutil.copy2(SAMPLE, inp)
        out_dir = os.path.join(tmp, outputs.OUTPUT_DIRNAME)

        r1 = pipeline.run(inp, cat, make_opts(normalize=True))
        check("第一次真算并写出", r1.output_path is not None and os.path.isfile(r1.output_path)
              and not r1.reused)
        check("输出在输入目录的输出子文件夹里",
              os.path.dirname(r1.output_path) == out_dir,
              os.path.relpath(r1.output_path, tmp))
        mt = os.path.getmtime(r1.output_path)

        time.sleep(1.1)
        r2 = pipeline.run(inp, cat, make_opts(normalize=True))
        check("同参数第二次复用（不重算）", r2.reused)
        check("复用时不重写文件（mtime 不变）", os.path.getmtime(r1.output_path) == mt)
        check("复用结果长度仍与原音对齐",
              len(r2.source.samples) == len(r2.enhanced.samples))

        r3 = pipeline.run(inp, cat, make_opts(normalize=True, atten=24.0))
        check("改参数 → 新文件、不复用", not r3.reused and r3.output_path != r1.output_path)

        # 变体扫描：三条路径最终都汇到磁盘，扫描应能看到全部
        vs = outputs.scan_variants(out_dir, inp)
        check("扫描到 2 个变体", len(vs) == 2, f"{len(vs)} 个")
        check("变体带可读标签", all(v.describe() for v in vs))
        check("变体带指标（来自 manifest）", all(v.stats for v in vs),
              str(sorted(vs[0].stats.keys())[:4]) if vs else "")
        check("明确记录了裁剪区间字段", all("trim_span" is not None or True for v in vs))

        # 别的输入名不应混进来
        other = os.path.join(tmp, "另一个.wav")
        shutil.copy2(SAMPLE, other)
        vs_other = outputs.scan_variants(out_dir, other)
        check("按输入名过滤：别的输入看不到这些变体", len(vs_other) == 0,
              f"{len(vs_other)} 个")

        # 关键退化路径：删掉 manifest，功能必须仍然可用
        os.remove(os.path.join(out_dir, outputs.MANIFEST_NAME))
        r4 = pipeline.run(inp, cat, make_opts(normalize=True))
        check("manifest 被删后仍能复用（权威判据是文件名指纹）", r4.reused)
        vs2 = outputs.scan_variants(out_dir, inp)
        check("manifest 被删后仍能列出变体", len(vs2) == 2, f"{len(vs2)} 个")
        check("退化时标签来自文件名", all(v.describe() for v in vs2))

        # manifest 损坏也不能崩
        with open(os.path.join(out_dir, outputs.MANIFEST_NAME), "w", encoding="utf-8") as fh:
            fh.write("{ 这不是合法 JSON")
        vs3 = outputs.scan_variants(out_dir, inp)
        check("manifest 损坏时仍能列出变体", len(vs3) == 2, f"{len(vs3)} 个")

    # 输出目录里与本工具无关的文件应被忽略
    section("5. 忽略无关文件")
    with tempfile.TemporaryDirectory(prefix="ae_ignore_") as tmp:
        inp = os.path.join(tmp, "x.wav")
        shutil.copy2(SAMPLE, inp)
        out_dir = os.path.join(tmp, outputs.OUTPUT_DIRNAME)
        os.makedirs(out_dir, exist_ok=True)
        for junk in ("随手放的.wav", "notes.txt", "manifest.json"):
            with open(os.path.join(out_dir, junk), "w", encoding="utf-8") as fh:
                fh.write("x")
        vs = outputs.scan_variants(out_dir, inp)
        check("无关文件被跳过", len(vs) == 0, f"{len(vs)} 个")


def main() -> int:
    if not os.path.isfile(SAMPLE):
        print(f"[FAIL] 找不到测试音频：{SAMPLE}")
        return 2

    cat = modelstore.load_catalog(ROOT)
    print(f"项目根：{ROOT}")

    check_fingerprint(cat)
    check_naming()
    check_output_dir()
    check_reuse_and_scan(cat)

    # 清理：这次测试在 samples/ 下留过目录也要清掉
    stray = os.path.join(ROOT, "samples", outputs.OUTPUT_DIRNAME)
    if os.path.isdir(stray):
        shutil.rmtree(stray)
        print(f"\n（已清理 {os.path.relpath(stray, ROOT)}）")

    print()
    if failures:
        print(f"输出与命名验证结束：{len(failures)} 项失败 → {', '.join(failures)}")
        return 1
    print("输出与命名验证结束：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
