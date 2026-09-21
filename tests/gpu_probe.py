"""GPU 场景探针：在**独立子进程**里跑真实链路并报告结果。

为什么必须独立进程：显式让 sherpa 试 CUDA 时，进程在**退出阶段**会段错误
（实测 2/2 复现，EXIT=139）。同一进程里测它会把整个测试套件带崩，所以隔离出来，
把退出码当作**观测结果**报回去——这也让"默认配置退出是否干净"变成可断言的事实。

三种模式（由 `tests/check_gpu.py` 调用，也可单独跑）：

    default       默认配置：带宽扩展 auto + 降噪 auto
                  —— 用户真正碰得到的那条，必须干净退出
    sherpa-cuda   先让 pip onnxruntime 上 CUDA，再显式请求 sherpa CUDA
                  —— 预期：CUDA 加载失败后回落 CPU（实测退出码 139，已记入 README）
    sherpa-only   只跑 sherpa，先 cpu 再 cuda
                  —— 没有 pip onnxruntime 干扰，此时 sherpa 的 CUDA 能真正生效，
                     所以可以量它的耗时与 CPU/GPU 等价性

输出：一行以 `PROBE ` 开头的 JSON（其余为 stderr 噪声，调用方按前缀解析）。

单跑示例：
    .venv\\Scripts\\python.exe tests\\gpu_probe.py sherpa-only
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bwe, cuda_env, engines, media_io, modelstore  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "speech_with_noise.wav")


def excerpt(seconds: float, rate: int = 16000) -> np.ndarray:
    """官方示例音频平铺到指定时长（确定性输入，不依赖用户素材）。"""
    data = media_io.decode(SAMPLE)
    base = np.asarray(data.samples, dtype=np.float32)
    if base.ndim > 1:
        base = media_io.to_mono(base)
    n = int(seconds * data.sample_rate)
    x = np.tile(base, int(np.ceil(n / max(1, len(base)))))[:n]
    if data.sample_rate != rate:
        x = media_io.resample(x, data.sample_rate, rate)
    return x


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"
    cat = modelstore.load_catalog(ROOT)
    out: dict = {"mode": mode, "gpu": cuda_env.nvidia_smi() is not None}

    if mode == "default":
        x = excerpt(5.0)
        bwe.upscale("novasr", cat, x, 16000, device="auto")
        out["bwe_desc"] = bwe.last_device().get("desc", "")
        eng = engines.build_engine(cat.engine("dpdfnet2"), cat)
        eng.process(x, 16000, engines.EngineParams(num_threads=2, sherpa_provider="auto"))
        out["denoise_desc"] = eng.provider_desc() or ""

    elif mode == "sherpa-cuda":
        # 先让 pip 的 onnxruntime 真的用上 CUDA（就像应用里带宽扩展先跑那样）
        bwe.upscale("novasr", cat, excerpt(5.0), 16000, device="auto")
        out["bwe_desc"] = bwe.last_device().get("desc", "")
        eng = engines.build_engine(cat.engine("dpdfnet2"), cat)
        eng.process(excerpt(5.0), 16000,
                    engines.EngineParams(num_threads=2, sherpa_provider="cuda"))
        out["denoise_desc"] = eng.provider_desc() or ""

    elif mode == "sherpa-only":
        x = excerpt(20.0)
        spec = cat.engine("dpdfnet2")
        results = {}
        for prov in ("cpu", "cuda"):
            eng = engines.build_engine(spec, cat)
            t0 = time.perf_counter()
            y, _r = eng.process(x, 16000,
                                engines.EngineParams(num_threads=2, sherpa_provider=prov))
            results[prov] = {
                "seconds": round(time.perf_counter() - t0, 3),
                "desc": eng.provider_desc() or "",
                "samples": np.asarray(y, dtype=np.float32),
            }
            eng.release()
        a, b = results["cpu"]["samples"], results["cuda"]["samples"]
        n = min(len(a), len(b))
        diff = np.abs(a[:n] - b[:n])
        rms = float(np.sqrt(np.mean(a[:n] ** 2))) or 1e-12
        out["cpu_seconds"] = results["cpu"]["seconds"]
        out["cuda_seconds"] = results["cuda"]["seconds"]
        out["cpu_desc"] = results["cpu"]["desc"]
        out["cuda_desc"] = results["cuda"]["desc"]
        out["max_abs_diff"] = float(diff.max())
        out["rel_rms_pct"] = round(float(np.sqrt(np.mean(diff ** 2))) / rms * 100.0, 5)
        out["cuda_really_used"] = "CUDA" in results["cuda"]["desc"]

    else:
        out["error"] = f"未知模式：{mode}"
        print("PROBE " + json.dumps(out, ensure_ascii=False), flush=True)
        return 2

    print("PROBE " + json.dumps(out, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
