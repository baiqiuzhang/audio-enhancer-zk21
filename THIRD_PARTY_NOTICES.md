# 第三方组件与模型来源声明

本仓库（AudioEnhancer_zk21）本身的代码以 **GNU GPL-3.0** 发布（见 [LICENSE](LICENSE)），
原因是它依赖 **PyQt5**，而 PyQt5 是 GPL-3.0 / 商业双许可。

**本仓库不包含任何模型权重。** 权重由界面或 `tests/download_models.py` 在运行时从上游地址下载，
来源记录在 [`models.json`](models.json) 里（含每个源的 size/sha256 校验值）。
下面逐项列出直接依赖与其许可证，以及各引擎权重的上游出处。

---

## 1. 直接 Python 依赖（`requirements.txt`）

| 包 | 固定版本 | 许可证 | 说明 |
|---|---|---|---|
| PyQt5 | 5.15.10 | **GPL-3.0** | 界面。**本项目选择 GPL-3.0 的直接原因**；亦可向 Riverbank 购买商业许可 |
| PyQt5-Qt5 | 5.15.2 | LGPL-3.0 | PyQt5 捆绑的 Qt5 运行库 |
| PyQt5-sip | 12.19.0 | BSD-2-Clause | PyQt5 的绑定生成器运行时 |
| sherpa-onnx | 1.13.8 | Apache-2.0 | 降噪 + VAD 核心。捆绑自己的 onnxruntime 动态库 |
| numpy | ≥1.26,<3 | BSD-3-Clause | 数值计算 |
| soundfile | 0.14.0 | BSD-3-Clause | wav/flac/ogg/opus/mp3 读写。wheel 内捆绑 **libsndfile**（LGPL-2.1） |
| av (PyAV) | 18.1.0 | BSD-3-Clause | 视频容器与 m4a/aac/wma 解码。wheel 内捆绑经 LGPL 配置构建的 **FFmpeg** |
| soxr | 1.1.0 | MIT（Python 绑定） | 重采样。wheel 内捆绑 **libsoxr**（LGPL-2.1） |
| sounddevice | 0.5.6 | MIT | 播放。wheel 内捆绑 **PortAudio**（MIT） |
| audiosronnx | ≥0.9.0a2 | Apache-2.0 | 带宽扩展引擎集合（`--pre` 安装）。带 onnxruntime、scipy |

> LGPL 组件（libsndfile / FFmpeg / libsoxr）是以**动态链接库**形式随 wheel 分发的，
> 与 GPL-3.0 分发兼容；其源代码可从各自上游获取。

## 2. 间接依赖（随上面的包自动安装）

Apache-2.0：`huggingface_hub`、`hf_xet`、`flatbuffers`
BSD-3-Clause：`httpx`、`httpcore`、`idna`、`scipy`、`protobuf`、`colorama`、`fsspec`、`pycparser`
BSD-3-Clause / MIT 双列：`tqdm`（MPL-2.0 AND MIT）、`packaging`
MIT：`h11`、`click`、`filelock`
MPL-2.0：`certifi`
PSF：`typing_extensions`

## 3. 可选 GPU 依赖（默认不安装）

`requirements.txt` 刻意不写 GPU 依赖，默认纯 CPU 即可运行。自行安装时：

| 包 | 版本 | 许可证 |
|---|---|---|
| onnxruntime-gpu | 1.26.0 | MIT |
| nvidia-cuda-runtime-cu12 / cublas / cufft / curand / cudnn / nvjitlink | 12.9 / 9.26 | **NVIDIA 专有许可**（随 wheel 分发的运行库） |
| sherpa-onnx 的 `+cuda12.cudnn9` wheel | 1.13.8 | Apache-2.0（内含 NVIDIA 运行库，同上） |

## 4. 模型权重（不在本仓库内，运行时下载）

每个引擎的上游与许可如下；具体的下载 URL、字节数与 sha256 见 [`models.json`](models.json)。

| 引擎 | `models.json` key | 上游 | 许可 |
|---|---|---|---|
| DSP 基线（传统处理） | `dsp_baseline` | 本仓库代码，无权重 | — |
| GTCRN | `gtcrn` | [k2-fsa / sherpa-onnx speech-enhancement-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models) | Apache-2.0 |
| DPDFNet baseline / 2 / 4 / 8 / 48kHz-HR | `dpdfnet_baseline`、`dpdfnet2`、`dpdfnet4`、`dpdfnet8`、`dpdfnet2_48khz_hr` | [Ceva-IP/DPDFNet](https://huggingface.co/Ceva-IP/DPDFNet) | 见上游仓库 |
| DeepFilterNet3 | `deepfilternet3` | [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)（外部 exe） | 见上游仓库 |
| NovaSR | `novasr` | [audiosr / audiosronnx](https://pypi.org/project/audiosronnx/) 上游 | Apache-2.0（随 audiosronnx） |
| LavaSR | `lavasr` | 同上 | 同上 |
| HiFi-GAN BWE | `hifiganbwe` | 同上 | MIT |

**示例音频**：`samples/speech_with_noise.wav` 取自
[sherpa-onnx 官方 release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models)（Apache-2.0）；
`samples/inp_16k.wav` 的 sha256 记录在 `samples/checksum.txt` 中。

## 5. 免责说明

- 各模型权重的许可证由**其上游作者**决定，不因本项目以 GPL-3.0 发布而改变。商用前请逐一核对上表来源。
- 「试听用」预设启用的带宽扩展（BWE）类是**生成式**模型，会编造原素材中不存在的高频内容；
  默认关闭，艺术/取证用途请勿开启。
