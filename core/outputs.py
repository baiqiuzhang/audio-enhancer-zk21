"""输出位置、命名规范与生成登记。

三件事：
1. **输出位置**：输入文件所在目录下的 `AudioEnhancer_zk21/`；不可写时回退到项目 `out/`
   并明确告知（不静默改变位置）。
2. **命名规范**：`{stem}__{chain}__{fp12}{ext}`。`chain` 是人类可读的阶段摘要，
   所以**即使 manifest 被删掉，只看文件名也知道用了哪个模型、什么参数**。
3. **登记**：`manifest.json` 记录完整参数与指标供界面显示。

关键分工：**判断"算过没有"的权威判据是文件名里的 `fp12`**，不依赖 manifest。
因此 manifest 损坏/被删不会让「避免二次生成」失效。

指纹用**文件名 + 大小**而不是完整路径：整个文件夹移走之后仍能复用。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

OUTPUT_DIRNAME = "AudioEnhancer_zk21"
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = 1

# 文件名总长上限（保守，避免路径过长；超出就退化为只留指纹）
MAX_FILENAME = 120
MAX_STEM = 60

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WS = re.compile(r"\s+")
# Windows 保留设备名
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _dir_lock(path: str) -> threading.Lock:
    """每个输出目录一把锁：批处理的多个 worker 会并发写同一个 manifest。"""
    with _locks_guard:
        lock = _locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _locks[path] = lock
        return lock


# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------

def output_dir_for(input_path: str, project_root: str) -> tuple[str, str | None]:
    """返回 (输出目录, 回退原因)。

    优先 `<输入文件所在目录>/AudioEnhancer_zk21/`；创建或写入失败则回退到
    `<project_root>/out/`，并把原因回传，供界面明确告知用户。
    """
    parent = os.path.dirname(os.path.abspath(input_path)) or os.getcwd()
    preferred = os.path.join(parent, OUTPUT_DIRNAME)
    try:
        os.makedirs(preferred, exist_ok=True)
    except OSError as exc:
        return _fallback(project_root, exc)

    # 探针文件名必须**唯一到线程级**：批处理的多个 worker 线程会同时调这里，
    # 若共用 `pid` 命名，先完成的那个删掉探针后，另一个的 remove 会报"文件不存在"，
    # 于是把明明可写的目录误判为不可写并回退——实测踩过这个坑。
    probe = os.path.join(
        preferred,
        f".ae_write_test_{os.getpid()}_{threading.get_ident()}_{time.monotonic_ns()}",
    )
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
    except OSError as exc:
        return _fallback(project_root, exc)
    finally:
        # 清理失败不影响"可写"的判定：文件都已经写成功了
        try:
            os.remove(probe)
        except OSError:
            pass

    return preferred, None


def _fallback(project_root: str, exc: OSError) -> tuple[str, str | None]:
    fallback = os.path.join(project_root, "out")
    try:
        os.makedirs(fallback, exist_ok=True)
    except OSError:
        pass
    return fallback, (
        f"无法写入输入目录下的 {OUTPUT_DIRNAME}/（{exc.strerror or exc}），"
        f"已回退到 {fallback}"
    )


def safe_stem(name: str, limit: int = MAX_STEM) -> str:
    """把输入文件名清洗成可用作文件名前缀的形式。"""
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = _ILLEGAL.sub("_", stem)
    stem = _WS.sub(" ", stem).strip().strip(".")
    if not stem:
        stem = "audio"
    if stem.upper() in _RESERVED:
        stem = f"_{stem}"
    if len(stem) > limit:
        stem = stem[:limit].rstrip()
    return stem


# ---------------------------------------------------------------------------
# 指纹
# ---------------------------------------------------------------------------

def _round(value):
    """浮点统一收敛到 6 位小数，避免 3 和 3.0000000000000004 生成不同指纹。"""
    if isinstance(value, float):
        return round(value, 6)
    return value


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return _round(value) if isinstance(value, float) else value


def _denoise_knobs(options, engine_kind: str | None) -> dict:
    """只保留该引擎**真正会用**的旋钮。

    否则会出现"用 GTCRN 时拖了一下降噪强度滑块，指纹就变了、结果被判定需要重算"
    这种假差异。engine_kind 由调用方从模型清单里给出。
    """
    eng = options.engine
    out: dict = {"id": options.engine_id}
    kind = engine_kind or ""
    if kind == "sherpa_dpdfnet":
        out["attenuation_limit_db"] = float(eng.attenuation_limit_db)
    elif kind == "dfn_exe":
        out["dfn_post_filter"] = bool(eng.dfn_post_filter)
        out["dfn_atten_lim_db"] = float(eng.dfn_atten_lim_db)
        out["dfn_compensate_delay"] = bool(eng.dfn_compensate_delay)
    elif kind == "dsp":
        out["dsp_gate_reduction_db"] = float(eng.dsp_gate_reduction_db)
        out["dsp_hp_cutoff_hz"] = float(getattr(eng, "dsp_hp_cutoff_hz", 0.0))
    return out


def spec_dict(input_path: str, options, stream_index: int | None = None,
              engine_kind: str | None = None) -> dict:
    """影响处理结果的参数快照。

    **刻意排除** num_threads、并行 worker 数、输出目录、界面状态——它们不改变结果，
    纳入会导致"改一下线程数就要重算一遍"。
    """
    try:
        size = os.path.getsize(input_path)
    except OSError:
        size = 0

    return _jsonable({
        "input": {
            # 身份用**绝对路径**（规范化、大小写折叠）而不是"文件名+大小"。
            # 原本想用文件名+大小换取"整个文件夹移走仍可复用"，但实测发现：
            # 两份同名同大小的文件（不同目录）会算出同一指纹、互相串用结果，
            # 甚至让"已生成"判定指向别人的产物。正确性优先。
            "path": os.path.normcase(os.path.abspath(input_path)),
            "name": os.path.basename(input_path),   # 留作可读记录
            "size": int(size),
            "stream_index": stream_index,
        },
        "pipeline": {
            "trim": {
                "enabled": bool(options.trim_silence),
                "dynamic_range_db": float(options.trim_dynamic_range_db),
                "pad_ms": float(options.trim_pad_ms),
            },
            "highpass_hz": float(options.highpass_hz),
            "denoise": _denoise_knobs(options, engine_kind),
            "bwe": {
                "id": options.bwe_id,
                "cutoff_hz": float(options.bwe_cutoff_hz),
            },
            "perceptual": {
                "shelf_enabled": bool(options.shelf_enabled),
                "shelf_hz": float(options.shelf_hz),
                "shelf_db": float(options.shelf_db),
                "mud_cut_hz": float(options.mud_cut_hz),
                "mud_cut_db": float(options.mud_cut_db),
                "mud_cut_width_oct": float(options.mud_cut_width_oct),
                "compressor_enabled": bool(options.compressor_enabled),
                "comp_threshold_dbfs": float(options.comp_threshold_dbfs),
                "comp_ratio": float(options.comp_ratio),
                "comp_attack_ms": float(options.comp_attack_ms),
                "comp_release_ms": float(options.comp_release_ms),
                "comp_makeup_db": float(options.comp_makeup_db),
            },
            "dereverb": {
                "enabled": bool(options.dereverb_enabled),
                "taps": int(options.dereverb_taps),
                "delay": int(options.dereverb_delay),
                "iterations": int(options.dereverb_iterations),
            },
            "normalize": {
                "enabled": bool(options.normalize),
                "target_dbfs": float(options.normalize_target_dbfs),
                "mode": options.normalize_mode,
            },
            "keep_channels": bool(options.keep_channels),
            "output_rate": options.output_rate,
            "output_ext": options.output_ext,
        },
    })


def fingerprint(input_path: str, options, stream_index: int | None = None,
                engine_kind: str | None = None, length: int = 12) -> str:
    """结果指纹：决定"算过没有"与"文件名"。"""
    payload = json.dumps(
        spec_dict(input_path, options, stream_index, engine_kind),
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# 命名
# ---------------------------------------------------------------------------

def filename_chain(options, engine_kind: str | None = None) -> str:
    """文件名里的阶段摘要（紧凑、可读）。

    只写非默认项，避免文件名被默认值淹掉。
    """
    eng = options.engine
    parts: list[str] = [f"dn-{options.engine_id}"]

    if engine_kind == "sherpa_dpdfnet" and float(eng.attenuation_limit_db) != 12.0:
        parts.append(f"att{eng.attenuation_limit_db:g}")
    if float(options.highpass_hz) not in (80.0, 0.0):
        parts.append(f"hp{options.highpass_hz:g}")
    if options.bwe_id:
        parts.append(f"bwe-{options.bwe_id}")
    if options.shelf_enabled:
        parts.append(f"eq{options.shelf_db:+g}")
    if options.mud_cut_db:
        parts.append(f"demud{options.mud_cut_db:+g}")
    if options.compressor_enabled:
        parts.append("cmp")
    if options.dereverb_enabled:
        parts.append("wpe")
    if options.normalize:
        parts.append(f"norm{options.normalize_target_dbfs:g}{options.normalize_mode}")
    parts.append(f"{options.output_rate}hz" if options.output_rate else "srchz")
    return "_".join(parts)


def display_label(options) -> str:
    """界面上显示的变体标签（比文件名更友好）。"""
    label_parts = [options.engine_id]
    if options.bwe_id:
        label_parts.append(f"+{options.bwe_id}")
    tail: list[str] = []
    if options.mud_cut_db:
        tail.append(f"去闷{options.mud_cut_db:+g}dB")
    if options.shelf_enabled or options.compressor_enabled:
        tail.append("听感增强")
    if not tail:
        tail.append("纯降噪")
    if options.dereverb_enabled:
        tail.append("去混响")
    rate = f"{options.output_rate // 1000} kHz" if options.output_rate else "源采样率"
    return f"{' '.join(label_parts)} · {'/'.join(tail)} · {rate}"


def build_filename(input_path: str, options, fp: str | None = None,
                   engine_kind: str | None = None) -> str:
    """`{stem}__{chain}__{fp12}{ext}`；过长时退化为 `{stem}__{fp12}{ext}`。"""
    fp = fp or fingerprint(input_path, options, engine_kind=engine_kind)
    ext = options.output_ext if options.output_ext.startswith(".") else f".{options.output_ext}"
    stem = safe_stem(input_path)
    full = f"{stem}__{filename_chain(options, engine_kind)}__{fp}{ext}"
    if len(full) > MAX_FILENAME:
        full = f"{stem}__{fp}{ext}"
    return full


# 注意：parse_filename 会先剥掉扩展名，所以这里匹配的是**不含扩展名**的结尾
_FP_RE = re.compile(r"__([0-9a-f]{8,32})$")


def parse_filename(filename: str) -> tuple[str, str | None]:
    """从 `{stem}__{chain}__{fp}{ext}` 反解 (输入名 stem, 指纹)。

    解不出指纹时返回 (stem, None)，调用方据此跳过非本工具生成的文件。
    """
    stem, _ext = os.path.splitext(filename)
    m = _FP_RE.search(stem)
    fp = m.group(1) if m else None
    head = stem[: m.start()] if m else stem
    input_stem = head.split("__", 1)[0] if "__" in head else head
    return input_stem, fp


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

@dataclass
class VariantEntry:
    fp: str
    output: str
    input_name: str = ""
    input_size: int = 0
    input_mtime: float = 0.0
    stream_index: int | None = None
    chain_label: str = ""
    created_at: str = ""
    params: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    trim_span: tuple[float, float] | None = None
    from_manifest: bool = True

    def path_in(self, output_dir: str) -> str:
        return os.path.join(output_dir, self.output)

    def size_bytes(self, output_dir: str) -> int:
        try:
            return os.path.getsize(self.path_in(output_dir))
        except OSError:
            return 0

    def describe(self) -> str:
        tail = []
        if self.stats:
            sr = self.stats.get("sample_rate")
            if sr:
                tail.append(f"{sr // 1000} kHz")
            rtf = self.stats.get("rtf")
            if rtf:
                tail.append(f"RTF {rtf:.2f}")
        return self.chain_label or self.output


def load_manifest(output_dir: str) -> dict:
    """读 manifest；缺失或损坏都返回空结构（不影响基于文件名的判据）。"""
    path = os.path.join(output_dir, MANIFEST_NAME)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            return data
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return {"schema": MANIFEST_SCHEMA, "entries": {}}


def save_manifest(output_dir: str, manifest: dict) -> None:
    """原子写：临时文件 + replace，避免并发读到半个文件。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, MANIFEST_NAME)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def register(output_dir: str, entry: VariantEntry) -> None:
    """把一次生成结果写进 manifest（并发安全）。"""
    with _dir_lock(output_dir):
        manifest = load_manifest(output_dir)
        manifest.setdefault("schema", MANIFEST_SCHEMA)
        manifest["entries"][entry.fp] = {
            "output": entry.output,
            "input_name": entry.input_name,
            "input_size": entry.input_size,
            "input_mtime": entry.input_mtime,
            "stream_index": entry.stream_index,
            "chain_label": entry.chain_label,
            "created_at": entry.created_at or time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": entry.params,
            "stats": entry.stats,
            "trim_span": list(entry.trim_span) if entry.trim_span else None,
        }
        save_manifest(output_dir, manifest)


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------

def expected_path(input_path: str, options, output_dir: str,
                  stream_index: int | None = None,
                  engine_kind: str | None = None) -> tuple[str, str]:
    """返回 (期望的输出绝对路径, 指纹)。"""
    fp = fingerprint(input_path, options, stream_index, engine_kind)
    return os.path.join(output_dir, build_filename(input_path, options, fp, engine_kind)), fp


def find_reusable(input_path: str, options, output_dir: str,
                  stream_index: int | None = None,
                  engine_kind: str | None = None) -> str | None:
    """参数完全相同时复用已有文件。

    **只看文件名里的指纹**，不依赖 manifest —— 所以 manifest 被删也能正常工作。
    """
    if not getattr(options, "reuse_existing", True):
        return None
    path, _fp = expected_path(input_path, options, output_dir, stream_index, engine_kind)
    return path if os.path.isfile(path) else None


def scan_variants(output_dir: str, input_path: str | None = None) -> list[VariantEntry]:
    """列出输出目录里已生成的变体（供试听区显示）。

    真值来自磁盘，因此"多选批量跑过"和"先后单跑"产生的变体同样能被发现，
    **应用重启后也依然在**。

    - 有 manifest：用它的元数据（参数、指标、裁剪区间）。
    - 没有 manifest：从文件名反解输入名与指纹，仍然能列出（只是没有指标）。
    """
    if not os.path.isdir(output_dir):
        return []

    manifest = load_manifest(output_dir)
    entries_by_fp: dict[str, dict] = manifest.get("entries", {})
    wanted_stem = safe_stem(input_path) if input_path else None

    out: list[VariantEntry] = []
    for name in sorted(os.listdir(output_dir)):
        if name == MANIFEST_NAME or name.startswith("."):
            continue
        full = os.path.join(output_dir, name)
        if not os.path.isfile(full):
            continue
        stem, fp = parse_filename(name)
        if fp is None:
            continue          # 不是本工具生成的文件，跳过
        if wanted_stem is not None and stem != wanted_stem:
            continue

        meta = entries_by_fp.get(fp)
        if meta:
            out.append(VariantEntry(
                fp=fp,
                output=meta.get("output", name),
                input_name=meta.get("input_name", ""),
                input_size=meta.get("input_size", 0),
                input_mtime=meta.get("input_mtime", 0.0),
                stream_index=meta.get("stream_index"),
                chain_label=meta.get("chain_label", ""),
                created_at=meta.get("created_at", ""),
                params=meta.get("params", {}),
                stats=meta.get("stats", {}),
                trim_span=tuple(meta["trim_span"]) if meta.get("trim_span") else None,
                from_manifest=True,
            ))
        else:
            # 没有 manifest 也要能列出：退化信息来自文件名本身
            _stem, parsed_fp = parse_filename(name)
            out.append(VariantEntry(
                fp=parsed_fp or "",
                output=name,
                input_name=os.path.basename(input_path) if input_path else "",
                chain_label=_label_from_filename(name),
                from_manifest=False,
            ))
    return out


def _label_from_filename(filename: str) -> str:
    """从 `{stem}__{chain}__{fp}{ext}` 里取出 chain 当标签。"""
    stem, _ext = os.path.splitext(filename)
    parts = stem.split("__")
    if len(parts) >= 3:
        return parts[1].replace("_", " ")
    return stem


def newest_variant(entries: list[VariantEntry], output_dir: str) -> VariantEntry | None:
    if not entries:
        return None

    def mtime(e: VariantEntry) -> float:
        try:
            return os.path.getmtime(e.path_in(output_dir))
        except OSError:
            return 0.0

    return max(entries, key=mtime)
