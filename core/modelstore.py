"""模型清单与下载管理。

设计要点：
- 每个文件可以有多个来源，按清单顺序尝试（镜像优先，因为本机 github.com 时通时断）。
- 同一模型在不同来源的导出字节数可能不同，所以 size/sha256 是**按源**记录的；
  校验只对照实际命中的那个源。已存在的文件只要与**任一**源的元数据吻合就视为可用。
- 断点续传：`<目标>.part<源序号>`，按源分别保存，避免跨源拼接出坏文件。
- 只用标准库（urllib），不引入额外依赖。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from .errors import Cancelled

MODELS_JSON = "models.json"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 AudioEnhancer_zk21"
)

ProgressCb = Callable[[int, int], None]   # (已下载字节, 总字节)；总字节未知时为 0
CancelCb = Callable[[], bool]
LogCb = Callable[[str], None]


class DownloadError(RuntimeError):
    """所有来源都失败。"""


class DownloadCorrupt(DownloadError):
    """下载完成但**校验不通过**（大小/sha256 不符）。

    与网络/IO 失败分开是为了决定"残片留不留"：坏残片必须丢弃（否则会拿它去续传，
    拼出一个永远修不好的文件），而网络中断留下的残片要**保留**——`_fetch` 会用
    `Range: bytes=<已有长度>-` 从断点续传，这正是本机这种时通时断网络下最值钱的一条。
    """


class DownloadCancelled(Cancelled):
    """下载被取消。继承 `Cancelled`，这样"取消"在所有层都是同一个类型（见 core/errors.py）。"""


# ---------------------------------------------------------------------------
# 清单结构
# ---------------------------------------------------------------------------

@dataclass
class Source:
    url: str
    size: int | None = None
    sha256: str | None = None

    @property
    def host(self) -> str:
        try:
            return urllib.parse.urlsplit(self.url).netloc
        except Exception:
            return "?"


@dataclass
class FileSpec:
    path: str                                  # 相对 models/ 的路径（可含 ..）
    sources: list[Source] = field(default_factory=list)


@dataclass
class EngineSpec:
    id: str
    name: str
    kind: str                                  # sherpa | dfn_exe | dsp | bwe
    files: list[FileSpec] = field(default_factory=list)
    sherpa_type: str | None = None             # gtcrn | dpdfnet
    sample_rate: int | None = None
    speed: str = ""
    note: str = ""
    default: bool = False
    bwe_default: bool = False                  # 仅 kind == "bwe" 有意义：带宽扩展的默认档

    @property
    def size_hint(self) -> int:
        """按第一个来源估算体积。"""
        total = 0
        for f in self.files:
            if f.sources and f.sources[0].size:
                total += f.sources[0].size
        return total


@dataclass
class ExtraSpec:
    id: str
    name: str
    files: list[FileSpec] = field(default_factory=list)
    required_by: str = ""
    size_hint_text: str = ""


@dataclass
class Catalog:
    engines: list[EngineSpec]
    extras: list[ExtraSpec]
    models_root: str

    def engine(self, engine_id: str) -> EngineSpec | None:
        return next((e for e in self.engines if e.id == engine_id), None)

    def extra(self, extra_id: str) -> ExtraSpec | None:
        return next((e for e in self.extras if e.id == extra_id), None)

    @property
    def default_engine(self) -> EngineSpec:
        return next((e for e in self.engines if e.default), self.engines[0])

    def resolve(self, rel: str) -> str:
        """把清单里的相对路径解析为绝对路径（`..` 允许指向项目内其它目录）。"""
        return os.path.normpath(os.path.join(self.models_root, rel))


def load_catalog(project_root: str) -> Catalog:
    models_root = os.path.join(project_root, "models")
    path = os.path.join(project_root, MODELS_JSON)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    engines: list[EngineSpec] = []
    for e in raw.get("engines", []):
        engines.append(
            EngineSpec(
                id=e["id"],
                name=e.get("name", e["id"]),
                kind=e.get("kind", "sherpa"),
                files=_parse_files(e.get("files", [])),
                sherpa_type=e.get("sherpa_type"),
                sample_rate=e.get("sample_rate"),
                speed=e.get("speed", ""),
                note=e.get("note", ""),
                default=bool(e.get("default")),
                bwe_default=bool(e.get("bwe_default")),
            )
        )

    extras: list[ExtraSpec] = []
    for x in raw.get("extras", []):
        extras.append(
            ExtraSpec(
                id=x["id"],
                name=x.get("name", x["id"]),
                files=_parse_files(x.get("files", [])),
                required_by=x.get("required_by", ""),
                size_hint_text=x.get("size_hint", ""),
            )
        )

    return Catalog(engines=engines, extras=extras, models_root=models_root)


def _parse_files(items: list[dict]) -> list[FileSpec]:
    out: list[FileSpec] = []
    for it in items:
        out.append(
            FileSpec(
                path=it["path"],
                sources=[
                    Source(
                        url=s["url"],
                        size=s.get("size"),
                        sha256=s.get("sha256"),
                    )
                    for s in it.get("sources", [])
                ],
            )
        )
    return out


# ---------------------------------------------------------------------------
# 校验 / 状态
# ---------------------------------------------------------------------------

def human_size(n: int | None) -> str:
    if not n:
        return "—"
    units = ["B", "KB", "MB", "GB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:.0f} {u}" if u == "B" else f"{v:.1f} {u}"
        v /= 1024
    return f"{n} B"


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _matches(path: str, src: Source) -> bool:
    """文件是否与某个来源的元数据吻合。未声明 sha256 时只比大小。"""
    try:
        if src.size is not None and os.path.getsize(path) != src.size:
            return False
        if src.sha256:
            return sha256_of(path) == src.sha256
        return src.size is not None
    except OSError:
        return False


def file_ready(fs: FileSpec, catalog: Catalog) -> bool:
    path = catalog.resolve(fs.path)
    if not os.path.isfile(path):
        return False
    return any(_matches(path, s) for s in fs.sources)


def unit_ready(files: list[FileSpec], catalog: Catalog) -> bool:
    return bool(files) and all(file_ready(f, catalog) for f in files)


def engine_ready(engine: EngineSpec, catalog: Catalog) -> bool:
    if engine.kind == "dsp":
        return True  # 纯 numpy 实现，无需下载
    return unit_ready(engine.files, catalog)


def missing_files(files: list[FileSpec], catalog: Catalog) -> list[FileSpec]:
    return [f for f in files if not file_ready(f, catalog)]


# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------

def download_file(
    fs: FileSpec,
    catalog: Catalog,
    *,
    progress: ProgressCb | None = None,
    cancel: CancelCb | None = None,
    log: LogCb | None = None,
    attempts_per_source: int = 2,
) -> str:
    """按顺序尝试各来源，直到某个来源下载并通过校验。

    返回目标绝对路径。所有来源都失败时抛 DownloadError，异常信息包含每个源的失败原因。
    """
    dest = catalog.resolve(fs.path)
    if file_ready(fs, catalog):
        if log:
            log(f"已存在且校验通过：{os.path.basename(dest)}")
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    errors: list[str] = []

    for idx, src in enumerate(fs.sources):
        if cancel and cancel():
            raise DownloadCancelled("已取消")

        part = f"{dest}.part{idx}"
        for attempt in range(1, attempts_per_source + 1):
            try:
                if log:
                    tag = f"（第 {attempt} 次）" if attempt > 1 else ""
                    log(f"下载 {os.path.basename(dest)} ← {src.host}{tag}")
                _fetch(src, part, progress=progress, cancel=cancel)
                _verify_or_raise(part, src)

                # 原子替换：先删旧目标，再改名
                if os.path.exists(dest):
                    os.remove(dest)
                shutil.move(part, dest)
                if log:
                    log(f"完成：{os.path.basename(dest)}  {human_size(os.path.getsize(dest))}")
                return dest
            except DownloadCancelled:
                raise
            except Exception as exc:
                errors.append(f"{src.host}: {exc}")
                if log:
                    log(f"失败：{src.host} → {exc}")
                keep = False
                if attempt < attempts_per_source:
                    # **只有校验失败才丢残片**（2026-10-01 修正）：早先这里无条件 _cleanup，
                    # 于是同源的第二次尝试永远从 0 开始，文档化的"断点续传"实际只在
                    # 取消/崩溃之后才有效。本机网络对 GitHub 时通时断，一个 200 MB–1.8 GB 的
                    # 权重下到 90% 掉线就要整段重来——保留残片 + `Range` 续传正是为此存在的。
                    keep = not isinstance(exc, DownloadCorrupt) and os.path.isfile(part)
                    if not keep:
                        _cleanup(part)
                    if log:
                        if keep:
                            log(f"网络中断，**保留残片 {human_size(os.path.getsize(part))}**，"
                                "下一次尝试从这里续传")
                        else:
                            log("校验失败，已丢弃残片（避免拿坏数据继续续传）")
                    time.sleep(1.0)
                else:
                    # 最后一次尝试失败：**保留**残片，下一次运行仍可从断点继续；
                    # 只在它已被判定为坏数据时清掉。
                    keep = not isinstance(exc, DownloadCorrupt) and os.path.isfile(part)
                    if not keep:
                        _cleanup(part)
                    elif log:
                        log(f"该来源最后一次尝试仍失败；已保留残片 {human_size(os.path.getsize(part))}"
                            f"（{part}），下次会从这里续传，可手动删除）")

    raise DownloadError(
        f"所有 {len(fs.sources)} 个来源均失败（{os.path.basename(dest)}）：\n  "
        + "\n  ".join(errors)
    )


def download_files(
    files: list[FileSpec],
    catalog: Catalog,
    **kw,
) -> list[str]:
    return [download_file(f, catalog, **kw) for f in files]


def _cleanup(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _fetch(src: Source, part: str, *, progress: ProgressCb | None, cancel: CancelCb | None) -> None:
    """把 src 下载到 part，已存在的部分用 Range 续传。"""
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    total = src.size or 0

    # 已下满（且源声明了大小）时无需再发请求
    if total and have == total:
        return
    if total and have > total:
        _cleanup(part)
        have = 0

    headers = {"User-Agent": _UA, "Accept": "*/*"}
    if have:
        headers["Range"] = f"bytes={have}-"

    req = urllib.request.Request(src.url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=45)
    except urllib.error.HTTPError as exc:
        # 416：请求区间越界，说明残片已完整或已损坏，丢弃重来
        if exc.code == 416 and have:
            _cleanup(part)
            raise DownloadError(f"HTTP 416，已丢弃残片，请重试") from exc
        raise DownloadError(f"HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(f"连接失败：{exc.reason}") from exc

    with resp:
        if have and resp.status != 206:
            # 服务端不支持续传，从头来
            _cleanup(part)
            have = 0

        remaining = resp.headers.get("Content-Length")
        if remaining and str(remaining).isdigit():
            total = have + int(remaining)
        elif not total:
            total = 0

        mode = "ab" if have else "wb"
        done = have
        if progress:
            progress(done, total)

        with open(part, mode) as fh:
            while True:
                if cancel and cancel():
                    raise DownloadCancelled("已取消")
                block = resp.read(1 << 16)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if progress:
                    progress(done, total)


def _verify_or_raise(path: str, src: Source) -> None:
    if src.size is not None:
        actual = os.path.getsize(path)
        if actual != src.size:
            raise DownloadCorrupt(f"大小不符：期望 {src.size:,} B，实得 {actual:,} B")
    if src.sha256:
        actual = sha256_of(path)
        if actual != src.sha256:
            raise DownloadCorrupt(f"sha256 不符：期望 {src.sha256[:12]}…，实得 {actual[:12]}…")


# ---------------------------------------------------------------------------
# 便利入口
# ---------------------------------------------------------------------------

def probe_sources(fs: FileSpec, timeout: int = 15) -> list[tuple[str, str]]:
    """逐个探测来源可达性，返回 [(host, 状态描述)]。用于诊断与「检查网络」功能。"""
    out: list[tuple[str, str]] = []
    for src in fs.sources:
        req = urllib.request.Request(src.url, headers={"User-Agent": _UA}, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out.append((src.host, f"HTTP {resp.status}"))
        except urllib.error.HTTPError as exc:
            # 部分镜像不支持 HEAD，返回 405 也算可达
            out.append((src.host, f"HTTP {exc.code}" + ("（可能不支持 HEAD）" if exc.code == 405 else "")))
        except Exception as exc:
            out.append((src.host, f"不可达：{type(exc).__name__}"))
    return out
