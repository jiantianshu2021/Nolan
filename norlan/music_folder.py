"""Bounded, local-only music-folder imports into the supervised music library.

The source folder and index are read-only. Every new byte sequence becomes an
unapproved candidate. Rejected/approved audio hashes remain deduplicated, so a
rescan cannot turn an old rejection into another approval request.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from urllib.parse import urlsplit

from . import core, media, music_library


class MusicFolderError(ValueError):
    """A user-actionable local validation error; no provider request was made."""

    safe_message = True


EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".flac"})
MAX_FILES = 100
MAX_ENTRIES = 1000
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_TOTAL_BYTES = 500 * 1024 * 1024
MAX_INDEX_BYTES = 256 * 1024


def _folder(value) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise MusicFolderError("请先选择本地音乐文件夹，或运行 nolan music-folder 项目目录 音乐目录 --scan。")
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise MusicFolderError("音乐文件夹不能是符号链接，请选择原始文件夹。")
    try:
        candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise MusicFolderError("音乐文件夹不存在或无法读取，请检查本机路径。") from None
    if not candidate.is_dir():
        raise MusicFolderError("请选择文件夹，不能把单个音乐文件当作曲库。")
    return candidate


def _text(value, maximum=500) -> str:
    return value.strip()[:maximum] if isinstance(value, str) else ""


def _texts(value, maximum=12, length=80) -> list[str]:
    if isinstance(value, str):
        value = [value]
    return [_text(item, length) for item in value[:maximum] if _text(item, length)] if isinstance(value, list) else []


def _url(value) -> str:
    value = _text(value, 2000)
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            return ""
    except ValueError:
        return ""
    return value


def _metadata(item: dict) -> dict:
    """Keep only bounded editorial/source fields, not arbitrary index payloads."""
    result = {key: text for key, limit in (("id", 100), ("title", 160), ("author", 160), ("source", 160),
              ("attribution", 2000), ("prompt", 1200)) if (text := _text(item.get(key), limit))}
    for key in ("source_url", "download_url", "license_url"):
        if url := _url(item.get(key)):
            result[key] = url
    license_value = item.get("license")
    if isinstance(license_value, str):
        result["license"] = {"name": _text(license_value, 160)}
    elif isinstance(license_value, dict):
        license_info = {key: text for key in ("name", "version") if (text := _text(license_value.get(key), 160))}
        if url := _url(license_value.get("url")):
            license_info["url"] = url
        for key in ("commercial_use", "adaptation", "attribution_required"):
            if type(license_value.get(key)) is bool:
                license_info[key] = license_value[key]
        if license_info:
            result["license"] = license_info
    editorial = item.get("editorial", {})
    if not isinstance(editorial, dict):
        editorial = {}
    normalized_editorial = {key: text for key in ("suggested_use", "avoid", "basis")
                             if (text := _text(editorial.get(key), 800))}
    moods = _texts(editorial.get("moods", item.get("mood", item.get("tags", []))))
    if moods:
        normalized_editorial["moods"] = moods
    energy = _text(item.get("energy"), 80)
    if energy:
        normalized_editorial["energy"] = energy
    if normalized_editorial:
        result["editorial"] = normalized_editorial
    source = item.get("source_metadata", {})
    if isinstance(source, dict):
        normalized_source = {key: values for key in ("instruments", "feel")
                               if (values := _texts(source.get(key)))}
        if text := _text(source.get("isrc"), 80):
            normalized_source["isrc"] = text
        if text := _text(source.get("catalogue_length"), 80):
            normalized_source["catalogue_length"] = text
        bpm = source.get("tempo_bpm")
        if isinstance(bpm, str):
            try:
                bpm = float(bpm.strip())
            except ValueError:
                bpm = None
        if type(bpm) in (int, float) and 0 < bpm <= 400 and math.isfinite(bpm):
            normalized_source["tempo_bpm"] = float(bpm)
        if normalized_source:
            result["source_metadata"] = normalized_source
    verification = item.get("verification", {})
    if isinstance(verification, dict):
        result["verification"] = {key: text for key in ("retrieved_at", "source_evidence_file")
                                   if (text := _text(verification.get(key), 500))}
    return result


def _index(folder: Path) -> dict[str, dict]:
    index = folder / "index.json"
    if not index.exists() and not index.is_symlink():
        return {}
    try:
        info = index.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_INDEX_BYTES:
            raise MusicFolderError("曲库 index.json 必须是不超过 256 KB 的普通 JSON 文件，不能是符号链接。")
        with os.fdopen(os.open(index, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), encoding="utf-8") as handle:
            raw = handle.read(MAX_INDEX_BYTES + 1)
        if len(raw.encode("utf-8")) > MAX_INDEX_BYTES:
            raise MusicFolderError("曲库 index.json 超过 256 KB，请精简曲目说明。")
        value = json.loads(raw)
    except MusicFolderError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise MusicFolderError("曲库 index.json 不是可读取的 UTF-8 JSON，请检查格式后重新扫描。") from None
    if not isinstance(value, dict) or not isinstance(value.get("tracks"), list) or len(value["tracks"]) > MAX_FILES:
        raise MusicFolderError("曲库 index.json 需要 tracks 列表，最多记录 100 首音乐。")
    result = {}
    for item in value["tracks"]:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            raise MusicFolderError("index.json 的每首音乐都需要 file 文件名。")
        filename = item["file"]
        if Path(filename).name != filename or filename in {"", ".", ".."} or "\\" in filename:
            raise MusicFolderError("index.json 只允许本文件夹内的单个文件名，不能包含子目录或外部路径。")
        if filename in result:
            raise MusicFolderError("index.json 包含重复文件名，请修正后扫描。")
        digest = item.get("sha256")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                   or any(char not in "0123456789abcdef" for char in digest)):
            raise MusicFolderError("index.json 中的 sha256 必须是 64 位小写文件摘要。")
        result[filename] = {**_metadata(item), "expected_sha256": digest}
    return result


def _entries(folder: Path) -> tuple[list[Path], list[dict]]:
    accepted, skipped = [], []
    total_size = 0
    try:
        with os.scandir(folder) as contents:
            for count, entry in enumerate(contents, 1):
                if count > MAX_ENTRIES:
                    raise MusicFolderError("音乐文件夹超过 1000 个目录项，请建立只存配乐的小曲库。")
                path = Path(entry.path)
                if path.suffix.lower() not in EXTENSIONS:
                    continue
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    skipped.append({"name": entry.name, "reason": "不是普通音频文件；符号链接和子目录不会导入。"})
                    continue
                if not 0 < info.st_size <= MAX_FILE_BYTES:
                    skipped.append({"name": entry.name, "reason": "音频不能为空，单首不能超过 100 MB。"})
                    continue
                accepted.append(path)
                total_size += info.st_size
                if len(accepted) > MAX_FILES or total_size > MAX_TOTAL_BYTES:
                    raise MusicFolderError("每个曲库最多 100 首、合计 500 MB；请先分成较小的音乐文件夹。")
    except MusicFolderError:
        raise
    except OSError:
        raise MusicFolderError("无法读取音乐文件夹，请在系统设置中允许 Nolan 访问该目录。") from None
    return sorted(accepted, key=lambda path: path.name.casefold()), skipped[:100]


def _snapshot(source: Path, destination: Path) -> str:
    """Open without following links and bound bytes even if the source changes."""
    digest = hashlib.sha256()
    copied = 0
    try:
        with os.fdopen(os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_FILE_BYTES:
                raise MusicFolderError("音频源文件已改变或超过 100 MB，请检查原文件。")
            with destination.open("xb") as target:
                while chunk := handle.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > MAX_FILE_BYTES:
                        raise MusicFolderError("音频源文件超过 100 MB，导入已停止。")
                    digest.update(chunk)
                    target.write(chunk)
            after = os.fstat(handle.fileno())
            if (after.st_size, after.st_mtime_ns) != (info.st_size, info.st_mtime_ns) or copied != info.st_size:
                raise MusicFolderError("扫描期间音频源文件发生改变，请文件保存完成后重新扫描。")
    except MusicFolderError:
        raise
    except OSError:
        raise MusicFolderError("无法安全读取音频源文件，可能已移动、删除或变为符号链接。") from None
    return digest.hexdigest()


def _prompt(metadata: dict) -> str:
    editorial = metadata.get("editorial", {})
    source = metadata.get("source_metadata", {})
    parts = [metadata.get("prompt", ""), editorial.get("suggested_use", "")]
    for label, values in (("情绪标签", editorial.get("moods", [])),
                          ("乐器", source.get("instruments", [])), ("素材方风格说明", source.get("feel", []))):
        if values:
            parts.append(label + "：" + "、".join(values))
    if source.get("tempo_bpm"):
        parts.append("素材方标注速度：" + str(source["tempo_bpm"]) + " BPM")
    if editorial.get("avoid"):
        parts.append("使用限制：" + editorial["avoid"])
    return "；".join(part for part in parts if part)[:450]


def _register(root, folder: Path) -> None:
    state = core.load_project(root)
    music = state["settings"].get("music", {})
    if music.get("mode") == "library" and music.get("folder") == str(folder):
        return
    with core.transaction(root) as state:
        state["settings"].setdefault("music", {}).update(mode="library", folder=str(folder))
        core.event(state, "music_folder_configured", folder=str(folder))


def configure_folder(root, folder) -> dict:
    """Register a local folder without scanning, starting work or changing media."""
    folder = _folder(folder)
    with core.run_lock(root):
        _register(root, folder)
    return core.load_project(root)


def scan_folder(root, folder=None) -> dict:
    """Manual API/CLI entry point. Production callers already holding the lock
    use scan_folder_locked instead; neither entry point starts production.
    """
    with core.run_lock(root):
        return scan_folder_locked(root, folder)


def scan_folder_locked(root, folder=None) -> dict:
    """Import new audio candidates; caller must hold core.run_lock(root)."""
    root = Path(root).resolve()
    state = core.load_project(root)
    folder = _folder(folder if folder is not None else state["settings"].get("music", {}).get("folder"))
    metadata = _index(folder)
    sources, skipped = _entries(folder)
    _register(root, folder)
    known = {track.get("sha256") for track in state.get("music_library", {}).get("tracks", [])}
    imported, duplicates = [], 0
    with tempfile.TemporaryDirectory(prefix=".music-folder-", dir=root) as raw:
        scratch = Path(raw)
        for index, source in enumerate(sources):
            item = metadata.get(source.name, {})
            try:
                snapshot = scratch / (str(index) + source.suffix.lower())
                digest = _snapshot(source, snapshot)
                if item.get("expected_sha256") and item["expected_sha256"] != digest:
                    raise MusicFolderError("文件摘要与 index.json 不一致，请核对下载原件。")
                if digest in known:
                    duplicates += 1
                    continue
                details = {key: value for key, value in item.items() if key != "expected_sha256"}
                provenance = {"source": "local_folder", "folder": str(folder), "filename": source.name,
                              "license_status": "declared" if details.get("license") else "unknown", **details}
                title = item.get("title") or source.stem[:160]
                track = music_library.import_track(root, snapshot, title, track_id="local-" + digest[:24],
                    prompt=_prompt(details), provenance=provenance)
                track["file"] = str(Path(track["file"]).relative_to(root))
                track["metadata"] = details
                with core.transaction(root) as current:
                    library = current.setdefault("music_library", {"version": 1, "tracks": []})
                    # Defensive recheck for callers that did not obey the lock contract.
                    if any(existing.get("sha256") == digest for existing in library["tracks"]):
                        duplicates += 1
                        known.add(digest)
                        continue
                    library["tracks"].append(track)
                    core.event(current, "music_candidate_imported", id=track["id"], version=track["version"],
                               sha256=digest, source="local_folder")
                known.add(digest)
                imported.append({key: track[key] for key in ("id", "title", "version", "sha256", "status", "duration")})
            except (MusicFolderError, music_library.MusicLibraryError) as exc:
                skipped.append({"name": source.name, "reason": str(exc)[:300]})
            except media.MediaError:
                skipped.append({"name": source.name, "reason": "文件不是可读取的独立音频，或本机缺少 FFmpeg/ffprobe。"})
            except (OSError, ValueError):
                skipped.append({"name": source.name, "reason": "音频无法导入，请检查格式、文件权限和 FFmpeg 安装。"})
    report = {"folder": str(folder), "imported": imported, "duplicates": duplicates,
              "skipped": skipped[:100], "total_candidates": len(sources),
              "scanned_at": datetime.now(timezone.utc).isoformat()}
    with core.transaction(root) as state:
        state["settings"]["music"]["folder_scan"] = report
    return report
