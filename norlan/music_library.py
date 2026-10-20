"""Human-approved music assets and deterministic episode cue rendering.

This module never contacts a provider or invents music. Imported/generated audio
is copied into an immutable local library; the caller persists state and records
that approval actions came from the human UI. A music approval binds the exact
track, version and bytes. A cue plan can only use those approved bytes.
"""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

from . import media


class MusicLibraryError(media.MediaError):
    """A local music validation error safe to display in the review room."""

    safe_message = True


_MAX_UPLOAD = 30 * 1024 * 1024
_MAX_LOCAL = 100 * 1024 * 1024
_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
_MIME_EXTENSIONS = {
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a", "audio/m4a": ".m4a", "audio/ogg": ".ogg",
    "application/ogg": ".ogg", "audio/flac": ".flac", "audio/x-flac": ".flac",
}


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _canonical_digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _number(value: object, name: str, *, minimum: float = 0.0,
            maximum: float = 7200.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MusicLibraryError(f"{name}必须是有限数值。")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise MusicLibraryError(f"{name}超出允许范围。")
    return value


def _identity(track_id: str | None, version: int) -> tuple[str, int]:
    track_id = track_id or "music-" + uuid.uuid4().hex[:16]
    if not isinstance(track_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", track_id):
        raise MusicLibraryError("音乐资产 ID 无效。")
    if isinstance(version, bool) or not isinstance(version, int) or not 1 <= version <= 10000:
        raise MusicLibraryError("音乐版本号无效。")
    return track_id, version


def _tracks(library: dict) -> list[dict]:
    if not isinstance(library, dict) or not isinstance(library.get("tracks", []), list):
        raise MusicLibraryError("音乐库格式无效。")
    tracks = library.get("tracks", [])
    if any(not isinstance(track, dict) for track in tracks):
        raise MusicLibraryError("音乐资产格式无效。")
    return tracks


def _find(library: dict, track_id: str, version: int | None = None) -> dict:
    matches = [track for track in _tracks(library) if track.get("id") == track_id
               and (version is None or track.get("version") == version)]
    if not matches:
        raise MusicLibraryError("找不到该音乐版本，请刷新后重试。")
    if len(matches) != 1:
        raise MusicLibraryError("音乐资产 ID 或版本重复，不能猜测需要审批的文件。")
    return matches[0]


def _track_path(track: dict, project: Path | None = None) -> Path:
    raw = track.get("file")
    if not isinstance(raw, str) or not raw:
        raise MusicLibraryError("音乐原件路径缺失。")
    path = Path(raw).expanduser()
    if project is not None:
        project = Path(project).expanduser().resolve()
        path = (path if path.is_absolute() else project / path).resolve()
        if not path.is_relative_to(project / "music" / "library"):
            raise MusicLibraryError("音乐原件必须保存在本项目受控音乐库。")
    elif not path.is_absolute():
        raise MusicLibraryError("音乐相对路径必须提供项目根目录。")
    return path.resolve()


def _verify(track: dict, *, approval: bool, project: Path | None = None) -> tuple[Path, dict]:
    path = _track_path(track, project)
    if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", str(track.get("sha256", ""))):
        raise MusicLibraryError("音乐原件缺失或哈希无效；请重新导入并试听。")
    if _digest(path) != track["sha256"]:
        raise MusicLibraryError("音乐文件已改变，原试听审批失效；请重新导入并确认。")
    metadata = media.probe(path)
    duration = _number(metadata.get("duration"), "音乐实测时长")
    if not metadata.get("has_audio") or metadata.get("has_video") or duration <= 0:
        raise MusicLibraryError("音乐资产必须是有有效时长的独立音频。")
    declared = _number(track.get("duration"), "音乐记录时长")
    if abs(declared - duration) > 0.05:
        raise MusicLibraryError("音乐时长与原记录不一致，不能沿用试听审批。")
    if approval and (track.get("status") != "approved"
                     or track.get("approved_sha256") != track["sha256"]
                     or track.get("approved_version") != track.get("version")):
        raise MusicLibraryError("配乐尚未经过本人试听确认，不能加入整集。")
    return path, metadata


def import_track(project: Path, source: Path, title: str, *, track_id: str | None = None,
                 version: int = 1, prompt: str = "", provenance: dict | None = None) -> dict:
    """Copy real audio and return an unapproved candidate; caller adds it to state.

    The source may be a finished provider download or a user-selected local file.
    No approval is inferred from an existing filename or matching audio hash.
    """
    track_id, version = _identity(track_id, version)
    if not isinstance(title, str) or not title.strip() or len(title) > 160:
        raise MusicLibraryError("请填写 1–160 字的音乐名称。")
    source = Path(source).expanduser().resolve()
    if (not source.is_file() or source.suffix.lower() not in _EXTENSIONS
            or not 0 < source.stat().st_size <= _MAX_LOCAL):
        raise MusicLibraryError("请导入不超过 100 MB 的 WAV、MP3、M4A、OGG 或 FLAC 音频。")
    destination = Path(project).expanduser().resolve() / "music" / "library"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".music-import-", dir=destination) as raw:
        temporary = Path(raw) / ("source" + source.suffix.lower())
        shutil.copyfile(source, temporary)
        metadata = media.probe(temporary)
        duration = _number(metadata.get("duration"), "音乐实测时长")
        if not metadata.get("has_audio") or metadata.get("has_video") or duration <= 0:
            raise MusicLibraryError("所选文件不是有效的独立音乐音频。")
        digest = _digest(temporary)
        target = destination / f"{track_id}-v{version}-{digest}{source.suffix.lower()}"
        if target.exists():
            if _digest(target) != digest:
                raise MusicLibraryError("音乐库目标已被修改，导入停止。")
        else:
            # Publish exclusively, never overwrite original/candidate media.
            os.link(temporary, target)
    return {
        "id": track_id, "title": title.strip(), "version": version,
        "file": str(target), "sha256": digest, "duration": duration,
        "status": "awaiting_approval", "prompt": str(prompt)[:4000],
        "provenance": dict(provenance or {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def import_data_url(project: Path, data_url: str, title: str, **options) -> dict:
    """Decode a bounded browser audio upload into an unapproved library asset."""
    if not isinstance(data_url, str) or len(data_url) > 4 * ((_MAX_UPLOAD + 2) // 3) + 100:
        raise MusicLibraryError("音乐上传上限为 30 MB。")
    match = re.fullmatch(r"data:([A-Za-z0-9.+/-]+);base64,([A-Za-z0-9+/=\r\n]+)", data_url)
    if not match or match[1].lower() not in _MIME_EXTENSIONS:
        raise MusicLibraryError("仅支持 WAV、MP3、M4A、OGG、FLAC 音频上传。")
    try:
        data = base64.b64decode(match[2].replace("\r", "").replace("\n", ""), validate=True)
    except (ValueError, binascii.Error):
        raise MusicLibraryError("音乐上传数据无效。") from None
    if not 0 < len(data) <= _MAX_UPLOAD:
        raise MusicLibraryError("音乐上传不能为空，且不能超过 30 MB。")
    project = Path(project).expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".music-upload-", dir=project) as raw:
        source = Path(raw) / ("upload" + _MIME_EXTENSIONS[match[1].lower()])
        source.write_bytes(data)
        return import_track(project, source, title, **options)


def approve_track(library: dict, track_id: str, version: int, sha256: str,
                  *, project: Path | None = None) -> dict:
    """Bind a human audition decision to the exact current version and file hash."""
    _identity(track_id, version)
    track = _find(library, track_id, version)
    if track.get("sha256") != sha256:
        raise MusicLibraryError("试听的音乐版本已更新，请刷新并试听当前版本。")
    _verify(track, approval=False, project=project)
    track.update(status="approved", approved_sha256=sha256, approved_version=version,
                 approved_at=datetime.now(timezone.utc).isoformat())
    track.pop("feedback", None)
    return track


def reject_track(library: dict, track_id: str, version: int, sha256: str,
                 feedback: str = "", *, project: Path | None = None) -> dict:
    """Record rejection only; this never submits another paid generation."""
    _identity(track_id, version)
    track = _find(library, track_id, version)
    if track.get("sha256") != sha256:
        raise MusicLibraryError("试听的音乐版本已更新，请刷新后再提交意见。")
    _verify(track, approval=False, project=project)
    track.update(status="rejected", feedback=str(feedback).strip()[:4000])
    for key in ("approved_sha256", "approved_version", "approved_at"):
        track.pop(key, None)
    return track


def approved_tracks(library: dict, *, project: Path | None = None) -> list[dict]:
    """Return a small verified catalog for GPT, excluding media and prior jobs."""
    result = []
    for track in _tracks(library):
        if track.get("status") != "approved":
            continue
        _verify(track, approval=True, project=project)
        item = {key: track.get(key) for key in ("id", "version", "duration", "sha256")}
        # A model needs the approved music's identity and concise intention, not
        # the original provider prompt. Twelve entries stay well under the
        # arrangement method's 20k context limit, leaving room for the episode.
        item.update(title=str(track.get("title", ""))[:120],
                    prompt=str(track.get("prompt", ""))[:480])
        editorial = track.get("provenance", {}).get("editorial", {})
        if isinstance(editorial, dict):
            tags = editorial.get("moods", [])
            if isinstance(tags, list):
                item["tags"] = [tag[:24] for tag in tags if isinstance(tag, str)][:8]
        result.append(item)
    return result


def select_catalog(catalog: list[dict], context: str, *, preferred_ids=(), previous_ids=(), limit=12) -> list[dict]:
    """Shortlist approved metadata locally; GPT never receives a whole folder.

    Matching descriptions is editorial assistance, not analysis of the waveform.
    Exact IDs used by the previous episode remain eligible for theme continuity.
    """
    if type(limit) is not int or not 1 <= limit <= 12 or not isinstance(context, str):
        raise MusicLibraryError("音乐目录选择参数无效。")
    terms = ("悬疑", "紧张", "危机", "工业", "科技", "系统", "动作", "战斗", "追逐", "节奏", "冷静", "低沉",
             "温暖", "情感", "悲伤", "希望", "励志", "胜利", "轻松", "幽默", "神秘", "压迫", "反转")
    wanted = {word for word in terms if word in context}
    preferred_ids, previous_ids = set(preferred_ids), set(previous_ids)
    ranked = []
    for index, track in enumerate(catalog):
        text = " ".join((str(track.get("title", "")), str(track.get("prompt", "")), " ".join(track.get("tags", []))))
        score = sum(3 for term in wanted if term in text)
        score += 10000 if track["id"] in preferred_ids else 0
        score += 4 if track["id"] in previous_ids else 0
        ranked.append((-score, index, track))
    return [dict(row[2]) for row in sorted(ranked, key=lambda row: row[:2])[:limit]]


def write_music_credits(plan: dict, library: dict, output: Path, *, project: Path) -> dict:
    """Publish attribution for only the cues actually used, alongside the film."""
    project = Path(project).resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(project / "exports") or output.suffix != ".txt":
        raise MusicLibraryError("配乐署名说明必须保存在本集导出目录内。")
    ids = list(dict.fromkeys(cue["track_id"] for cue in plan["cues"]))
    lines = ["本集配乐署名与来源", "", "处理说明：本集使用下列音乐的剪辑片段，并调整音量、淡入淡出和混音。",
             "发布本集时请保留需要署名的作者、曲名、来源与许可链接（可放在发布说明中）。", ""]
    for number, tid in enumerate(ids, 1):
        track = _find(library, tid)
        _verify(track, approval=True, project=project)
        source = track.get("provenance", {})
        license = source.get("license", {})
        license = license if isinstance(license, dict) else {"name": str(license)}
        lines.extend([f"{number}. {track['title']}", "作者：" + str(source.get("author") or "用户提供，未登记作者"),
                      "来源：" + str(source.get("source_url") or "本地音乐文件"),
                      "许可：" + str(license.get("name") or "用户提供，许可信息尚未登记")])
        if license.get("url"):
            lines.append("许可链接：" + str(license["url"]))
        if source.get("attribution"):
            lines.append("署名文本：" + str(source["attribution"]))
        lines.append("")
    data = ("\n".join(lines) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != data:
            raise MusicLibraryError("已有配乐署名文件与本版内容不一致，没有覆盖。")
    else:
        with tempfile.TemporaryDirectory(prefix=".music-credits-", dir=output.parent) as raw:
            temporary = Path(raw) / "credits.txt"
            temporary.write_bytes(data)
            os.link(temporary, output)
    return {"file": str(output), "sha256": _digest(output), "track_ids": ids}


def validate_cue_plan(plan: dict, library: dict, duration: float,
                      *, project: Path | None = None) -> dict:
    """Check actual source bounds and approval, without implicit loops or stretch.

    Overlapping cues must form explicit fade-out/fade-in crossfades; no more than
    two cues may play simultaneously. Empty cues are allowed as a deliberate
    silent arrangement, but callers must not label it as a scored episode.
    """
    duration = _number(duration, "本集实测时长")
    if duration <= 0 or not isinstance(plan, dict):
        raise MusicLibraryError("配乐表或本集时长无效。")
    cues = plan.get("cues")
    if not isinstance(cues, list) or len(cues) > 64:
        raise MusicLibraryError("配乐表需要提供最多 64 个音乐段。")
    checked = []
    cache = {}
    for cue in cues:
        if not isinstance(cue, dict):
            raise MusicLibraryError("配乐段格式无效。")
        if set(cue) - {"track_id", "track_version", "track_sha256", "start", "end",
                       "source_start", "gain_db", "fade_in", "fade_out", "reason"}:
            raise MusicLibraryError("配乐表包含未支持的参数，不能执行任意滤镜或循环。")
        track_id = cue.get("track_id")
        if not isinstance(track_id, str):
            raise MusicLibraryError("音乐段缺少有效资产 ID。")
        if track_id not in cache:
            track = _find(library, track_id)
            _verify(track, approval=True, project=project)
            cache[track_id] = track
        track = cache[track_id]
        if (cue.get("track_version", track["version"]) != track["version"]
                or cue.get("track_sha256", track["sha256"]) != track["sha256"]):
            raise MusicLibraryError("配乐表引用的音乐版本已改变，需要重新编排。")
        start = _number(cue.get("start"), "音乐入点", maximum=duration)
        end = _number(cue.get("end"), "音乐出点", maximum=duration)
        if end <= start:
            raise MusicLibraryError("音乐出点必须晚于入点。")
        source_start = _number(cue.get("source_start", 0), "音乐素材入点")
        length = end - start
        if source_start + length > track["duration"] + 0.000001:
            raise MusicLibraryError("音乐取用范围超过实测素材时长；请换段编排，不能隐式循环或拉伸。")
        gain = _number(cue.get("gain_db", -16), "音乐增益", minimum=-60, maximum=0)
        fade_in = _number(cue.get("fade_in", min(0.6, length / 4)), "音乐淡入", maximum=length)
        fade_out = _number(cue.get("fade_out", min(0.6, length / 4)), "音乐淡出", maximum=length)
        if fade_in + fade_out > length + 0.000001:
            raise MusicLibraryError("音乐淡入淡出合计不能超过该音乐段时长。")
        checked.append({
            "track_id": track_id, "track_version": track["version"],
            "track_sha256": track["sha256"], "start": start, "end": end,
            "source_start": source_start, "gain_db": gain,
            "fade_in": fade_in, "fade_out": fade_out,
            "reason": str(cue.get("reason", ""))[:500],
        })
    checked.sort(key=lambda cue: (cue["start"], cue["end"], cue["track_id"]))
    for index, cue in enumerate(checked):
        active = [prior for prior in checked[:index] if prior["end"] > cue["start"]]
        if len(active) > 1:
            raise MusicLibraryError("同一时刻最多允许两段音乐交叉衔接。")
        if active:
            prior = active[0]
            overlap = min(prior["end"], cue["end"]) - cue["start"]
            if (prior["end"] > cue["end"] or prior["fade_out"] + 0.000001 < overlap
                    or cue["fade_in"] + 0.000001 < overlap):
                raise MusicLibraryError("重叠音乐必须安排完整的淡出与淡入交叉衔接。")
    result = {"version": 1, "duration": duration, "summary": str(plan.get("summary", ""))[:2000],
              "cues": checked}
    result["fingerprint"] = _canonical_digest(result)
    return result


def render_music_bed(plan: dict, library: dict, duration: float, output: Path,
                     *, project: Path | None = None, normalize: bool = False) -> dict:
    """Render the explicit cue plan as an episode-length stereo WAV.

    Deliberate gaps remain silent. Existing artifacts are reused only when their
    source-bound fingerprint and bytes match the manifest. Video is untouched.
    The final episode mixer should apply a gain of 1.0 to this already-scaled bed.
    """
    plan = validate_cue_plan(plan, library, duration, project=project)
    duration = plan["duration"]
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() != ".wav":
        raise MusicLibraryError("配乐工作轨请保存为 WAV 文件。")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    sources = [_find(library, cue["track_id"]) for cue in plan["cues"]]
    # One immutable normalization reference per approved track, shared by all
    # its cues. Never independently level adjacent excerpts and create a jump.
    normalization = {}
    if normalize:
        from .music_analysis import normalization_filter
        for track in sources:
            if track["id"] not in normalization:
                normalization[track["id"]] = normalization_filter(
                    _track_path(track, project), 0, track["duration"],
                    cache_dir=Path(project) / "cache" / "music-analysis" if project else output.parent / ".music-analysis",
                    target_lufs=-18, peak_dbfs=-2)
    render_policy = {"version": 2 if normalize else 1, "normalization": normalization}
    if output in [_track_path(track, project) for track in sources]:
        raise MusicLibraryError("配乐合成不能覆盖音乐原件。")
    if output.exists() or manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest["fingerprint"] == plan["fingerprint"] and output.is_file()
                    and manifest.get("render_policy", {"version": 1, "normalization": {}}) == render_policy
                    and manifest["sha256"] == _digest(output)):
                actual = media.probe(output)
                if actual["has_audio"] and abs(actual["duration"] - duration) <= 0.001:
                    return manifest
        except (OSError, ValueError, KeyError):
            pass
        raise MusicLibraryError("已有配乐工作轨与当前安排不一致，请使用新的版本文件名。")
    with tempfile.TemporaryDirectory(prefix=".music-render-", dir=output.parent) as raw:
        temporary = Path(raw)
        candidate = temporary / "music.wav"
        args = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        graph = [f"[0:a]atrim=duration={duration:.9f},asetpts=PTS-STARTPTS[silence]"]
        labels = ["[silence]"]
        for index, (cue, track) in enumerate(zip(plan["cues"], sources), 1):
            args.extend(["-protocol_whitelist", "file,pipe", "-i", str(_track_path(track, project))])
            length = cue["end"] - cue["start"]
            # FFmpeg treats d=0 as "use default sample count", not as a zero-
            # duration fade. Omit that filter entirely for an intentional cut.
            fades = []
            if cue["fade_in"]:
                fades.append(f"afade=t=in:st=0:d={cue['fade_in']:.9f}")
            if cue["fade_out"]:
                fades.append(f"afade=t=out:st={length-cue['fade_out']:.9f}:d={cue['fade_out']:.9f}")
            fade_filters = "".join(value + "," for value in fades)
            norm_filter = normalization.get(track["id"], {}).get("filter", "anull")
            graph.append(
                f"[{index}:a:0]atrim=start={cue['source_start']:.9f}:duration={length:.9f},"
                "asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,"
                f"{norm_filter},"
                f"volume={cue['gain_db']:.9f}dB,"
                f"{fade_filters}"
                f"adelay={round(cue['start']*48000)}S:all=1[cue{index}]"
            )
            labels.append(f"[cue{index}]")
        graph.append("".join(labels) + f"amix=inputs={len(labels)}:duration=first:"
                     "dropout_transition=0:normalize=0,alimiter=limit=0.95:level=false:latency=1,"
                     f"apad,atrim=duration={duration:.9f}[music]")
        media._ffmpeg([*args, "-filter_complex", ";".join(graph), "-map", "[music]",
                       "-t", f"{duration:.9f}", "-ar", "48000", "-ac", "2",
                       "-c:a", "pcm_s16le", str(candidate)])
        metadata = media.probe(candidate)
        if not metadata["has_audio"] or abs(metadata["duration"] - duration) > 0.001:
            raise MusicLibraryError("配乐工作轨时长核对失败，未发布结果。")
        for track in sources:
            _verify(track, approval=True, project=project)
        result = {
            "output": str(output), "duration": metadata["duration"], "sha256": _digest(candidate),
            "fingerprint": plan["fingerprint"], "cue_plan": plan,
            "render_policy": render_policy,
            "music_present": bool(plan["cues"]),
            "warnings": ["音乐按已确认素材编排；剧情适配、对白可懂度及衔接听感仍须本人听审。"],
        }
        staged_manifest = temporary / "manifest.json"
        staged_manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.link(candidate, output)
        os.link(staged_manifest, manifest_path)
    return result
