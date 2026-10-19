"""A private, localhost-only review room. No remote assets or dependencies."""
from __future__ import annotations

import fcntl
import hashlib
import json
import mimetypes
import re
import secrets
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .regroup import PUBLIC_ERRORS as SEGMENT_ERRORS


MAX_BODY = 32_768
MAX_REFERENCE_BODY = 12 * 1024 * 1024
INPUT_ERRORS = frozenset({
    "这张参考图片已经上传，无需重复添加。",
    "每项资产最多上传 9 张参考图片。",
    "参考图片每张不能超过 8 MiB。",
    "参考图片不能为空，每张不能超过 8 MiB。",
    "参考图片仅支持 PNG、JPEG 或 WebP 的有效图片数据。",
    "图片实际格式与声明类型不一致，请使用 PNG、JPEG 或 WebP 图片。",
    "参考图片无法完整解码，请重新导出为有效的 PNG、JPEG 或 WebP。",
    "请提供有效的静态参考图：单边不超过 8192 像素，总像素不超过 4000 万。",
    "验证参考图片需要 FFmpeg（包含 ffprobe），请先安装。",
    "由 GPT 决定时请清空自定义描述和参考图，或选择由你提供描述。",
    "请填写资产描述，或提供至少一张参考图片。",
    "资产输入已更新，请刷新网页后重新确认。",
    "尚有在途或结果不明的调用，请等待落盘或完成对账后再修改资产输入。",
    "请先批准新的分段方案，或选择保留原分段，再继续制作。",
    "请先完成人工要求返工的上游片段并批准新版本，再确认这段视频的衔接",
    "衔接复核来源缺失，请先检查上游片段记录",
    "生成失败记录尚未核实，不能再次提交；请先核实原调用结果",
    "本集仍有未经人工批准当前版本的视频片段",
    "视频文件不属于当前版本，请先恢复原生成任务并刷新后重新审阅",
    "请先确认或取消新增资产方案。",
    "请在片段下确认提示词后再生成。",
    "请先查看资产失败原因，并明确提出重试意见。",
})
MEDIA_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".m4v", ".mov", ".webm", ".wav", ".mp3", ".m4a", ".ogg", ".srt", ".vtt"})


def _core():
    from . import core
    return core


def _run(project_dir: Path) -> dict:
    from .pipeline import run
    return run(project_dir)


def _pick(value: dict, keys: tuple[str, ...]) -> dict:
    return {key: value[key] for key in keys if key in value}


def _public_cost_sync_result(value: dict) -> dict:
    """Expose counts and bounded job identifiers, never arbitrary exceptions."""
    if not isinstance(value, dict) or not isinstance(value.get("errors"), list):
        raise ValueError("费用同步摘要无效")
    result = {}
    for key in ("synced", "skipped", "unavailable"):
        count = value.get(key)
        if type(count) is not int or count < 0:
            raise ValueError("费用同步摘要无效")
        result[key] = count
    result["error_count"] = len(value["errors"])
    errors = []
    for item in value["errors"][:100]:
        job_id = item.get("job_id", "") if isinstance(item, dict) else ""
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", job_id):
            job_id = ""
        errors.append({"job_id": job_id, "error": "该调用用量暂时无法同步，请检查任务 ID、权限或服务状态。"})
    result["errors"] = errors
    return result


def _version(value):
    return value if type(value) is int and value >= 0 else 0


def _valid_sha(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _public_timestamps(record):
    from .progress import _date
    return {key: record[key] for key in ("created_at", "completed_at")
            if isinstance(record.get(key), str) and len(record[key]) <= 64 and _date(record[key])}


def _public_video_path(value, prefix):
    if not isinstance(value, str) or len(value) > 512 or not value.startswith(prefix + "/") or "\\" in value or "\x00" in value:
        return False
    parts = value.split("/")
    return (not any(part in {"", ".", ".."} for part in parts)
            and Path(value).suffix.lower() in {".mp4", ".m4v", ".mov", ".webm"})


def _file_version(shot, file):
    match = re.fullmatch(re.escape(str(shot.get("id", ""))) + r"-v([1-9]\d*)\.(?:mp4|m4v|mov|webm)",
                         Path(file).name) if isinstance(file, str) else None
    return int(match[1]) if match else None


def _current_shot_artifact(shot):
    if not shot.get("file") or not shot.get("sha256"):
        return False
    file_version = _file_version(shot, shot["file"])
    return file_version is None or file_version == _version(shot.get("version"))


def _bound_shot_job(state, shot):
    job = state.get("jobs", {}).get(shot.get("job_id"), {})
    if not isinstance(job, dict) or job.get("kind") != "video" or job.get("item_id") != shot.get("id"):
        return {}
    version = _file_version(shot, job.get("output"))
    return job if version is None or version == _version(shot.get("version")) else {}


def _public_shot_failures(state, shot):
    """Expose only sanitized diagnostics, bound to the current attempt or history."""
    from .failures import sanitize_failure
    from .core import execution_unsettled
    job = _bound_shot_job(state, shot)
    unknown = job.get("state") == "uncertain" or job.get("accounting") == "unknown"
    recoverable = bool(not unknown and job.get("task_id") and job.get("state") in {"submitted", "download_pending"}
                       and isinstance(job.get("failure"), dict) and job["failure"])
    no_video = not _current_shot_artifact(shot)
    failed = no_video and shot.get("status") != "awaiting_prompt_approval" and (shot.get("status") == "generation_failed" or
                          shot.get("status") == "review_required" or
                          job.get("state") == "failed")

    def diagnostic(value, uncertain=False):
        result = sanitize_failure(value) if isinstance(value, dict) and value else {}
        if not result or not result.get("message"):
            result = {"stage": "unknown", "message": "旧版本未保存具体失败原因。", "uncertain": uncertain}
        if uncertain:
            result["uncertain"] = True
        return result

    result = {}
    if unknown or failed or recoverable:
        result["failure"] = diagnostic(job.get("failure"), unknown)
        result["failure_recoverable"] = recoverable
        if unknown and not job.get("failure"):
            result["failure"]["message"] = "调用结果尚未核实，不能再次提交。请先核实原任务结果。"
        unsettled = any(execution_unsettled(item) for item in state.get("jobs", {}).values() if isinstance(item, dict))
        result["can_retry"] = bool(failed and shot.get("generation_error") and job.get("state") == "failed"
                                   and job.get("accounting") in {"spent", "released"}
                                   and not result["failure"].get("uncertain") and not unsettled
                                   and _version(shot.get("version")) > 0)
        result["generation_error"] = "本次没有可审视频。不会自动再次提交；请查看失败原因后决定是否重试一次。"
        if recoverable:
            result["generation_error"] = "原任务暂未恢复完成。请查看错误详情；继续操作只恢复原任务，不会重新生成。"
    history, seen = [], set()
    for attempt in shot.get("attempts", []):
        if not isinstance(attempt, dict):
            continue
        job_id = attempt.get("job_id")
        if not isinstance(job_id, str) or job_id in seen:
            continue
        seen.add(job_id)
        previous = state.get("jobs", {}).get(job_id, {})
        if (not isinstance(previous, dict) or previous.get("kind") != "video"
                or previous.get("item_id") != shot.get("id")):
            continue
        version = _file_version(shot, previous.get("output")) or _version(attempt.get("version"))
        details = [value for value in previous.get("failure_history", []) if isinstance(value, dict)]
        if not details and previous.get("state") in {"failed", "uncertain"}:
            details = [previous.get("failure")]
        for value in details:
            detail = diagnostic(value, previous.get("state") == "uncertain")
            if job_id == shot.get("job_id") and detail == result.get("failure"):
                continue
            row = {"failure": detail, **_public_timestamps(previous)}
            if version:
                row["version"] = version
            history.append(row)
    result["failure_history"] = history
    return result


def _public_shot_continuity(state, episode, shot):
    """Describe the saved engine references without exposing source URLs."""
    result = {}
    manifest = shot.get("reference_manifest")
    context = manifest.get("context") if isinstance(manifest, dict) else None
    previous = context.get("previous_actual") if isinstance(context, dict) else None
    if isinstance(previous, dict):
        mode = previous.get("reference_mode")
        source_id = previous.get("shot_id")
        source_index = next((index + 1 for index, item in enumerate(episode.get("shots", []))
                             if isinstance(item, dict) and item.get("id") == source_id), None)
        key, role = ("video_references", "previous_original_video") if mode == "original_video" else (
            "image_references", "previous_official_tail")
        references = context.get(key)
        referenced = isinstance(references, list) and any(
            isinstance(item, dict) and item.get("role") == role and item.get("shot_id") == source_id
            and type(item.get("index")) is int and item["index"] > 0 and _valid_sha(item.get("sha256"))
            for item in references)
        if source_index and referenced and mode in {"original_video", "official_tail"}:
            label = (f"衔接参考：第 {source_index} 段原始视频" if mode == "original_video"
                     else f"衔接参考：第 {source_index} 段官方尾帧")
            result["continuity_reference"] = {"mode": mode, "label": label}
    job = _bound_shot_job(state, shot)
    if _current_shot_artifact(shot) and isinstance(job.get("official_tail_error"), dict) and job["official_tail_error"]:
        result["official_tail_warning"] = "本段视频可正常审阅；官方尾帧尚未保存，后续衔接需先恢复原任务素材。"
    return result


def _public_prompt_candidate(value):
    """Expose the exact human-review prompt, never request config or URLs."""
    if not isinstance(value, dict):
        return None
    result = _pick(value, ("id", "version", "revision", "prompt", "dialogue", "continuity_notes", "request_hash", "duration"))
    manifest = value.get("reference_manifest")
    context = manifest.get("context", manifest) if isinstance(manifest, dict) else {}
    context = context if isinstance(context, dict) else {}
    references = {}
    for key in ("image_references", "audio_references", "video_references"):
        rows = context.get(key, [])
        if isinstance(rows, list):
            references[key] = [_pick(row, ("index", "role", "asset_id", "character_id", "shot_id", "version"))
                               for row in rows[:20] if isinstance(row, dict)]
    previous = context.get("previous_actual")
    if isinstance(previous, dict):
        references["previous_actual"] = _pick(previous, ("shot_id", "reference_mode", "same_scene"))
    assets = context.get("assets")
    if isinstance(assets, list):
        from .core import is_internal_voice_source
        internal_ids = {asset.get("id") for asset in assets if isinstance(asset, dict) and is_internal_voice_source(asset)}
        references["assets"] = []
        for asset in assets[:100]:
            if not isinstance(asset, dict) or is_internal_voice_source(asset):
                continue
            item = _pick(asset, ("id", "kind", "name", "version"))
            source = _public_voice_source(asset, assets)
            if source:
                item["voice_source"] = source
            references["assets"].append(item)
        references["image_references"] = [row for row in references.get("image_references", [])
                                           if row.get("asset_id") not in internal_ids]
    result["reference_manifest"] = references
    return result


def _public_voice_source(asset, assets, audio_policy=None):
    """An offscreen source is descriptive voice metadata, never an approval card."""
    from .core import is_internal_voice_source
    if asset.get("kind") != "voice":
        return None
    source = next((item for item in assets if isinstance(item, dict)
                   and item.get("id") == asset.get("character_id") and is_internal_voice_source(item)), None)
    if source is None:
        return None
    from .failures import sanitize_failure
    safe = sanitize_failure({"message": source.get("name") or "画外播报声源",
                             "provider_message": source.get("prompt", "")})
    result = {"name": safe["message"], "description": safe.get("provider_message", ""), "offscreen": True}
    broadcast = audio_policy.get("system_broadcast") if isinstance(audio_policy, dict) else None
    result["purpose"] = "系统通知与任务播报" if isinstance(broadcast, dict) and broadcast.get("voice_id") == asset.get("id") else "画外播报"
    return result


def _public_asset_amendment(value, state=None):
    if not isinstance(value, dict) or value.get("status") != "candidate":
        return None
    result = _pick(value, ("id", "status", "target_shot_id", "target_version", "feedback", "created_at"))
    from .core import is_internal_voice_source
    state = state if isinstance(state, dict) else {}
    additions = [asset for asset in value.get("additions", []) if isinstance(asset, dict)]
    assets = list(state.get("assets", [])) + additions
    internal_ids = {asset.get("id") for asset in assets if is_internal_voice_source(asset)}
    result["additions"] = []
    for asset in additions:
        if is_internal_voice_source(asset):
            continue
        item = _pick(asset, ("id", "kind", "name", "prompt", "character_id", "audition_text", "audition_reason"))
        source = _public_voice_source(asset, assets, value.get("audio_policy", state.get("audio_policy")))
        if source:
            item["voice_source"] = source
        result["additions"].append(item)
    result["shot_updates"] = []
    for shot in value.get("shot_updates", []):
        if isinstance(shot, dict):
            item = _pick(shot, ("id", "add_asset_ids", "prompt", "dialogue", "continuity_notes"))
            if isinstance(item.get("add_asset_ids"), list):
                item["add_asset_ids"] = [identifier for identifier in item["add_asset_ids"] if identifier not in internal_ids]
            result["shot_updates"].append(item)
    audio_policy = value.get("audio_policy")
    broadcast = audio_policy.get("system_broadcast") if isinstance(audio_policy, dict) else None
    if isinstance(broadcast, dict):
        result["audio_policy"] = {"system_broadcast": _pick(broadcast, ("enabled", "character_id", "voice_id", "instruction"))}
    return result


def _public_asset_failure(state, asset):
    """Keep failure diagnostics attached to the failed asset's own media job."""
    if asset.get("status") != "generation_failed":
        return {}
    from .failures import sanitize_failure
    from .core import execution_unsettled
    value = asset.get("generation_error")
    diagnostic = sanitize_failure({"message": value}) if isinstance(value, str) and value else {}
    result = {"generation_error": diagnostic.get("message") or "本次资产生成失败，旧版本保留，等待你处理。"}
    job = state.get("jobs", {}).get(asset.get("job_id"), {})
    expected = "voice" if asset.get("kind") == "voice" else "image"
    if (not isinstance(job, dict) or job.get("kind") != expected or job.get("item_id") != asset.get("id")):
        result["can_retry"] = False
        return result
    failure = sanitize_failure(job.get("failure")) if isinstance(job.get("failure"), dict) else {}
    uncertain = job.get("state") == "uncertain" or job.get("accounting") == "unknown"
    if uncertain:
        failure.update(uncertain=True)
    if failure:
        result["failure"] = failure
    unsettled = any(execution_unsettled(item) for item in state.get("jobs", {}).values() if isinstance(item, dict))
    result["can_retry"] = bool(not uncertain and not unsettled and not failure.get("uncertain")
                               and job.get("state") == "failed" and job.get("accounting") in {"spent", "released"}
                               and _version(asset.get("version")) > 0)
    return result


def _public_shot_report(record, version):
    """Only publish an advisory report bound to this exact video's hash."""
    review = record.get("review")
    if (not isinstance(review, dict) or not _valid_sha(record.get("sha256"))
            or review.get("sha256") != record["sha256"]
            or ("version" in review and review["version"] != version)):
        return None
    verdict = review.get("verdict")
    identity = json.dumps([record.get("file"), version, record["sha256"]], ensure_ascii=True)
    result = {"version": version, "key": "report-" + hashlib.sha256(identity.encode()).hexdigest(),
              "verdict": verdict if isinstance(verdict, str) and verdict in {"pass", "fail", "uncertain"} else "uncertain"}
    result["issues"] = [value[:8000] for value in review.get("issues", [])[:100] if isinstance(value, str)] if isinstance(review.get("issues"), list) else []
    for key in ("observed_end_state", "repair_prompt", "scope"):
        if isinstance(review.get(key), str):
            result[key] = review[key][:16000]
    for key in ("audio_checked", "whole_video_checked", "audio_assessed", "report_unavailable", "review_interrupted"):
        if type(review.get(key)) is bool:
            result[key] = review[key]
    return result


def _public_shot_history(shot):
    history, seen = [], set()
    for record in shot.get("history", []):
        if (not isinstance(record, dict) or not _public_video_path(record.get("file"), "media")
                or not _valid_sha(record.get("sha256"))):
            continue
        identity = (record["file"], record["sha256"])
        if identity in seen:
            continue
        seen.add(identity)
        item = {"file": record["file"], **_public_timestamps(record)}
        file_version = _file_version(shot, record["file"])
        item["version"] = file_version or _version(record.get("version"))
        report = _public_shot_report(record, item["version"])
        if report and (not record.get("version") or record["version"] == item["version"]):
            item["review"] = report
        history.append(item)
    return history


def _public_rough_cuts(episode):
    """Expose explicit cut files and distinguish current from obsolete footage."""
    from .core import human_approved_shot

    def available(shot):
        return (shot.get("status") in {"accepted", "reviewing", "review_required"}
                and _public_video_path(shot.get("file"), "media")
                and _valid_sha(shot.get("sha256")) and _current_shot_artifact(shot))

    output, seen = [], set()
    shots = episode.get("shots", [])
    sources = [(record, False) for record in episode.get("rough_cuts", [])]
    sources += [(record, True) for record in episode.get("reviews", [])]
    for record, legacy in sources:
        if (not isinstance(record, dict) or not _public_video_path(record.get("file"), "exports")
                or not _valid_sha(record.get("sha256")) or not _valid_sha(record.get("key"))):
            continue
        identity = (record["file"], record["sha256"])
        if identity in seen:
            continue
        seen.add(identity)
        count = record.get("shot_count")
        manifest = record.get("shots")
        current = False
        public_manifest = []
        if not legacy:
            valid = (type(count) is int and 0 < count <= len(shots) and isinstance(manifest, list)
                     and len(manifest) == count and all(isinstance(row, dict) and isinstance(row.get("id"), str)
                         and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", row["id"])
                         and type(row.get("version")) is int and row["version"] > 0
                         and _valid_sha(row.get("sha256")) for row in manifest))
            if valid:
                public_manifest = [_pick(row, ("id", "version")) for row in manifest]
                current = all(row["id"] == shot.get("id") and row["version"] == _version(shot.get("version"))
                              and row["sha256"] == shot.get("sha256") and available(shot)
                              for row, shot in zip(manifest, shots[:count]))
        else:
            match = re.fullmatch(re.escape(str(episode.get("id", ""))) + r"-rough-([1-9]\d*)-([0-9a-f]{16})\.mp4",
                                 Path(record["file"]).name)
            count = int(match[1]) if match else 0
            if match and 0 < count <= len(shots) and match[2] == record["key"][:16]:
                originals = shots[:count]
                if all(available(shot) for shot in originals):
                    calculated = hashlib.sha256("-".join(shot["sha256"] for shot in originals).encode()).hexdigest()
                    current = calculated == record["key"]
                    if current:
                        public_manifest = [{"id": shot["id"], "version": _version(shot.get("version"))} for shot in originals]
        item = {"key": record["key"], "file": record["file"], **_public_timestamps(record)}
        item.update(shot_count=count if type(count) is int and 0 < count <= len(shots) else 0,
                    current=current, legacy=legacy, shots=public_manifest)
        # A cut is immutable, but approval can change after it was rendered.
        # Derive pending clips from the live prefix, never the saved cut label.
        item["pending_shot_ids"] = [shot["id"] for shot in shots[:count] if not human_approved_shot(shot)] if current else []
        matched = next((review for review in reversed(episode.get("reviews", []))
                        if isinstance(review, dict) and review.get("key") == record["key"]
                        and review.get("file") == record["file"] and review.get("sha256") == record["sha256"]), None)
        if matched:
            report = _public_shot_report({"file": record["file"], "sha256": record["sha256"], "review": matched}, None)
            if report:
                item["review"] = report
        output.append(item)
    return output


def _shot_progress(state, episode, shot, episode_status, stamp):
    """Show durable phases and elapsed time, never invented provider percent."""
    from .progress import _elapsed
    version = _version(shot.get("version"))
    artifact = _current_shot_artifact(shot)
    status = shot.get("status")
    job = _bound_shot_job(state, shot)
    hint = episode.get("progress_stage", {})
    hint = hint if isinstance(hint, dict) and hint.get("shot_id") == shot.get("id") else {}
    phase = hint.get("stage")
    job_phase = {"submitting": "shot_submission", "submitted": "shot_generation",
                 "download_pending": "shot_download", "completed": "shot_checks"}.get(job.get("state"))
    labels = {"shot_preparation": "正在整理参考与编写提示词", "shot_submission": "正在提交生成请求",
              "shot_generation": "视频生成中", "shot_download": "视频已生成，正在下载",
              "shot_checks": "已生成，正在检查技术参数", "shot_review": "已生成，正在检查画面与衔接"}
    active = job.get("state") in {"submitting", "submitted", "download_pending"}
    result = {"stage": "queued", "label": "待返工" if version else "待生成", "version": version,
              "target_version": version, "elapsed_seconds": None, "stage_elapsed_seconds": None,
              "elapsed_basis": "生成任务用时", "completed": artifact, "accepted": status == "accepted" and artifact,
              "is_working": False, "waiting_for_human": False, "has_active_task": active}
    if status == "accepted" and artifact:
        result.update(stage="accepted", label="已生成，已通过")
    elif job.get("state") == "uncertain" or job.get("accounting") == "unknown":
        result.update(stage="unknown", label="调用结果待核实", waiting_for_human=True)
    elif status == "awaiting_prompt_approval":
        candidate = shot.get("prompt_candidate", {})
        result.update(stage="awaiting_prompt_approval", label="提示词已准备，等待你确认生成", waiting_for_human=True,
                      target_version=_version(candidate.get("version")) or version + 1)
    elif status == "generation_failed" or (status == "review_required" and not artifact) or job.get("state") == "failed":
        result.update(stage="failed", label="生成失败，等待处理", waiting_for_human=True)
    elif status == "review_required":
        report = _public_shot_report(shot, version) if artifact else None
        interrupted = report and report.get("report_unavailable") is True and report.get("review_interrupted") is True
        result.update(stage="review_required", label="视频已生成，GPT 报告中断，等待你审批" if interrupted else "已生成，等待你审批", waiting_for_human=True)
    else:
        if status == "reviewing" and artifact:
            phase = "shot_review"
        elif job_phase and not (job.get("state") == "completed" and phase in {"shot_checks", "shot_review"}):
            phase = job_phase
        elif phase not in labels and episode_status.get("current_shot") == shot.get("id"):
            phase = episode_status.get("stage")
        if phase in labels:
            result.update(stage=phase, label=labels[phase], is_working=True)
            if phase == "shot_preparation":
                result["target_version"] = version + 1
        if state.get("paused") or state.get("status") == "paused":
            if result["is_working"] or active:
                result.update(stage="paused", label="已暂停跟踪；原任务可能仍在生成" if active else "已暂停", is_working=False)
        elif state.get("status") == "blocked" and result["is_working"]:
            result.update(stage="blocked", label="制作已停止，请查看项目提示", is_working=False, waiting_for_human=True)
    # A completed provider job is frozen at its own update time. Subsequent
    # episode work must not keep increasing an earlier shot's displayed cost in time.
    if job.get("created_at"):
        end = job.get("updated_at") if job.get("state") in {"completed", "failed", "uncertain"} else stamp
        result["elapsed_seconds"] = _elapsed(job["created_at"], end)
    elif phase == "shot_preparation" and result["is_working"]:
        result["elapsed_seconds"] = _elapsed(hint.get("started_at"), stamp)
        result["elapsed_basis"] = "准备阶段用时"
    if result["is_working"]:
        stage_start = hint.get("started_at") if hint.get("stage") == result["stage"] else job.get("updated_at") or job.get("created_at")
        result["stage_elapsed_seconds"] = _elapsed(stage_start, stamp)
    return result


def public_state(state: dict) -> dict:
    """Keep credentials, source novel, provider responses and settings off the UI API."""
    result = _pick(state, ("id", "title", "mode", "status", "paused", "created_at", "updated_at"))
    from .core import is_nonblocking_advisory_unknown
    result["pending_advisory_count"] = sum(is_nonblocking_advisory_unknown(job)
                                            for job in state.get("jobs", {}).values() if isinstance(job, dict))
    result["pending_advisory_notice"] = (f"有 {result['pending_advisory_count']} 次 GPT 审查调用的费用或额度待核实，未自动重试。"
                                         if result["pending_advisory_count"] else "")
    result["plan"] = _pick(state.get("plan", {}), ("status", "version", "style", "ratio", "feedback"))
    if state.get("last_error"):
        message = str(state["last_error"])[:1500]
        message = re.sub(r"(?i)\b(?:sk|ark)[-_][A-Za-z0-9_-]{12,}", "[已隐藏凭据]", message)
        message = re.sub(r"(?i)(api[_ -]?key|authorization|secret|token)\s*[:=]\s*[^\s,;]+", r"\1=[已隐藏]", message)
        result["last_error"] = message
    from .billing import summarize_costs
    budget = state.get("budget", {})
    result["budget"] = _pick(budget, ("limit", "spent", "reserved", "unknown", "currency", "available"))
    result["budget"].update(summarize_costs(state.get("jobs", {}), budget.get("currency", "CNY")))
    from .asset_inputs import input_state, has_started, confirmed
    from .progress import episode_progress
    from .core import now
    result["assets"] = []
    from .core import is_internal_voice_source
    for asset in state.get("assets", []):
        if is_internal_voice_source(asset):
            continue
        item = _pick(asset, ("id", "kind", "name", "prompt", "status", "version", "file", "audio_file", "feedback", "audition_profile", "voice_only", "character_id"))
        source = _public_voice_source(asset, state.get("assets", []), state.get("audio_policy"))
        if source:
            item["voice_source"] = source
        data = input_state(asset)
        item["creation_input"] = _pick(data, ("mode", "description", "revision", "confirmed"))
        item["creation_input"]["references"] = [_pick(ref, ("id", "file", "mime", "bytes")) for ref in data.get("references", [])]
        item["input_editable"] = not has_started(state, asset) and not (asset.get("kind") == "character" and asset.get("voice_only") is True)
        item["input_required"] = item["input_editable"] and not confirmed(asset)
        if asset.get("status") == "generation_failed":
            item.pop("file", None)
            item.pop("audio_file", None)
            item.update(_public_asset_failure(state, asset))
        result["assets"].append(item)
    result["asset_amendments"] = [candidate for value in state.get("asset_amendments", [])
                                  if (candidate := _public_asset_amendment(value, state)) is not None]
    episodes = []
    stamp = now()
    for episode in state.get("episodes", []):
        item = _pick(episode, ("id", "title", "summary", "status", "version", "file", "feedback", "subtitle_status", "music_status", "subtitles_burned", "render_warnings", "review_note", "subtitle_file"))
        if episode.get("file"):
            report = _public_shot_report(episode, _version(episode.get("version")))
            if report:
                item["review"] = report
        item["has_subtitles"] = bool(episode.get("render_manifest", {}).get("subtitles"))
        if episode.get("preview_error"):
            item["preview_error"] = "累计拼接预览暂未生成成功；片段已保留，可以重试本地拼接后再决定。"
        item["progress"] = episode_progress(state, episode, now=stamp)
        item["rough_cuts"] = _public_rough_cuts(episode)
        proposal = episode.get("segment_proposal")
        if isinstance(proposal, dict) and proposal.get("status") == "candidate":
            item["segment_proposal"] = _pick(proposal, ("id", "status", "source_ids", "created_at"))
            item["segment_proposal"]["shots"] = []
            for proposed in proposal.get("shots", []):
                entry = _pick(proposed, ("id", "duration", "prompt", "dialogue", "source_shot_ids"))
                if isinstance(proposed.get("subshots"), list):
                    entry["subshots"] = [_pick(part, ("start", "end", "prompt")) for part in proposed["subshots"]]
                item["segment_proposal"]["shots"].append(entry)
        item["shots"] = []
        for shot in episode.get("shots", []):
            entry = _pick(shot, ("id", "status", "file", "prompt", "duration"))
            if isinstance(shot.get("subshots"), list):
                entry["subshots"] = [_pick(part, ("start", "end", "prompt"))
                                     for part in shot["subshots"] if isinstance(part, dict)]
            entry["version"] = _version(shot.get("version"))
            amendment_id = shot.get("pending_asset_amendment")
            if amendment_id and any(value.get("id") == amendment_id and value.get("status") in {"candidate", "awaiting_assets"}
                                    for value in state.get("asset_amendments", []) if isinstance(value, dict)):
                entry["asset_amendment_pending"] = True
            if shot.get("status") == "awaiting_prompt_approval":
                candidate = _public_prompt_candidate(shot.get("prompt_candidate"))
                if candidate:
                    entry["prompt_candidate"] = candidate
            entry.update(_public_shot_failures(state, shot))
            entry.update(_public_shot_continuity(state, episode, shot))
            if shot.get("continuity_review"):
                entry["continuity_review"] = "前段已返工，请重新审看片段接续。当前视频保留，由你决定是否需要重做。"
            if not _current_shot_artifact(shot):
                entry.pop("file", None)
            else:
                report = _public_shot_report(shot, entry["version"])
                if report:
                    entry["review"] = report
            entry["progress"] = _shot_progress(state, episode, shot, item["progress"], stamp)
            if entry["progress"]["stage"] == "failed":
                entry["status"] = "generation_failed"
            entry["history"] = _public_shot_history(shot)
            item["shots"].append(entry)
        episodes.append(item)
    result["episodes"] = episodes
    if result.get("status") in {"review_required", "generation_failed"} and any(
            episode["progress"]["stage"] == "failed" for episode in episodes):
        result["status"] = "generation_failed"
    return result


def _media_paths(state: dict) -> set[str]:
    paths = set()
    items = list(state.get("assets", [])) + list(state.get("episodes", []))
    for episode in state.get("episodes", []):
        items.extend(episode.get("shots", []))
        paths.update(cut["file"] for cut in _public_rough_cuts(episode))
        for shot in episode.get("shots", []):
            paths.update(record["file"] for record in _public_shot_history(shot))
    for item in items:
        for ref in item.get("creation_input", {}).get("references", []):
            if isinstance(ref, dict) and isinstance(ref.get("file"), str):
                paths.add(ref["file"])
        for key in ("file", "audio_file", "subtitle_file"):
            path = item.get(key)
            if isinstance(path, str):
                paths.add(path)
    return paths


def resolve_media(project_dir: Path, requested: str, state: dict) -> Path | None:
    """Only explicitly referenced media can be served; never follow a symlink."""
    if not requested or "\\" in requested or "\x00" in requested:
        return None
    parts = requested.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    if requested not in _media_paths(state) or Path(requested).suffix.lower() not in MEDIA_SUFFIXES:
        return None
    root = project_dir.resolve()
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            return None
    try:
        if not path.is_file() or not path.resolve().is_relative_to(root):
            return None
    except OSError:
        return None
    return path


@dataclass
class _Context:
    project_dir: Path
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    lock: threading.Lock = field(default_factory=threading.Lock)
    mutation_lock: threading.Lock = field(default_factory=threading.Lock)
    cost_sync_lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    worker_error: str | None = None
    cost_sync_running: bool = False
    cost_sync_result: dict | None = None
    cost_sync_error: str | None = None

    def external_run_active(self) -> bool:
        """Probe the existing production lock without creating or changing it."""
        path = self.project_dir / ".run.lock"
        if not path.is_file():
            return False
        try:
            with path.open("r") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                fcntl.flock(lock, fcntl.LOCK_UN)
        except OSError:
            return False
        return False

    def state(self) -> dict:
        state = public_state(_core().load_project(self.project_dir))
        with self.lock:
            state["dashboard"] = {"running": self.running or self.external_run_active(), "error": self.worker_error}
        with self.cost_sync_lock:
            state["dashboard"]["cost_sync"] = {"running": self.cost_sync_running,
                                              "result": self.cost_sync_result, "error": self.cost_sync_error}
        return state

    def launch_cost_sync(self) -> bool:
        """Query usage in a separate worker; never resume or run production."""
        with self.cost_sync_lock:
            if self.cost_sync_running:
                return False
            self.cost_sync_running = True
            self.cost_sync_result = None
            self.cost_sync_error = None

        def worker():
            try:
                from .costs import sync_usage
                result = _public_cost_sync_result(sync_usage(self.project_dir, confirm_no_video_input=False))
                with self.cost_sync_lock:
                    self.cost_sync_result = result
            except Exception:
                with self.cost_sync_lock:
                    self.cost_sync_error = "费用同步未完成，请检查本机方舟凭据、网络和任务权限。"
            finally:
                with self.cost_sync_lock:
                    self.cost_sync_running = False

        try:
            threading.Thread(target=worker, name="norlan-cost-sync", daemon=True).start()
        except Exception:
            with self.cost_sync_lock:
                self.cost_sync_running = False
                self.cost_sync_error = "费用同步未能启动，请稍后重试。"
            raise RuntimeError("费用同步未能启动，请稍后重试。") from None
        return True

    def launch(self) -> bool:
        from .core import is_internal_voice_source
        with self.lock:
            if self.running:
                return False
            state = _core().load_project(self.project_dir)
            if any(asset.get("status") == "generation_failed" and not is_internal_voice_source(asset)
                   for asset in state.get("assets", [])):
                raise ValueError("请先查看资产失败原因，并明确提出重试意见。")
            if any(isinstance(episode.get("segment_proposal"), dict)
                   and episode["segment_proposal"].get("status") == "candidate"
                   for episode in state.get("episodes", [])):
                raise ValueError("请先批准新的分段方案，或选择保留原分段，再继续制作。")
            if any(isinstance(value, dict) and value.get("status") == "candidate"
                   for value in state.get("asset_amendments", [])):
                raise ValueError("请先确认或取消新增资产方案。")
            if any(shot.get("status") == "awaiting_prompt_approval"
                   for episode in state.get("episodes", []) for shot in episode.get("shots", [])):
                raise ValueError("请在片段下确认提示词后再生成。")
            self.running = True
            self.worker_error = None
        try:
            _core().set_paused(self.project_dir, False)
        except Exception:
            with self.lock:
                self.running = False
            raise

        def worker():
            try:
                _run(self.project_dir)
            except Exception:
                # Never put raw provider errors, credentials or tracebacks in HTML.
                with self.lock:
                    self.worker_error = "执行已停止。请查看项目状态，或在终端运行 nolan run 检查原因。"
            finally:
                with self.lock:
                    self.running = False
        threading.Thread(target=worker, name="norlan-pipeline", daemon=True).start()
        return True


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    context: _Context


class _Handler(BaseHTTPRequestHandler):
    server: ReviewServer
    server_version = "Nolan"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, *_args):
        pass

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _headers(self, status: int, content_type: str, length: int, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        nonce = self.server.context.nonce
        self.send_header("Content-Security-Policy", f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; img-src 'self'; media-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        for key, value in (extra or {}).items():
            self.send_header(key, str(value))
        self.end_headers()

    def _json(self, status: int, value: dict):
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(payload))
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _error(self, status: int, message: str):
        self._json(status, {"error": message})

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._host_ok():
            return self._error(403, "仅允许本机访问。")
        path = urlsplit(self.path).path
        context = self.server.context
        try:
            if path == "/":
                html = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
                payload = html.replace("{{TOKEN}}", context.token).replace("{{NONCE}}", context.nonce).encode("utf-8")
                self._headers(200, "text/html; charset=utf-8", len(payload))
                if self.command != "HEAD":
                    self.wfile.write(payload)
            elif path == "/api/state":
                self._json(200, context.state())
            elif path.startswith("/media/"):
                state = _core().load_project(context.project_dir)
                media = resolve_media(context.project_dir, unquote(path[len("/media/"):]), state)
                if media is None:
                    return self._error(404, "媒体不存在或不可访问。")
                self._media(media)
            else:
                self._error(404, "未找到此页面。")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self._error(500, "项目状态暂时无法读取。")

    def _media(self, path: Path):
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        headers = {"Accept-Ranges": "bytes"}
        if path.suffix.lower() in {".srt", ".vtt"}:
            headers["Content-Disposition"] = 'attachment; filename="subtitles' + path.suffix.lower() + '"'
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if not match or not size or not any(match.groups()):
                self._headers(416, "application/octet-stream", 0, {"Content-Range": f"bytes */{size}"})
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                start = max(size - int(last), 0)
            if start > end or start >= size:
                self._headers(416, "application/octet-stream", 0, {"Content-Range": f"bytes */{size}"})
                return
            status = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        length = max(0, end - start + 1)
        self._headers(status, mimetypes.guess_type(path.name)[0] or "application/octet-stream", length, headers)
        if self.command == "HEAD":
            return
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                block = stream.read(min(65_536, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def do_POST(self):
        if not self._host_ok():
            return self._error(403, "仅允许本机访问。")
        path = urlsplit(self.path).path
        if path not in {"/api/action", "/api/asset-reference"}:
            return self._error(404, "未找到此操作。")
        context = self.server.context
        if not secrets.compare_digest(self.headers.get("X-Norlan-Token", "").encode("utf-8"), context.token.encode("ascii")):
            return self._error(403, "会话已失效，请刷新页面。")
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + self.headers.get("Host", ""):
            return self._error(403, "不允许跨站操作。")
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            return self._error(415, "操作需要 JSON 数据。")
        if self.headers.get("Transfer-Encoding"):
            return self._error(400, "不支持此请求格式。")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._error(400, "请求长度无效。")
        limit = MAX_REFERENCE_BODY if path == "/api/asset-reference" else MAX_BODY
        if length <= 0 or length > limit:
            return self._error(413, "请求过大或为空。")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError
            action = data.get("action")
            with context.mutation_lock:
                if path == "/api/asset-reference" or action in {"asset_input", "remove_asset_reference"}:
                    item_id, version, revision = data.get("id"), data.get("version"), data.get("input_revision")
                    if (not isinstance(item_id, str) or not 0 < len(item_id) <= 128
                            or type(version) is not int or version < 0
                            or type(revision) is not int or revision < 0):
                        raise ValueError
                    if path == "/api/asset-reference":
                        _core().add_asset_reference(context.project_dir, item_id, version, revision, data.get("data_url"))
                    elif action == "asset_input":
                        _core().set_asset_input(context.project_dir, item_id, version, revision, data.get("mode"), data.get("description", ""))
                    else:
                        _core().remove_asset_reference(context.project_dir, item_id, version, revision, data.get("reference_id"))
                    return self._json(200, context.state())
                if action == "asset_inputs_gpt_remaining":
                    version = data.get("plan_version")
                    if type(version) is not int or version < 1:
                        raise ValueError
                    _core().set_remaining_asset_inputs_gpt(context.project_dir, version)
                    return self._json(200, context.state())
                if action == "segment_plan":
                    from .regroup import decide_segments
                    item_id, proposal_id, approve = data.get("id"), data.get("proposal_id"), data.get("approve")
                    if (not isinstance(item_id, str) or not 0 < len(item_id) <= 64
                            or not isinstance(proposal_id, str) or not 0 < len(proposal_id) <= 64
                            or type(approve) is not bool):
                        raise ValueError
                    decide_segments(context.project_dir, item_id, proposal_id, approve)
                    return self._json(200, context.state())
                if action in {"approve_prompt", "reject_prompt"}:
                    item_id, candidate_id, request_hash = data.get("id"), data.get("candidate_id"), data.get("request_hash")
                    if (not isinstance(item_id, str) or not 0 < len(item_id) <= 128
                            or not isinstance(candidate_id, str) or not 0 < len(candidate_id) <= 128
                            or not _valid_sha(request_hash)):
                        raise ValueError
                    if action == "approve_prompt":
                        _core().approve_prompt(context.project_dir, item_id, candidate_id, request_hash)
                    else:
                        feedback = data.get("feedback")
                        if not isinstance(feedback, str) or not feedback.strip() or len(feedback) > 8000:
                            raise ValueError
                        _core().reject_prompt(context.project_dir, item_id, candidate_id, request_hash, feedback.strip())
                    return self._json(200, context.state())
                if action == "asset_amendment":
                    amendment_id, decision = data.get("id"), data.get("decision")
                    if (not isinstance(amendment_id, str) or not 0 < len(amendment_id) <= 128
                            or decision not in {"approve", "reject"}):
                        raise ValueError
                    _core().apply_asset_amendment(context.project_dir, amendment_id, decision)
                    return self._json(200, context.state())
                if action == "sync_costs":
                    if not context.launch_cost_sync():
                        return self._error(409, "费用正在同步，请稍候，无需重复启动。")
                    return self._json(202, context.state())
                if action == "run":
                    if not context.launch():
                        return self._error(409, "制作正在进行，无需重复启动。")
                    return self._json(202, context.state())
                if action == "pause":
                    if type(data.get("paused")) is not bool:
                        raise ValueError
                    _core().set_paused(context.project_dir, data["paused"])
                elif action in {"approve", "reject", "captions"}:
                    kind, item_id, version = data.get("kind"), data.get("id"), data.get("version")
                    if kind not in {"plan", "asset", "episode", "shot"} or not isinstance(item_id, str) or not 0 < len(item_id) <= 128:
                        raise ValueError
                    if type(version) is not int or version < 1:
                        raise ValueError
                    if action == "approve":
                        _core().approve(context.project_dir, kind, item_id, version)
                    else:
                        feedback, shot_ids = data.get("feedback"), data.get("shot_ids")
                        if not isinstance(feedback, str) or not feedback.strip() or len(feedback) > 8000:
                            raise ValueError
                        if shot_ids is not None and (not isinstance(shot_ids, list) or len(shot_ids) > 100 or any(not isinstance(shot, str) or not 0 < len(shot) <= 128 for shot in shot_ids)):
                            raise ValueError
                        if action == "captions":
                            if kind != "episode":
                                raise ValueError
                            _core().request_caption_fix(context.project_dir, item_id, version, feedback.strip())
                        else:
                            _core().reject(context.project_dir, kind, item_id, version, feedback.strip(), shot_ids)
                else:
                    raise ValueError
            self._json(200, context.state())
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            message = str(exc)
            self._error(409, message if message in INPUT_ERRORS or message in SEGMENT_ERRORS else "状态已变化或参数无效，请刷新后重试。")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self._error(500, "操作未完成，请刷新后检查项目状态。")


def make_server(project_dir: Path, port: int = 8765) -> ReviewServer:
    """Create an unstarted server, useful for embedding and isolated tests."""
    root = Path(project_dir).expanduser().resolve()
    _core().load_project(root)
    server = ReviewServer(("127.0.0.1", port), _Handler)
    server.context = _Context(root)
    return server


def serve(project_dir: Path, port: int = 8765, open_browser: bool = True) -> None:
    server = make_server(project_dir, port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Nolan 审片室：{url}\n按 Ctrl+C 关闭审片室。")
    try:
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                print("浏览器未自动打开，请复制上方地址访问审片室。")
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if server.context.running:
            _core().set_paused(server.context.project_dir, True)
        server.server_close()
