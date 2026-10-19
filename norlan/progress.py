"""Derive episode progress from durable production milestones.

Percentages describe Nolan's workflow, never a provider's internal generation
progress or an estimated time remaining. This module does no I/O and does not
advance jobs. The caller may pass ``now`` to show a live elapsed-time counter.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math


_ACTIVE_JOB_STATES = frozenset({"submitting", "submitted", "download_pending"})
_SHOT_STAGES = {
    "shot_preparation": "整理已批准参考并编写片段提示词",
    "shot_submission": "正在提交视频生成请求",
    "shot_generation": "等待视频引擎生成",
    "shot_download": "视频已生成，正在下载",
    "shot_checks": "提取画面与声音证据，检查技术参数",
    "shot_review": "检查抽样画面及前后镜头衔接",
}
_POST_STAGES = {
    "music": (90.0, "处理本集配乐"),
    "subtitles": (93.0, "处理本集字幕与时间轴"),
    "rendering": (96.0, "合成整集画面、配乐和字幕"),
    "transcribing": (91.0, "本地识别本集实际对白和字幕时间轴"),
    "correcting_subtitles": (93.0, "GPT 校对字幕文字与断句"),
    "arranging_music": (95.0, "GPT 复用已批准音乐，编排本集配乐"),
    "assembling": (90.0, "锁定已批准片段，拼接本集原声粗剪"),
    "recognizing": (91.0, "本地识别本集实际对白和字幕时间轴"),
    "proofreading": (93.0, "GPT 校对字幕文字与断句"),
}


def _date(value):
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    # Existing local project timestamps use UTC ISO strings. Treat a legacy
    # naive timestamp as UTC so elapsed time does not depend on host timezone.
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _elapsed(start, now):
    first, last = _date(start), _date(now)
    return max(0, int((last - first).total_seconds())) if first and last else None


def _weight(shot):
    value = shot.get("duration")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return 1.0
    return float(value)


def _accepted(shot):
    return shot.get("status") == "accepted" and bool(shot.get("file") and shot.get("sha256"))


def episode_progress(state, episode, *, now=None):
    """Return a public, JSON-safe progress description without mutating state.

    Completed, accepted shot durations form 90% of progress. Entering subtitles
    after soundtrack handling establishes 93%; entering final rendering after
    subtitles establishes 96%. Only a recorded final render awaiting human
    review, or already approved, reaches 100%. These are workflow weights, not
    elapsed-time estimates. A rejection can reduce the displayed percentage.

    Optional pipeline hints use ``episode.progress_stage`` with ``stage``,
    ``shot_id``, ``started_at`` and ``updated_at``. Hints must be cleared when a
    render is invalidated. Missing hints are supported for existing projects.
    ``now`` accepts an ISO timestamp or datetime; without it elapsed counters
    are null, keeping this function deterministic for the supplied state.
    """
    from .core import asset_ready, is_internal_voice_source, blocking_unknown_jobs
    public_assets = [asset for asset in state.get("assets", []) if not is_internal_voice_source(asset)]
    shots = episode.get("shots", [])
    weights = [_weight(shot) for shot in shots]
    accepted = [_accepted(shot) for shot in shots]
    completed = sum(accepted)
    total = len(shots)
    completed_duration = sum(weight for weight, done in zip(weights, accepted) if done)
    total_duration = sum(weights)
    percent = round(90 * completed_duration / total_duration, 1) if total_duration else 0.0
    all_complete = total > 0 and completed == total
    current = next((shot for shot, done in zip(shots, accepted) if not done), None)
    current_index = next((index for index, done in enumerate(accepted, 1) if not done), None)
    hint = episode.get("progress_stage", {})
    hint = hint if isinstance(hint, dict) else {}
    jobs = state.get("jobs", {})
    jobs = jobs if isinstance(jobs, dict) else {}
    shot_jobs = [jobs.get(shot.get("job_id"), {}) for shot in shots]
    episode_jobs = [job for job in jobs.values() if isinstance(job, dict) and
                    job.get("item_id") == episode.get("id") and job.get("kind") in {"music", "transcription"}]
    relevant_jobs = [job for job in shot_jobs + episode_jobs if isinstance(job, dict)]
    active_jobs = [job for job in relevant_jobs if job.get("state") in _ACTIVE_JOB_STATES]
    active_job = jobs.get(current.get("job_id"), {}) if current else {}
    active_job = active_job if isinstance(active_job, dict) else {}
    start = hint.get("started_at") or active_job.get("created_at")
    timestamps = [value for value in (hint.get("updated_at"), episode.get("updated_at"),
                                      active_job.get("updated_at"), state.get("updated_at")) if _date(value)]
    updated_at = max(timestamps, key=_date) if timestamps else None
    # Public output intentionally excludes raw provider messages and job payloads.
    result = {
        "percent": percent,
        "stage": "queued",
        "label": "等待开始本集制作",
        "current_shot": current.get("id") if current else None,
        "current_shot_index": current_index,
        "completed_shots": completed,
        "total_shots": total,
        "completed_segments": completed,
        "total_segments": total,
        "planned_camera_shots": sum(len(shot.get("subshots") or [None]) for shot in shots),
        "completed_duration": round(completed_duration, 3),
        "total_duration": round(total_duration, 3),
        "elapsed_seconds": _elapsed(start, now),
        "episode_elapsed_seconds": _elapsed(episode.get("progress_started_at"), now),
        "waiting_for_human": False,
        "has_active_task": bool(active_jobs),
        "is_working": False,
        "updated_at": updated_at,
        "basis": "按已验收片段的计划时长及后期步骤计算；不代表视频引擎的内部进度或剩余时间。",
    }

    def finish(stage, label, *, waiting=False, working=False, value=None):
        result.update(stage=stage, label=label, waiting_for_human=waiting, is_working=working)
        if value is not None:
            result["percent"] = value
        return result

    post = episode.get("postproduction", {})
    post = post if isinstance(post, dict) else {}
    post_enabled = state.get("settings", {}).get("postproduction", {}).get("enabled") is True
    if all_complete and episode.get("file") and episode.get("sha256") and episode.get("status") in {"awaiting_review", "approved"} and (not post_enabled or post.get("final_ready") is True):
        if episode["status"] == "approved":
            return finish("approved", "本集已通过人工审批", value=100.0)
        return finish("awaiting_review", "本集已合成，请观看成片并审批", waiting=True, value=100.0)

    if state.get("plan", {}).get("status") != "approved":
        return finish("awaiting_plan", "等待分集与分镜规划审批", waiting=True)
    if episode.get("segment_proposal", {}).get("status") == "candidate":
        return finish("awaiting_segment_plan", "等待你确认剩余内容的分段调整方案", waiting=True)
    if any(isinstance(value, dict) and value.get("status") == "candidate"
           for value in state.get("asset_amendments", [])):
        return finish("awaiting_asset_amendment", "等待你确认新增资产方案；确认后先制作并审批新资产", waiting=True)
    episodes = state.get("episodes", [])
    index = next((index for index, item in enumerate(episodes) if item.get("id") == episode.get("id")), 0)
    if any(item.get("status") != "approved" for item in episodes[:index]):
        return finish("awaiting_previous_episode", "等待前集成片通过人工审批", waiting=True)
    if state.get("status") == "awaiting_asset_input":
        return finish("awaiting_asset_input", "等待选择资产创作方式", waiting=True)
    if any(asset.get("status") == "generation_failed" for asset in public_assets):
        return finish("asset_generation_failed", "资产生成失败；请查看具体原因并明确提出重试意见", waiting=True)
    if any(not asset_ready(asset) for asset in public_assets):
        return finish("awaiting_assets", "等待人物、场景和声音资产全部通过审批", waiting=True)
    if state.get("paused") or state.get("status") == "paused":
        label = "已暂停；已提交的视频任务可能仍在生成" if active_jobs else "本集制作已暂停"
        return finish("paused", label)
    if blocking_unknown_jobs(state):
        return finish("unknown", "存在结果不明的调用，需核实原任务后继续", waiting=True)
    if all_complete and post_enabled:
        post_status = post.get("status", "")
        if post_status == "awaiting_music_approval":
            return finish("awaiting_music_approval", "请在本集后期试听并确认新音乐，才能混入成片", waiting=True, value=94.0)
        if post_status in {"configuration_required", "awaiting_music", "awaiting_music_selection", "postproduction_configuration"}:
            return finish("postproduction_configuration", "本集后期需要补充配置或音乐素材，请查看本集后期", waiting=True, value=percent)
        if post_status in {"failed", "error"} or post.get("error"):
            return finish("postproduction_failed", "本集后期已停止；视频保留，请查看本集后期的具体原因", waiting=True, value=percent)
        if post_status in _POST_STAGES:
            value, label = _POST_STAGES[post_status]
            return finish(post_status, label, working=True, value=value)
    prompt_shot = next((shot for shot in shots if shot.get("status") == "awaiting_prompt_approval"), None)
    if prompt_shot:
        result["current_shot"] = prompt_shot.get("id")
        result["current_shot_index"] = shots.index(prompt_shot) + 1
        return finish("awaiting_prompt_approval", f"片段 {result['current_shot_index']}/{total} 提示词已准备，请确认后生成", waiting=True)
    failed_shot = next((shot for shot in shots if shot.get("status") == "generation_failed" or
                       (shot.get("status") == "review_required" and not (shot.get("file") and shot.get("sha256")))), None)
    if failed_shot:
        result["current_shot"] = failed_shot.get("id")
        result["current_shot_index"] = shots.index(failed_shot) + 1
        return finish("failed", f"片段 {result['current_shot_index']}/{total} 生成失败，没有可审视频；请查看失败原因", waiting=True)
    review_shot = next((shot for shot in shots if shot.get("status") == "review_required"), None)
    if review_shot:
        result["current_shot"] = review_shot.get("id")
        result["current_shot_index"] = shots.index(review_shot) + 1
        report = review_shot.get("review", {})
        interrupted = (isinstance(report, dict) and report.get("report_unavailable") is True
                       and report.get("review_interrupted") is True and report.get("sha256") == review_shot.get("sha256")
                       and report.get("version") == review_shot.get("version"))
        label = f"片段 {result['current_shot_index']}/{total} GPT 报告中断，视频可人工审片" if interrupted else f"片段 {result['current_shot_index']}/{total} 需要人工复核"
        return finish("review_required", label, waiting=True)
    if state.get("status") == "blocked":
        return finish("blocked", "制作暂时停止，请查看错误说明后继续", waiting=True)
    if active_job.get("state") == "failed":
        return finish("failed", f"片段 {current_index}/{total} 生成失败，等待处理", waiting=True)

    stage = hint.get("stage", "")
    if all_complete:
        if stage in _POST_STAGES:
            value, label = _POST_STAGES[stage]
            return finish(stage, label, working=True, value=value)
        if stage == "batch_assembly":
            return finish(stage, "拼接分段粗剪，准备检查镜头衔接", working=True)
        if stage == "batch_review":
            return finish(stage, "检查分段粗剪的抽样画面与镜头衔接", working=True)
        return finish("postproduction", "片段已全部验收，等待整集后期合成", working=episode.get("status") == "generating")
    if stage in {"batch_assembly", "batch_review"}:
        label = "拼接已完成镜头的分段粗剪" if stage == "batch_assembly" else "检查分段粗剪的抽样画面与镜头衔接"
        return finish(stage, label, working=True)

    # Job states outrank the optional display hint, which may lag a durable
    # submission/download write. Only the current attempt is consulted.
    job_stage = {
        "submitting": "shot_submission", "submitted": "shot_generation",
        "download_pending": "shot_download", "completed": "shot_checks",
    }.get(active_job.get("state"))
    if current and current.get("status") == "reviewing":
        job_stage = "shot_review"
    if job_stage:
        stage = job_stage if stage not in {"shot_checks", "shot_review"} or active_job.get("state") != "completed" else stage
    elif stage not in _SHOT_STAGES or hint.get("shot_id") not in {None, current.get("id") if current else None}:
        stage = "shot_preparation" if current and current.get("status") == "generating" else "queued"
    if stage in _SHOT_STAGES:
        return finish(stage, f"片段 {current_index}/{total}：{_SHOT_STAGES[stage]}", working=True)
    return result
