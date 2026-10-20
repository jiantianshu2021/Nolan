"""Recover a narrowly proven, never-submitted legacy video request locally.

The former 64 MB check ran after reservation but before opening an HTTP request.
Only its exact durable signature permits releasing that reservation. Recovery
restores an unchanged prompt for a fresh human decision; it never makes calls.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import math
import uuid

from . import core, prompt_gate, storyboard_revision


OVERSIZED_REQUEST_MESSAGE = "参考素材编码后的请求超过 64 MB，请减少或压缩参考图和音频。"
RECOVERY_POLICY = 1


def _remote_evidence(value):
    if not isinstance(value, dict):
        return True
    if any(value.get(key) for key in (
            "task_id", "request_id", "http_status", "provider_code", "provider_status",
            "output", "origin_video", "last_frame", "last_frame_url", "url", "usage",
            "bill_reference", "bill_basis", "bill_verified_at", "usage_recorded_at")):
        return True
    return any(value.get(key) is not None for key in ("usage_cost", "actual_cost"))


def _known_local_failure(job):
    failure = job.get("failure")
    if (job.get("kind") != "video" or job.get("state") != "failed"
            or job.get("accounting") != "spent" or not isinstance(failure, dict)
            or failure.get("stage") != "local_validation"
            or failure.get("message") != OVERSIZED_REQUEST_MESSAGE
            or failure.get("uncertain") is not False or job.get("uncertain") is True or _remote_evidence(job)
            or _remote_evidence(failure)):
        return False
    billing = job.get("billing", {})
    if _remote_evidence(billing):
        return False
    history = job.get("failure_history", [])
    if not isinstance(history, list):
        return False
    return all(isinstance(row, dict) and row.get("stage") == "local_validation"
               and row.get("message") == OVERSIZED_REQUEST_MESSAGE
               and row.get("uncertain") is False and not _remote_evidence(row)
               for row in history)


def _identity(shot, candidate):
    return {"candidate_id": candidate.get("id"), "request_hash": candidate.get("request_hash"),
            "version": shot.get("version"), "revision": shot.get("revision", 0)}


def _decline(state, job, shot, reason):
    candidate = shot.get("prompt_candidate", {})
    identity = _identity(shot, candidate if isinstance(candidate, dict) else {})
    previous = job.get("local_submission_recovery", {})
    if previous.get("status") == "blocked" and previous.get("identity") == identity and previous.get("reason") == reason:
        return
    job["local_submission_recovery"] = {"policy": RECOVERY_POLICY, "status": "blocked",
        "reason": reason, "identity": identity, "checked_at": core.now()}
    core.event(state, "local_video_submission_recovery_blocked", id=shot["id"], job_id=job["id"], reason=reason)


def _validated_candidate(root, state, shot, job):
    candidate, approval = shot.get("prompt_candidate"), shot.get("prompt_approval")
    if (shot.get("status") != "generation_failed" or shot.get("file") or shot.get("sha256")
            or shot.get("job_id") != job.get("id") or job.get("item_id") != shot.get("id")
            or not isinstance(candidate, dict) or not isinstance(approval, dict)
            or candidate.get("status") != "approved" or approval.get("decision") != "generate"
            or approval.get("candidate_id") != candidate.get("id")
            or approval.get("request_hash") != candidate.get("request_hash")
            or type(shot.get("version")) is not int or shot["version"] < 1
            or type(candidate.get("version")) is not int
            or type(candidate.get("revision")) is not int
            or candidate.get("version") != shot["version"]
            or candidate.get("revision") != shot.get("revision", 0)
            or ("video_model" in approval and approval["video_model"] != candidate.get("video_model"))
            or not isinstance(candidate.get("prompt"), str) or not candidate["prompt"].strip()):
        raise core.GateError("原失败任务与已确认提示词的版本绑定不完整，未自动恢复。")
    if not any(isinstance(attempt, dict) and attempt.get("job_id") == job["id"]
               and attempt.get("version") == shot["version"]
               and attempt.get("revision", 0) == shot.get("revision", 0)
               for attempt in shot.get("attempts", [])):
        raise core.GateError("原失败任务缺少匹配的版本尝试记录，未自动恢复。")
    if shot.get("future_continuity_binding"):
        raise core.GateError("原提示词绑定了后段接续草稿，需要同时核对后段；本次未改动相邻片段。")
    if shot.get("continuity_revision"):
        raise core.GateError("本片段仍绑定上游接续草稿，需先核对其来源；本次未改动相邻片段。")
    # Submission advances the displayed version before the adapter runs. The
    # original consent hash was computed against the preceding version.
    original_state = copy.deepcopy(state)
    original_shot = core._find(original_state, "shot", shot["id"])
    original_shot["version"] = candidate["version"] - 1
    # Restoring a confirmation panel is harmless while paused. Keep the real
    # pause setting; only validate immutable consent inputs in this local view.
    original_state["paused"] = False
    prompt_gate._validate_inputs(root, original_state, original_shot, candidate)
    if candidate.get("manual_prompt") and approval.get("source") != "human_manual_edit":
        raise core.GateError("手动定稿缺少原人工编辑凭据，未自动恢复。")
    return copy.deepcopy(candidate)


def _restore(state, shot, job, candidate, root):
    stamp = core.now()
    original_identity = _identity(shot, candidate)
    old_revision = shot.get("revision", 0)
    if type(old_revision) is not int or old_revision < 0:
        raise core.GateError("原失败片段的返修版本无效，未自动恢复。")
    shot.setdefault("prompt_history", []).append({**copy.deepcopy(candidate),
        "status": "local_request_not_submitted", "retired_at": stamp, "failed_job_id": job["id"]})
    shot["revision"] = old_revision + 1
    # A storyboard can be rebound entirely inside this segment. Its camera
    # definition and original pre-draft snapshot remain unchanged.
    if shot.get("storyboard_revision") or shot.get("storyboard_revision_binding"):
        binding = shot.get("storyboard_revision_binding")
        if not isinstance(binding, dict):
            raise core.GateError("局部分镜绑定缺失，未自动恢复。")
        binding.update(signature=storyboard_revision.feedback_signature(shot),
                       revision=shot["revision"], version=shot["version"])
        storyboard_revision.public(shot)
    if candidate.get("manual_prompt"):
        # Keep the strict manual source/version checks intact. Rebind a copy of
        # the original non-manual source to the new local version, preserving
        # both original histories and both exact prompt text digests.
        old_marker = candidate["manual_prompt"]
        source = copy.deepcopy(prompt_gate._manual_source(shot, old_marker, candidate["prompt"]))
        source_origin = {"job_id": job["id"], "id": source["id"],
            "request_hash": source["request_hash"], "version": source["version"],
            "revision": source.get("revision", 0)}
        source.update(id="prompt-" + uuid.uuid4().hex[:12], status="superseded_by_manual_edit",
            version=shot["version"] + 1, revision=shot["revision"], created_at=stamp,
            superseded_at=stamp, video_model=candidate.get("video_model"),
            local_submission_recovery_origin=source_origin)
        source.pop("approved_at", None)
        prompt_gate._validate_candidate_timing(shot, source)
        source["request_hash"] = prompt_gate._hash(prompt_gate.specification(
            state, shot, source["prompt"], source["reference_manifest"], source.get("refined")))
        shot["prompt_history"].append(source)
        marker = {**copy.deepcopy(old_marker), "source_candidate_id": source["id"],
                  "source_request_hash": source["request_hash"]}
        candidate["manual_prompt"] = marker
        candidate["refined"] = prompt_gate._manual_refined(candidate["prompt"], marker)
    candidate.update(id="prompt-" + uuid.uuid4().hex[:12], status="candidate",
        version=shot["version"] + 1, revision=shot["revision"], created_at=stamp)
    candidate.pop("approved_at", None)
    candidate["request_hash"] = prompt_gate._hash(prompt_gate.specification(
        state, shot, candidate["prompt"], candidate["reference_manifest"], candidate.get("refined")))
    shot.update(status="awaiting_prompt_approval", prompt_candidate=candidate)
    for key in ("prompt_approval", "job_id", "generation_error"):
        shot.pop(key, None)
    validation_state = copy.deepcopy(state)
    validation_state["paused"] = False
    prompt_gate._validate_inputs(root, validation_state, core._find(validation_state, "shot", shot["id"]), candidate)
    job.update(accounting="released", updated_at=stamp,
        local_submission_recovery={"policy": RECOVERY_POLICY, "status": "restored",
            "reason": "请求在本机 64 MB 检查时被拒绝，未建立供应商请求；原文等待重新确认。",
            "identity": original_identity, "restored_candidate_id": candidate["id"],
            "restored_version": candidate["version"], "restored_at": stamp,
            "public_summary": "原请求未提交；已保留完整提示词，请确认后再生成。"})
    shot["local_submission_recovery"] = {"job_id": job["id"], "at": stamp,
        "candidate_id": candidate["id"], "public_summary": job["local_submission_recovery"]["public_summary"]}
    core.event(state, "local_video_submission_recovered", id=shot["id"], job_id=job["id"],
        original_amount=job["amount"], original_failure=copy.deepcopy(job["failure"]),
        original_candidate_id=original_identity["candidate_id"], candidate_id=candidate["id"],
        version=candidate["version"], revision=shot["revision"])
    state.update(status="awaiting_prompt_approval", last_error=None)


def recover_oversized_requests(root, already_locked=False):
    """Release only proven legacy local failures and restore human prompt gates.

    Returns restored shot IDs. A declined candidate retains its charged ledger
    and exact prompt, with one bounded explanation. No request is submitted.
    """
    recovered = []
    with (nullcontext() if already_locked else core.run_lock(root)), core.transaction(root) as state:
        if any(core.execution_unsettled(job) for job in state["jobs"].values()):
            return recovered
        targets = [(shot["id"], shot.get("job_id"))
                   for episode in state["episodes"] for shot in episode["shots"]
                   if episode.get("status") != "approved" and shot.get("status") == "generation_failed"]
        for shot_id, job_id in targets:
            job = state["jobs"].get(job_id, {})
            if not job or job.get("id") != job_id or not _known_local_failure(job):
                continue
            amount = job.get("amount")
            if (isinstance(amount, bool) or not isinstance(amount, (int, float))
                    or not math.isfinite(amount) or amount < 0):
                continue
            shot = core._find(state, "shot", shot_id)
            try:
                candidate = _validated_candidate(root, state, shot, job)
                # Validate and stage every change in an isolated snapshot so an
                # invalid new binding cannot release funds or partially mutate.
                staged = copy.deepcopy(state)
                _restore(staged, core._find(staged, "shot", shot_id), staged["jobs"][job_id], candidate, root)
            except (core.GateError, ValueError, KeyError, TypeError, AttributeError, OSError):
                # Keep a useful bounded explanation without persisting arbitrary
                # exception messages, input prose, or filesystem paths.
                reason = ("原提示词绑定后段接续草稿，需同时核对后段；本次未改动相邻片段。"
                          if shot.get("future_continuity_binding") or shot.get("continuity_revision") else
                          "原提示词、参考素材或模型的版本绑定未通过核对；已保留原文和费用记录，未自动恢复。")
                _decline(state, job, shot, reason)
                continue
            state.clear()
            state.update(staged)
            recovered.append(shot_id)
    return recovered
