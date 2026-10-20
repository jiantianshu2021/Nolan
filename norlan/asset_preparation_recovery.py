"""Undo one legacy pre-reservation version bump, never a submitted request.

Before asset prompt consent existed, the asset version advanced before budget
reservation. A never-approved zero media budget could therefore leave the first
asset at ``generating / 1`` without ever permitting a provider POST. Only that
narrow, corroborated state can return to local preparation and human consent.
Missing jobs alone are deliberately insufficient evidence.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
import os
from pathlib import Path
import re

from . import core


RECOVERY_POLICY = 1
MAX_SCAN_ENTRIES = 10000
_KINDS = {"character", "location", "prop", "voice"}
_EVIDENCE_FIELDS = (
    "file", "sha256", "job_id", "attempts", "history", "evidence",
    "audio_file", "audio_sha256", "generation_prompt", "generation_request",
    "generation_prompt_approval", "generation_prompt_history", "generation_model",
    "generation_error", "approval", "approved_at", "approved_version", "approved_sha256",
    "task_id", "request_id", "url", "output", "usage", "billing", "provider_response",
    "last_frame", "last_frame_url", "origin_video", "bill_reference", "http_status",
    "asset_preparation_recovery",
)


def _zero(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value == 0)


def _budget_never_enabled(state):
    budget, jobs, events = state.get("budget"), state.get("jobs"), state.get("events")
    if (state.get("mode") != "live" or state.get("paused") is True or not isinstance(budget, dict)
            or budget.get("enforcement") == "disabled"
            or budget.get("approved") is not False or not _zero(budget.get("limit"))
            or not isinstance(jobs, dict) or not isinstance(events, list)):
        return False
    # The complete project must still be before media production, rather than
    # merely lacking a ledger row for this particular asset.
    if any(not isinstance(job, dict) or job.get("kind") != "llm"
           or core.execution_unsettled(job) for job in jobs.values()):
        return False
    if not any(isinstance(row, dict) and row.get("action") == "project_created"
               and row.get("mode") == "live" for row in events):
        return False
    for row in events:
        if not isinstance(row, dict):
            return False
        if row.get("action") == "configuration_updated" and not _zero(row.get("budget_limit")):
            return False
        if row.get("action") == "call_reserved" and row.get("kind") != "llm":
            return False
        if row.get("action") == "human_approved" and row.get("kind") in {"asset", "shot", "episode"}:
            return False
        if row.get("action") in {"human_approved_asset_prompt", "human_approved_prompt"}:
            return False
    return True


def _has_disk_evidence(root, asset_id):
    """Fail closed on versioned outputs anywhere inside this project.

    Never follow a symlink to scan another directory. Any symlink encountered
    inside output/evidence trees also blocks recovery, as its target cannot be
    dismissed as absent production evidence.
    """
    prefix, scanned = asset_id + "-v", 0
    def fail(error):
        raise error
    try:
        for directory, folders, files in os.walk(root, followlinks=False,
                                                 onerror=fail):
            relative = Path(directory).relative_to(root)
            output_tree = bool(relative.parts and relative.parts[0] in {"media", "evidence", "exports"})
            for name in folders + files:
                scanned += 1
                if scanned > MAX_SCAN_ENTRIES:
                    return True
                child = Path(directory) / name
                if name.startswith(prefix) or (output_tree and child.is_symlink()):
                    return True
    except (OSError, ValueError):
        return True
    return False


def _eligible(root, asset):
    identifier = asset.get("id") if isinstance(asset, dict) else None
    return (isinstance(identifier, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", identifier)
            and asset.get("kind") in _KINDS and asset.get("voice_only") is not True
            and asset.get("status") == "generating" and type(asset.get("version")) is int
            and asset["version"] == 1 and not any(asset.get(field) for field in _EVIDENCE_FIELDS)
            and all(asset.get(field) is None for field in ("actual_cost", "usage_cost"))
            and not _has_disk_evidence(root, identifier))


def recover(root, already_locked=False):
    """Return locally restored IDs; never approve, reserve, submit or retry.

    Preserve all directions, feedback, ledger entries and top-level failure
    diagnostics. The normal preparation flow must still produce a complete
    prompt candidate and obtain the human's explicit model/prompt confirmation.
    """
    root, restored = Path(root).resolve(), []
    with (nullcontext() if already_locked else core.run_lock(root)), core.transaction(root) as state:
        if not _budget_never_enabled(state):
            return restored
        for asset in state.get("assets", []):
            if not _eligible(root, asset):
                continue
            stamp = core.now()
            reason = "旧版在预算检查前提前增加资产版本；媒体预算从未批准，未登记或提交媒体请求。已返回提示词与模型确认，未开始生成。"
            proof = {"mode": "live", "budget_approved": False, "budget_limit": 0,
                     "media_ledger_jobs": 0, "media_reservation_events": 0,
                     "asset_generation_evidence": False, "versioned_disk_evidence": False,
                     "reserve_before_post": True}
            asset.update(status="planned", version=0,
                         asset_preparation_recovery={"policy": RECOVERY_POLICY,
                             "status": "local_preparation_restored", "restored_at": stamp,
                             "original_status": "generating", "original_version": 1,
                             "reason": reason, "proof": proof, "requires_human_prompt_confirmation": True})
            core.event(state, "asset_preparation_recovered", id=asset["id"],
                       policy=RECOVERY_POLICY, original_status="generating", original_version=1,
                       restored_status="planned", restored_version=0, proof=proof,
                       reason=reason, requires_human_prompt_confirmation=True)
            restored.append(asset["id"])
    return restored
