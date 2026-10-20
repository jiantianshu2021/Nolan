"""Small, reversible continuity drafts for the immediate unstarted next segment.

These edits never authorize media. Both segments still use their own complete
prompt approval. Completed/started work is immutable at this boundary.
"""
from __future__ import annotations

import copy

from . import storyboard_revision as camera
from .segments import validate_subshots


PLANNING_POLICY_VERSION = 2
EDIT_FIELDS = ("prompt", "continuity_notes", "subshots")
FEEDBACK_SEPARATOR = "\n\n用户对规划阻塞的补充意见："


def feedback_text(shot):
    """Keep the audit record intact; send each exact repeated instruction once."""
    unique = []
    for part in shot.get("feedback", "").split(FEEDBACK_SEPARATOR):
        text = part.strip()
        if text and text not in unique:
            unique.append(text)
    return FEEDBACK_SEPARATOR.join(unique)


def next_shot(context, shot):
    episodes = context.get("episodes", context.get("plan", {}).get("episodes", []))
    for episode in episodes:
        ordered = episode.get("shots", [])
        for index, candidate in enumerate(ordered):
            if candidate.get("id") == shot.get("id"):
                return ordered[index + 1] if index + 1 < len(ordered) else None
    return None


def unstarted(shot, *, allow_continuity_draft=False):
    return (isinstance(shot, dict) and shot.get("status") == "planned"
            and type(shot.get("version", 0)) is int and shot.get("version", 0) == 0
            and type(shot.get("revision", 0)) is int and shot.get("revision", 0) == 0
            and isinstance(shot.get("feedback", ""), str) and not shot.get("feedback", "").strip()
            and not any(shot.get(key) for key in (
                "file", "job_id", "attempts", "prompt_candidate", "prompt_approval",
                "pending_asset_amendment", "storyboard_revision", "storyboard_revision_binding"))
            and (allow_continuity_draft or not shot.get("continuity_revision")))


def eligible_next(context, shot):
    candidate = next_shot(context, shot)
    return candidate if unstarted(candidate) else None


def baseline(shot):
    return {**camera.baseline(shot), "asset_ids": copy.deepcopy(shot.get("asset_ids", [])),
            "feedback": shot.get("feedback", "")}


def validate_updates(context, shot, updates):
    """Bind only the next untouched plan; never accept arbitrary model fields."""
    if not isinstance(updates, list) or len(updates) > 1:
        raise ValueError("接续调整只能涉及同集紧邻、尚未开始的下一个片段。")
    if not updates:
        return []
    neighbor = eligible_next(context, shot)
    update = updates[0]
    fields = {"id", "summary", *EDIT_FIELDS}
    if not neighbor or not isinstance(update, dict) or set(update) != fields or update["id"] != neighbor["id"]:
        raise ValueError("只能自动协调同集紧邻的未生成片段；已有视频、提示词或人工修改不能被覆盖。")
    for key in ("summary", "prompt", "continuity_notes"):
        value = update[key]
        if not isinstance(value, str) or not value.strip() or len(value) > (2000 if key == "summary" else 12000) or "\x00" in value:
            raise ValueError("接续调整缺少有效说明或完整提示词。")
    proposed = update["subshots"]
    validate_subshots({**neighbor, "subshots": proposed})
    original = neighbor.get("subshots") or [{"start": 0, "end": neighbor["duration"]}]
    if (len(original) != len(proposed) or any(
            (before.get("start"), before.get("end")) != (after.get("start"), after.get("end"))
            for before, after in zip(original, proposed))):
        raise ValueError("后段接续只调整动作文字，须保留其镜头数量、切点和总时长。")
    return [{"update": copy.deepcopy(update), "before": baseline(neighbor)}]


def planning_recovery_available(state, shot):
    """Legacy creative blockers get one policy migration, not another form."""
    from . import core
    analysis = shot.get("revision_analysis", {})
    if not isinstance(analysis, dict):
        return False
    policy = analysis.get("planning_policy_version", 0)
    blockers = analysis.get("blockers")
    return (shot.get("status") == "planned" and not any(shot.get(k) for k in ("file", "job_id", "prompt_candidate", "prompt_approval"))
            and isinstance(shot.get("feedback"), str) and bool(shot["feedback"].strip())
            and type(shot.get("revision", 0)) is int and shot.get("revision", 0) >= 0
            and type(shot.get("version", 0)) is int and shot.get("version", 0) >= 0
            and analysis.get("signature") == camera.feedback_signature(shot)
            and isinstance(blockers, list) and bool(blockers)
            and all(isinstance(text, str) and text.strip() for text in blockers)
            and type(policy) is int and policy < PLANNING_POLICY_VERSION
            and not any(core.execution_unsettled(job) for job in state.get("jobs", {}).values()))


def dependency(state, shot):
    neighbor = next_shot(state, shot)
    return camera._hash({"next": baseline(neighbor) if neighbor else None,
                         "editable": unstarted(neighbor) if neighbor else False})


def apply(state, shot, record):
    """Called in the same transaction as the target camera draft."""
    from . import core
    entries = record.get("future_continuity_updates", [])
    if not entries:
        return False
    signature = camera.feedback_signature(shot)
    if record.get("signature") != signature:
        raise core.GateError("接续调整与当前返修意见不符，未提交视频。")
    existing = shot.get("future_continuity_binding")
    if existing:
        if existing.get("signature") != signature or existing.get("updates") != entries:
            raise core.GateError("接续调整草稿已改变，请重新准备完整提示词。")
        public_updates(state, shot, strict=True)
        return False
    for entry in entries:
        update = entry["update"]
        try:
            validated = validate_updates(state, shot, [update])
        except ValueError as exc:
            raise core.GateError(str(exc)) from None
        neighbor = next_shot(state, shot)
        if validated[0]["before"] != entry["before"]:
            raise core.GateError("下一个片段的规划已改变，旧接续草稿不可套用。")
        for key in EDIT_FIELDS:
            neighbor[key] = copy.deepcopy(update[key])
        neighbor["continuity_revision"] = {"source_shot_id": shot["id"], "signature": signature,
            "summary": update["summary"], "after_hash": camera._hash(baseline(neighbor))}
        core.event(state, "future_continuity_drafted", id=neighbor["id"], source_shot_id=shot["id"], signature=signature)
    shot["future_continuity_binding"] = {"signature": signature, "updates": copy.deepcopy(entries)}
    return True


def public_updates(state, shot, *, strict=False):
    """Show only the summaries whose saved drafts still match this feedback."""
    binding = shot.get("future_continuity_binding")
    if not binding:
        return []
    valid = binding.get("signature") == camera.feedback_signature(shot)
    summaries = []
    for entry in binding.get("updates", []):
        update = entry["update"]
        neighbor = next_shot(state, shot)
        marker = neighbor.get("continuity_revision", {}) if neighbor else {}
        matches = (neighbor and neighbor["id"] == update["id"] and marker.get("source_shot_id") == shot["id"]
                   and unstarted(neighbor, allow_continuity_draft=True)
                   and marker.get("signature") == binding.get("signature")
                   and marker.get("summary") == update["summary"]
                   and marker.get("after_hash") == camera._hash(baseline(neighbor)))
        valid = valid and bool(matches)
        summaries.append({"id": update["id"], "summary": update["summary"]})
    if not valid and strict:
        from .core import GateError
        raise GateError("已保存的后段接续草稿与当前规划不符，不能沿用旧提示词确认。")
    return summaries if valid else []


def discard(state, shot, *, reason, stamp):
    """Undo only untouched neighbor drafts, leaving any started media alone."""
    binding = shot.pop("future_continuity_binding", None)
    if not binding:
        return
    for entry in binding.get("updates", []):
        neighbor = next_shot(state, shot)
        marker = neighbor.get("continuity_revision", {}) if neighbor else {}
        if (neighbor and neighbor["id"] == entry["update"]["id"]
                and marker.get("source_shot_id") == shot["id"]
                and marker.get("signature") == binding.get("signature")
                and unstarted(neighbor, allow_continuity_draft=True)
                and marker.get("after_hash") == camera._hash(baseline(neighbor))):
            old = entry["before"]["definition"]
            for key in EDIT_FIELDS:
                if key in old:
                    neighbor[key] = copy.deepcopy(old[key])
                else:
                    neighbor.pop(key, None)
            neighbor.pop("continuity_revision", None)
    shot.setdefault("future_continuity_history", []).append({**copy.deepcopy(binding), "reason": reason, "superseded_at": stamp})
