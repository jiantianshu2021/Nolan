"""Local, resumable camera redesign under the existing final-prompt gate.

One segment remains one provider request.  This module only stages its internal
camera definition; it never approves an asset, prompt, or generated video.
"""
from __future__ import annotations

import copy
import hashlib
import json

from .segments import validate_subshots


DEFINITION_FIELDS = ("id", "scene_id", "duration", "prompt", "dialogue", "subshots",
                     "continuity_notes", "source_shot_ids", "source_shot_instructions")


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def feedback_signature(shot):
    return hashlib.sha256(json.dumps([shot.get("revision", 0), shot.get("feedback", "")],
                                    ensure_ascii=False).encode()).hexdigest()


def definition(shot):
    return {key: copy.deepcopy(shot[key]) for key in DEFINITION_FIELDS if key in shot}


def baseline(shot):
    return {"definition": definition(shot), "version": shot.get("version", 0),
            "revision": shot.get("revision", 0)}


def validate_plan(shot, proposal):
    """Strictly validate the only editable structural fields, without mutation."""
    if proposal is None:
        return None
    if not isinstance(proposal, dict) or set(proposal) != {"summary", "duration", "subshots"}:
        raise ValueError("局部分镜方案只能包含 summary、duration 和 subshots，不能改变片段、场景或人物对白。")
    summary = proposal["summary"]
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 2000 or "\x00" in summary:
        raise ValueError("局部分镜调整说明须为不超过 2000 字的非空文本。")
    if type(proposal["duration"]) is not int or not 4 <= proposal["duration"] <= 15:
        raise ValueError("局部分镜时长须为 4–15 秒整数；超过单次生成上限须先明确调整方案，不能截短或加速。")
    effective = {**shot, "duration": proposal["duration"], "subshots": copy.deepcopy(proposal["subshots"])}
    validate_subshots(effective)
    return copy.deepcopy(proposal)


def discard(shot, *, reason, stamp, state=None):
    """Retire current metadata; undo a draft only if it was never submitted.

    Generated/failed versions keep the camera definition they actually used.
    A rejected *prompt* returns to its exact preceding definition for replanning.
    """
    if state is not None:
        from . import future_continuity
        future_continuity.discard(state, shot, reason=reason, stamp=stamp)
    public = shot.pop("storyboard_revision", None)
    binding = shot.pop("storyboard_revision_binding", None)
    if not public and not binding:
        return
    unsubmitted = (isinstance(binding, dict) and not shot.get("file") and not shot.get("job_id")
                   and shot.get("version", 0) == binding.get("version"))
    shot.setdefault("storyboard_revision_history", []).append({
        "storyboard_revision": copy.deepcopy(public), "binding": copy.deepcopy(binding),
        "definition": definition(shot), "status": "superseded_draft" if unsubmitted else "superseded_version",
        "superseded_at": stamp, "reason": reason})
    if unsubmitted:
        original = binding.get("before", {}).get("definition", {})
        if "subshots" in original:
            shot["subshots"] = copy.deepcopy(original["subshots"])
        else:
            shot.pop("subshots", None)
        if "duration" in original:
            shot["duration"] = original["duration"]


def apply(root, shot_id, record):
    """Apply one verified/cached plan once after any asset gate has completed."""
    from . import core
    proposal = record.get("storyboard_revision")
    if proposal is None:
        if not record.get("future_continuity_updates"):
            return False
        with core.transaction(root) as state:
            from . import future_continuity
            shot = core._find(state, "shot", shot_id)
            if shot.get("future_continuity_binding"):
                return future_continuity.apply(state, shot, record)
            if (shot.get("status") != "planned" or shot.get("file") or shot.get("job_id")
                    or shot.get("prompt_candidate") or shot.get("prompt_approval")
                    or shot.get("pending_asset_amendment")
                    or any(item.get("status") in {"candidate", "awaiting_assets"} for item in state.get("asset_amendments", []))
                    or any(core.execution_unsettled(job) for job in state["jobs"].values())
                    or any(not core.asset_ready(asset) for asset in state["assets"])):
                raise core.GateError("请先完成原任务及资产审批，再准备接续草稿。")
            return future_continuity.apply(state, shot, record)
    with core.transaction(root) as state:
        shot = core._find(state, "shot", shot_id)
        proposal = validate_plan(shot, proposal)
        signature = feedback_signature(shot)
        if signature != record.get("signature"):
            raise core.GateError("局部分镜意见已改变，请重新分析本次返修；尚未生成视频。")
        existing = shot.get("storyboard_revision_binding")
        if isinstance(existing, dict) and existing.get("signature") == signature:
            expected = {"summary": proposal["summary"], "before_count": existing["before_count"],
                        "before_duration": existing["before"]["definition"]["duration"],
                        "duration": proposal["duration"], "subshots": proposal["subshots"]}
            if (shot.get("storyboard_revision") != expected or shot.get("subshots") != proposal["subshots"]
                    or shot.get("duration") != proposal["duration"] or existing.get("plan_hash") != _hash(proposal)):
                raise core.GateError("已保存的局部分镜草稿发生变化，请提交新的返修意见。")
            from . import future_continuity
            future_continuity.apply(state, shot, record)
            return False
        if (shot.get("status") != "planned" or shot.get("file") or shot.get("job_id")
                or shot.get("prompt_candidate") or shot.get("prompt_approval")
                or any(core.execution_unsettled(job) for job in state["jobs"].values())):
            raise core.GateError("片段已有生成任务或待确认提示词，不能改写内部镜头；尚未提交新视频。")
        if (shot.get("pending_asset_amendment")
                or any(item.get("status") in {"candidate", "awaiting_assets"}
                       for item in state.get("asset_amendments", []))
                or any(not core.asset_ready(asset) for asset in state["assets"])):
            raise core.GateError("请先完成人工资产审批，再应用本次局部分镜并准备提示词。")
        before = baseline(shot)
        if record.get("storyboard_baseline") != before:
            raise core.GateError("片段定义或版本已变化，原局部分镜方案已过期，请提交新的返修意见。")
        before_count = len(shot.get("subshots", [])) or 1
        public = {"summary": proposal["summary"], "before_count": before_count, "before_duration": shot["duration"],
                  "duration": proposal["duration"], "subshots": copy.deepcopy(proposal["subshots"])}
        shot["duration"] = proposal["duration"]
        shot["subshots"] = copy.deepcopy(proposal["subshots"])
        shot["storyboard_revision"] = public
        shot["storyboard_revision_binding"] = {
            "signature": signature, "version": shot.get("version", 0),
            "revision": shot.get("revision", 0), "before_count": before_count,
            "before": before, "plan_hash": _hash(proposal), "applied_at": core.now()}
        shot.pop("refined_prompt", None)
        from . import future_continuity
        future_continuity.apply(state, shot, record)
        owner = next(ep for ep in state["episodes"] if any(item is shot for item in ep["shots"]))
        core._invalidate_render(owner, "局部分镜或时长已调整，须按新视频重做拼接和后期。")
        core.event(state, "storyboard_revision_drafted", id=shot_id, signature=signature,
                   before_count=before_count, after_count=len(proposal["subshots"]),
                   before_duration=before["definition"]["duration"], duration=proposal["duration"])
    return True


def public(shot):
    """Validate and return the small current draft bound by prompt consent."""
    value = shot.get("storyboard_revision")
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {"summary", "before_count", "before_duration", "duration", "subshots"}
            or type(value["before_count"]) is not int or not 1 <= value["before_count"] <= 3):
        raise ValueError("局部分镜草稿元数据无效，请重新准备完整提示词。")
    proposal = {"summary": value["summary"], "duration": value["duration"], "subshots": value["subshots"]}
    validate_plan(shot, proposal)
    binding = shot.get("storyboard_revision_binding", {})
    if (not isinstance(binding, dict) or binding.get("signature") != feedback_signature(shot)
            or binding.get("revision") != shot.get("revision", 0)
            or binding.get("before_count") != value["before_count"]
            or type(binding.get("version")) is not int
            or binding.get("version") not in {shot.get("version", 0), shot.get("version", 0) - 1}
            or binding.get("plan_hash") != _hash(proposal)
            or binding.get("before", {}).get("definition", {}).get("duration") != value["before_duration"]
            or value["duration"] != shot.get("duration") or value["subshots"] != shot.get("subshots")):
        raise ValueError("局部分镜草稿与当前返修记录不一致，请重新准备完整提示词。")
    return copy.deepcopy(value)
