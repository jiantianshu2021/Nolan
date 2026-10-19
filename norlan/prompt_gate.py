"""Human consent for the exact next video request, bound to immutable inputs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import uuid

from . import core


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def specification(state, shot, prompt, manifest):
    bindings = []
    for identifier in shot["asset_ids"]:
        asset = core._find(state, "asset", identifier)
        bindings.append({key: copy.deepcopy(asset.get(key)) for key in
                         ("id", "version", "status", "sha256", "audio_sha256", "voice_only_digest")})
    ark = state["settings"]["ark"]
    return {"shot_id": shot["id"], "current_version": shot.get("version", 0),
            "revision": shot.get("revision", 0), "source_prompt": shot["prompt"],
            "source_dialogue": shot.get("dialogue"), "feedback": shot.get("feedback", ""),
            "duration": shot["duration"], "scene_id": shot["scene_id"],
            "subshots": copy.deepcopy(shot.get("subshots")), "prompt": prompt,
            "references": copy.deepcopy(manifest), "assets": bindings,
            "audio_policy": copy.deepcopy(state.get("audio_policy", {})),
            "provider": {"mode": state["mode"], "ratio": state["plan"].get("ratio", "16:9"),
                         **{key: ark.get(key) for key in ("provider", "base_url", "video_model", "resolution", "generate_audio")}}}


def _validate_inputs(root, state, shot, candidate):
    if state["paused"]:
        raise core.GateError("项目已暂停，请先明确恢复，再确认提示词。")
    if any(not core.asset_ready(a) for a in state["assets"]):
        raise core.GateError("尚有新增或变更资产未批准，不能确认视频生成。")
    if any(a.get("status") in {"candidate", "awaiting_assets"} for a in state.get("asset_amendments", [])):
        raise core.GateError("资产变更尚未完成，请先确认新增资产。")
    if (candidate.get("revision") != shot.get("revision", 0)
            or candidate.get("version") != shot.get("version", 0) + 1):
        raise core.GateError("提示词候选已过期，请重新准备。")
    for identifier in shot["asset_ids"]:
        core._check_artifact(root, core._find(state, "asset", identifier))
    manifest = candidate["reference_manifest"]
    context = manifest.get("context", {})
    for key, info_key, hash_key in (("images", "image_references", "sha256"),
                                    ("audio", "audio_references", "sha256")):
        files, information = manifest.get(key, []), context.get(info_key, [])
        if len(files) != len(information):
            raise core.GateError("提示词的素材清单不完整，请重新准备。")
        for relative, info in zip(files, information):
            path = core.safe_path(root, relative)
            if not path.is_file() or core.digest(path) != info.get(hash_key):
                raise core.GateError("提示词引用的素材内容发生变化，请重新准备并确认。")
    previous = context.get("previous_actual")
    if previous:
        source = core._find(state, "shot", previous["shot_id"])
        if not core.human_approved_shot(source) or source.get("sha256") != previous.get("sha256"):
            raise core.GateError("前段批准版本已经变化，请重新准备提示词。")
        core._check_artifact(root, source)
    actual = _hash(specification(state, shot, candidate["prompt"], manifest))
    if actual != candidate.get("request_hash"):
        raise core.GateError("提示词、素材或模型配置已变化，请重新准备并确认。")


def prepare(root, shot_id, refined, prompt, manifest):
    with core.transaction(root) as state:
        shot = core._find(state, "shot", shot_id)
        shot.setdefault("version", 0)
        if shot.get("job_id"):
            raise core.GateError("已有视频任务，不能为同一版本另建生成候选。")
        candidate = {"id": "prompt-" + uuid.uuid4().hex[:12], "status": "candidate",
                     "version": shot.get("version", 0) + 1, "revision": shot.get("revision", 0),
                     "prompt": prompt, "dialogue": refined.get("dialogue", ""),
                     "continuity_notes": refined.get("continuity_notes", ""),
                     "refined": copy.deepcopy(refined), "reference_manifest": copy.deepcopy(manifest),
                     "created_at": core.now()}
        candidate["request_hash"] = _hash(specification(state, shot, prompt, manifest))
        shot.update(status="awaiting_prompt_approval", prompt_candidate=candidate)
        shot.pop("prompt_approval", None)
        state.update(status="awaiting_prompt_approval", last_error=None)
        core.event(state, "prompt_prepared", id=shot_id, candidate_id=candidate["id"],
                   version=candidate["version"], request_hash=candidate["request_hash"])
    return candidate


def _bound(shot, candidate_id, request_hash):
    candidate = shot.get("prompt_candidate", {})
    if (shot.get("status") != "awaiting_prompt_approval" or candidate.get("status") != "candidate"
            or not isinstance(candidate_id, str) or candidate.get("id") != candidate_id
            or not isinstance(request_hash, str) or candidate.get("request_hash") != request_hash):
        raise core.GateError("提示词候选已更新，请刷新后确认。")
    return candidate


def approve(root, shot_id, candidate_id, request_hash):
    with core.run_lock(root), core.transaction(root) as state:
        shot = core._find(state, "shot", shot_id)
        candidate = _bound(shot, candidate_id, request_hash)
        _validate_inputs(root, state, shot, candidate)
        candidate.update(status="approved", approved_at=core.now())
        shot["prompt_approval"] = {"candidate_id": candidate_id, "request_hash": request_hash,
                                   "at": core.now(), "decision": "generate"}
        shot["status"] = "prompt_approved"
        state.update(status="ready", last_error=None)
        core.event(state, "human_approved_prompt", id=shot_id, candidate_id=candidate_id, request_hash=request_hash)
    return core.load_project(root)


def reject(root, shot_id, candidate_id, request_hash, feedback):
    if not isinstance(feedback, str) or not feedback.strip() or len(feedback) > 12000:
        raise ValueError("请填写不超过12000字的提示词修改意见。")
    with core.run_lock(root), core.transaction(root) as state:
        shot = core._find(state, "shot", shot_id)
        candidate = _bound(shot, candidate_id, request_hash)
        shot.setdefault("prompt_history", []).append(copy.deepcopy(candidate))
        shot.pop("prompt_candidate", None)
        shot.pop("prompt_approval", None)
        shot["feedback"] = (shot.get("feedback", "") + "\n提示词修改意见：" + feedback.strip()).strip()
        shot.update(status="planned", revision=shot.get("revision", 0) + 1)
        shot.pop("revision_analysis", None)
        state.update(status="ready", last_error=None)
        core.event(state, "human_rejected_prompt", id=shot_id, candidate_id=candidate_id)
    return core.load_project(root)


def approved(root, shot):
    state = core.load_project(root)
    current = core._find(state, "shot", shot["id"])
    candidate, approval = current.get("prompt_candidate", {}), current.get("prompt_approval", {})
    if (current.get("status") != "prompt_approved" or candidate.get("status") != "approved"
            or approval.get("candidate_id") != candidate.get("id")
            or approval.get("request_hash") != candidate.get("request_hash")
            or approval.get("decision") != "generate"):
        raise core.GateError("请先在网页确认本次提示词并点击生成。")
    try:
        _validate_inputs(root, state, current, candidate)
    except core.GateError:
        invalidate(root, current["id"])
        raise
    return copy.deepcopy(candidate)


def invalidate(root, shot_id):
    """Discard stale consent while leaving a visible route to prepare again."""
    with core.transaction(root) as state:
        shot = core._find(state, "shot", shot_id)
        candidate = shot.pop("prompt_candidate", None)
        if candidate:
            shot.setdefault("prompt_history", []).append({**candidate, "status": "invalidated"})
        shot.pop("prompt_approval", None)
        shot["status"] = "planned"
        core.event(state, "prompt_consent_invalidated", id=shot_id)
