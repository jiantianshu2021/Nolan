"""Local, version-bound human consent for one exact asset generation.

Preparing and reading candidates never accesses credentials or providers. The
dashboard owns the explicit approval call; no model/MCP approval tool exposes it.
Existing media and already submitted jobs remain on their original recovery path.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import uuid

from . import asset_models, core


PROMPT_LIMIT = 100000
_ASSET_FIELDS = ("id", "kind", "name", "prompt", "version", "feedback", "creation_input",
                 "character_id", "voice_only", "audition_speaker", "audition_text",
                 "audition_reason", "audition_source", "audition_profile", "source_episode_id",
                 "source_file", "source_sha256")
_CONFIG_FIELDS = ("provider", "base_url", "api_key_env", "image_model", "image_size",
                  "video_model", "ratio", "resolution", "generate_audio", "video_watermark")
_WIRE_FIELDS = ("image_model", "image_size", "video_model", "ratio", "resolution",
                "generate_audio", "video_watermark")
_STATES = {"awaiting_asset_prompt_approval", "asset_prompt_approved"}
_ACTIVE = {"submitting", "submitted", "download_pending", "completed", "uncertain"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _prompt(value):
    if not isinstance(value, str) or not value.strip() or len(value) > PROMPT_LIMIT or "\x00" in value:
        raise core.GateError("资产最终提示词须为非空文本，最多 100000 字。")
    return value


def _asset(state, asset_or_id):
    identifier = asset_or_id.get("id") if isinstance(asset_or_id, dict) else asset_or_id
    return core._find(state, "asset", identifier)


def _manifest(root, references):
    """Hash the actual ordered local image inputs, never prose or remote URLs."""
    if isinstance(references, dict):
        if references.get("audio"):
            raise core.GateError("资产生成候选不接受额外声音参考。")
        values = references.get("images", [])
        expected = references.get("context", {}).get("image_references", [])
    else:
        values, expected = references, None
    if not isinstance(values, (list, tuple)) or len(values) > 9:
        raise core.GateError("资产生成候选须提供实际参考图片列表，最多 9 张。")
    result = {"images": [], "audio": [], "context": {"image_references": [], "audio_references": []}}
    if expected is not None and len(expected) != len(values):
        raise core.GateError("资产提示词的实际参考素材清单不完整。")
    for index, value in enumerate(values):
        try:
            path = Path(value)
            relative = str(path.resolve().relative_to(Path(root).resolve())) if path.is_absolute() else str(path)
            path = core.safe_path(root, relative)
            if not path.is_file():
                raise ValueError
            digest = core.digest(path)
        except (TypeError, ValueError, OSError):
            raise core.GateError("资产实际参考素材缺失或不在当前项目中，请重新确认。") from None
        if expected is not None and (not isinstance(expected[index], dict)
                                     or expected[index].get("sha256") != digest):
            raise core.GateError("资产提示词引用的素材内容已改变，请重新确认。")
        result["images"].append(relative)
        result["context"]["image_references"].append({"file": relative, "sha256": digest})
    return result


def _source(root, state, asset):
    result = {"asset": {key: copy.deepcopy(asset.get(key)) for key in _ASSET_FIELDS},
              "plan": {key: copy.deepcopy(state.get("plan", {}).get(key))
                       for key in ("status", "version", "style", "ratio")},
              "mode": state.get("mode")}
    relative = asset.get("source_file") or "source/novel.txt"
    source = core.safe_path(root, relative)
    if not source.is_file():
        raise core.GateError("资产对应的原剧本文件缺失，不能使用旧提示词生成。")
    source_hash = core.digest(source)
    if asset.get("source_sha256") and source_hash != asset["source_sha256"]:
        raise core.GateError("资产对应的分集原文校验失败，未开始生成。")
    result["source"] = {"file": relative, "sha256": source_hash}
    if asset.get("kind") == "voice":
        character = core._find(state, "asset", asset.get("character_id"))
        if character.get("kind") != "character" or not core.asset_ready(character):
            raise core.GateError("声音试音须先绑定已批准的角色形象或已登记的内部画外声源。")
        core._check_artifact(root, character)
        result["character"] = {key: copy.deepcopy(character.get(key))
                               for key in ("id", "kind", "version", "file", "sha256", "voice_only",
                                           "voice_only_digest", "internal_role")}
    return result


def _actual_references(root, state, asset):
    if asset.get("kind") == "voice":
        character = core._find(state, "asset", asset.get("character_id"))
        return [] if core.is_internal_voice_source(character) else [str(character.get("file", ""))]
    from .asset_inputs import reference_paths
    return [str(path.relative_to(Path(root).resolve())) for path in reference_paths(root, asset)]


def _configuration(state, asset, model):
    config = asset_models.effective_config(state, asset, model)
    return {key: copy.deepcopy(config.get(key)) for key in _CONFIG_FIELDS if key in config}


def _specification(root, state, asset, candidate):
    return {"source_manifest": _source(root, state, asset), "target_version": candidate["target_version"],
            "prompt": candidate["prompt"], "duration": candidate.get("duration"),
            "reference_manifest": _manifest(root, candidate["reference_manifest"]),
            "provider": _configuration(state, asset, candidate.get("model"))}


def _can_prepare(state, asset):
    if core.is_internal_voice_source(asset):
        raise core.GateError("内部画外声源无需生成资产，请确认对应真实声音资产。")
    if state.get("plan", {}).get("status") != "approved":
        raise core.GateError("请先批准策划，再准备资产生成提示词。")
    if asset.get("status") in {"approved", "candidate", "internal"}:
        raise core.GateError("已有资产候选或已批准素材，不能补做审批或重复生成。")
    if asset.get("status") == "generation_failed":
        raise core.GateError("资产生成已失败，请先提交明确的返修意见，不能自动重新生成。")
    job = state.get("jobs", {}).get(asset.get("job_id"), {})
    if job and (job.get("state") in _ACTIVE or core.execution_unsettled(job)):
        raise core.GateError("资产已有在途或已完成任务，请恢复原任务，不能新建生成审批。")
    from .asset_inputs import confirmed, has_started
    if not has_started(state, asset) and not confirmed(asset):
        raise core.GateError("请先确认资产描述或由 GPT 决定的生成方式，再查看完整提示词。")


def _valid(root, state, asset, candidate):
    if not isinstance(candidate, dict):
        return False
    if asset.get("status") not in _STATES or candidate.get("status") not in {"candidate", "approved"}:
        return False
    expected_status = "approved" if asset.get("status") == "asset_prompt_approved" else "candidate"
    if candidate.get("status") != expected_status:
        return False
    if expected_status == "approved":
        approval = asset.get("generation_prompt_approval", {})
        if (approval.get("candidate_id") != candidate.get("id")
                or approval.get("request_hash") != candidate.get("request_hash")
                or approval.get("model") != candidate.get("model")
                or approval.get("target_version") != candidate.get("target_version")
                or approval.get("decision") != "generate"):
            return False
    version = asset.get("version", 0)
    if (type(version) is not int or candidate.get("source_version") != version
            or candidate.get("target_version") != version + 1 or candidate.get("version") != version + 1):
        return False
    if not isinstance(candidate.get("request_hash"), str) or candidate.get("sha256") != candidate["request_hash"]:
        return False
    _prompt(candidate.get("prompt"))
    try:
        specification = _specification(root, state, asset, candidate)
    except (KeyError, TypeError):
        return False
    if candidate.get("source_manifest") != specification["source_manifest"]:
        return False
    if specification["reference_manifest"]["images"] != _actual_references(root, state, asset):
        return False
    if candidate.get("generation_config") != {key: value for key, value in specification["provider"].items()
                                             if key in _WIRE_FIELDS}:
        return False
    return candidate["request_hash"] == _hash(specification)


def current(root, asset_or_id):
    """Return an unchanged prepared candidate; safe for repeated Continue."""
    state = core.load_project(root)
    asset = _asset(state, asset_or_id)
    candidate = asset.get("generation_prompt")
    if not _valid(root, state, asset, candidate):
        return None
    return copy.deepcopy(candidate)


def prepare(root, asset_id, prompt, references, duration=None):
    """Prepare one exact request without a provider, reservation or version bump.

    An unchanged source keeps its existing candidate (including exact human
    edits). New source/feedback/input/identity invalidates and archives old
    consent, leaving the user a fresh complete prompt to confirm.
    """
    _prompt(prompt)
    if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float))
                                 or not math.isfinite(duration) or duration <= 0):
        raise core.GateError("资产试音时长无效。")
    with core.transaction(root) as state:
        asset = core._find(state, "asset", asset_id)
        _can_prepare(state, asset)
        manifest = _manifest(root, references)
        if manifest["images"] != _actual_references(root, state, asset):
            raise core.GateError("提示词参考素材须与当前已确认的资产输入一致，不能遗漏或替换。")
        existing = asset.get("generation_prompt")
        if _valid(root, state, asset, existing):
            return copy.deepcopy(existing)
        version = asset.get("version", 0)
        if type(version) is not int or version < 0:
            raise core.GateError("资产版本记录无效，不能准备生成。")
        config = asset_models.effective_config(state, asset)
        model = config.get("video_model" if asset.get("kind") == "voice" else "image_model")
        candidate = {"id": "asset-prompt-" + uuid.uuid4().hex[:12], "status": "candidate",
                     "version": version + 1, "target_version": version + 1, "source_version": version,
                     "prompt": prompt, "duration": duration, "reference_manifest": manifest,
                     "model": model, "created_at": core.now()}
        candidate["source_manifest"] = _source(root, state, asset)
        candidate["generation_config"] = {key: copy.deepcopy(config[key]) for key in _WIRE_FIELDS if key in config}
        candidate["request_hash"] = _hash(_specification(root, state, asset, candidate))
        candidate["sha256"] = candidate["request_hash"]
        if existing:
            asset.setdefault("generation_prompt_history", []).append({**copy.deepcopy(existing),
                "status": "invalidated", "invalidated_at": core.now()})
        asset.pop("generation_prompt_approval", None)
        asset.update(status="awaiting_asset_prompt_approval", generation_prompt=candidate)
        state.update(status="awaiting_asset_prompt_approval", last_error=None)
        core.event(state, "asset_prompt_prepared", id=asset_id, candidate_id=candidate["id"],
                   target_version=candidate["target_version"], request_hash=candidate["request_hash"])
    return copy.deepcopy(candidate)


def approve(root, asset_id, candidate_id, request_hash, model=None, prompt=None):
    """Human dashboard action: adopt exact text/model and authorize one version."""
    if prompt is not None:
        _prompt(prompt)
    with core.run_lock(root), core.transaction(root) as state:
        asset = core._find(state, "asset", asset_id)
        _can_prepare(state, asset)
        candidate = asset.get("generation_prompt", {})
        if (asset.get("status") != "awaiting_asset_prompt_approval" or candidate.get("status") != "candidate"
                or not isinstance(candidate_id, str) or candidate.get("id") != candidate_id
                or not isinstance(request_hash, str) or candidate.get("request_hash") != request_hash):
            raise core.GateError("资产提示词候选已更新或已经确认，请刷新后重试。")
        if state.get("paused"):
            raise core.GateError("项目已暂停，请先恢复制作再确认资产提示词。")
        if core.blocking_unknown_jobs(state):
            raise core.GateError("存在结果不明的调用，请先核实；本次未提交资产生成。")
        if not _valid(root, state, asset, candidate):
            raise core.GateError("资产提示词、来源或参考素材已改变，请重新准备并确认。")
        selected = candidate.get("model") if model is None else model
        try:
            config = asset_models.validate_choice(state, asset, selected, candidate.get("duration"), candidate=candidate)
        except ValueError as exc:
            raise core.GateError(str(exc)) from None
        stamp = core.now()
        if (prompt is not None and prompt != candidate["prompt"]) or selected != candidate.get("model"):
            original = copy.deepcopy(candidate)
            asset.setdefault("generation_prompt_history", []).append({**original,
                "status": "superseded_by_human", "superseded_at": stamp})
            candidate = copy.deepcopy(candidate)
            candidate.update(id="asset-prompt-" + uuid.uuid4().hex[:12], created_at=stamp,
                             source_candidate_id=original["id"], source_request_hash=original["request_hash"])
            if prompt is not None:
                candidate.update(prompt=prompt, manual_prompt=True)
        asset["generation_model"] = selected
        candidate.update(model=selected, status="approved", approved_at=stamp)
        candidate["generation_config"] = {key: copy.deepcopy(config[key]) for key in _WIRE_FIELDS if key in config}
        candidate["request_hash"] = _hash(_specification(root, state, asset, candidate))
        candidate["sha256"] = candidate["request_hash"]
        asset.update(status="asset_prompt_approved", generation_prompt=candidate,
                     generation_prompt_approval={"candidate_id": candidate["id"],
                         "request_hash": candidate["request_hash"], "model": selected,
                         "target_version": candidate["target_version"], "decision": "generate", "at": stamp,
                         "source": "human_manual_edit" if candidate.get("manual_prompt") else "human_confirmation"})
        state.update(status="ready", last_error=None)
        core.event(state, "human_approved_asset_prompt", id=asset_id, candidate_id=candidate["id"],
                   target_version=candidate["target_version"], request_hash=candidate["request_hash"], model=selected)
    return core.load_project(root)


def approved(root, asset_or_id):
    """Return exact, unconsumed consent or None; never creates consent itself."""
    state = core.load_project(root)
    asset = _asset(state, asset_or_id)
    candidate = asset.get("generation_prompt", {})
    approval = asset.get("generation_prompt_approval", {})
    if (asset.get("status") != "asset_prompt_approved" or candidate.get("status") != "approved"
            or approval.get("candidate_id") != candidate.get("id")
            or approval.get("request_hash") != candidate.get("request_hash")
            or approval.get("model") != candidate.get("model")
            or approval.get("target_version") != candidate.get("target_version")
            or approval.get("decision") != "generate"):
        return None
    if state.get("paused"):
        raise core.GateError("项目已暂停，不能开始资产生成。")
    _can_prepare(state, asset)
    if not _valid(root, state, asset, candidate):
        return None
    try:
        config = asset_models.validate_choice(state, asset, candidate.get("model"),
                                             candidate.get("duration"), candidate=candidate)
    except ValueError as exc:
        raise core.GateError(str(exc)) from None
    if candidate.get("generation_config") != {key: config[key] for key in _WIRE_FIELDS if key in config}:
        return None
    return copy.deepcopy(candidate)
