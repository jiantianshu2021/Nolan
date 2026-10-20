"""Human direct edits of completed asset prompts, without a GPT rewrite.

Editor snapshots are read-only. The explicit submit atomically requests one
revision and records exact-text consent; the normal production runner then owns
submission, accounting and recovery. No provider or model is called here.
"""
from __future__ import annotations

import copy
import hashlib
import math
import uuid

from . import asset_models, asset_prompt_gate as gate, core


def _editable(root, state, asset_id, version):
    asset = core._find(state, "asset", asset_id)
    core._check_version(asset, version)
    if type(version) is not int or version < 1:
        raise core.GateError("资产尚未生成，请在卡片内确认首次生成提示词。")
    if core.is_internal_voice_source(asset):
        raise core.GateError("内部画外声源无需生成，请修改对应声音资产。")
    if state.get("mode") != "live" or state.get("plan", {}).get("status") != "approved":
        raise core.GateError("请先批准真实制作项目的策划，再修改资产提示词。")
    if asset.get("status") not in {"candidate", "generation_failed"}:
        raise core.GateError("此资产已更新或不在待审阶段，请刷新后操作。")
    if state.get("paused"):
        raise core.GateError("项目已暂停，请先恢复制作再提交资产修改。")
    if any(core.execution_unsettled(job) for job in state.get("jobs", {}).values()):
        raise core.GateError("尚有在途或结果不明的调用，请等待落盘或核实后再修改资产。")
    if asset.get("status") == "candidate" or asset.get("file"):
        core._check_artifact(root, asset)
    if asset.get("status") == "generation_failed":
        failed = state.get("jobs", {}).get(asset.get("job_id"), {})
        expected = "voice" if asset.get("kind") == "voice" else "image"
        if (not asset.get("generation_error") or failed.get("state") != "failed"
                or failed.get("accounting") not in {"spent", "released"}
                or failed.get("kind") != expected or failed.get("item_id") != asset_id):
            raise core.GateError("资产失败记录尚未核实，不能再次提交；请先核实原调用结果。")
    # Asset replacement has historically invalidated downstream plans. Preserve
    # that bookkeeping for unstarted plans, but never silently retire real media.
    affected = False
    for episode in state.get("episodes", []):
        selected = [shot for shot in episode.get("shots", []) if asset_id in shot.get("asset_ids", [])]
        if selected or affected:
            if episode.get("file") or any(
                    shot.get("file") or shot.get("job_id") or shot.get("version", 0) > 0
                    for shot in episode.get("shots", [])):
                raise core.GateError("此资产已关联制作过的视频，请先处理相关视频的资产返修，现有视频已保留。")
            affected = True
    if asset.get("kind") == "character" and any(
            item.get("kind") == "voice" and item.get("character_id") == asset_id
            and (item.get("file") or item.get("job_id") or item.get("version", 0) > 0)
            for item in state.get("assets", [])):
        raise core.GateError("此角色已有关联声音素材，请先处理关联资产的返修，现有声音已保留。")
    return asset


def _template(state, asset):
    """Prefer the actual saved request, with a deterministic legacy fallback."""
    job = state.get("jobs", {}).get(asset.get("job_id"), {})
    for source, request in (
            ("generation_request", asset.get("generation_request")),
            ("job_request", job.get("asset_generation")),
            ("generation_prompt", asset.get("generation_prompt"))):
        if (isinstance(request, dict) and isinstance(request.get("prompt"), str)
                and request["prompt"].strip()
                and request.get("target_version", asset["version"]) == asset["version"]):
            return request["prompt"], request, source, ""
    if asset.get("kind") == "voice" and isinstance(asset.get("audition_prompt"), str) \
            and asset["audition_prompt"].strip():
        return asset["audition_prompt"], {}, "audition_prompt", ""
    from .asset_inputs import compose_prompt
    if asset.get("kind") == "voice":
        from .voice import build_audition_prompt
        character = core._find(state, "asset", asset.get("character_id"))
        prompt = build_audition_prompt({**asset, "prompt": compose_prompt(asset),
                                       "voice_only": character.get("voice_only") is True})
    else:
        prompt = compose_prompt(asset, state["plan"].get("style", ""))
    return prompt, {}, "legacy_template", "旧版本未保存完整生成请求；已用现有资产记录还原完整模板，请核对后提交。"


def _snapshot(root, state, asset_id, version):
    asset = _editable(root, state, asset_id, version)
    source = gate._source(root, state, asset)
    prompt, saved, prompt_source, notice = _template(state, asset)
    gate._prompt(prompt)
    references = gate._manifest(root, gate._actual_references(root, state, asset))
    if saved.get("reference_manifest") is not None:
        if gate._manifest(root, saved["reference_manifest"]) != references:
            raise core.GateError("原生成参考素材与当前素材不一致，请先核实；未替换或删减参考素材。")
    duration = None
    if asset.get("kind") == "voice":
        from .voice import audition_duration
        duration = saved.get("duration")
        if duration is None:
            job = state.get("jobs", {}).get(asset.get("job_id"), {})
            duration = job.get("duration")
            if duration is None:
                duration = job.get("video_request", {}).get("duration")
            if duration is None:
                duration = audition_duration(asset)
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0):
            raise core.GateError("原声音试片时长记录无效，请先核实。")
    model_key = "video_model" if asset.get("kind") == "voice" else "image_model"
    model = saved.get("model") or asset_models.effective_config(state, asset).get(model_key)
    view_asset = {**asset, "generation_model": model}
    specification = {
        "source": source, "status": asset["status"], "version": version,
        "artifact": {key: copy.deepcopy(asset.get(key)) for key in
                     ("file", "sha256", "audio_file", "audio_sha256", "job_id", "generation_error")},
        "prompt": prompt, "prompt_source": prompt_source, "duration": duration,
        "reference_manifest": references, "model": model,
        "configuration": gate._configuration(state, view_asset, model),
        "saved_request": copy.deepcopy(saved),
    }
    result = {"id": asset_id, "version": version, "target_version": version + 1,
              "sha256": gate._hash(specification), "prompt": prompt, "model": model,
              "duration": duration, "reference_manifest": references,
              "prompt_source": prompt_source, "source_notice": notice}
    result["models"] = asset_models.options(state, view_asset, result)
    return result


def editor_snapshot(root, asset_id, version):
    """Read a complete editable template without changing state or approving."""
    return _snapshot(root, core.load_project(root), asset_id, version)


def edit_and_approve(root, asset_id, version, sha256, prompt, model=None):
    """Adopt exact human text/model and authorize one asset revision atomically."""
    gate._prompt(prompt)
    with core.run_lock(root), core.transaction(root) as state:
        snapshot = _snapshot(root, state, asset_id, version)
        if not isinstance(sha256, str) or snapshot["sha256"] != sha256:
            raise core.GateError("资产、提示词或参考素材已更新，请重新打开卡片编辑器后提交。")
        asset = core._find(state, "asset", asset_id)
        selected = snapshot["model"] if model is None else model
        view_asset = {**asset, "generation_model": snapshot["model"]}
        try:
            config = asset_models.validate_choice(state, view_asset, selected,
                                                  snapshot["duration"], candidate=snapshot)
        except ValueError as exc:
            raise core.GateError(str(exc)) from None
        stamp = core.now()
        reason = "用户直接修改完整提示词并提交生成。"
        original = copy.deepcopy(asset.get("generation_prompt"))
        core._reject_asset(state, asset, version, reason)
        core.event(state, "human_rejected", kind="asset", id=asset_id, version=version,
                   feedback=reason, shot_ids=None)
        # Both original and edited text are retained without feeding the edit
        # into feedback interpretation, voice selection or prompt composition.
        if original:
            asset.setdefault("generation_prompt_history", []).append({
                **original, "status": "superseded_by_human", "superseded_at": stamp})
        candidate = {
            "id": "asset-prompt-" + uuid.uuid4().hex[:12], "status": "approved",
            "version": version + 1, "target_version": version + 1, "source_version": version,
            "prompt": prompt, "duration": snapshot["duration"],
            "reference_manifest": copy.deepcopy(snapshot["reference_manifest"]),
            "model": selected, "created_at": stamp, "approved_at": stamp,
            "manual_prompt": True, "manual_revision": True,
            "editor_sha256": sha256, "original_prompt": snapshot["prompt"],
            "original_prompt_sha256": hashlib.sha256(snapshot["prompt"].encode()).hexdigest(),
            "prompt_source": snapshot["prompt_source"],
        }
        asset["generation_model"] = selected
        candidate["source_manifest"] = gate._source(root, state, asset)
        candidate["generation_config"] = {key: copy.deepcopy(config[key])
                                          for key in gate._WIRE_FIELDS if key in config}
        candidate["request_hash"] = gate._hash(gate._specification(root, state, asset, candidate))
        candidate["sha256"] = candidate["request_hash"]
        asset.update(status="asset_prompt_approved", generation_prompt=candidate,
                     generation_prompt_approval={
                         "candidate_id": candidate["id"], "request_hash": candidate["request_hash"],
                         "model": selected, "target_version": version + 1, "decision": "generate",
                         "at": stamp, "source": "human_manual_edit"})
        state.update(status="ready", last_error=None)
        core.event(state, "human_approved_asset_prompt", id=asset_id, candidate_id=candidate["id"],
                   target_version=version + 1, request_hash=candidate["request_hash"],
                   model=selected, source="human_manual_edit", editor_sha256=sha256)
    return core.load_project(root)
