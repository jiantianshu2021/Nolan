"""Prepare one source-backed episode without replacing a finished production.

The persistent candidate is a human gate. It cannot generate media or install
assets/shots until an explicit, version-bound human decision adopts the plan.
"""
from __future__ import annotations

import copy
import math
import re
from contextlib import nullcontext
from pathlib import Path

from . import core
from .asset_inputs import empty_input
from .brain import BrainError, EPISODE_PLAN_SCHEMA, _validate
from .voice import AuditionSelectionError, validate_audition


_DEFINITION_FIELDS = ("id", "kind", "name", "prompt", "character_id", "voice_only",
                      "audition_speaker", "audition_text", "audition_reason", "audition_source")
_CATALOG_FIELDS = ("id", "kind", "name", "character_id", "voice_only")
_ASSET_BINDING_FIELDS = _DEFINITION_FIELDS + ("status", "version", "file", "sha256", "audio_file", "audio_sha256",
                                             "approved_version", "approved_sha256", "approved_at")
_TECHNICAL_FEEDBACK = (
    "仅校正当前缓存方案的结构错误：补齐镜头文字中已明确写出的资产ID引用；每段最多8项视觉资产，"
    "另为官方衔接尾帧预留1个图像位置，最多3项声音。把非本镜实际出镜或使用的资产从对应镜头参考清单移走，"
    "实际需要时合理重分镜头节拍，不能省略真实所需素材或用文字冒充参考。"
    "只保留当前集原稿情节、人物对白、原集编号标题和原稿总时长，不为参考数量上限删减剧情或增加虚构慢动作。"
    "仅复用视觉风格，旧第1集专用的结尾约10秒慢动作等叙事安排不适用于本集，除非本集原稿明确要求。"
    "本次只是未批准的程序结构校正，不是人工反馈或审批，不生成任何媒体。"
)
_NEGATIVE_REFERENCE = re.compile(r"不要引用|不要使用|禁止|不引用|不使用|无需|不需要|不用|不得|勿", re.I)


def _target(state, episode_id):
    episode = core._find(state, "episode", episode_id)
    if (episode.get("shots") or episode.get("file") or episode.get("job_id")
            or episode.get("version", 0) != 0
            or episode.get("status") not in {"awaiting_episode_plan", "episode_plan_failed"}):
        raise core.GateError("本集已有分镜、视频或制作记录，不能用新分集方案覆盖；请使用局部返修。")
    return episode


def _source(root, episode):
    relative, expected = episode.get("source_file"), episode.get("source_sha256")
    if not isinstance(relative, str) or not relative.startswith("source/") or not isinstance(expected, str):
        raise core.GateError("本集尚未登记独立原稿和来源哈希，请先导入本集剧本。")
    path = core.safe_path(root, relative)
    if not path.is_file() or core.digest(path) != expected:
        raise core.GateError("本集原稿缺失或来源哈希已改变，请核实导入内容；不会重用旧方案。")
    text = path.read_text(encoding="utf-8")
    if not text.strip() or len(text) > 60000:
        raise core.GateError("本集原稿为空或超过单集规划上限；请整理本集来源，不能静默截断正文。")
    return text


def _idle(state):
    if state.get("paused"):
        raise core.GateError("项目已暂停，请先继续项目再准备本集方案。")
    if state.get("plan", {}).get("status") != "approved":
        raise core.GateError("请先批准项目的原始策划，再准备后续分集。")
    if any(core.execution_unsettled(job) for job in state.get("jobs", {}).values()):
        raise core.GateError("尚有在途或结果不明的调用，请先核实原任务，不能重复准备分集。")
    if any(item.get("status") in {"candidate", "awaiting_assets"} for item in state.get("asset_amendments", [])):
        raise core.GateError("请先完成已有资产增补，再准备新一集。")


def _baseline(root, state, episode):
    """Bind local facts, without putting histories or prior scripts into GPT."""
    previous = None
    for item in state["episodes"][:state["episodes"].index(episode)]:
        if item.get("status") != "approved":
            raise core.GateError("前面的分集尚未由你批准最终成片，不能开始这一集。")
        if not item.get("sha256") or item.get("approved_sha256") != item["sha256"]:
            raise core.GateError("前集最终审批与当前文件不符，请先核实前集。")
        previous = item
    if previous:
        core._check_artifact(root, previous)
    previous_binding = ({key: copy.deepcopy(previous.get(key)) for key in
                         ("id", "version", "file", "sha256", "approved_sha256", "approved_at")}
                        if previous else None)
    definitions = [{key: copy.deepcopy(asset.get(key)) for key in _ASSET_BINDING_FIELDS} for asset in state["assets"]]
    identity = {key: copy.deepcopy(episode.get(key)) for key in
                ("id", "title", "version", "source_file", "source_sha256", "duration", "source_duration")}
    return {"episode": identity, "previous_episode": previous_binding,
            "assets_sha256": core._json_digest(definitions),
            "style": state["plan"].get("style", ""), "ratio": state["plan"].get("ratio", "16:9"),
            "audio_policy": copy.deepcopy(state.get("audio_policy", {}))}


def _expected_duration(episode):
    value = episode.get("source_duration", episode.get("duration"))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 4 <= value <= 180:
        raise core.GateError("原稿分集时长须为4–180秒，不能改用默认时长。")
    return value


def _signature(baseline, feedback, technical=None):
    fields = {"baseline": baseline, "feedback": feedback}
    if technical is not None:
        fields["technical_repair"] = technical
    return core._json_digest(fields)


def _completed_cache(state, signature):
    cached = next((job for job in reversed(list(state.get("jobs", {}).values()))
        if job.get("item_id") == "plan_episode" and job.get("state") == "completed"
        and job.get("post_cache_key") == signature), None)
    if cached is not None and ("post_result" not in cached
            or core._json_digest(cached["post_result"]) != cached.get("post_result_hash")):
        raise core.GateError("已完成分集规划缓存缺失或被改动，不能重复调用，请核实原记录。")
    return cached


def planning_context(state, episode):
    """Only the current source, a catalog and previous accepted ending are used."""
    index = state["episodes"].index(episode)
    previous = state["episodes"][index - 1] if index else None
    ending = None
    if previous:
        last = (previous.get("shots") or [{}])[-1]
        observation = last.get("review", {})
        ending = {"id": previous["id"], "title": previous["title"],
                  "summary": previous.get("summary", ""),
                  "ending_dialogue": last.get("dialogue", ""),
                  "continuity_notes": last.get("continuity_notes", ""),
                  "observed_end_state": observation.get("observed_end_state", "") if isinstance(observation, dict) else ""}
    return {"episode": {"id": episode["id"], "title": episode["title"],
                        "expected_duration": _expected_duration(episode)},
            "style": state["plan"].get("style", ""), "ratio": state["plan"].get("ratio", "16:9"),
            "audio_policy": copy.deepcopy(state.get("audio_policy", {})), "previous_episode": ending,
            "existing_assets": [{key: copy.deepcopy(asset.get(key, False if key == "voice_only" else ""))
                                 for key in _CATALOG_FIELDS} for asset in state["assets"]]}


def validate_proposal(state, episode, result, script):
    """Validate only the target/new definitions, retaining old assets verbatim."""
    _validate(result, EPISODE_PLAN_SCHEMA)
    proposed, additions = result["episode"], result["new_assets"]
    if proposed["id"] != episode["id"] or proposed["title"] != episode["title"]:
        raise core.GateError("分集方案改变了原集编号或标题，不能采用。")
    expected = _expected_duration(episode)
    total = sum(shot["duration"] for shot in proposed["shots"])
    if total > 180 or (expected is not None and abs(total - expected) > 0.001):
        raise core.GateError("分镜总时长与本集原稿时长不符；不能自动缩短、拆集或修改其他集。")
    old_ids = {item["id"] for item in state["assets"] + state["episodes"]}
    old_ids.update(shot["id"] for item in state["episodes"] for shot in item.get("shots", []))
    if any(asset["id"] in old_ids for asset in additions):
        raise core.GateError("新资产 ID 与现有记录冲突；分集方案不能重定义既有资产。")
    old_identity = {(asset["kind"], asset["name"].strip()) for asset in state["assets"]}
    existing_voice_characters = {asset.get("character_id") for asset in state["assets"] if asset["kind"] == "voice"}
    if any((asset["kind"], asset["name"].strip()) in old_identity
           or (asset["kind"] == "voice" and asset.get("character_id") in existing_voice_characters)
           for asset in additions):
        raise core.GateError("方案重复建立已有角色、场景或声音，请复用现有资产 ID。")
    if any(shot["id"] in old_ids for shot in proposed["shots"]):
        raise core.GateError("新分镜 ID 与现有记录冲突；不能改写其他集。")
    # Old auditions may predate today's selection policy. Validate references
    # using minimal aliases; never revalidate or replace their approved media.
    old_assets = [{"id": asset["id"], "kind": asset["kind"], "name": "沿用已有资产", "prompt": "沿用已批准定义",
                   "character_id": asset.get("character_id", "")} for asset in state["assets"]]
    merged = old_assets + copy.deepcopy(additions)
    core.validate_plan({"title": state["title"], "ratio": state["plan"].get("ratio", "16:9"),
                        "assets": merged, "episodes": [proposed]})
    by_id = {asset["id"]: asset for asset in state["assets"] + additions}
    used = {ref for shot in proposed["shots"] for ref in shot["asset_ids"]}
    if any(asset["id"] not in used for asset in additions):
        raise core.GateError("新资产须确实用于本集，不能批量生成尚未需要的后续资产。")
    for shot in proposed["shots"]:
        if by_id.get(shot["scene_id"], {}).get("kind") != "location" or shot["scene_id"] not in shot["asset_ids"]:
            raise core.GateError("分镜 scene_id 须引用本片段已列入参考的实际场景资产。")
        visual_count = sum(by_id[ref]["kind"] != "voice" and not core.is_internal_voice_source(by_id[ref])
                           for ref in shot["asset_ids"])
        voice_count = sum(by_id[ref]["kind"] == "voice" for ref in shot["asset_ids"])
        if visual_count > 8 or voice_count > 3:
            raise core.GateError(f"分镜 {shot['id']} 的视觉参考为{visual_count}项、声音为{voice_count}项；"
                                 "每段最多8项视觉资产和3项声音，须为衔接尾帧预留图像位置，不能静默丢弃真实参考。")
    for asset in additions:
        if asset["kind"] == "voice" and asset.get("audition_text", "").strip():
            character = by_id[asset["character_id"]]
            try:
                validate_audition(asset, script, asset.get("audition_speaker", "").strip() or character["name"])
            except AuditionSelectionError as exc:
                raise core.GateError(str(exc)) from None
    return result


def normalize_proposal(result):
    """Fill exact new-asset IDs already written in the same shot's directions.

    This repairs a mechanical omission only. Names, dialogue, similar IDs and
    inferred story requirements do not authorize a new asset reference. The
    original model result/cache is retained unchanged, then strict validation
    still checks the complete normalized proposal.
    """
    _validate(result, EPISODE_PLAN_SCHEMA)
    normalized = copy.deepcopy(result)
    patterns = [(asset["id"], re.compile(r"(?<![A-Za-z0-9_-])" + re.escape(asset["id"]) + r"(?![A-Za-z0-9_-])"))
                for asset in normalized["new_assets"]]
    repairs = []
    for shot in normalized["episode"]["shots"]:
        text = "\n".join([shot["prompt"], shot["continuity_notes"],
                          *(subshot["prompt"] for subshot in shot["subshots"])])
        sentences = re.split(r"[。！？!?；;\n]", text)
        added = []
        for identifier, pattern in patterns:
            matching = [sentence for sentence in sentences if pattern.search(sentence)]
            if (identifier not in shot["asset_ids"] and matching
                    and not any(_NEGATIVE_REFERENCE.search(sentence) for sentence in matching)):
                added.append(identifier)
        if added:
            shot["asset_ids"].extend(added)
            repairs.append({"shot_id": shot["id"], "asset_ids": added,
                            "reason": "将本片段文字中明确写出的新增资产ID补入参考清单；未改动剧情、时长或既有资产。"})
    return normalized, repairs


def _save_candidate(root, episode_id, signature, baseline, result, repairs, *, recovered=False, revised=False):
    with core.transaction(root) as current:
        target = _target(current, episode_id)
        _source(root, target)
        pending = target.get("planning", {})
        if pending.get("signature") != signature or _baseline(root, current, target) != baseline:
            raise core.GateError("分集来源、资产或前集审批已改变，旧候选不能采用。")
        if revised:
            pending["revision"] += 1
        pending.update(status="candidate", episode=copy.deepcopy(result["episode"]),
                       new_assets=copy.deepcopy(result["new_assets"]), summary=result["episode"]["summary"],
                       result_sha256=core._json_digest(result), local_repairs=copy.deepcopy(repairs),
                       completed_at=core.now())
        pending.pop("error", None)
        if recovered:
            pending["recovered"] = True
        target["status"] = "awaiting_episode_plan"
        current.update(status="awaiting_episode_plan", last_error=None)
        core.event(current, "episode_plan_cache_recovered" if recovered else "episode_plan_prepared",
                   episode_id=episode_id, revision=pending["revision"])
    return core.load_project(root)


def prepare_episode(root, episode_id, *, planner=None, already_locked=False):
    """Save a reusable candidate; a crash/uncertain call cannot replay itself."""
    root = Path(root).resolve()
    with (nullcontext() if already_locked else core.run_lock(root)):
        state = core.load_project(root)
        _idle(state)
        episode = _target(state, episode_id)
        script = _source(root, episode)
        baseline = _baseline(root, state, episode)
        prior = episode.get("planning", {})
        feedback = prior.get("feedback", "")
        technical = prior.get("technical_repair") if prior.get("status") != "rejected" else None
        signature = _signature(baseline, feedback, technical)
        if (prior.get("status") in {"preparing", "failed"} and prior.get("signature") == signature
                and prior.get("baseline") == baseline):
            # A process may exit after the budgeted caller stored the complete
            # answer but before this proposal was written. Recover only that
            # hash-verified answer; never invoke GPT to repair a missing cache.
            cached = _completed_cache(state, signature)
            if cached is not None:
                try:
                    result, repairs = normalize_proposal(cached["post_result"])
                    validate_proposal(state, episode, result, script)
                except (BrainError, core.GateError, core.PlanValidationError):
                    if prior.get("status") != "failed":
                        raise
                else:
                    # A failed call is not a general retry escape. Only an
                    # actual deterministic reference repair can reopen it.
                    if prior.get("status") == "preparing" or repairs:
                        return _save_candidate(root, episode_id, signature, baseline, result, repairs, recovered=True)
        if prior.get("status") in {"preparing", "uncertain"}:
            raise core.GateError(prior.get("error") or "本集方案已有未核实的准备记录，不会重复请求；请先核对原调用。")
        if prior.get("status") == "failed":
            raise core.GateError(prior.get("error") or "本集方案已有失败记录，不会重复请求；请核对任务后提交新的调整意见。")
        if prior.get("signature") == signature:
            if prior.get("status") == "candidate":
                original = {"episode": prior.get("episode"), "new_assets": prior.get("new_assets")}
                if core._json_digest(original) != prior.get("result_sha256"):
                    raise core.GateError("已保存分集候选校验失败，请核实记录，不能重复调用。")
                result, repairs = normalize_proposal(original)
                validate_proposal(state, episode, result, script)
                if repairs:
                    return _save_candidate(root, episode_id, signature, baseline, result, repairs, revised=True)
                if state.get("status") != "awaiting_episode_plan" or state.get("last_error"):
                    with core.transaction(root) as current:
                        current.update(status="awaiting_episode_plan", last_error=None)
                    return core.load_project(root)
                return state
        revision = prior.get("revision", 0) + 1
        with core.transaction(root) as current:
            target = _target(current, episode_id)
            if _baseline(root, current, target) != baseline:
                raise core.GateError("分集来源或前集审批已改变，请刷新后重试。")
            if target.get("planning"):
                target.setdefault("planning_history", []).append(copy.deepcopy(target["planning"]))
            target["planning"] = {"status": "preparing", "revision": revision, "signature": signature,
                                  "baseline": baseline, "feedback": feedback, "created_at": core.now()}
            current.update(status="preparing_episode_plan", last_error=None)
            core.event(current, "episode_plan_started", episode_id=episode_id, revision=revision)
        try:
            context = planning_context(state, episode)
            if feedback:
                context["feedback"] = feedback
            if planner is None:
                from .pipeline import _brain
                result = _brain(root, "plan_episode", script, context, _cache_key=signature)
            else:
                result = planner(script, context)
            result, repairs = normalize_proposal(result)
            validate_proposal(state, episode, result, script)
            _save_candidate(root, episode_id, signature, baseline, result, repairs)
        except Exception as exc:
            message = str(exc) if isinstance(exc, (BrainError, core.GateError, core.PlanValidationError)) else "本集方案准备失败，请核对任务记录后提交新的调整意见。"
            with core.transaction(root) as current:
                target = core._find(current, "episode", episode_id)
                pending = target.get("planning", {})
                if pending.get("signature") == signature and pending.get("status") == "preparing":
                    pending.update(status="uncertain" if getattr(exc, "uncertain", False) else "failed", error=message)
                    target["status"] = "episode_plan_failed"
                    current.update(status="blocked", last_error=message)
            raise
    return core.load_project(root)


def repair_cached_structure(root, episode_id, *, planner=None, already_locked=False):
    """Allow one explicit technical rewrite of a fully cached failed proposal.

    This is not a human approval/feedback route and is never called by normal
    resume. The attempt is durable before GPT, with ordinary budget/cache
    accounting and no media submission. All results still wait for the human.
    """
    root = Path(root).resolve()
    with (nullcontext() if already_locked else core.run_lock(root)):
        state = core.load_project(root)
        _idle(state)
        episode = _target(state, episode_id)
        script = _source(root, episode)
        baseline = _baseline(root, state, episode)
        prior = episode.get("planning", {})
        if prior.get("status") != "failed":
            raise core.GateError("只有已核实完成、但结构校验失败的分集方案可进行一次程序校正。")
        if episode.get("technical_repair_count", 0) or prior.get("technical_repair_count", 0):
            raise core.GateError("本集已使用一次程序结构校正，不会反复调用；请提交人工调整意见。")
        if prior.get("baseline") != baseline or prior.get("signature") != _signature(baseline, prior.get("feedback", "")):
            raise core.GateError("原稿、资产或前集审批已改变，不能校正旧缓存方案。")
        cached = _completed_cache(state, prior["signature"])
        if cached is None:
            raise core.GateError("缺少已完成且校验一致的原始规划缓存，不会重新请求或猜测调用结果。")
        _, initial_repairs = normalize_proposal(cached["post_result"])
        # Mechanical additions may exceed the output schema's reference cap.
        # Send the intact schema-valid raw proposal for structural correction;
        # do not exhaust the single repair attempt on a local input rejection.
        proposal = copy.deepcopy(cached["post_result"])
        technical = {"attempt": 1, "source_signature": prior["signature"],
                     "source_result_sha256": cached["post_result_hash"]}
        feedback = prior.get("feedback", "")
        signature = _signature(baseline, feedback, technical)
        revision = prior["revision"] + 1
        with core.transaction(root) as current:
            target = _target(current, episode_id)
            if target.get("planning") != prior or _baseline(root, current, target) != baseline:
                raise core.GateError("本集规划或来源已改变，未提交结构校正。")
            target.setdefault("planning_history", []).append(copy.deepcopy(prior))
            target["technical_repair_count"] = 1
            target["planning"] = {"status": "preparing", "revision": revision, "signature": signature,
                                  "baseline": baseline, "feedback": feedback, "created_at": core.now(),
                                  "technical_repair_count": 1, "technical_repair": technical,
                                  "technical_feedback": _TECHNICAL_FEEDBACK,
                                  "initial_local_repairs": initial_repairs}
            target["status"] = "awaiting_episode_plan"
            current.update(status="preparing_episode_plan", last_error=None)
            core.event(current, "episode_plan_technical_repair_started", episode_id=episode_id, revision=revision)
        try:
            context = planning_context(state, episode)
            if feedback:
                context["feedback"] = feedback
            context.update(proposal_to_repair=proposal, technical_feedback=_TECHNICAL_FEEDBACK)
            if planner is None:
                from .pipeline import _brain
                result = _brain(root, "plan_episode", script, context, _cache_key=signature)
            else:
                result = planner(script, context)
            result, repairs = normalize_proposal(result)
            validate_proposal(state, episode, result, script)
            _save_candidate(root, episode_id, signature, baseline, result, repairs)
        except Exception as exc:
            message = str(exc) if isinstance(exc, (BrainError, core.GateError, core.PlanValidationError)) else "本集方案结构校正失败；已保留一次校正记录，不会自动重复调用。"
            with core.transaction(root) as current:
                target = core._find(current, "episode", episode_id)
                pending = target.get("planning", {})
                if pending.get("signature") == signature and pending.get("status") == "preparing":
                    pending.update(status="uncertain" if getattr(exc, "uncertain", False) else "failed", error=message)
                    target["status"] = "episode_plan_failed"
                    current.update(status="blocked", last_error=message)
            raise
    return core.load_project(root)


def decide_episode_plan(root, episode_id, revision, signature, decision, feedback=""):
    """Adopt/reject a plan on behalf of an explicit human web decision only."""
    if decision not in {"approve", "reject"}:
        raise ValueError("请选择采用或调整本集方案。")
    if not isinstance(feedback, str) or len(feedback) > 12000 or (decision == "reject" and not feedback.strip()):
        raise ValueError("调整本集方案时请填写意见，最多12000字。")
    with core.run_lock(root), core.transaction(root) as state:
        _idle(state)
        episode = _target(state, episode_id)
        pending = episode.get("planning", {})
        if (type(revision) is not int or pending.get("revision") != revision or not isinstance(signature, str)
                or pending.get("signature") != signature or pending.get("status") not in {"candidate", "failed"}):
            raise core.GateError("本集方案版本已改变，请刷新后重新审阅。")
        if pending.get("status") != "candidate" and decision != "reject":
            raise core.GateError("没有可批准的本集候选，请先提交调整意见。")
        if decision == "reject":
            pending.update(status="rejected", feedback=feedback.strip(), decided_at=core.now())
            episode["status"] = "awaiting_episode_plan"
            state.update(status="ready", last_error=None)
        else:
            script = _source(root, episode)
            if _baseline(root, state, episode) != pending.get("baseline"):
                raise core.GateError("本集来源、既有资产或前集审批已改变，请重新准备分集方案。")
            result = {"episode": pending.get("episode"), "new_assets": pending.get("new_assets")}
            if core._json_digest(result) != pending.get("result_sha256"):
                raise core.GateError("分集候选内容校验失败，不能采用。")
            validate_proposal(state, episode, result, script)
            used = {ref for shot in result["episode"]["shots"] for ref in shot["asset_ids"]}
            for asset in state["assets"]:
                if asset["id"] in used and core.asset_ready(asset):
                    core._check_artifact(root, asset)
            for definition in result["new_assets"]:
                asset = {"character_id": "", **copy.deepcopy(definition), "status": "planned", "version": 0,
                         "history": [], "feedback": "", "creation_input": empty_input(),
                         "source_episode_id": episode_id, "source_file": episode["source_file"],
                         "source_sha256": episode["source_sha256"]}
                if core.is_internal_voice_source(asset):
                    core._register_voice_source(asset)
                state["assets"].append(asset)
            episode.update(copy.deepcopy(result["episode"]))
            episode.update(status="planned", version=0, history=[], feedback="", reviews=[],
                           duration=sum(shot["duration"] for shot in episode["shots"]))
            for shot in episode["shots"]:
                shot.update(status="planned", version=0, attempts=[], revision=0, feedback="")
            pending.update(status="approved", decided_at=core.now(), human_decision="approve")
            state.update(status="ready", last_error=None)
        core.event(state, "human_episode_plan_decided", episode_id=episode_id, revision=revision, decision=decision)
    return core.load_project(root)
