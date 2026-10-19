"""Human-reviewed grouping of a pristine tail into longer generation segments.

No model or provider calls happen here. Existing calls, paid attempts, media,
assets and accounting records are never rewritten. A saved proposal is inert
until a human explicitly approves its still-current source snapshot.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from . import core
from .segments import validate_subshots


PUBLIC_ERRORS = frozenset({
    "请先暂停制作并等待在途任务结束，再提出剩余生成片段的分组方案。",
    "请先批准当前策划，再提出剩余生成片段的分组方案。",
    "尚有在途或结果不明的调用，请先恢复原任务或核实结果，再调整剩余生成片段。",
    "本集已有最终成片或已进入成片审批，不能通过剩余片段分组修改。",
    "分组须为完整未启动尾部的连续 ID 分组，每组1–3个原生成片段。",
    "分组必须按原顺序完整覆盖尚未启动的尾部，不能遗漏、重复或移动已制作内容。",
    "方案没有合并任何生成片段，保留原计划即可。",
    "无法为合并后的生成片段分配唯一 ID，请重新提出方案。",
    "只能合并同一场景的连续生成片段；跨场景应保留独立任务。",
    "合并后的生成片段总时长须为10–15秒；不能截断、加速或延长原计划凑时长。",
    "生成片段引用的资产不存在，请先核对原策划。",
    "合并后的实际参考超过8张资产图片或3个声音；还须为前镜承接画面保留一个图片位置。",
    "生成片段引用声音时必须同时引用其绑定角色，不能以提示词替代参考图。",
    "原生成片段的内部镜头时间表无效，请先核对策划。",
    "原生成片段的对白字段无效，请先核对原规划，不能静默丢弃对白。",
    "原生成片段的返工意见字段无效，请先核对原规划，不能静默丢弃修改要求。",
    "合并后须保留1–3个有效内部镜头，且连续覆盖总时长；过多镜头或过长提示不能静默删减。",
    "新分组未通过完整策划的资产引用、声音绑定或时间表校验，请检查原方案。",
    "本集已有待审批的分组方案，请先在网页批准或保留原方案。",
    "规划存档路径不能是符号链接。",
    "该方案的旧规划存档已存在且内容不同，不能覆盖；请核对后重新提出方案。",
    "分组方案已变化或已处理，请刷新网页重新审阅。",
    "原片段、资产或策划已变化，此方案已过期；请保留原方案并重新提出分组。",
    "待审批方案内容已变化，不能应用；请重新提出并审阅方案。",
})


_CLOCK = r"(?:\d{1,2}:)?\d{2}:\d{2}(?:\.\d{1,3})?"
_LEADING_RANGE = re.compile(r"^\s*(?:\[|【)?\s*" + _CLOCK + r"\s*[-–—~～至到]\s*" + _CLOCK
                            + r"\s*(?:\]|】)?\s*[。.,，:：;；]?\s*")


def _clean_prompt(prompt):
    """Remove only an explicit leading clock range, retaining original prose."""
    return _LEADING_RANGE.sub("", prompt, count=1).strip()


def _guard(state, episode_id, *, require_paused=False):
    if require_paused and state.get("paused") is not True:
        raise core.GateError("请先暂停制作并等待在途任务结束，再提出剩余生成片段的分组方案。")
    if state["plan"].get("status") != "approved":
        raise core.GateError("请先批准当前策划，再提出剩余生成片段的分组方案。")
    if any(job.get("accounting") in {"reserved", "unknown"} or job.get("state") in
           {"submitting", "submitted", "download_pending", "uncertain"} for job in state["jobs"].values()):
        raise core.GateError("尚有在途或结果不明的调用，请先恢复原任务或核实结果，再调整剩余生成片段。")
    episode = core._find(state, "episode", episode_id)
    if episode.get("file") or episode.get("status") in {"awaiting_review", "approved"}:
        raise core.GateError("本集已有最终成片或已进入成片审批，不能通过剩余片段分组修改。")
    return episode


def _pristine(state, shot):
    version = shot.get("version", 0)
    revision = shot.get("revision", 0)
    return (shot.get("status") == "planned" and type(version) is int and version == 0
            and type(revision) is int and revision >= 0
            and "refined_prompt" not in shot and "evidence" not in shot
            and not any(shot.get(key) for key in ("file", "history", "attempts", "job_id"))
            and not any(job.get("item_id") == shot["id"] for job in state["jobs"].values()))


def _tail(state, episode):
    start = len(episode["shots"])
    while start > 0 and _pristine(state, episode["shots"][start - 1]):
        start -= 1
    return start, episode["shots"][start:]


def _base_hash(state, episode):
    value = {"plan": state["plan"], "assets": state["assets"], "shots": episode["shots"],
             "episode": {key: episode.get(key) for key in ("id", "status", "version", "file", "sha256")}}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _groups(groups, tail):
    ids = [shot["id"] for shot in tail]
    if (not ids or not isinstance(groups, list) or not groups or len(groups) > len(ids)
            or any(not isinstance(group, list) or not 1 <= len(group) <= 3
                   or any(not isinstance(value, str) for value in group) for group in groups)):
        raise core.GateError("分组须为完整未启动尾部的连续 ID 分组，每组1–3个原生成片段。")
    if [value for group in groups for value in group] != ids:
        raise core.GateError("分组必须按原顺序完整覆盖尚未启动的尾部，不能遗漏、重复或移动已制作内容。")
    if not any(len(group) > 1 for group in groups):
        raise core.GateError("方案没有合并任何生成片段，保留原计划即可。")


def _unique_lines(values):
    seen, result = set(), []
    for value in values:
        for line in str(value or "").splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                result.append(line)
    return "\n".join(result)


def _dialogue(shot):
    value = shot.get("dialogue")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise core.GateError("原生成片段的对白字段无效，请先核对原规划，不能静默丢弃对白。")
    return value.strip()


def _feedback(shot):
    value = shot.get("feedback")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise core.GateError("原生成片段的返工意见字段无效，请先核对原规划，不能静默丢弃修改要求。")
    return value


def _new_id(first_id, proposal_id, index, occupied):
    for attempt in range(100):
        suffix = hashlib.sha256(f"{proposal_id}:{index}:{attempt}".encode()).hexdigest()[:16]
        value = first_id[:43] + "-seg-" + suffix
        if value not in occupied:
            occupied.add(value)
            return value
    raise core.GateError("无法为合并后的生成片段分配唯一 ID，请重新提出方案。")


def _build(state, tail, groups, proposal_id):
    _groups(groups, tail)
    by_id = {shot["id"]: shot for shot in tail}
    assets = {asset["id"]: asset for asset in state["assets"]}
    occupied = set(assets) | {episode["id"] for episode in state["episodes"]}
    occupied.update(shot["id"] for episode in state["episodes"] for shot in episode["shots"])
    result = []
    for index, group in enumerate(groups):
        originals = [by_id[value] for value in group]
        if len(originals) == 1:
            result.append(copy.deepcopy(originals[0]))
            continue
        if len({shot["scene_id"] for shot in originals}) != 1:
            raise core.GateError("只能合并同一场景的连续生成片段；跨场景应保留独立任务。")
        duration = sum(shot["duration"] for shot in originals)
        if not 10 <= duration <= 15:
            raise core.GateError("合并后的生成片段总时长须为10–15秒；不能截断、加速或延长原计划凑时长。")
        references = list(dict.fromkeys(ref for shot in originals for ref in shot["asset_ids"]))
        if any(ref not in assets for ref in references):
            raise core.GateError("生成片段引用的资产不存在，请先核对原策划。")
        voices = [assets[ref] for ref in references if assets[ref]["kind"] == "voice"]
        images = [ref for ref in references if assets[ref]["kind"] != "voice"]
        if len(images) > 8 or len(voices) > 3:
            raise core.GateError("合并后的实际参考超过8张资产图片或3个声音；还须为前镜承接画面保留一个图片位置。")
        if any(voice.get("character_id") not in references for voice in voices):
            raise core.GateError("生成片段引用声音时必须同时引用其绑定角色，不能以提示词替代参考图。")
        subshots, offset = [], 0
        for shot in originals:
            try:
                validate_subshots(shot)
            except ValueError:
                raise core.GateError("原生成片段的内部镜头时间表无效，请先核对策划。") from None
            if "subshots" in shot:
                source = shot["subshots"]
            else:
                prompt = _clean_prompt(shot["prompt"])
                dialogue = _dialogue(shot)
                if dialogue:
                    prompt += "\n本内部镜头须完整说出以下原定对白（仅说一次，不移到其他镜头）：\n" + dialogue
                source = [{"start": 0, "end": shot["duration"], "prompt": prompt}]
            feedback = _feedback(shot)
            for sub_index, sub in enumerate(source):
                prompt = _clean_prompt(sub["prompt"])
                if sub_index == 0 and feedback:
                    prompt += (f"\n原生成片段 {shot['id']} 的返工意见（仅适用于本生成片段 {offset}–{offset + shot['duration']} 秒的来源段；"
                               "覆盖该来源段的内部镜头，保留各句对白的归属，不作为每镜新增对白）：\n" + feedback)
                subshots.append({"start": offset + sub["start"], "end": offset + sub["end"], "prompt": prompt})
            offset += shot["duration"]
        candidate = {"id": _new_id(originals[0]["id"], proposal_id, index, occupied),
                     "duration": duration, "prompt": "\n".join(_clean_prompt(shot["prompt"]) for shot in originals),
                     "dialogue": "\n".join(text for shot in originals if (text := _dialogue(shot))),
                     "asset_ids": references, "scene_id": originals[0]["scene_id"],
                     "continuity_notes": _unique_lines(shot.get("continuity_notes") for shot in originals),
                     "source_shot_ids": list(group), "subshots": subshots,
                     "source_shot_instructions": [{"id": shot["id"], "revision": shot.get("revision", 0),
                                                    "feedback": _feedback(shot)} for shot in originals],
                     "status": "planned", "version": 0, "revision": 0, "attempts": [], "history": [],
                     "feedback": "\n".join(f"{shot['id']}：{text}" for shot in originals if (text := _feedback(shot)))}
        try:
            validate_subshots(candidate)
        except ValueError:
            raise core.GateError("合并后须保留1–3个有效内部镜头，且连续覆盖总时长；过多镜头或过长提示不能静默删减。") from None
        result.append(candidate)
    return result


def _validate_result(state, episode, start, replacement):
    episodes = copy.deepcopy(state["episodes"])
    next(item for item in episodes if item["id"] == episode["id"])["shots"] = episode["shots"][:start] + replacement
    try:
        core.validate_plan({"title": state["title"], "ratio": state["plan"].get("ratio", "16:9"),
                            "assets": state["assets"], "episodes": episodes})
    except core.PlanValidationError:
        raise core.GateError("新分组未通过完整策划的资产引用、声音绑定或时间表校验，请检查原方案。") from None


def propose_segments(root, episode_id, groups):
    """Save an inert proposal; the dashboard alone may request human approval."""
    with core.run_lock(root), core.transaction(root) as state:
        episode = _guard(state, episode_id, require_paused=True)
        previous = episode.get("segment_proposal")
        if previous and previous.get("status") == "candidate":
            raise core.GateError("本集已有待审批的分组方案，请先在网页批准或保留原方案。")
        start, tail = _tail(state, episode)
        proposal_id = "segments-" + uuid.uuid4().hex
        shots = _build(state, tail, groups, proposal_id)
        _validate_result(state, episode, start, shots)
        if previous:
            episode.setdefault("segment_proposal_history", []).append(copy.deepcopy(previous))
        episode["segment_proposal"] = {
            "id": proposal_id, "status": "candidate", "source_ids": [shot["id"] for shot in tail],
            "shots": shots, "base_hash": _base_hash(state, episode), "created_at": core.now(),
            "groups": copy.deepcopy(groups), "base_plan_version": state["plan"]["version"],
            "note": "仅重组尚未启动的连续尾部；同场景只是必要条件，剧情与动作是否适合合并仍需审阅。减少任务次数不代表费用必然降低。",
        }
        core.event(state, "segment_proposal_created", episode_id=episode_id, proposal_id=proposal_id,
                   source_ids=episode["segment_proposal"]["source_ids"])
    return core.load_project(root)


def _snapshot(root, state, episode, proposal):
    relative = "plans/" + proposal["id"] + "-before.json"
    path = core.safe_path(root, relative)
    if (Path(root) / relative).is_symlink() or (Path(root) / "plans").is_symlink():
        raise core.GateError("规划存档路径不能是符号链接。")
    value = {"proposal_id": proposal["id"], "base_hash": proposal["base_hash"],
             "plan": state["plan"], "assets": state["assets"], "episodes": state["episodes"]}
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise core.GateError("该方案的旧规划存档已存在且内容不同，不能覆盖；请核对后重新提出方案。")
    else:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return {"file": relative, "sha256": core.digest(path)}


def decide_segments(root, episode_id, proposal_id, approve: bool):
    """Apply only the exact still-current proposal that the human reviewed."""
    if type(approve) is not bool or not isinstance(proposal_id, str) or not re.fullmatch(r"segments-[0-9a-f]{32}", proposal_id):
        raise ValueError("分组方案审批必须提供明确方案 ID 和布尔决定。")
    with core.run_lock(root), core.transaction(root) as state:
        episode = _guard(state, episode_id)
        proposal = episode.get("segment_proposal", {})
        if proposal.get("id") != proposal_id or proposal.get("status") != "candidate":
            raise core.GateError("分组方案已变化或已处理，请刷新网页重新审阅。")
        if not approve:
            proposal.update(status="rejected", decided_at=core.now())
            core.event(state, "human_segment_proposal_rejected", episode_id=episode_id, proposal_id=proposal_id)
        else:
            start, tail = _tail(state, episode)
            if (proposal.get("base_hash") != _base_hash(state, episode)
                    or proposal.get("source_ids") != [shot["id"] for shot in tail]):
                raise core.GateError("原片段、资产或策划已变化，此方案已过期；请保留原方案并重新提出分组。")
            expected = _build(state, tail, proposal.get("groups"), proposal_id)
            if proposal.get("shots") != expected:
                raise core.GateError("待审批方案内容已变化，不能应用；请重新提出并审阅方案。")
            _validate_result(state, episode, start, expected)
            archive = _snapshot(root, state, episode, proposal)
            episode["shots"] = episode["shots"][:start] + copy.deepcopy(expected)
            episode.setdefault("segment_plan_history", []).append({"proposal_id": proposal_id,
                "source_ids": proposal["source_ids"], "applied_at": core.now(), "snapshot": archive})
            for key in ("progress_stage", "progress_started_at"):
                episode.pop(key, None)
            proposal.update(status="approved", decided_at=core.now())
            core.event(state, "human_segment_proposal_approved", episode_id=episode_id, proposal_id=proposal_id,
                       source_ids=proposal["source_ids"], snapshot=archive)
    return core.load_project(root)
