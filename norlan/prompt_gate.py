"""Human consent for the exact next video request, bound to immutable inputs."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

from . import core


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _timing_text(value, label, maximum):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise core.GateError(f"时间安排的{label}须为非空文本，最多{maximum}字。")
    return value.strip()


def _dialogue_by_speaker(dialogue):
    """Read explicit speaker labels; never infer a new voice from prose."""
    if not isinstance(dialogue, str):
        raise core.GateError("原定对白须为文本，不能在时间安排中改写。")
    speakers = {}
    for line in dialogue.splitlines():
        if not line.strip():
            continue
        matched = re.fullmatch(r"\s*([^：:\n]{1,60})[：:]\s*(.+?)\s*", line)
        if not matched:
            raise core.GateError("时间安排需要原定对白逐行标明说话人，不能猜测声音归属。")
        speaker, text = matched.group(1).strip(), matched.group(2).strip()
        speakers[speaker] = speakers.get(speaker, "") + text
    return speakers


def replace_broadcast_dialogue(dialogue, speaker, spoken_text):
    """Replace only the bound system speaker; preserve other lines verbatim."""
    lines, inserted = [], False
    for line in dialogue.splitlines():
        matched = re.fullmatch(r"\s*([^：:\n]{1,60})[：:]\s*(.+?)\s*", line)
        if matched and matched.group(1).strip() == speaker:
            if not inserted:
                lines.append(f"{speaker}：{spoken_text}")
                inserted = True
        else:
            lines.append(line)
    return "\n".join(lines)


def validate_broadcast_proposal(shot, refined, context=None):
    """Check source/visual retention and protect all non-system dialogue.

    A trusted project context additionally verifies the designated voice. A
    model's semantic summary remains a draft for final human prompt approval;
    these structural checks cannot prove semantic faithfulness.
    """
    proposal = refined.get("broadcast_proposal") if isinstance(refined, dict) else None
    if proposal is None:
        return None
    fields = {"speaker", "source_text", "spoken_text", "display_text", "summary"}
    if not isinstance(proposal, dict) or set(proposal) != fields:
        raise core.GateError("系统简播方案须包含说话人、来源全文、播报内容、画面全文和取舍说明。")
    for key in fields:
        _timing_text(proposal[key], key, 60 if key == "speaker" else 12000)
    speaker = proposal["speaker"]
    original = _dialogue_by_speaker(shot.get("dialogue", ""))
    if original.get(speaker) != proposal["source_text"]:
        raise core.GateError("系统简播来源必须等于原计划中该说话人的完整原文，不能编造或漏掉规则。")
    if proposal["display_text"] != proposal["source_text"]:
        raise core.GateError("系统详细规则须在画面上完整保留，不能随播报一并删去。")
    if "\n" in proposal["spoken_text"] or "\r" in proposal["spoken_text"]:
        raise core.GateError("系统简播内容须为单一说话人的文本，不能夹带其他对白行。")
    source_numbers = set(re.findall(r"\d+(?:\.\d+)?", proposal["source_text"]))
    spoken_numbers = set(re.findall(r"\d+(?:\.\d+)?", proposal["spoken_text"]))
    if spoken_numbers - source_numbers:
        raise core.GateError("系统简播不得增加或改动原文数值。")
    expected = replace_broadcast_dialogue(shot.get("dialogue", ""), speaker, proposal["spoken_text"])
    if refined.get("dialogue") != expected:
        raise core.GateError("只能调整指定系统声源的播报内容，其他角色对白及顺序须逐字保留。")
    if shot.get("subshots") and refined.get("timing_proposal") is None:
        raise core.GateError("系统简播须同时提供一致的摄影与声音安排，不能保留冲突的旧完整播报时窗。")
    if context is not None:
        from .broadcast import policy_for
        policy = policy_for(context)
        assets = context.get("assets", context.get("plan", {}).get("assets", []))
        by_id = {asset.get("id"): asset for asset in assets if isinstance(asset, dict)}
        voice = by_id.get(policy.get("voice_id"), {})
        character = by_id.get(policy.get("character_id"), {})
        if (policy.get("enabled") is not True or policy.get("editorial_mode") != "key_points"
                or speaker not in policy.get("speaker_labels", [])
                or policy.get("voice_id") not in shot.get("asset_ids", [])
                or voice.get("kind") != "voice" or character.get("kind") != "character"
                or voice.get("character_id") != character.get("id")):
            raise core.GateError("简播仅适用于项目指定且已引用的系统声源，不能改写人物台词。")
    return copy.deepcopy(proposal)


def validate_timing_proposal(shot, refined):
    """Validate a draft delivery schedule without changing the approved plan.

    Camera cuts and total duration remain canonical. Only subshot text and the
    continuous, explicitly attributed audio schedule are proposed for the one
    final-prompt confirmation. Character dialogue stays byte-for-byte fixed;
    a separately validated system editorial draft can select key information.
    """
    if not isinstance(refined, dict):
        raise core.GateError("最终提示词候选须为对象。")
    broadcast = validate_broadcast_proposal(shot, refined)
    proposal = refined.get("timing_proposal")
    if proposal is None:
        return None
    if set(refined) - {"prompt", "dialogue", "continuity_notes", "timing_proposal", "broadcast_proposal"}:
        raise core.GateError("时间安排不能添加时长、资产引用或其他计划字段。")
    if not isinstance(proposal, dict) or set(proposal) != {"subshots", "summary", "audio_windows"}:
        raise core.GateError("时间安排仅能包含内部镜头文字、调整说明和声音时窗。")
    original = shot.get("subshots")
    proposed = proposal["subshots"]
    if not isinstance(original, list) or not original or not isinstance(proposed, list) or len(proposed) != len(original):
        raise core.GateError("时间安排不能增加或删去原定内部镜头。")
    duration = shot.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration):
        raise core.GateError("原定片段时长无效，不能准备时间安排。")
    from .segments import validate_subshots
    effective = copy.deepcopy(shot)
    effective["subshots"] = copy.deepcopy(proposed)
    try:
        validate_subshots(effective)
    except ValueError as exc:
        raise core.GateError(str(exc)) from None
    if any((before.get("start"), before.get("end")) != (after["start"], after["end"])
           for before, after in zip(original, proposed)):
        raise core.GateError("时间安排不能改变摄影镜头的起止时间或切点。")
    _timing_text(proposal["summary"], "调整说明", 4000)
    if broadcast is None and refined.get("dialogue") != shot.get("dialogue", ""):
        raise core.GateError("时间安排不能改写、增加或省略原定对白。")
    expected = _dialogue_by_speaker(refined["dialogue"] if broadcast else shot.get("dialogue", ""))
    windows = proposal["audio_windows"]
    if not isinstance(windows, list) or len(windows) > 30:
        raise core.GateError("声音时窗须为不超过30项的列表。")
    spoken, previous_end = {}, 0
    for window in windows:
        if not isinstance(window, dict) or set(window) != {"start", "end", "speaker", "text"}:
            raise core.GateError("每个声音时窗只能包含 start、end、speaker 和 text。")
        start, end = window["start"], window["end"]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (start, end)) or not 0 <= start < end <= duration:
            raise core.GateError("声音时窗必须完整位于原片段时长内，可以跨内部摄影切点。")
        speaker = _timing_text(window["speaker"], "说话人", 60)
        text = _timing_text(window["text"], "对白", 12000)
        if speaker not in expected:
            raise core.GateError("声音时窗出现了原定对白之外的说话人。")
        if start < previous_end:
            raise core.GateError("声音时窗须按时间排序，不能重叠或用播报覆盖其他角色对白。")
        previous_end = end
        spoken[speaker] = spoken.get(speaker, "") + text
    if spoken != expected:
        raise core.GateError("声音时窗须逐字覆盖全部原定对白，不增加、重复或漏掉任何角色台词。")
    return copy.deepcopy(proposal)


def effective_shot(shot, refined):
    """Return draft timing for this candidate only; leave canonical shot intact."""
    proposal = validate_timing_proposal(shot, refined)
    effective = copy.deepcopy(shot)
    if proposal is not None:
        effective["subshots"] = copy.deepcopy(proposal["subshots"])
    if refined.get("broadcast_proposal") is not None:
        effective["dialogue"] = refined["dialogue"]
    return effective


def candidate_timeline_prompt(shot, refined):
    """Render one consistent camera/audio schedule, including cross-cut speech."""
    proposal = validate_timing_proposal(shot, refined)
    broadcast = validate_broadcast_proposal(shot, refined)
    from .segments import _seconds, timeline_prompt
    if proposal is None and broadcast is None:
        return timeline_prompt(shot)
    if proposal is None:
        return ("本次候选的系统播报与画面文字：\n"
                f"剧情系统界面完整显示（可分页，保留阅读时间）：{broadcast['display_text']}\n"
                f"仅以下内容由{broadcast['speaker']}画外播报：{broadcast['spoken_text']}\n"
                "GPT按自然语速安排，不覆盖人物对白；此安排尚待最终提示词审批。")
    lines = [f"本次候选的摄影与声音时间表（总时长{_seconds(shot['duration'])}秒，一次连续生成）：",
             "以下秒点是生成目标，不保证精确到秒；摄影镜头数量、顺序和切点保持原计划，以下文字替代旧内部镜头的 UI 与声音安排。"]
    for index, subshot in enumerate(proposal["subshots"], 1):
        lines.append(f"内部镜头{index}，{_seconds(subshot['start'])}–{_seconds(subshot['end'])}秒：{subshot['prompt'].strip()}")
    lines.append("独立连续声音时窗（从本片段0秒起算；音频可以跨摄影切点连续说完，切镜不重说）：")
    for window in proposal["audio_windows"]:
        lines.append(f"{_seconds(window['start'])}–{_seconds(window['end'])}秒，{window['speaker'].strip()}：{window['text'].strip()}")
    if not proposal["audio_windows"]:
        lines.append("本段无原定对白，不添加新播报。")
    if broadcast:
        lines.extend([f"剧情任务框/系统界面完整显示以下详细规则（可分页，留足阅读时间）：{broadcast['display_text']}",
                      f"画面全文不全部朗读，系统声仅播上述选定关键内容：{broadcast['spoken_text']}",
                      "人物原对白逐字保留；上述系统简播替代旧完整朗读要求，不同时朗读详细规则。"])
    lines.extend(["本次实际对白按上述连续声音时窗自然说完；不加速、不截断、不跨生成片段重复。" if broadcast else
                  "各说话人的原定对白逐字保留，按上述连续声音时窗自然说完；不加速、不截断、不跨生成片段重复。",
                  "摄影切点不终止正在进行的对白；不重复动作，人物、空间轴线、道具和环境底声连续，使用本次真实已批准的图像与声音参考。",
                  "不生成背景配乐，不添加额外画面字幕；剧情中的 UI/任务框文字与播音按本次候选执行。"])
    return "\n".join(lines)


def specification(state, shot, prompt, manifest, refined=None):
    broadcast = validate_broadcast_proposal(shot, refined or {}, state)
    bindings = []
    for identifier in shot["asset_ids"]:
        asset = core._find(state, "asset", identifier)
        bindings.append({key: copy.deepcopy(asset.get(key)) for key in
                         ("id", "version", "status", "sha256", "audio_sha256", "voice_only_digest")})
    ark = state["settings"]["ark"]
    specification = {"shot_id": shot["id"], "current_version": shot.get("version", 0),
            "revision": shot.get("revision", 0), "source_prompt": shot["prompt"],
            "source_dialogue": shot.get("dialogue"), "feedback": shot.get("feedback", ""),
            "duration": shot["duration"], "scene_id": shot["scene_id"],
            "subshots": copy.deepcopy(shot.get("subshots")), "prompt": prompt,
            "references": copy.deepcopy(manifest), "assets": bindings,
            "audio_policy": copy.deepcopy(state.get("audio_policy", {})),
            "provider": {"mode": state["mode"], "ratio": state["plan"].get("ratio", "16:9"),
                         "video_watermark": ark.get("video_watermark", False),
                         **{key: ark.get(key) for key in ("provider", "base_url", "video_model", "resolution", "generate_audio")}}}
    from .storyboard_revision import public
    try:
        redesign = public(shot)
    except ValueError as exc:
        raise core.GateError(str(exc)) from None
    if redesign is not None:
        specification["storyboard_revision"] = redesign
    from .future_continuity import public_updates
    updates = public_updates(state, shot, strict=True)
    if updates:
        specification["future_continuity_updates"] = updates
        specification["future_continuity_binding"] = copy.deepcopy(shot["future_continuity_binding"])
    proposal = validate_timing_proposal(shot, refined or {})
    if proposal is not None:
        specification["timing_proposal"] = proposal
        specification["timing_refinement"] = {key: copy.deepcopy(refined.get(key))
                                              for key in ("prompt", "dialogue", "continuity_notes")}
    if broadcast is not None:
        specification["broadcast_proposal"] = broadcast
        specification["broadcast_refinement"] = {key: copy.deepcopy(refined.get(key))
                                                 for key in ("prompt", "dialogue", "continuity_notes")}
    return specification


def _validate_candidate_timing(shot, candidate):
    from .storyboard_revision import public
    try:
        redesign = public(shot)
    except ValueError as exc:
        raise core.GateError(str(exc)) from None
    if candidate.get("storyboard_revision") != redesign:
        raise core.GateError("局部分镜说明已变化，请重新准备并确认完整提示词。")
    refined = candidate.get("refined", {})
    broadcast = validate_broadcast_proposal(shot, refined)
    if broadcast != candidate.get("broadcast_proposal"):
        raise core.GateError("系统简播或画面全文已变化，请重新准备并确认完整提示词。")
    if broadcast is not None and (candidate.get("dialogue") != refined.get("dialogue")
            or candidate.get("continuity_notes") != refined.get("continuity_notes")):
        raise core.GateError("系统简播的对白或说明已变化，请重新准备并确认完整提示词。")
    proposal = validate_timing_proposal(shot, refined)
    if proposal is None and broadcast is None:
        if any(key in candidate for key in ("timing_proposal", "timing_summary", "effective_subshots")):
            raise core.GateError("提示词的时间安排元数据已变化，请重新准备并确认。")
        return
    if proposal is not None and (candidate.get("timing_proposal") != proposal
            or candidate.get("timing_summary") != proposal["summary"]
            or candidate.get("effective_subshots") != proposal["subshots"]
            or candidate.get("dialogue") != refined.get("dialogue")
            or candidate.get("continuity_notes") != refined.get("continuity_notes")):
        raise core.GateError("提示词的时间安排说明已变化，请重新准备并确认。")
    timeline = candidate_timeline_prompt(shot, refined)
    prompt = candidate.get("prompt")
    if not isinstance(prompt, str) or prompt.count(timeline) != 1:
        raise core.GateError("最终提示词须包含唯一完整的候选摄影与声音时间表。")
    from .segments import timeline_prompt
    old_timeline = timeline_prompt(shot)
    if old_timeline and old_timeline in prompt:
        raise core.GateError("最终提示词仍包含旧声音时间限制，请重新准备一致的候选。")


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
    _validate_candidate_timing(shot, candidate)
    from .future_continuity import public_updates
    if candidate.get("future_continuity_updates", []) != public_updates(state, shot, strict=True):
        raise core.GateError("后续接续调整说明已改变，请重新准备并确认完整提示词。")
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
    actual = _hash(specification(state, shot, candidate["prompt"], manifest, candidate.get("refined")))
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
        from .storyboard_revision import public
        try:
            redesign = public(shot)
        except ValueError as exc:
            raise core.GateError(str(exc)) from None
        if redesign is not None:
            candidate["storyboard_revision"] = redesign
        from .future_continuity import public_updates
        updates = public_updates(state, shot, strict=True)
        if updates:
            candidate["future_continuity_updates"] = updates
        proposal = validate_timing_proposal(shot, refined)
        broadcast = validate_broadcast_proposal(shot, refined, state)
        if broadcast is not None:
            candidate["broadcast_proposal"] = broadcast
        if proposal is not None:
            candidate.update(timing_proposal=copy.deepcopy(proposal), timing_summary=proposal["summary"],
                             effective_subshots=copy.deepcopy(proposal["subshots"]))
        _validate_candidate_timing(shot, candidate)
        candidate["request_hash"] = _hash(specification(state, shot, prompt, manifest, refined))
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
        from .storyboard_revision import discard
        discard(shot, reason=feedback.strip(), stamp=core.now(), state=state)
        shot["feedback"] = (shot.get("feedback", "") + "\n提示词修改意见：" + feedback.strip()).strip()
        shot.update(status="planned", revision=shot.get("revision", 0) + 1)
        analysis = shot.pop("revision_analysis", None)
        if analysis:
            shot.setdefault("revision_analysis_history", []).append({**copy.deepcopy(analysis),
                "superseded_at": core.now(), "resolution": "human_rejected_prompt"})
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
