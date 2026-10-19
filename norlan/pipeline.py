"""Resumable production, with human decisions for every generated segment."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import uuid
from pathlib import Path

from . import core, media
from .billing import price_snapshot
from .voice import (AuditionSelectionError, audition_duration, audition_profile,
                    audition_selection_key, build_audition_prompt, validate_audition)


def demo_plan():
    """A conspicuously synthetic two-episode workflow fixture, not AI footage."""
    assets = [
        {"id": "hero", "kind": "character", "name": "演示人物卡", "prompt": "DEMO placeholder, blue character identity card"},
        {"id": "station", "kind": "location", "name": "演示场景卡", "prompt": "DEMO placeholder, railway station environment"},
    ]
    return {"title": "Nolan · 两集流程演示", "style": "流程占位素材，不代表 AI 漫剧质量", "ratio": "16:9",
        "assets": assets,
        "episodes": [{"id": f"E{i:03}", "title": f"第 {i} 集：流程演示", "summary": "用于验证资产审批、逐集放行和局部返工。",
          "music_prompt": "无配乐；本演示只验证制作流程。",
          "shots": [{"id": f"E{i:03}-S{j:03}", "duration": 4,
                     "prompt": f"DEMO Episode {i} Shot {j} / synthetic test clip",
                     "dialogue": "", "asset_ids": ["hero", "station"],
                     "scene_id": "station", "continuity_notes": "保持同一场景和顺序；演示不模拟人物生成。"}
                    for j in (1, 2)]} for i in (1, 2)]}


def _set(root, **values):
    with core.transaction(root) as s:
        s.update(values)


def _gate(root):
    state = core.load_project(root)
    if state["paused"]:
        raise core.GateError("已暂停；在途任务可能仍由供应商继续执行")
    unknown = [j["id"] for j in state["jobs"].values() if j["accounting"] == "unknown"]
    if unknown:
        raise core.GateError("存在结果不明的调用，请先对账，避免重复计费：" + ", ".join(unknown))
    return state


def _pending_segment_plan(state):
    return any(episode.get("segment_proposal", {}).get("status") == "candidate"
               for episode in state.get("episodes", []))


def _new_work_gate(state):
    if _pending_segment_plan(state):
        raise core.GateError("剩余生成段方案等待人工批准；批准前不会提交新的付费调用。")


def _brain(root, method, *args, **kwargs):
    from .brain import Brain
    from .connection import production_brain_config
    s = _gate(root)
    _new_work_gate(s)
    if s["mode"] == "demo":
        raise core.GateError("演示项目不调用在线模型；请新建 live 项目进行真实制作")
    config = production_brain_config(s["settings"]["brain"])
    amount = s["settings"]["caps"]["llm"]
    if (config.get("provider") == "openai" or config.get("billing") == "api") and (
            amount <= 0 or not s["budget"]["approved"]):
        raise core.GateError("GPT API 制作调用需要你先设置项目预算及 caps.llm 每次调用费用上限；尚未提交请求。")
    job_id = "gpt-" + uuid.uuid4().hex[:12]
    core.reserve(root, job_id, "llm", method, amount)
    try:
        result = getattr(Brain(config), method)(*args, **kwargs)
        core.update_job(root, job_id, state="completed", accounting="spent")
        return result
    except Exception:
        # A response/schema error does not imply a free request.
        core.update_job(root, job_id, state="failed", accounting="spent")
        raise


def _ark(root):
    from .providers import ArkProvider
    s = core.load_project(root)
    config = {**s["settings"]["ark"], "ratio": s["plan"].get("ratio", "16:9")}
    return ArkProvider(config)


def _rel(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _relative_evidence(root, evidence):
    result = copy.deepcopy(evidence)
    for key in ("tail_frame", "audio"):
        if result.get(key):
            result[key] = _rel(root, result[key])
    result["frames"] = [_rel(root, f) for f in result.get("frames", [])]
    result["frame_hashes"] = {_rel(root, f): core.digest(f) for f in evidence.get("frames", [])}
    if evidence.get("audio"):
        result["audio_sha256"] = core.digest(evidence["audio"])
    return result


def _archive_stale_shot_file(shot):
    """Remove a prior version from the current slot, preserving its provenance."""
    file = shot.get("file")
    match = re.fullmatch(re.escape(shot["id"]) + r"-v(\d+)\.(?:mp4|m4v|mov|webm)", Path(file).name) if file else None
    if not match or int(match.group(1)) == shot.get("version"):
        return False
    if not any(record.get("file") == file for record in shot.get("history", [])):
        old = {key: copy.deepcopy(shot[key]) for key in ("file", "sha256", "evidence", "review") if key in shot}
        old["version"] = int(match.group(1))
        shot.setdefault("history", []).append(old)
    for key in ("file", "sha256", "evidence", "review", "approved_at", "approved_sha256", "approved_version"):
        shot.pop(key, None)
    return True


def _recover(root):
    # Before a POST starts its intent was committed. A killed process must not
    # interpret the absence of a response as absence of a provider-side job.
    with core.transaction(root) as s:
        for job in s["jobs"].values():
            if job["accounting"] == "reserved" and job["state"] == "submitting":
                job.update(accounting="unknown", state="uncertain")
                core.event(s, "interrupted_submission", id=job["id"])
        # Older releases cleared a failed candidate and queued an automatic
        # retry. Restore that exact candidate for a human decision instead. An
        # explicit rejection advances revision, so it is not undone here.
        for ep in s["episodes"]:
            for shot in ep["shots"]:
                if shot["status"] != "accepted" and _archive_stale_shot_file(shot):
                    active = s["jobs"].get(shot.get("job_id"), {})
                    if active.get("state") in {"submitted", "download_pending", "completed"}:
                        shot["status"] = "generating"
                    else:
                        shot.update(status="review_required", generation_error=
                            "本次视频生成未得到可用结果。不会自动再次提交；请填写意见后由你决定是否重试。")
                    core.event(s, "stale_candidate_archived", id=shot["id"], version=shot.get("version"))
                if shot["status"] != "planned" or shot.get("file") or shot.get("job_id"):
                    continue
                history = shot.get("history", [])
                attempts = shot.get("attempts", [])
                if not history or not attempts:
                    continue
                candidate = copy.deepcopy(history[-1])
                if "version" not in candidate:
                    match = re.fullmatch(re.escape(shot["id"]) + r"-v(\d+)\.mp4", Path(candidate.get("file", "")).name)
                    if match:
                        candidate["version"] = int(match.group(1))
                if (candidate.get("version") != shot.get("version")
                        or candidate.get("review", {}).get("verdict") != "fail"
                        or attempts[-1].get("revision", 0) != shot.get("revision", 0)):
                    continue
                core._check_artifact(root, candidate)
                shot.update({key: copy.deepcopy(candidate[key]) for key in
                             ("file", "sha256", "version", "evidence", "review") if key in candidate})
                shot.update(status="review_required", feedback="")
                core.event(s, "automatic_retry_cancelled", id=shot["id"], version=shot["version"])


def _save_asset(root, asset_id, **values):
    with core.transaction(root) as s:
        core._find(s, "asset", asset_id).update(values)


def _save_shot(root, shot_id, **values):
    with core.transaction(root) as s:
        core._find(s, "shot", shot_id).update(values)


def _save_episode(root, ep_id, **values):
    with core.transaction(root) as s:
        core._find(s, "episode", ep_id).update(values)


def _progress(root, ep_id, stage, shot_id=None):
    """Persist actual phase transitions; elapsed time never increases percent."""
    with core.transaction(root) as state:
        ep = core._find(state, "episode", ep_id)
        previous = ep.get("progress_stage", {})
        if previous.get("stage") == stage and previous.get("shot_id") == shot_id:
            return
        stamp = core.now()
        ep.setdefault("progress_started_at", stamp)
        ep["progress_stage"] = {"stage": stage, "shot_id": shot_id, "started_at": stamp, "updated_at": stamp}


def _job_progress(root, job, stage):
    if job.get("kind") != "video":
        return
    state = core.load_project(root)
    ep = next((e for e in state["episodes"] if any(sh["id"] == job["item_id"] for sh in e["shots"])), None)
    if ep:
        _progress(root, ep["id"], stage, job["item_id"])


def _poll_video(root, job_id, output):
    provider = _ark(root)
    started = time.monotonic()
    while True:
        s = core.load_project(root)
        job = s["jobs"][job_id]
        if not job.get("task_id"):
            raise core.GateError("视频任务没有供应商 ID，必须先对账")
        if job["state"] != "download_pending":
            _job_progress(root, job, "shot_generation")
        result = provider.poll_video(job["task_id"])
        if result["status"] == "completed":
            core.record_usage(root, job_id, result)
            # Retain ID before download. An expired/download-failed URL is never
            # permission to resubmit the paid generation task.
            core.update_job(root, job_id, state="download_pending", accounting="spent")
            _job_progress(root, job, "shot_download")
            if not result.get("url"):
                raise core.GateError("供应商任务已完成，但未返回可下载地址；用量已保存，继续时只查询原任务。")
            provider.download(result["url"], output)
            core.update_job(root, job_id, state="completed", output=_rel(root, output))
            return
        if result["status"] == "failed":
            core.record_usage(root, job_id, result)
            core.update_job(root, job_id, state="failed", accounting="spent")
            raise core.GateError("供应商报告任务失败；预算占用不代表实际扣费，重新生成须由你明确决定。")
        # A pause stops waiting/new submissions; the saved task is polled again
        # after resume. It may continue charging at the provider.
        _gate(root)
        if time.monotonic() - started >= s["settings"]["max_poll_seconds"]:
            raise core.GateError("供应商仍在生成，任务 ID 已保存；稍后继续将查询原任务")
        time.sleep(s["settings"]["poll_seconds"])


def _video(root, kind, item_id, output, duration, prompt, references, audio_references=None, old_job=None):
    s = _gate(root)
    if old_job:
        job = s["jobs"][old_job]
        if job.get("output") and core.safe_path(root, job["output"]) != Path(output).resolve():
            raise core.GateError("恢复任务输出路径与已登记版本不符，禁止复制旧任务为新版本")
        if job["state"] == "completed" and Path(output).is_file():
            return old_job
        if job.get("task_id") and job["state"] != "failed":
            _poll_video(root, old_job, output)
            return old_job
        if job["accounting"] == "unknown":
            raise core.GateError("未知任务必须先核实，不能再次提交")
    _new_work_gate(s)
    provider = _ark(root) if s["mode"] == "live" else None
    if provider:
        provider.preflight("video")
    job_id = f"{kind}-{item_id}-{uuid.uuid4().hex[:8]}"
    amount = 0 if s["mode"] == "demo" else duration * s["settings"]["caps"]["video_second"]
    pricing = price_snapshot({**s["settings"]["ark"], "currency": s["budget"]["currency"]}, kind)
    core.reserve(root, job_id, kind, item_id, amount, pricing=pricing if s["mode"] == "live" else None)
    with core.transaction(root) as current:
        obj = core._find(current, "asset" if kind == "voice" else "shot", item_id)
        obj["job_id"] = job_id
        obj.setdefault("attempts", []).append({"job_id": job_id, "revision": obj.get("revision", 0), "time": core.now()})
        if kind == "voice":
            current["jobs"][job_id]["audition_profile"] = copy.deepcopy(obj.get("audition_profile", {}))
            current["jobs"][job_id]["duration"] = duration
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if s["mode"] == "demo":
        media.synthetic_clip(output, prompt, duration, index=len(s["jobs"]))
        core.update_job(root, job_id, state="completed", accounting="spent", output=_rel(root, output))
        return job_id
    try:
        _job_progress(root, {"kind": kind, "item_id": item_id}, "shot_submission")
        result = provider.submit_video(prompt, duration, references,
                                       audio_references=audio_references or [])
    except Exception as exc:
        uncertain = getattr(exc, "uncertain", True)
        core.update_job(root, job_id, state="uncertain" if uncertain else "failed",
                        accounting="unknown" if uncertain else "spent")
        raise
    core.update_job(root, job_id, state="submitted", task_id=result["id"])
    _poll_video(root, job_id, output)
    return job_id


def _select_audition(root, asset, character):
    """Select a sourced excerpt once per voice/feedback/source change."""
    novel = (Path(root) / "source/novel.txt").read_text(encoding="utf-8")
    signature = audition_selection_key(asset, character["name"], core.digest(Path(root) / "source/novel.txt"))
    if asset.get("audition_selection_key") == signature and asset.get("audition_selection_error"):
        raise core.GateError(asset["audition_selection_error"])

    def invalid_selection(message):
        _save_asset(root, asset["id"], audition_selection_key=signature, audition_selection_error=message)
        raise core.GateError(message)

    needs_selection = not asset.get("audition_text") or (asset["status"] == "rejected" and asset.get("audition_selection_key") != signature)
    if needs_selection and asset.get("audition_selection_key") != signature:
        from .asset_inputs import compose_prompt
        selection = _brain(root, "select_voice_audition", novel, {**asset, "prompt": compose_prompt(asset)}, character)
        if not isinstance(selection, dict) or any(not isinstance(selection.get(key), str) or len(selection[key]) > 4000
                for key in ("audition_text", "audition_reason", "audition_source")):
            invalid_selection("声音选句返回无效；尚未提交试音，请在 Nolan 中检查原文选句。")
        candidate = {**asset, **{key: selection[key] for key in ("audition_text", "audition_reason", "audition_source")}}
        if candidate["audition_text"].strip():
            try:
                validate_audition(candidate, novel, character["name"])
            except AuditionSelectionError as exc:
                invalid_selection(str(exc))
        elif candidate["audition_source"].strip() or not candidate["audition_reason"].strip():
            invalid_selection("没有合适试音句时必须说明原因，并将台词和来源留空；尚未提交试音。")
        values = {key: candidate[key] for key in ("audition_text", "audition_reason", "audition_source")}
        values["audition_selection_key"] = signature
        values["audition_selection_error"] = ""
        _save_asset(root, asset["id"], **values)
        asset = {**asset, **values}
    try:
        validate_audition(asset, novel, character["name"])
    except AuditionSelectionError as exc:
        raise core.GateError(str(exc) + (" 选句说明：" + asset["audition_reason"] if asset.get("audition_reason") else "")) from None
    return asset


def _assets(root):
    s = _gate(root)
    from .asset_inputs import pending_assets, reference_paths, compose_prompt
    waiting_for_input = bool(pending_assets(s))
    for original in s["assets"]:
        s = _gate(root)
        asset = core._find(s, "asset", original["id"])
        if asset["status"] in {"approved", "candidate"}:
            continue
        active_job = s["jobs"].get(asset.get("job_id", ""))
        recovering = bool(asset["status"] != "rejected" and active_job and active_job["state"] in {"submitted", "download_pending", "completed"})
        # An upgraded project can already have a submitted task while later
        # assets still lack the new choice. Recover that task without any new
        # POST so its reservation cannot permanently block the user's choices.
        if waiting_for_input and not recovering:
            continue
        if asset["kind"] == "voice" and not recovering:
            # A voice audition must use approved visual identity references.
            if any(a["status"] != "approved" for a in s["assets"] if a["kind"] != "voice"):
                continue
        if not recovering:
            _new_work_gate(s)
            total_attempts = sum(j["item_id"] == asset["id"] and j["kind"] in {"voice", "image"} for j in s["jobs"].values())
            if total_attempts >= s["settings"]["max_attempts"]:
                raise core.GateError("资产已达到尝试上限；请调整方案或显式提高上限：" + asset["id"])
            references = reference_paths(root, asset)
            if asset["kind"] == "voice" and s["mode"] == "live":
                character = core._find(s, "asset", asset["character_id"])
                core._check_artifact(root, character)
                asset = _select_audition(root, asset, character)
            version = asset["version"] + 1
            with core.transaction(root) as current:
                a = core._find(current, "asset", asset["id"])
                if a.get("file"):
                    a["history"].append({k: a[k] for k in ("file", "sha256", "version")})
                a.update(version=version, status="generating")
                a.pop("job_id", None)
                asset = copy.deepcopy(a)
        version = asset["version"]
        suffix = ".mp4" if asset["kind"] == "voice" else ".png"
        out = Path(root) / "media" / "assets" / f"{asset['id']}-v{version}{suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        if recovering and active_job.get("output") and core.safe_path(root, active_job["output"]) != out.resolve():
            raise core.GateError("恢复素材任务的输出路径与候选版本不符，不会覆盖旧素材")
        prompt = compose_prompt(asset, s['plan'].get('style', ''))
        if asset["kind"] == "voice":
            character = next((a for a in s["assets"] if a["id"] == asset.get("character_id") and a["kind"] == "character"), None)
            if character is None or character["status"] != "approved":
                raise core.GateError("声音试音必须绑定一个已批准的角色，不能猜测说话人")
            core._check_artifact(root, character)
            images = [core.safe_path(root, character["file"])]
            voice_asset = {**asset, "prompt": compose_prompt(asset)}
            voice_prompt = asset.get("audition_prompt", "已提交声音试片，仅恢复原任务") if recovering else (
                "DEMO / SYNTHETIC PLACEHOLDER voice audition" if s["mode"] == "demo" else build_audition_prompt(voice_asset))
            duration = 5 if recovering or s["mode"] == "demo" else audition_duration(asset)
            if not recovering:
                _save_asset(root, asset["id"], audition_profile=audition_profile(voice_asset), audition_prompt=voice_prompt)
            _video(root, "voice", asset["id"], out, duration,
                voice_prompt, images,
                old_job=asset.get("job_id") if recovering else None)
            ev = _relative_evidence(root, media.extract_evidence(out, Path(root) / "evidence" / f"{asset['id']}-v{version}"))
            if not ev.get("audio"):
                raise core.GateError("试音没有声音轨，不能批准为声音参考")
            _save_asset(root, asset["id"], audio_file=ev["audio"], audio_sha256=ev["audio_sha256"], evidence=ev)
        elif not recovering:
            provider = _ark(root) if s["mode"] == "live" else None
            if provider:
                provider.preflight("image")
            job_id = f"image-{asset['id']}-{uuid.uuid4().hex[:8]}"
            amount = 0 if s["mode"] == "demo" else s["settings"]["caps"]["image"]
            pricing = price_snapshot({**s["settings"]["ark"], "currency": s["budget"]["currency"]}, "image")
            core.reserve(root, job_id, "image", asset["id"], amount, pricing=pricing if s["mode"] == "live" else None)
            with core.transaction(root) as current:
                a = core._find(current, "asset", asset["id"])
                a["job_id"] = job_id
                a.setdefault("attempts", []).append({"job_id": job_id, "time": core.now()})
                current["jobs"][job_id]["input_references"] = copy.deepcopy(a.get("creation_input", {}).get("references", []))
            try:
                if s["mode"] == "demo":
                    media.synthetic_asset(out, asset["id"] + " / DEMO", "#2563eb" if asset["kind"] == "character" else "#059669")
                else:
                    result = provider.generate_image(prompt, out, references=references,
                        on_usage=lambda usage: core.record_usage(root, job_id, usage))
                    core.record_usage(root, job_id, result)
                core.update_job(root, job_id, state="completed", accounting="spent", output=_rel(root, out))
            except Exception as exc:
                uncertain = getattr(exc, "uncertain", True)
                core.update_job(root, job_id, state="uncertain" if uncertain else "failed",
                                accounting="unknown" if uncertain else "spent")
                raise
        if not out.is_file():
            raise core.GateError("已登记成功的素材文件缺失，请从供应商取回；不会重复生成")
        _save_asset(root, asset["id"], status="candidate", file=_rel(root, out), sha256=core.digest(out))
    s = core.load_project(root)
    if pending_assets(s):
        _set(root, status="awaiting_asset_input")
        return False
    if any(a["status"] != "approved" for a in s["assets"]):
        _set(root, status="awaiting_assets")
        return False
    for asset in s["assets"]:
        core._check_artifact(root, asset)
    return True


def _shot_context(root, ep, shot, previous):
    s = _gate(root)
    assets = [a for a in s["assets"] if a["id"] in shot["asset_ids"]]
    images, voices, image_manifest, voice_manifest = [], [], [], []
    for asset in assets:
        if asset["status"] != "approved":
            raise core.GateError("镜头引用的资产尚未批准")
        core._check_artifact(root, asset)
        if asset["kind"] == "voice":
            if asset.get("character_id") not in shot["asset_ids"]:
                raise core.GateError("镜头声音参考未绑定到同镜引用的角色，不能猜测说话人")
            voices.append(core.safe_path(root, asset["audio_file"]))
            voice_manifest.append({"index": len(voices), "asset_id": asset["id"], "character_id": asset["character_id"], "sha256": asset["audio_sha256"]})
        else:
            images.append(core.safe_path(root, asset["file"]))
            image_manifest.append({"index": len(images), "role": "approved_asset", "asset_id": asset["id"], "sha256": asset["sha256"]})
    if shot.get("dialogue") and s["settings"]["require_voice_reference"] and not voices:
        raise core.GateError(f"{shot['id']} 有对白但未引用已批准的声音资产；请补充规划或显式关闭声音参考要求")
    continuity = None
    if previous:
        if previous["status"] != "accepted":
            raise core.GateError("前镜头未被接受，不能成为连续性参考")
        core._check_artifact(root, previous)
        report = previous.get("review", {})
        report = report if (report.get("sha256") == previous["sha256"] and
                           report.get("version", previous.get("version")) == previous.get("version")) else {}
        continuity = {"shot_id": previous["id"], "sha256": previous["sha256"],
            "observed_end_state": report.get("observed_end_state", "未核实"),
            "same_scene": previous["scene_id"] == shot["scene_id"]}
        if core.human_approved_shot(previous):
            continuity["human_decision"] = copy.deepcopy(previous.get("human_decision") or {
                "decision": "approved", "version": previous.get("version"), "sha256": previous["sha256"],
                "at": previous.get("approved_at"), "source": "legacy_human_approval"})
            continuity["accepted_deviations"] = copy.deepcopy(report.get("issues", []))
            continuity["acceptance_note"] = "用户已接受这个视频版本；报告中的疑点未必被用户逐项证实。以实际已接受画面承接，不擅自纠正其既有细节。"
        if continuity["same_scene"]:
            tail = previous["evidence"]["tail_frame"]
            path = core.safe_path(root, tail)
            if not path.is_file() or core.digest(path) != previous["evidence"].get("frame_hashes", {}).get(tail):
                raise core.GateError("前镜尾帧缺失或内容改变，不能沿用未核实的画面参考")
            images.append(path)
            image_manifest.append({"index": len(images), "role": "previous_actual_tail", "shot_id": previous["id"], "sha256": core.digest(path)})
    if len(images) > 9 or len(voices) > 3:
        raise core.GateError("参考素材超过模型限制，不能静默省略人物或声音参考")
    if voices:
        voices, reference_details = media.prepare_voice_references(voices, Path(root) / "media" / "voice-references")
        for reference, details in zip(voice_manifest, reference_details, strict=True):
            if reference["sha256"] != details["source_sha256"]:
                raise core.GateError("声音原件在准备参考时发生变化，不能提交未批准的音频")
            reference.update(details)
    context = {"episode": {k: ep.get(k) for k in ("id", "title", "summary", "feedback")},
        "style": s["plan"].get("style"), "assets": [{k: a.get(k) for k in
            ("id", "kind", "name", "prompt", "character_id", "version", "sha256")} for a in assets],
        "previous_actual": continuity, "feedback": shot.get("feedback", ""),
        "image_references": image_manifest, "audio_references": voice_manifest,
        "audio_strategy": "使用批准的声音参考，镜头不生成背景配乐；配乐在整集时间线统一添加。",
        "limitations": "尾帧是实际画面参考；不能保证身份、口型和音色绝对一致。"}
    return context, images, voices


def _shot(root, ep_id, shot_id, previous):
    s = _gate(root)
    ep = core._find(s, "episode", ep_id)
    shot = core._find(s, "shot", shot_id)
    if "subshots" in shot:
        from .segments import validate_subshots
        validate_subshots(shot)
    if shot["status"] == "accepted":
        core._check_artifact(root, shot)
        if core.human_approved_shot(shot):
            return True
        # A legacy GPT pass is advice, not a human decision.
        _save_shot(root, shot_id, status="review_required")
        _set(root, status="review_required")
        return False
    if shot["status"] == "review_required":
        _set(root, status="review_required")
        return False
    active = s["jobs"].get(shot.get("job_id", ""))
    recovering = bool(active and active["state"] in {"submitted", "download_pending", "completed"})
    attempted = bool(active and active.get("state") == "failed") or any(
        attempt.get("revision", 0) == shot.get("revision", 0) for attempt in shot.get("attempts", []))
    if not recovering and attempted:
        _save_shot(root, shot_id, status="review_required", generation_error=
                   "本次视频生成未得到可用结果。不会自动再次提交；请填写意见后由你决定是否重试。")
        _set(root, status="review_required")
        return False
    if not recovering:
        _new_work_gate(s)
        _progress(root, ep_id, "shot_preparation", shot_id)
    context, images, voices = _shot_context(root, ep, shot, previous)
    if not recovering:
        if s["mode"] == "demo":
            refined = {"prompt": shot["prompt"] + " " + shot.get("feedback", ""), "dialogue": "", "continuity_notes": "Synthetic demo"}
        else:
            refined = _brain(root, "refine_shot", shot, context, images)
        version = shot.get("version", 0) + 1
        _save_shot(root, shot_id, refined_prompt=refined, version=version, status="generating",
            reference_manifest={"images": [_rel(root, f) for f in images],
                "audio": [_rel(root, f) for f in voices], "context": context})
        shot = core._find(core.load_project(root), "shot", shot_id)
    version = shot["version"]
    out = Path(root) / "media" / ep_id / f"{shot_id}-v{version}.mp4"
    if shot.get("file") and shot["file"] != _rel(root, out):
        # A legacy interrupted retry may still carry v2's file under version 3.
        # Keep the old bytes/history, but never allow approving them as v3.
        with core.transaction(root) as current:
            stored = core._find(current, "shot", shot_id)
            if not _archive_stale_shot_file(stored):
                raise core.GateError("当前视频文件与生成任务路径不一致，请先核实原任务；不会重新提交。")
    prompt = shot["refined_prompt"]["prompt"]
    if "subshots" in shot:
        from .segments import timeline_prompt
        prompt += "\n" + timeline_prompt(shot)
    # Refined text must not be treated as approved story changes.
    if shot.get("dialogue"):
        prompt += "\n必须使用的原定对白：" + shot["dialogue"]
    prompt += "\n不添加画面字幕，不添加背景配乐。"
    try:
        _video(root, "video", shot_id, out, shot["duration"], prompt, images, voices,
               old_job=shot.get("job_id") if recovering else None)
    except Exception:
        current = core.load_project(root)
        current_shot = core._find(current, "shot", shot_id)
        failed = current["jobs"].get(current_shot.get("job_id", ""), {})
        if failed.get("state") != "failed" or failed.get("accounting") == "unknown":
            raise
        _save_shot(root, shot_id, status="review_required", generation_error=
                   "本次视频生成未得到可用结果。不会自动再次提交；请填写意见后由你决定是否重试。")
        _set(root, status="review_required")
        return False
    _progress(root, ep_id, "shot_checks", shot_id)
    evidence_dir = Path(root) / "evidence" / f"{shot_id}-v{version}"
    evidence = (media.extract_evidence(out, evidence_dir, subshots=shot["subshots"])
                if "subshots" in shot else media.extract_evidence(out, evidence_dir))
    relative = _relative_evidence(root, evidence)
    _save_shot(root, shot_id, file=_rel(root, out), sha256=core.digest(out), evidence=relative, status="reviewing")
    # Review the seam too: prior tail + current sampled frames, tied to hashes.
    review_images = images + [Path(f) for f in evidence["frames"]]
    review_context = {**context, "shot": shot, "actual": relative,
        "current_frame_image_indices": list(range(len(images) + 1, len(review_images) + 1)),
        "current_frame_samples": [{"image_index": len(images) + index + 1, "time_seconds": instant}
                                  for index, instant in enumerate(evidence.get("sample_times", []))],
        "review_scope": "当前镜头首/中/尾抽样画面与前镜实际尾帧。没有审听音频；不可宣称整段动作或音色已核实。"}
    if "subshots" in shot:
        review_context.update(
            review_scope="按计划子镜中点及切点前后采样，包含实测首帧、最后一帧与前段实际尾帧参考。"
                         "只是计划点抽样，未检测或确认实际切镜；未审听音频，不可宣称完整动作、口型或音色已核实。",
            repair_unit="一个生成段对应一次视频任务；仅当用户选择返工时才重新生成整段，不能仅替换段内某一子镜。")
    _progress(root, ep_id, "shot_review", shot_id)
    try:
        review = ({"verdict": "pass", "issues": [], "observed_end_state": "Synthetic demo only",
                   "repair_prompt": "", "scope": "synthetic smoke test"} if s["mode"] == "demo"
                  else _brain(root, "review", review_context, review_images))
    except Exception:
        # A failed report must not hide a paid-for candidate or cause another
        # video/report request on every resume. The user can review the video.
        if core.load_project(root)["paused"]:
            raise
        review = {"verdict": "uncertain", "issues": ["审查报告未完成，请自行观看当前视频后决定通过或返工。"],
                  "observed_end_state": "未核实", "repair_prompt": "", "scope": "sampled_frames_only",
                  "report_unavailable": True}
    review.update(sha256=core.digest(out), version=version, advisory=True,
                  audio_checked=False, whole_video_checked=False)
    _save_shot(root, shot_id, review=review, status="review_required")
    _set(root, status="review_required")
    return False

def _soundtrack(root, ep):
    """Music is one episode cue; frame-only reviewers never certify its sound."""
    s = _gate(root)
    music_cfg = s["settings"]["music"]
    mode = music_cfg.get("mode", "none")
    if mode == "none":
        return None
    if mode == "file":
        source = Path(music_cfg.get("file", "")).expanduser().resolve()
        if not source.is_file():
            raise core.GateError("本地配乐文件不存在")
        import shutil
        target = Path(root) / "media" / "music" / (core.digest(source) + source.suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        return target
    if mode != "volc":
        raise core.GateError("配乐模式必须为 none / file / volc")
    if s["mode"] == "demo":
        raise core.GateError("演示项目不能提交在线配乐任务，请新建 live 项目")
    from .music import MusicProvider
    provider = MusicProvider(music_cfg)
    provider._credentials()
    if ep.get("music_file"):
        return core.safe_path(root, ep["music_file"])
    duration = sum(media.probe(core.safe_path(root, sh["file"]))["duration"] for sh in ep["shots"])
    if duration > 120:
        raise core.GateError("本集超过纯音乐接口的 120 秒上限；请提供足够时长的本地配乐，或拆分分集")
    duration = max(30, int(duration + 0.999))
    out = Path(root) / "media" / "music" / f"{ep['id']}-{uuid.uuid4().hex[:8]}.mp3"
    out.parent.mkdir(parents=True, exist_ok=True)
    job = s["jobs"].get(ep.get("music_job_id", ""))
    if not job:
        _new_work_gate(s)
        job_id = "music-" + ep["id"] + "-" + uuid.uuid4().hex[:8]
        cap = music_cfg.get("call_cap", 0)
        core.reserve(root, job_id, "music", ep["id"], cap)
        _save_episode(root, ep["id"], music_job_id=job_id, pending_music_file=_rel(root, out))
        try:
            result = provider.submit(ep.get("music_prompt", "适合剧情的无歌词背景音乐"), duration)
        except Exception as exc:
            uncertain = getattr(exc, "uncertain", True)
            core.update_job(root, job_id, state="uncertain" if uncertain else "failed", accounting="unknown" if uncertain else "spent")
            raise
        job = core.update_job(root, job_id, state="submitted", task_id=result["id"])
    else:
        out = core.safe_path(root, ep["pending_music_file"])
        if job["state"] == "failed":
            raise core.GateError("配乐任务失败，请检查供应商账单后配置新的配乐方案")
    started = time.monotonic()
    while True:
        result = provider.poll(job["task_id"])
        if result["status"] == "completed":
            core.update_job(root, job["id"], state="download_pending", accounting="spent")
            provider.download(result["url"], out)
            core.update_job(root, job["id"], state="completed", output=_rel(root, out))
            _save_episode(root, ep["id"], music_file=_rel(root, out))
            return out
        if result["status"] == "failed":
            core.update_job(root, job["id"], state="failed", accounting="spent")
            raise core.GateError("配乐生成失败")
        _gate(root)
        if time.monotonic() - started > s["settings"]["max_poll_seconds"]:
            raise core.GateError("配乐任务仍在生成，稍后可恢复原任务")
        time.sleep(s["settings"]["poll_seconds"])


def _captions(root, ep):
    """Transcribe the actual locked cut, once per content hash, before music."""
    s = _gate(root)
    config = s["settings"].get("subtitles", {"mode": "none"})
    clip_hashes = [sh["sha256"] for sh in ep["shots"]]
    if ep.get("caption_feedback") and ep.get("caption_source_clips") != clip_hashes:
        # A text correction refers to the old audio timeline. A later picture
        # revision invalidates it even if the feedback is still queued.
        _save_episode(root, ep["id"], caption_feedback="", caption_source=[],
                      caption_source_clips=[], verified_subtitles=None, subtitle_clip_hashes=[])
        ep = {**ep, "caption_feedback": "", "verified_subtitles": None}
    if ep.get("caption_feedback"):
        segments = _brain(root, "correct_subtitles", ep["caption_source"], ep["caption_feedback"])
        _save_episode(root, ep["id"], verified_subtitles=segments, subtitle_clip_hashes=clip_hashes,
                      caption_feedback="", caption_source=[], caption_source_clips=[])
        return segments, "human_requested_text_edit_needs_review"
    if ep.get("verified_subtitles") is not None:
        if ep.get("subtitle_clip_hashes") == clip_hashes:
            return ep["verified_subtitles"], "human_requested_text_edit_needs_review"
        _save_episode(root, ep["id"], verified_subtitles=None, subtitle_clip_hashes=[])
    if config.get("mode", "none") == "none":
        return None, "disabled"
    if config.get("mode") != "openai":
        raise core.GateError("字幕模式只能是 none / openai")
    if s["mode"] == "demo":
        raise core.GateError("演示项目不能提交在线转录任务，请新建 live 项目")
    from .transcription import Transcriber
    transcriber = Transcriber(config)
    # Validate credentials before reserving a paid call.
    transcriber._key()
    fingerprint = hashlib.sha256(json.dumps({"clips": [sh["sha256"] for sh in ep["shots"]],
        "model": config.get("model", "whisper-1"), "language": config.get("language")}, sort_keys=True).encode()).hexdigest()
    output = Path(root) / "evidence" / f"{ep['id']}-transcript-{fingerprint}.json"
    existing = [j for j in s["jobs"].values() if j["kind"] == "transcription" and
                j.get("fingerprint") == fingerprint and j["item_id"] == ep["id"]]
    if output.is_file():
        # JSON is durable before marking the call spent, so a killed process can
        # recover the response without issuing another billable transcription.
        result = json.loads(output.read_text(encoding="utf-8"))
        if existing and existing[-1]["state"] == "completed" and core.digest(output) == existing[-1].get("result_sha256"):
            return result["segments"], "automatic_asr_needs_review"
        raise core.GateError("字幕缓存未完成登记或已改变，请核实原调用；不会重复转录")
    if existing and existing[-1]["state"] in {"submitting", "uncertain", "completed"}:
        raise core.GateError("原字幕调用结果需要核实，不能自动重复提交")
    if len(existing) >= s["settings"]["max_attempts"]:
        raise core.GateError("本版字幕转录已达到尝试上限")
    # Use the same normalization and timeline as the final export; do not use
    # planned shot durations as speech timestamps.
    dry_cut = Path(root) / "exports" / f"{ep['id']}-dialogue-{fingerprint[:16]}.mp4"
    if not dry_cut.is_file():
        media.compose([core.safe_path(root, sh["file"]) for sh in ep["shots"]], dry_cut)
    evidence = media.extract_evidence(dry_cut, Path(root) / "evidence" / f"asr-{fingerprint[:16]}")
    if not evidence.get("audio"):
        raise core.GateError("成片没有实际音轨，不能生成对白字幕")
    job_id = f"asr-{ep['id']}-{uuid.uuid4().hex[:8]}"
    _new_work_gate(s)
    core.reserve(root, job_id, "transcription", ep["id"], config.get("call_cap", 0))
    core.update_job(root, job_id, fingerprint=fingerprint, result_file=_rel(root, output))
    try:
        segments = transcriber.transcribe(Path(evidence["audio"]))
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps({"segments": segments, "audio_sha256": core.digest(evidence["audio"])},
                                   ensure_ascii=False), encoding="utf-8")
        temp.replace(output)
        core.update_job(root, job_id, state="completed", accounting="spent", result_sha256=core.digest(output),
                        request_id=transcriber.last_request_id)
        return segments, "automatic_asr_needs_review"
    except Exception as exc:
        uncertain = getattr(exc, "uncertain", True)
        core.update_job(root, job_id, state="uncertain" if uncertain else "failed", accounting="unknown" if uncertain else "spent")
        raise


def _rough_cut(root, episode, count):
    """Publish a local cumulative cut before any batch model review.

    Source hashes preserve the old batch-review identity for existing projects.
    Registered outputs are immutable and verified before reuse; an unregistered
    file left by an interruption is never mistaken for a completed render.
    """
    shots = episode["shots"][:count]
    if not shots or len(shots) != count or any(shot["status"] != "accepted" for shot in shots):
        raise core.GateError("累计粗剪只能使用已经验收的连续镜头。")
    for shot in shots:
        core._check_artifact(root, shot)
    key = hashlib.sha256("-".join(shot["sha256"] for shot in shots).encode()).hexdigest()
    existing = next((cut for cut in episode.get("rough_cuts", []) if cut.get("key") == key), None)
    if existing:
        core._check_artifact(root, existing)
        return copy.deepcopy(existing)
    legacy = next((review for review in episode.get("reviews", [])
                   if review.get("key") == key and review.get("file") and review.get("sha256")), None)
    if legacy:
        core._check_artifact(root, legacy)
        path = core.safe_path(root, legacy["file"])
    else:
        _progress(root, episode["id"], "batch_assembly", shots[-1]["id"])
        path = Path(root) / "exports" / f"{episode['id']}-rough-{count}-{key[:16]}.mp4"
        if path.exists():
            path = path.with_name(f"{path.stem}-{uuid.uuid4().hex[:8]}{path.suffix}")
        media.compose([core.safe_path(root, shot["file"]) for shot in shots], path)
    record = {"key": key, "file": _rel(root, path), "sha256": core.digest(path),
              "shot_count": count,
              "shots": [{"id": shot["id"], "version": shot.get("version", 0), "sha256": shot["sha256"]}
                        for shot in shots], "created_at": core.now()}
    if count >= 2:
        with core.transaction(root) as state:
            cuts = core._find(state, "episode", episode["id"]).setdefault("rough_cuts", [])
            if not any(cut.get("key") == key for cut in cuts):
                cuts.append(record)
    return record


def _episode(root, ep_id):
    s = _gate(root)
    ep = core._find(s, "episode", ep_id)
    if ep["status"] == "awaiting_review" and all(core.human_approved_shot(shot) for shot in ep["shots"]):
        _set(root, status="awaiting_episode_review")
        return
    _save_episode(root, ep_id, status="generating")
    index_of_episode = s["episodes"].index(ep)
    prior = s["episodes"][index_of_episode - 1] if index_of_episode else None
    if prior and prior["status"] != "approved":
        raise core.GateError("前集尚未人工批准，不能生成下一集")
    if prior:
        core._check_artifact(root, prior)
    previous = prior["shots"][-1] if prior else None
    for index, original in enumerate(ep["shots"]):
        if not _shot(root, ep_id, original["id"], previous):
            _set(root, status="review_required")
            return
        fresh = core._find(core.load_project(root), "episode", ep_id)
        previous = fresh["shots"][index]
        count = index + 1
        rough_cut = _rough_cut(root, fresh, count) if count >= 2 else None
        if count % s["settings"]["batch_size"] == 0 or count == len(ep["shots"]):
            batch_key = hashlib.sha256("-".join(sh["sha256"] for sh in fresh["shots"][:count]).encode()).hexdigest()
            if any(r.get("key") == batch_key for r in fresh["reviews"]):
                continue
            rough_cut = rough_cut or _rough_cut(root, fresh, count)
            path = core.safe_path(root, rough_cut["file"])
            ev = media.extract_evidence(path, Path(root) / "evidence" / f"rough-{ep_id}-{batch_key[:16]}")
            _progress(root, ep_id, "batch_review", original["id"])
            try:
                review = ({"verdict": "pass", "issues": [], "observed_end_state": "Synthetic demo",
                           "repair_prompt": ""} if s["mode"] == "demo" else
                          _brain(root, "review", {"episode": ep_id, "shots": fresh["shots"][:count],
                              "scope": "分段粗剪抽帧检查；听觉未评估，运动过程未完整评估。",
                              "evidence": _relative_evidence(root, ev)}, [Path(f) for f in ev["frames"]]))
            except Exception:
                if core.load_project(root)["paused"]:
                    raise
                review = {"verdict": "uncertain", "issues": ["拼接审查报告未完成，请自行观看累计粗剪检查衔接。"],
                          "observed_end_state": "未核实", "repair_prompt": "", "report_unavailable": True}
            review.update(key=batch_key, file=_rel(root, path), sha256=core.digest(path),
                          advisory=True, audio_checked=False, whole_video_checked=False)
            with core.transaction(root) as current:
                core._find(current, "episode", ep_id)["reviews"].append(review)
            # The rough-cut report is advisory too. It never overwrites a
            # segment's report or revokes that segment's human approval.
    fresh = core._find(core.load_project(root), "episode", ep_id)
    _gate(root)
    _progress(root, ep_id, "music")
    music = _soundtrack(root, fresh)
    _progress(root, ep_id, "subtitles")
    captions, subtitle_status = _captions(root, fresh)
    version = fresh["version"] + 1
    out = Path(root) / "exports" / f"{ep_id}-v{version}.mp4"
    _progress(root, ep_id, "rendering")
    result = media.compose([core.safe_path(root, sh["file"]) for sh in fresh["shots"]], out,
                           subtitles=captions, music=music)
    _save_episode(root, ep_id, version=version, file=_rel(root, out), sha256=core.digest(out),
        status="awaiting_review", duration=result["duration"],
        subtitle_status=subtitle_status,
        subtitle_file=_rel(root, result["subtitles"]) if result.get("subtitles") else None,
        subtitles_burned=result.get("burned_subtitles", False), render_warnings=result.get("warnings", []),
        music_status="mixed" if music else "absent",
        review_note="请观看完整成片并审听对白、口型、音色和音乐。自动检查仅覆盖技术参数与抽样画面。",
        render_manifest={"clips": [{"id": sh["id"], "sha256": sh["sha256"]} for sh in fresh["shots"]],
                         "music": _rel(root, music) if music else None,
                         "subtitles": captions or []})
    _set(root, status="awaiting_episode_review")


def run(root):
    """Advance to the next human/cost gate. Never auto-approve any gate."""
    root = Path(root).expanduser().resolve()
    acquired = False
    try:
        with core.run_lock(root):
            acquired = True
            _recover(root)
            s = _gate(root)
            _set(root, last_error=None)
            if _pending_segment_plan(s) and not any(
                    job.get("state") in {"submitted", "download_pending"} for job in s["jobs"].values()):
                _set(root, status="awaiting_segment_plan")
                return core.load_project(root)
            if s["plan"]["status"] in {"pending", "rejected"}:
                novel = (root / "source/novel.txt").read_text(encoding="utf-8")
                if s["settings"].get("creative_brief"):
                    novel += "\n\n用户创作要求：" + s["settings"]["creative_brief"]
                if s["plan"].get("feedback"):
                    novel += "\n\n用户对上版规划的意见：" + s["plan"]["feedback"]
                plan = demo_plan() if s["mode"] == "demo" else _brain(root, "plan", novel)
                core.install_plan(root, plan)
                return core.load_project(root)
            if s["plan"]["status"] != "approved":
                _set(root, status="awaiting_plan")
                return core.load_project(root)
            if not _assets(root):
                return core.load_project(root)
            s = _gate(root)
            for ep in s["episodes"]:
                if ep["status"] != "approved":
                    _episode(root, ep["id"])
                    return core.load_project(root)
            _set(root, status="complete")
    except core.GateError as exc:
        if not acquired:
            return core.load_project(root)
        current = core.load_project(root)
        status = "paused" if current["paused"] else "awaiting_segment_plan" if _pending_segment_plan(current) else "blocked"
        _set(root, status=status, last_error=str(exc))
    except Exception as exc:
        # Adapters sanitize provider errors; avoid leaking credentials in reprs.
        message = str(exc) if getattr(exc, "safe_message", False) or isinstance(exc, (ValueError, FileNotFoundError)) else type(exc).__name__ + "：执行失败，请检查配置和本地文件"
        _set(root, status="blocked", last_error=message)
    return core.load_project(root)
