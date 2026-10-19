"""Resumable production, with human decisions for every generated segment."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
import uuid
from pathlib import Path

from . import core, media, official, prompt_gate
from .brain import task_asset_summary, task_context_summary, task_shot_summary
from .billing import price_snapshot
from .failures import exception_failure, sanitize_failure
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
    unknown = [j["id"] for j in core.blocking_unknown_jobs(state)]
    if unknown:
        raise core.GateError("存在结果不明的调用，请先对账，避免重复计费：" + ", ".join(unknown))
    return state


def _pending_segment_plan(state):
    return any(episode.get("segment_proposal", {}).get("status") == "candidate"
               for episode in state.get("episodes", []))


def _new_work_gate(state):
    if any(item.get("status") == "candidate" for item in state.get("asset_amendments", [])):
        raise core.GateError("新增资产方案等待确认；尚未提交新的生成任务。")
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
    target = (args[0].get("advisory_target") if method == "review" and args and isinstance(args[0], dict) else None)
    submitted = False

    def before_submit():
        nonlocal submitted
        if submitted:
            raise core.GateError("同一次 GPT 工作不能重复提交；请先核实原调用。")
        core.reserve(root, job_id, "llm", method, amount, advisory_target=target)
        submitted = True

    client = Brain(config)
    client.before_submit = before_submit
    try:
        result = getattr(client, method)(*args, **kwargs)
        if not submitted:
            # For example, review() returns an explicitly uncertain local
            # report when there are no frames. It consumes no model request.
            return result
        cached = {"review_result": copy.deepcopy(result), "review_result_hash": core._json_digest(result)} if method == "review" else {}
        core.update_job(root, job_id, state="completed", accounting="spent", **cached)
        return result
    except Exception as exc:
        if submitted:
            # Local validation made no request. Once submission begins, an
            # uncertain outcome must block a repeat instead of being spent.
            uncertain = getattr(exc, "uncertain", False) is True
            failure = exception_failure(exc, "submission")
            failure["stage"] = "submission"
            core.update_job(root, job_id, state="uncertain" if uncertain else "failed",
                            accounting="unknown" if uncertain else "spent", failure=failure)
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


def _unavailable_review(message="GPT 审查中断，报告不可用。请自行观看当前视频，决定通过或返工。"):
    return {"verdict": "uncertain", "issues": [message], "observed_end_state": "未核实", "repair_prompt": "",
            "scope": "sampled_frames_only", "report_unavailable": True, "review_interrupted": True,
            "advisory": True, "audio_checked": False, "whole_video_checked": False}


def _cached_review(state, target):
    """A persisted answer is reusable only for the exact completed artifact."""
    from .brain import BrainError, REVIEW_SCHEMA, _validate
    for job in reversed(list(state["jobs"].values())):
        if (job.get("kind") != "llm" or job.get("item_id") != "review" or job.get("state") != "completed"
                or job.get("advisory_target") != target or not isinstance(job.get("review_result"), dict)):
            continue
        result = job["review_result"]
        try:
            if core._json_digest(result) != job.get("review_result_hash"):
                continue
            fields = set(REVIEW_SCHEMA["properties"])
            if set(result) - fields - {"scope", "audio_assessed"}:
                continue
            _validate({key: result[key] for key in fields if key in result}, REVIEW_SCHEMA)
            if result.get("scope", "sampled_frames_only") != "sampled_frames_only" or result.get("audio_assessed", False) is not False:
                continue
        except (BrainError, ValueError, TypeError):
            continue
        return {**copy.deepcopy(result), "recovered_from_cache": True, "advisory": True,
                "audio_checked": False, "whole_video_checked": False}
    return None


def _shot_review_target(shot):
    return {"kind": "shot", "id": shot["id"], "version": shot["version"], "sha256": shot["sha256"],
            "file": shot["file"], "video_job_id": shot["job_id"]}


def _cut_review_target(episode_id, cut):
    return {"kind": "rough_cut", "id": episode_id, "key": cut["key"],
            "sha256": cut["sha256"], "file": cut["file"]}


def _recover_advisory_reports(root, state):
    """Restore human review without repeating an interrupted GPT or media call."""
    recovered = {"shots": [], "rough_cuts": []}
    for job in state["jobs"].values():
        if (job.get("kind") == "llm" and job.get("item_id") == "review" and not job.get("task_id")
                and job.get("state") == "uncertain" and job.get("accounting") == "unknown"
                and job.get("purpose") in {None, "advisory_review"}
                and not core.is_nonblocking_advisory_unknown(job)):
            # This acknowledges only the review method. Legacy jobs did not
            # record a target, so never invent one from nearby timestamps.
            job["advisory_recovery"] = {"status": "interrupted", "no_retry": True,
                "detected_at": core.now(), "result": "unavailable", "accounting_unresolved": True}
            core.event(state, "advisory_review_interrupted", id=job["id"], accounting=job["accounting"])
    for episode in state["episodes"]:
        for shot in episode["shots"]:
            if shot.get("status") != "reviewing":
                continue
            job = state["jobs"].get(shot.get("job_id"), {})
            if (job.get("kind") != "video" or job.get("item_id") != shot["id"]
                    or job.get("state") != "completed" or not job.get("output")):
                raise core.GateError("审查中断后的原视频任务尚未核实，不能直接转入人工审批；请先恢复原任务。")
            core._check_shot_artifact(root, state, shot)
            review = _cached_review(state, _shot_review_target(shot)) or _unavailable_review()
            review.update(version=shot["version"], sha256=shot["sha256"])
            shot.update(review=review, status="review_required")
            recovered["shots"].append(shot["id"])
            core.event(state, "advisory_review_restored", id=shot["id"], version=shot["version"],
                       from_cache=bool(review.get("recovered_from_cache")))
        hint = episode.get("progress_stage", {})
        for cut in episode.get("rough_cuts", []):
            if any(review.get("key") == cut.get("key") and review.get("sha256") == cut.get("sha256")
                   for review in episode.get("reviews", [])):
                continue
            count = cut.get("shot_count", len(cut.get("shots", [])))
            if type(count) is not int or not 0 < count <= len(episode["shots"]):
                continue
            prefix = episode["shots"][:count]
            manifest = [{"id": item["id"], "version": item.get("version", 0), "sha256": item.get("sha256")} for item in prefix]
            if not all(core.human_approved_shot(item) for item in prefix) or cut.get("shots") != manifest:
                continue
            target = _cut_review_target(episode["id"], cut)
            relevant_job = any(job.get("kind") == "llm" and job.get("item_id") == "review"
                               and job.get("advisory_target") == target for job in state["jobs"].values())
            legacy_hint = hint.get("stage") == "batch_review" and hint.get("shot_id") == prefix[-1]["id"]
            if not relevant_job and not legacy_hint:
                continue
            for item in prefix:
                core._check_shot_artifact(root, state, item)
            core._check_artifact(root, cut)
            review = _cached_review(state, target) or _unavailable_review("GPT 拼接审查中断，报告不可用。请自行播放累计拼接检查衔接。")
            review.update(key=cut["key"], file=cut["file"], sha256=cut["sha256"])
            episode.setdefault("reviews", []).append(review)
            recovered["rough_cuts"].append(cut["key"])
            core.event(state, "advisory_cut_review_restored", id=episode["id"], key=cut["key"],
                       from_cache=bool(review.get("recovered_from_cache")))
    return recovered


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
                    job = s["jobs"].get(shot.get("job_id", ""), {})
                    if (shot.get("status") == "review_required" and not shot.get("file")
                            and job.get("state") == "failed" and job.get("kind") == "video"
                            and job.get("item_id") == shot["id"]):
                        shot["status"] = "generation_failed"
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
        recovered = _recover_advisory_reports(root, s)
        legacy_single = []
        for episode in s["episodes"]:
            hint = episode.get("progress_stage", {})
            if (len(episode["shots"]) == 1 and hint.get("stage") == "batch_review"
                    and hint.get("shot_id") == episode["shots"][0]["id"]
                    and core.human_approved_shot(episode["shots"][0])):
                shot = episode["shots"][0]
                manifest = [{"id": shot["id"], "version": shot["version"], "sha256": shot["sha256"]}]
                if not any(cut.get("shots") == manifest for cut in episode.get("rough_cuts", [])):
                    legacy_single.append(episode["id"])
    # Older releases did not register one-clip cuts. Rebuild that local cut
    # from the approved original outside the state transaction; never reissue
    # its missing GPT report or pretend that the old call had a target binding.
    for episode_id in legacy_single:
        episode = core._find(core.load_project(root), "episode", episode_id)
        cut = _rough_cut(root, episode, 1)
        with core.transaction(root) as state:
            episode = core._find(state, "episode", episode_id)
            if not any(review.get("key") == cut["key"] for review in episode.get("reviews", [])):
                review = _cached_review(state, _cut_review_target(episode_id, cut)) or _unavailable_review(
                    "GPT 拼接审查中断，报告不可用。请自行播放本集检查衔接。")
                review.update(key=cut["key"], file=cut["file"], sha256=cut["sha256"])
                episode.setdefault("reviews", []).append(review)
                recovered["rough_cuts"].append(cut["key"])
                core.event(state, "legacy_single_cut_review_restored", id=episode_id, key=cut["key"])
    return recovered


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


def _record_video_failure(root, job_id, failure, **updates):
    """Keep bounded, redacted diagnostics on the exact call, including recovery."""
    detail = sanitize_failure(failure)
    with core.transaction(root) as state:
        job = state["jobs"][job_id]
        if job.get("failure") != detail:
            job.setdefault("failure_history", []).append({**detail, "recorded_at": core.now()})
        job.update(updates, failure=detail, updated_at=core.now())
        core.event(state, "video_call_failed", id=job_id, stage=detail.get("stage"),
                   http_status=detail.get("http_status"), provider_code=detail.get("provider_code"))


def _shot_generation_failed(root, shot_id):
    state = core.load_project(root)
    shot = core._find(state, "shot", shot_id)
    job = state["jobs"].get(shot.get("job_id", ""), {})
    if job.get("kind") != "video" or job.get("item_id") != shot_id:
        job = {}
    failure = sanitize_failure(job.get("failure")) if job.get("failure") else {}
    message = failure.get("message") or "旧版本未保存具体失败原因；本次没有可审视频，需要你决定是否重试。"
    _save_shot(root, shot_id, status="generation_failed", generation_error=message)
    _set(root, status="generation_failed")
    return False


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
        try:
            result = provider.poll_video(job["task_id"])
        except Exception as exc:
            _record_video_failure(root, job_id, exception_failure(exc, "poll"))
            raise
        if result["status"] == "completed":
            core.record_usage(root, job_id, result)
            # Retain ID before download. An expired/download-failed URL is never
            # permission to resubmit the paid generation task.
            core.update_job(root, job_id, state="download_pending", accounting="spent")
            _job_progress(root, job, "shot_download")
            if not result.get("url"):
                _record_video_failure(root, job_id, {"stage": "download", "message": "供应商任务已完成，但未返回可下载地址；继续时只查询原任务。", "uncertain": False})
                raise core.GateError("供应商任务已完成，但未返回可下载地址；用量已保存，继续时只查询原任务。")
            try:
                original = job.get("origin_video", {})
                if original:
                    if not Path(output).is_file() or core.digest(output) != original.get("sha256"):
                        raise core.GateError("已保存的原始视频缺失或发生变化；未覆盖已有原件。")
                else:
                    provider.download(result["url"], output)
                    core.update_job(root, job_id, output=_rel(root, output), origin_video={
                        "file": _rel(root, output), "sha256": core.digest(output), "task_id": job["task_id"],
                        "created_at": result.get("created_at") or job["created_at"]})
            except Exception as exc:
                _record_video_failure(root, job_id, exception_failure(exc, "download"))
                raise
            if job.get("kind") == "video":
                try:
                    official.cache_tail(root, job_id, result, provider)
                except Exception as exc:
                    # A missing ancillary image cannot hide a paid-for video
                    # or authorize another POST. Resolve it by GET before the
                    # next segment needs an official continuity reference.
                    official.tail_error(root, job_id, exc)
            core.update_job(root, job_id, state="completed", output=_rel(root, output), failure=None)
            return
        if result["status"] == "failed":
            core.record_usage(root, job_id, result)
            _record_video_failure(root, job_id, result.get("failure") or {
                "stage": "generation", "message": "供应商报告视频生成失败。", "provider_code": result.get("error"), "uncertain": False},
                state="failed", accounting="spent")
            raise core.GateError("供应商报告任务失败；预算占用不代表实际扣费，重新生成须由你明确决定。")
        # A pause stops waiting/new submissions; the saved task is polled again
        # after resume. It may continue charging at the provider.
        _gate(root)
        if time.monotonic() - started >= s["settings"]["max_poll_seconds"]:
            raise core.GateError("供应商仍在生成，任务 ID 已保存；稍后继续将查询原任务")
        time.sleep(s["settings"]["poll_seconds"])


def _video(root, kind, item_id, output, duration, prompt, references, audio_references=None, old_job=None,
           previous_video=None, reference_duration=0):
    s = _gate(root)
    if old_job:
        job = s["jobs"][old_job]
        if job.get("output") and core.safe_path(root, job["output"]) != Path(output).resolve():
            raise core.GateError("恢复任务输出路径与已登记版本不符，禁止复制旧任务为新版本")
        if job["state"] == "completed" and Path(output).is_file():
            origin = job.get("origin_video")
            if origin and (origin.get("file") != _rel(root, output)
                           or origin.get("task_id") != job.get("task_id")
                           or origin.get("sha256") != core.digest(output)):
                raise core.GateError("已完成任务的原始视频发生变化；未覆盖或采用修改后的文件。")
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
    if previous_video is not None and (isinstance(reference_duration, bool)
            or not isinstance(reference_duration, (int, float)) or not math.isfinite(reference_duration)
            or not 2 <= reference_duration <= 15):
        raise core.GateError("原始视频参考时长尚未核实；未提交新任务。")
    amount = 0 if s["mode"] == "demo" else (duration + reference_duration) * s["settings"]["caps"]["video_second"]
    pricing = price_snapshot({**s["settings"]["ark"], "currency": s["budget"]["currency"]}, kind,
                             has_video_input=previous_video is not None)
    core.reserve(root, job_id, kind, item_id, amount, pricing=pricing if s["mode"] == "live" else None)
    with core.transaction(root) as current:
        obj = core._find(current, "asset" if kind == "voice" else "shot", item_id)
        obj["job_id"] = job_id
        obj.setdefault("attempts", []).append({"job_id": job_id, "revision": obj.get("revision", 0),
                                               "version": obj.get("version", 0), "time": core.now()})
        current["jobs"][job_id].update(duration=duration, reference_video_duration=reference_duration,
                                       return_last_frame=s["mode"] == "live")
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
        video_options = {"audio_references": audio_references or []}
        if previous_video is not None:
            video_options["previous_video"] = previous_video
        result = provider.submit_video(prompt, duration, references, **video_options)
    except Exception as exc:
        uncertain = getattr(exc, "uncertain", True)
        detail = {**exception_failure(exc, "submission"), "uncertain": uncertain}
        _record_video_failure(root, job_id, detail,
                              state="uncertain" if uncertain else "failed",
                              accounting="unknown" if uncertain else "spent")
        raise
    core.update_job(root, job_id, state="submitted", task_id=result["id"],
                    request_id=sanitize_failure({"request_id": result.get("request_id")}).get("request_id"), failure=None)
    _poll_video(root, job_id, output)
    return job_id


def _select_audition(root, asset, character):
    """Select a sourced excerpt once per voice/feedback/source change."""
    novel = (Path(root) / "source/novel.txt").read_text(encoding="utf-8")
    speaker = asset.get("audition_speaker") or character.get("audition_speaker") or character["name"]
    signature = audition_selection_key(asset, speaker, core.digest(Path(root) / "source/novel.txt"))
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
                validate_audition(candidate, novel, speaker)
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
        validate_audition(asset, novel, speaker)
    except AuditionSelectionError as exc:
        raise core.GateError(str(exc) + (" 选句说明：" + asset["audition_reason"] if asset.get("audition_reason") else "")) from None
    return asset


def _asset_generation_failed(root, asset_id):
    state = core.load_project(root)
    asset = core._find(state, "asset", asset_id)
    job = state["jobs"].get(asset.get("job_id"), {})
    if job.get("item_id") != asset_id or job.get("kind") not in {"voice", "image"} or job.get("state") != "failed":
        raise core.GateError("资产失败记录与当前版本不匹配，请核对原调用。")
    detail = sanitize_failure(job.get("failure")) if job.get("failure") else {}
    message = detail.get("message") or "旧资产调用未保存具体原因；不会再次提交，请填写意见后决定是否重试。"
    _save_asset(root, asset_id, status="generation_failed", generation_error=message)
    _set(root, status="asset_generation_failed", last_error=message)
    return False


def _assets(root):
    s = _gate(root)
    core.normalize_voice_sources(root, already_locked=True)
    s = _gate(root)
    from .asset_inputs import pending_assets, reference_paths, compose_prompt
    waiting_for_input = bool(pending_assets(s))
    for original in s["assets"]:
        s = _gate(root)
        asset = core._find(s, "asset", original["id"])
        if core.is_internal_voice_source(asset):
            if not core.asset_ready(asset):
                raise core.GateError("内部画外声源定义尚未登记，不能生成声音")
            core._check_artifact(root, asset)
            continue
        if asset["status"] in {"approved", "candidate"}:
            continue
        active_job = s["jobs"].get(asset.get("job_id", ""))
        if active_job and active_job.get("state") == "failed" and asset["status"] != "rejected":
            return _asset_generation_failed(root, asset["id"])
        recovering = bool(asset["status"] != "rejected" and active_job and active_job["state"] in {"submitted", "download_pending", "completed"})
        # An upgraded project can already have a submitted task while later
        # assets still lack the new choice. Recover that task without any new
        # POST so its reservation cannot permanently block the user's choices.
        if waiting_for_input and not recovering:
            continue
        if asset["kind"] == "voice" and not recovering:
            # A voice audition must use approved visual identity references.
            if any(not core.asset_ready(a) for a in s["assets"] if a["kind"] != "voice"):
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
                a.pop("generation_error", None)
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
            if character is None or not core.asset_ready(character):
                raise core.GateError("声音试音必须绑定已批准的可见角色或已登记的内部声源，不能猜测说话人")
            core._check_artifact(root, character)
            images = [] if character.get("voice_only") is True else [core.safe_path(root, character["file"])]
            voice_asset = {**asset, "prompt": compose_prompt(asset), "voice_only": character.get("voice_only") is True}
            voice_prompt = asset.get("audition_prompt", "已提交声音试片，仅恢复原任务") if recovering else (
                "DEMO / SYNTHETIC PLACEHOLDER voice audition" if s["mode"] == "demo" else build_audition_prompt(voice_asset))
            duration = 5 if recovering or s["mode"] == "demo" else audition_duration(asset)
            if not recovering:
                _save_asset(root, asset["id"], audition_profile=audition_profile(voice_asset), audition_prompt=voice_prompt)
            try:
                _video(root, "voice", asset["id"], out, duration,
                    voice_prompt, images,
                    old_job=asset.get("job_id") if recovering else None)
            except Exception:
                latest = core.load_project(root)
                failed_asset = core._find(latest, "asset", asset["id"])
                failed = latest["jobs"].get(failed_asset.get("job_id"), {})
                if failed.get("state") == "failed" and failed.get("accounting") != "unknown":
                    return _asset_generation_failed(root, asset["id"])
                raise
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
                                accounting="unknown" if uncertain else "spent", failure=exception_failure(exc, "submission"))
                if not uncertain:
                    return _asset_generation_failed(root, asset["id"])
                raise
        if not out.is_file():
            raise core.GateError("已登记成功的素材文件缺失，请从供应商取回；不会重复生成")
        _save_asset(root, asset["id"], status="candidate", file=_rel(root, out), sha256=core.digest(out))
    s = core.load_project(root)
    if pending_assets(s):
        _set(root, status="awaiting_asset_input")
        return False
    if any(not core.asset_ready(a) for a in s["assets"]):
        _set(root, status="awaiting_assets")
        return False
    for asset in s["assets"]:
        core._check_artifact(root, asset)
    return True


def _previous_actual_context(previous, shot):
    """One accepted predecessor, without embedding its complete saved report."""
    report = previous.get("review", {})
    report = report if (report.get("sha256") == previous.get("sha256") and
                       report.get("version", previous.get("version")) == previous.get("version")) else {}
    summary = {"shot_id": previous["id"], "version": previous.get("version"), "sha256": previous.get("sha256"),
        "file": previous.get("file"), "observed_end_state": report.get("observed_end_state", "未核实"),
        "same_scene": previous.get("scene_id") == shot.get("scene_id")}
    if core.human_approved_shot(previous):
        summary["human_decision"] = task_shot_summary(previous).get("human_decision") or {
            "decision": "approved", "version": previous.get("version"), "sha256": previous.get("sha256"),
            "at": previous.get("approved_at"), "source": "legacy_human_approval"}
        summary["accepted_deviations"] = copy.deepcopy(report.get("issues", []))
        summary["acceptance_note"] = "用户已接受这个视频版本；报告中的疑点未必被用户逐项证实。以实际已接受画面承接，不擅自纠正其既有细节。"
    return summary


def _shot_context(root, ep, shot, previous):
    s = _gate(root)
    assets = [a for a in s["assets"] if a["id"] in shot["asset_ids"]]
    images, voices, image_manifest, voice_manifest = [], [], [], []
    for asset in assets:
        if not core.asset_ready(asset):
            raise core.GateError("镜头引用的资产尚未批准")
        core._check_artifact(root, asset)
        if asset["kind"] == "voice":
            if asset.get("character_id") not in shot["asset_ids"]:
                raise core.GateError("镜头声音参考未绑定到同镜引用的角色，不能猜测说话人")
            voices.append(core.safe_path(root, asset["audio_file"]))
            voice_manifest.append({"index": len(voices), "asset_id": asset["id"], "character_id": asset["character_id"], "sha256": asset["audio_sha256"]})
        elif asset.get("voice_only") is not True:
            images.append(core.safe_path(root, asset["file"]))
            image_manifest.append({"index": len(images), "role": "approved_asset", "asset_id": asset["id"], "sha256": asset["sha256"]})
    if shot.get("dialogue") and s["settings"]["require_voice_reference"] and not voices:
        raise core.GateError(f"{shot['id']} 有对白但未引用已批准的声音资产；请补充规划或显式关闭声音参考要求")
    continuity = None
    if previous:
        if previous["status"] != "accepted":
            raise core.GateError("前镜头未被接受，不能成为连续性参考")
        core._check_artifact(root, previous)
        continuity = _previous_actual_context(previous, shot)
        if continuity["same_scene"]:
            override = shot.get("continuity_reference", {})
            use_video = (isinstance(override, dict) and override.get("mode") == "original_video"
                         and override.get("revision") == shot.get("revision", 0))
            if s["mode"] == "live" and use_video:
                if (override.get("source_shot_id") != previous["id"]
                        or override.get("source_sha256") != previous["sha256"]):
                    raise core.GateError("原始视频参考授权与当前批准版本不一致。")
                continuity["reference_mode"] = "original_video"
            else:
                if s["mode"] == "live":
                    path = official.get_tail(root, previous, _ark(root))
                    role = "previous_official_tail"
                    continuity["reference_mode"] = "official_tail"
                else:
                    tail = previous["evidence"]["tail_frame"]
                    path = core.safe_path(root, tail)
                    if not path.is_file() or core.digest(path) != previous["evidence"].get("frame_hashes", {}).get(tail):
                        raise core.GateError("前镜尾帧缺失或内容改变，不能沿用未核实的画面参考")
                    role = "previous_actual_tail"
                    continuity["reference_mode"] = "synthetic_demo"
                images.append(path)
                image_manifest.append({"index": len(images), "role": role, "shot_id": previous["id"], "sha256": core.digest(path)})
    if len(images) > 9 or len(voices) > 3:
        raise core.GateError("参考素材超过模型限制，不能静默省略人物或声音参考")
    if voices:
        voices, reference_details = media.prepare_voice_references(voices, Path(root) / "media" / "voice-references")
        for reference, details in zip(voice_manifest, reference_details, strict=True):
            if reference["sha256"] != details["source_sha256"]:
                raise core.GateError("声音原件在准备参考时发生变化，不能提交未批准的音频")
            reference.update(details)
    context = {"episode": {k: ep.get(k) for k in ("id", "title", "summary", "feedback")},
        "style": s["plan"].get("style"), "assets": [task_asset_summary(a) for a in assets],
        "previous_actual": continuity, "feedback": shot.get("feedback", ""),
        "image_references": image_manifest, "audio_references": voice_manifest,
        "audio_strategy": "使用批准的声音参考，镜头不生成背景配乐；配乐在整集时间线统一添加。",
        "audio_policy": copy.deepcopy(s.get("audio_policy", {})),
        "limitations": "尾帧是实际画面参考；不能保证身份、口型和音色绝对一致。"}
    return context, images, voices


def _observation_images(root, context, images, previous):
    """Local evidence may inform GPT, but it is never an Ark image reference."""
    observed = list(images)
    manifest = [{**item, "generation_image_index": item["index"], "sent_to_video_engine": True}
                for item in context.get("image_references", [])]
    if (previous and (context.get("previous_actual") or {}).get("reference_mode") == "original_video"):
        tail = previous.get("evidence", {}).get("tail_frame")
        path = core.safe_path(root, tail) if tail else None
        if path and path.is_file() and core.digest(path) == previous["evidence"].get("frame_hashes", {}).get(tail):
            observed.append(path)
            manifest.append({"index": len(observed), "role": "previous_observation_tail", "shot_id": previous["id"],
                             "sha256": core.digest(path), "sent_to_video_engine": False})
    context["observation_images"] = manifest
    return observed


def _final_shot_prompt(state, shot, refined, context):
    prompt = refined["prompt"]
    if "subshots" in shot:
        from .segments import timeline_prompt
        prompt += "\n" + timeline_prompt(shot)
    if shot.get("dialogue"):
        prompt += "\n必须使用的原定对白：" + shot["dialogue"]
    prompt += "\n不添加画面字幕，不添加背景配乐。"
    if (context.get("previous_actual") or {}).get("reference_mode") == "original_video":
        prompt += "\n视频1是已批准的上一段原始视频。仅承接其结尾继续本段剧情，保持人物、场景、动作方向和声音连续；不要重演已完成的动作或重复已说完的台词。"
    policy = state.get("audio_policy", {}).get("system_broadcast", {})
    if policy.get("enabled") is True and policy.get("voice_id") in shot["asset_ids"]:
        reference = next((item for item in context.get("audio_references", [])
                          if item.get("asset_id") == policy["voice_id"]), None)
        if not reference:
            raise core.GateError("系统播报必须传入已批准的女声参考，不能仅靠文字描述。")
        prompt += (f"\n系统播音规则（覆盖旧分镜中与之冲突的无配音或不念字幕要求）：使用音频{reference['index']}"
                   "作为固定系统女声，仅作画外播报，不生成可见播报员。" + policy.get("instruction", "") +
                   "任务框播音与画面字幕是不同要求；不添加画面字幕不代表取消播音。只播读本片段明确安排的系统通知，"
                   "不重复试音台词，不覆盖人物对白，不通过加速或截断强行压缩。")
        prompt += "\n如果本片段没有明确系统通知，则该系统声源保持静默，不凭空增加播报内容；弹幕、直播标题与普通道具文字不自动当作系统通知。"
    return prompt


def _save_revision_analysis(root, shot_id, record):
    """Keep superseded diagnoses while binding the current one to its feedback."""
    with core.transaction(root) as state:
        shot = core._find(state, "shot", shot_id)
        old = shot.get("revision_analysis")
        if isinstance(old, dict) and old and old.get("signature") != record.get("signature"):
            history = shot.setdefault("revision_analysis_history", [])
            if not any(item.get("signature") == old.get("signature") for item in history):
                history.append({**copy.deepcopy(old), "superseded_at": core.now()})
        shot["revision_analysis"] = copy.deepcopy(record)


def _revision_analysis(root, shot):
    """Resolve a human request for new assets before preparing a paid rework."""
    if not shot.get("feedback", "").strip():
        return True
    # Parse the meaning once per human revision. A keyword rule would miss
    # requests such as "让另一个女人说这些" or "换到沙漠里".
    signature = hashlib.sha256(json.dumps([shot.get("revision", 0), shot["feedback"]], ensure_ascii=False).encode()).hexdigest()
    old = shot.get("revision_analysis", {})
    if old.get("signature") == signature:
        if old.get("amendment_id"):
            amendment = next((a for a in core.load_project(root).get("asset_amendments", [])
                              if a.get("id") == old["amendment_id"]), {})
            if amendment.get("status") == "rejected":
                raise core.GateError("新增资产方案已取消。请修改返修意见后重新提出方案；尚未生成返修视频。")
            return amendment.get("status") in {"applied", "activated"}
        if old.get("blockers"):
            raise core.GateError("；".join(old["blockers"]))
        return True
    state = _gate(root)
    if state["mode"] == "demo":
        return True
    context = task_context_summary({"assets": state["assets"], "episodes": state["episodes"],
               "novel": (Path(root) / "source/novel.txt").read_text(encoding="utf-8"),
               "audio_policy": state.get("audio_policy", {})})
    ordered = [item for episode in state["episodes"] for item in episode["shots"]]
    target_index = next((index for index, item in enumerate(ordered) if item["id"] == shot["id"]), None)
    if target_index is not None and target_index > 0:
        previous = ordered[target_index - 1]
        if previous.get("status") == "accepted" and core.human_approved_shot(previous):
            core._check_artifact(root, previous)
            context["previous_actual"] = _previous_actual_context(previous, shot)
    result = _brain(root, "plan_revision", task_shot_summary(shot), context, shot["feedback"])
    if not isinstance(result, dict) or type(result.get("needs_assets")) is not bool:
        raise core.GateError("返修需求分析结果不完整，尚未提交视频。")
    record = {"signature": signature, "blockers": result.get("blockers", []), "summary": result.get("summary", "")}
    if record["blockers"]:
        _save_revision_analysis(root, shot["id"], record)
        raise core.GateError("；".join(record["blockers"]))
    policy = result.get("audio_policy") or {}
    previous_policy = state.get("audio_policy", {})
    policy_changed = bool(policy.get("system_broadcast", {}).get("enabled")
                          or previous_policy.get("system_broadcast", {}).get("enabled")) and policy != previous_policy
    references_changed = any(set(update.get("add_asset_ids", [])) - set(core._find(state, "shot", update["id"])["asset_ids"])
                             for update in result.get("shot_updates", []))
    if result["needs_assets"] or references_changed or policy_changed:
        changed = core.propose_asset_amendment(root, shot["id"], shot.get("version", 0), shot["feedback"],
                    result["additions"], result["shot_updates"], audio_policy=policy or None, already_locked=True)
        amendment = changed["asset_amendments"][-1]
        record["amendment_id"] = amendment["id"]
        _save_revision_analysis(root, shot["id"], record)
        _set(root, status="awaiting_asset_amendment")
        return False
    _save_revision_analysis(root, shot["id"], record)
    return True


def _system_voice_reference(root, shot):
    """A user-approved global voice rule must never silently lose its source.

    Carry the fixed reference even in a silent segment; the final prompt makes
    it explicit that only written system notifications may be spoken. This
    avoids guessing from keywords whether a task interface appears on screen.
    """
    state = core.load_project(root)
    policy = state.get("audio_policy", {}).get("system_broadcast", {})
    if policy.get("enabled") is not True:
        return shot
    required = [policy.get("character_id"), policy.get("voice_id")]
    for identifier in required:
        asset = core._find(state, "asset", identifier)
        if not core.asset_ready(asset):
            raise core.GateError("固定系统播报的内部声源未登记或声音尚未人工批准，不能继续准备片段。")
        core._check_artifact(root, asset)
    if not set(required) <= set(shot["asset_ids"]):
        _save_shot(root, shot["id"], asset_ids=list(dict.fromkeys(shot["asset_ids"] + required)))
        return core._find(core.load_project(root), "shot", shot["id"])
    return shot


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
        _refresh_candidate_preview(root, ep_id)
        _set(root, status="review_required")
        return False
    active = s["jobs"].get(shot.get("job_id", ""))
    if active and (active.get("kind") != "video" or active.get("item_id") != shot_id):
        raise core.GateError("生成任务与当前片段不匹配，请先核对记录；未提交新任务。")
    if shot["status"] == "generation_failed" or (shot["status"] == "review_required" and not shot.get("file")
            and active and active.get("state") == "failed"):
        return _shot_generation_failed(root, shot_id)
    if shot["status"] == "review_required":
        _refresh_candidate_preview(root, ep_id)
        _set(root, status="review_required")
        return False
    if shot["status"] == "awaiting_prompt_approval":
        _set(root, status="awaiting_prompt_approval")
        return False
    recovering = bool(active and active["state"] in {"submitted", "download_pending", "completed"})
    attempted = bool(active and active.get("state") == "failed") or any(
        attempt.get("revision", 0) == shot.get("revision", 0) for attempt in shot.get("attempts", []))
    if not recovering and attempted:
        return _shot_generation_failed(root, shot_id)
    approved_candidate = prompt_gate.approved(root, shot) if not recovering and shot.get("status") == "prompt_approved" else None
    if not recovering:
        _new_work_gate(s)
        shot = _system_voice_reference(root, shot)
        _progress(root, ep_id, "shot_preparation", shot_id)
        if not _revision_analysis(root, shot):
            return False
    previous_video, reference_duration = None, 0
    if recovering:
        # Already-submitted jobs need only their original task ID, even after
        # source URLs/records have expired. Do not resolve inputs again.
        saved = shot.get("reference_manifest", {})
        context = copy.deepcopy(saved.get("context", {}))
        images, voices = [], []
        observations = []
        observed_manifest = context.get("observation_images", context.get("image_references", []))
        for relative, info in zip(saved.get("observations", saved.get("images", [])), observed_manifest):
            path = core.safe_path(root, relative)
            if path.is_file() and core.digest(path) == info.get("sha256"):
                observations.append(path)
        if len(observations) != len(observed_manifest):
            context["observation_images"] = []
            context["observation_warning"] = "部分历史参考图已不可读取；本次只对实际可读的抽样图出具报告。"
            observations = []
    else:
        context, images, voices = _shot_context(root, ep, shot, previous)
        if (context.get("previous_actual") or {}).get("reference_mode") == "original_video":
            previous_video, reference_duration, video_manifest = official.resolve_video(root, previous, _ark(root))
            context["video_references"] = [video_manifest]
        observations = _observation_images(root, context, images, previous)
    if not recovering and shot.get("status") == "prompt_approved":
        candidate = approved_candidate
        refined = candidate["refined"]
        prompt = candidate["prompt"]
        manifest = {"images": [_rel(root, f) for f in images], "audio": [_rel(root, f) for f in voices],
                    "context": context, "observations": [_rel(root, f) for f in observations]}
        actual_hash = prompt_gate._hash(prompt_gate.specification(core.load_project(root), shot, prompt, manifest))
        if actual_hash != candidate["request_hash"]:
            prompt_gate.invalidate(root, shot_id)
            raise core.GateError("实际生成素材与已确认提示词不一致，请重新准备提示词。")
        version = candidate["version"]
        _save_shot(root, shot_id, refined_prompt=refined, version=version, status="generating", generation_error=None,
                   reference_manifest=manifest)
        shot = core._find(core.load_project(root), "shot", shot_id)
    elif not recovering:
        if s["mode"] == "demo":
            refined = {"prompt": shot["prompt"] + " " + shot.get("feedback", ""), "dialogue": "", "continuity_notes": "Synthetic demo"}
        else:
            refined = _brain(root, "refine_shot", task_shot_summary(shot), task_context_summary(context), observations)
        if s["mode"] == "live":
            manifest = {"images": [_rel(root, f) for f in images], "audio": [_rel(root, f) for f in voices],
                        "context": context, "observations": [_rel(root, f) for f in observations]}
            prompt_gate.prepare(root, shot_id, refined, _final_shot_prompt(s, shot, refined, context), manifest)
            return False
        version = shot.get("version", 0) + 1
        _save_shot(root, shot_id, refined_prompt=refined, version=version, status="generating", generation_error=None,
            reference_manifest={"images": [_rel(root, f) for f in images],
                "audio": [_rel(root, f) for f in voices], "context": context,
                "observations": [_rel(root, f) for f in observations]})
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
    prompt = (shot.get("prompt_candidate", {}).get("prompt")
              if shot.get("prompt_approval") else _final_shot_prompt(s, shot, shot["refined_prompt"], context))
    try:
        _video(root, "video", shot_id, out, shot["duration"], prompt, images, voices,
               old_job=shot.get("job_id") if recovering else None,
               previous_video=previous_video, reference_duration=reference_duration)
    except Exception:
        current = core.load_project(root)
        current_shot = core._find(current, "shot", shot_id)
        failed = current["jobs"].get(current_shot.get("job_id", ""), {})
        if failed.get("state") != "failed" or failed.get("accounting") == "unknown":
            raise
        return _shot_generation_failed(root, shot_id)
    _progress(root, ep_id, "shot_checks", shot_id)
    evidence_dir = Path(root) / "evidence" / f"{shot_id}-v{version}"
    evidence = (media.extract_evidence(out, evidence_dir, subshots=shot["subshots"])
                if "subshots" in shot else media.extract_evidence(out, evidence_dir))
    relative = _relative_evidence(root, evidence)
    _save_shot(root, shot_id, file=_rel(root, out), sha256=core.digest(out), evidence=relative, status="reviewing")
    # Human approval needs the seam in motion, not just a standalone clip.
    # This local preview is published before the advisory GPT call completes.
    _refresh_candidate_preview(root, ep_id)
    # Review the seam too: prior tail + current sampled frames, tied to hashes.
    review_images = observations + [Path(f) for f in evidence["frames"]]
    has_prior_frame = any(item.get("role") in {"previous_actual_tail", "previous_official_tail", "previous_observation_tail"}
                          for item in context.get("observation_images", []) if isinstance(item, dict))
    prior_scope = ("含前段实际尾帧参考。" if has_prior_frame else "未提供前段实际画面，不能核实跨段画面衔接。")
    current_shot = core._find(core.load_project(root), "shot", shot_id)
    review_context = {**task_context_summary(context), "shot": task_shot_summary(current_shot), "actual": relative, "advisory_target": _shot_review_target(current_shot),
        "current_frame_image_indices": list(range(len(observations) + 1, len(review_images) + 1)),
        "current_frame_samples": [{"image_index": len(observations) + index + 1, "time_seconds": instant}
                                  for index, instant in enumerate(evidence.get("sample_times", []))],
        "review_scope": "当前镜头首/中/尾抽样画面。" + prior_scope + "没有审听音频；不可宣称整段动作或音色已核实。"}
    if "subshots" in shot:
        review_context.update(
            review_scope="按计划子镜中点及切点前后采样，包含实测首帧和最后一帧。" + prior_scope +
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


def _rough_cut(root, episode, count, *, include_pending=False):
    """Publish a local cumulative cut before any batch model review.

    Candidate previews are allowed only when explicitly requested here; they
    never approve clips or become a generation reference. Source identities,
    versions and hashes bind a cut to its exact timeline. Legacy hash-only cuts
    remain reusable when their saved manifest matches the current sources.
    """
    shots = episode["shots"][:count]
    allowed = {"accepted", "reviewing", "review_required"} if include_pending else {"accepted"}
    if not shots or len(shots) != count or any(shot["status"] not in allowed for shot in shots):
        raise core.GateError("累计粗剪需要连续且已落盘的当前版本；待审预览不会批准视频。")
    state = core.load_project(root)
    for shot in shots:
        core._check_shot_artifact(root, state, shot)
    manifest = [{"id": shot["id"], "version": shot.get("version", 0), "sha256": shot["sha256"]} for shot in shots]
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    legacy_key = hashlib.sha256("-".join(shot["sha256"] for shot in shots).encode()).hexdigest()
    existing = next((cut for cut in episode.get("rough_cuts", [])
                     if cut.get("key") in {key, legacy_key} and cut.get("shots") == manifest), None)
    if existing:
        core._check_artifact(root, existing)
        return copy.deepcopy(existing)
    legacy = next((review for review in episode.get("reviews", [])
                   if review.get("key") in {key, legacy_key} and review.get("file") and review.get("sha256")
                   and not any(cut.get("file") == review["file"] and cut.get("shots") != manifest
                               for cut in episode.get("rough_cuts", []))), None)
    if legacy:
        core._check_artifact(root, legacy)
        path = core.safe_path(root, legacy["file"])
        key = legacy["key"]
    else:
        _progress(root, episode["id"], "batch_assembly", shots[-1]["id"])
        path = Path(root) / "exports" / f"{episode['id']}-rough-{count}-{key[:16]}.mp4"
        if path.exists():
            path = path.with_name(f"{path.stem}-{uuid.uuid4().hex[:8]}{path.suffix}")
        media.compose([core.safe_path(root, shot["file"]) for shot in shots], path)
    record = {"key": key, "file": _rel(root, path), "sha256": core.digest(path),
              "shot_count": count,
              "shots": manifest, "created_at": core.now()}
    with core.transaction(root) as state:
        cuts = core._find(state, "episode", episode["id"]).setdefault("rough_cuts", [])
        if not any(cut.get("key") == key for cut in cuts):
            cuts.append(record)
    return record


def _refresh_candidate_preview(root, ep_id):
    """Join the longest available prefix locally, without changing approvals.

    A preserved downstream clip may still need continuity review. Include its
    current bytes so a repaired middle clip can be judged on both seams. Missing
    or still-generating clips terminate the prefix; history is never substituted.
    Preview failures preserve the paid-for clips and their human review gate.
    """
    state = core.load_project(root)
    episode = core._find(state, "episode", ep_id)
    count = 0
    for shot in episode["shots"]:
        if (shot.get("status") not in {"accepted", "reviewing", "review_required"}
                or not shot.get("file") or not shot.get("sha256") or not core._shot_file_matches_version(shot)):
            break
        count += 1
    if count < 2:
        if episode.get("preview_error"):
            _save_episode(root, ep_id, preview_error=None)
        return None
    try:
        cut = _rough_cut(root, episode, count, include_pending=True)
    except Exception:
        _save_episode(root, ep_id, preview_error=
                      "累计拼接预览暂未生成成功；片段已保留，可以重试本地拼接后再决定。")
        return None
    if episode.get("preview_error"):
        _save_episode(root, ep_id, preview_error=None)
    return cut


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
            return
        fresh = core._find(core.load_project(root), "episode", ep_id)
        previous = fresh["shots"][index]
        count = index + 1
        rough_cut = _rough_cut(root, fresh, count) if count >= 2 else None
        if count % s["settings"]["batch_size"] == 0 or count == len(ep["shots"]):
            rough_cut = rough_cut or _rough_cut(root, fresh, count)
            batch_key = rough_cut["key"]
            if any(r.get("key") == batch_key for r in fresh["reviews"]):
                continue
            path = core.safe_path(root, rough_cut["file"])
            ev = media.extract_evidence(path, Path(root) / "evidence" / f"rough-{ep_id}-{batch_key[:16]}")
            _progress(root, ep_id, "batch_review", original["id"])
            try:
                review = ({"verdict": "pass", "issues": [], "observed_end_state": "Synthetic demo",
                           "repair_prompt": ""} if s["mode"] == "demo" else
                          _brain(root, "review", {"episode": ep_id, "shots": [task_shot_summary(item) for item in fresh["shots"][:count]],
                              "advisory_target": _cut_review_target(ep_id, rough_cut),
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
            recovered = _recover(root)
            s = _gate(root)
            _set(root, last_error=None)
            if recovered["shots"] or recovered["rough_cuts"]:
                _set(root, status="review_required")
                return core.load_project(root)
            if any(item.get("status") == "candidate" for item in s.get("asset_amendments", [])):
                _set(root, status="awaiting_asset_amendment")
                return core.load_project(root)
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
            for amendment in core.load_project(root).get("asset_amendments", []):
                if amendment.get("status") == "awaiting_assets":
                    core.activate_asset_amendment(root, amendment["id"])
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
