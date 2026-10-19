"""Resumable production, with a human gate before every next episode."""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from pathlib import Path

from . import core, media


def demo_plan():
    """A conspicuously synthetic two-episode workflow fixture, not AI footage."""
    assets = [
        {"id": "hero", "kind": "character", "name": "演示人物卡", "prompt": "DEMO placeholder, blue character identity card"},
        {"id": "station", "kind": "location", "name": "演示场景卡", "prompt": "DEMO placeholder, railway station environment"},
    ]
    return {"title": "Norlan · 两集流程演示", "style": "流程占位素材，不代表 AI 漫剧质量", "ratio": "16:9",
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


def _brain(root, method, *args, **kwargs):
    from .brain import Brain
    s = _gate(root)
    job_id = "gpt-" + uuid.uuid4().hex[:12]
    amount = 0 if s["mode"] == "demo" else s["settings"]["caps"]["llm"]
    core.reserve(root, job_id, "llm", method, amount)
    try:
        result = getattr(Brain(s["settings"]["brain"]), method)(*args, **kwargs)
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


def _recover(root):
    # Before a POST starts its intent was committed. A killed process must not
    # interpret the absence of a response as absence of a provider-side job.
    with core.transaction(root) as s:
        for job in s["jobs"].values():
            if job["accounting"] == "reserved" and job["state"] == "submitting":
                job.update(accounting="unknown", state="uncertain")
                core.event(s, "interrupted_submission", id=job["id"])


def _save_asset(root, asset_id, **values):
    with core.transaction(root) as s:
        core._find(s, "asset", asset_id).update(values)


def _save_shot(root, shot_id, **values):
    with core.transaction(root) as s:
        core._find(s, "shot", shot_id).update(values)


def _save_episode(root, ep_id, **values):
    with core.transaction(root) as s:
        core._find(s, "episode", ep_id).update(values)


def _poll_video(root, job_id, output):
    provider = _ark(root)
    started = time.monotonic()
    while True:
        s = core.load_project(root)
        job = s["jobs"][job_id]
        if not job.get("task_id"):
            raise core.GateError("视频任务没有供应商 ID，必须先对账")
        result = provider.poll_video(job["task_id"])
        if result["status"] == "completed":
            # Retain ID before download. An expired/download-failed URL is never
            # permission to resubmit the paid generation task.
            core.update_job(root, job_id, state="download_pending", accounting="spent")
            provider.download(result["url"], output)
            core.update_job(root, job_id, state="completed", output=_rel(root, output))
            return
        if result["status"] == "failed":
            core.update_job(root, job_id, state="failed", accounting="spent")
            raise core.GateError("供应商报告任务失败，已按上限保守计账；再次运行可在重试限制内返工")
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
    provider = _ark(root) if s["mode"] == "live" else None
    if provider:
        provider.preflight("video")
    job_id = f"{kind}-{item_id}-{uuid.uuid4().hex[:8]}"
    amount = 0 if s["mode"] == "demo" else duration * s["settings"]["caps"]["video_second"]
    core.reserve(root, job_id, kind, item_id, amount)
    with core.transaction(root) as current:
        obj = core._find(current, "asset" if kind == "voice" else "shot", item_id)
        obj["job_id"] = job_id
        obj.setdefault("attempts", []).append({"job_id": job_id, "revision": obj.get("revision", 0), "time": core.now()})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if s["mode"] == "demo":
        media.synthetic_clip(output, prompt, duration, index=len(s["jobs"]))
        core.update_job(root, job_id, state="completed", accounting="spent", output=_rel(root, output))
        return job_id
    try:
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


def _assets(root):
    s = _gate(root)
    for original in s["assets"]:
        s = _gate(root)
        asset = core._find(s, "asset", original["id"])
        if asset["status"] in {"approved", "candidate"}:
            continue
        if asset["kind"] == "voice":
            # A voice audition must use approved visual identity references.
            if any(a["status"] != "approved" for a in s["assets"] if a["kind"] != "voice"):
                continue
        active_job = s["jobs"].get(asset.get("job_id", ""))
        recovering = bool(asset["status"] != "rejected" and active_job and active_job["state"] in {"submitted", "download_pending", "completed"})
        if not recovering:
            total_attempts = sum(j["item_id"] == asset["id"] and j["kind"] in {"voice", "image"} for j in s["jobs"].values())
            if total_attempts >= s["settings"]["max_attempts"]:
                raise core.GateError("资产已达到尝试上限；请调整方案或显式提高上限：" + asset["id"])
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
        prompt = f"{s['plan'].get('style','')}\n{asset['prompt']}\n修改意见：{asset.get('feedback','')}"
        if asset["kind"] == "voice":
            character = next((a for a in s["assets"] if a["id"] == asset.get("character_id") and a["kind"] == "character"), None)
            if character is None or character["status"] != "approved":
                raise core.GateError("声音试音必须绑定一个已批准的角色，不能猜测说话人")
            core._check_artifact(root, character)
            images = [core.safe_path(root, character["file"])]
            _video(root, "voice", asset["id"], out, 5,
                prompt + "\n角色声音试音：说一句简短中文对白，只保留清晰人声，无配乐、无其他说话人。", images,
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
            core.reserve(root, job_id, "image", asset["id"], amount)
            with core.transaction(root) as current:
                a = core._find(current, "asset", asset["id"])
                a["job_id"] = job_id
                a.setdefault("attempts", []).append({"job_id": job_id, "time": core.now()})
            try:
                if s["mode"] == "demo":
                    media.synthetic_asset(out, asset["id"] + " / DEMO", "#2563eb" if asset["kind"] == "character" else "#059669")
                else:
                    provider.generate_image(prompt, out)
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
        continuity = {"shot_id": previous["id"], "sha256": previous["sha256"],
            "observed_end_state": previous.get("review", {}).get("observed_end_state", "未核实"),
            "same_scene": previous["scene_id"] == shot["scene_id"]}
        if continuity["same_scene"]:
            tail = previous["evidence"]["tail_frame"]
            path = core.safe_path(root, tail)
            if not path.is_file() or core.digest(path) != previous["evidence"].get("frame_hashes", {}).get(tail):
                raise core.GateError("前镜尾帧缺失或内容改变，不能沿用未核实的画面参考")
            images.append(path)
            image_manifest.append({"index": len(images), "role": "previous_actual_tail", "shot_id": previous["id"], "sha256": core.digest(path)})
    if len(images) > 9 or len(voices) > 3:
        raise core.GateError("参考素材超过模型限制，不能静默省略人物或声音参考")
    context = {"episode": {k: ep.get(k) for k in ("id", "title", "summary", "feedback")},
        "style": s["plan"].get("style"), "assets": [{k: a.get(k) for k in
            ("id", "kind", "name", "prompt", "character_id", "version", "sha256")} for a in assets],
        "previous_actual": continuity, "feedback": shot.get("feedback", ""),
        "image_references": image_manifest, "audio_references": voice_manifest,
        "audio_strategy": "使用批准的声音参考，镜头不生成背景配乐；配乐在整集时间线统一添加。",
        "limitations": "尾帧是实际画面参考；不能保证身份、口型和音色绝对一致。"}
    return context, images, voices


def _shot(root, ep_id, shot_id, previous):
    while True:
        s = _gate(root)
        ep = core._find(s, "episode", ep_id)
        shot = core._find(s, "shot", shot_id)
        if shot["status"] == "accepted":
            core._check_artifact(root, shot)
            return True
        if shot["status"] == "review_required":
            _set(root, status="review_required")
            return False
        total_attempts = sum(j["item_id"] == shot_id and j["kind"] == "video" for j in s["jobs"].values())
        active = s["jobs"].get(shot.get("job_id", ""))
        recovering = bool(active and active["state"] in {"submitted", "download_pending", "completed"})
        if not recovering and total_attempts >= s["settings"]["max_attempts"]:
            if shot.get("file"):
                _save_shot(root, shot_id, status="review_required")
            raise core.GateError("自动返工达到上限，请人工审查：" + shot_id)
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
        prompt = shot["refined_prompt"]["prompt"]
        # Refined text must not be treated as approved story changes.
        if shot.get("dialogue"):
            prompt += "\n必须使用的原定对白：" + shot["dialogue"]
        prompt += "\n不添加画面字幕，不添加背景配乐。"
        _video(root, "video", shot_id, out, shot["duration"], prompt, images, voices,
               old_job=shot.get("job_id") if recovering else None)
        evidence = media.extract_evidence(out, Path(root) / "evidence" / f"{shot_id}-v{version}")
        relative = _relative_evidence(root, evidence)
        _save_shot(root, shot_id, file=_rel(root, out), sha256=core.digest(out), evidence=relative, status="reviewing")
        # Review the seam too: prior tail + current sampled frames, tied to hashes.
        review_images = images + [Path(f) for f in evidence["frames"]]
        review_context = {**context, "shot": shot, "actual": relative,
            "current_frame_image_indices": list(range(len(images) + 1, len(review_images) + 1)),
            "review_scope": "当前镜头首/中/尾抽样画面与前镜实际尾帧。没有审听音频；不可宣称整段动作或音色已核实。"}
        review = ({"verdict": "pass", "issues": [], "observed_end_state": "Synthetic demo only",
                   "repair_prompt": "", "scope": "synthetic smoke test"} if s["mode"] == "demo"
                  else _brain(root, "review", review_context, review_images))
        review.update(sha256=core.digest(out), audio_checked=False, whole_video_checked=False)
        verdict = review.get("verdict", "uncertain")
        if verdict == "pass":
            _save_shot(root, shot_id, review=review, status="accepted")
            return True
        _save_shot(root, shot_id, review=review, status="review_required")
        if verdict != "fail":
            _set(root, status="review_required")
            return False
        # A failed sample never becomes canon. Retry only this shot, bounded by
        # all attempts, including provider failures and human revisions.
        shot = core._find(core.load_project(root), "shot", shot_id)
        count = sum(j["item_id"] == shot_id and j["kind"] == "video" for j in core.load_project(root)["jobs"].values())
        if count >= s["settings"]["max_attempts"]:
            return False
        with core.transaction(root) as current:
            sh = core._find(current, "shot", shot_id)
            sh.setdefault("history", []).append({"file": sh["file"], "sha256": sh["sha256"], "review": review})
            sh.update(status="planned", feedback=review.get("repair_prompt") or "；".join(review.get("issues", [])))
            sh.pop("job_id", None)


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


def _episode(root, ep_id):
    s = _gate(root)
    ep = core._find(s, "episode", ep_id)
    if ep["status"] == "awaiting_review":
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
        if count % s["settings"]["batch_size"] == 0 or count == len(ep["shots"]):
            batch_key = hashlib.sha256("-".join(sh["sha256"] for sh in fresh["shots"][:count]).encode()).hexdigest()
            if any(r.get("key") == batch_key for r in fresh["reviews"]):
                continue
            path = Path(root) / "exports" / f"{ep_id}-rough-{count}-{batch_key[:16]}.mp4"
            media.compose([core.safe_path(root, sh["file"]) for sh in fresh["shots"][:count]], path)
            ev = media.extract_evidence(path, Path(root) / "evidence" / f"rough-{ep_id}-{batch_key[:16]}")
            review = ({"verdict": "pass", "issues": [], "observed_end_state": "Synthetic demo",
                       "repair_prompt": ""} if s["mode"] == "demo" else
                      _brain(root, "review", {"episode": ep_id, "shots": fresh["shots"][:count],
                          "scope": "分段粗剪抽帧检查；听觉未评估，运动过程未完整评估。",
                          "evidence": _relative_evidence(root, ev)}, [Path(f) for f in ev["frames"]]))
            review.update(key=batch_key, file=_rel(root, path), sha256=core.digest(path),
                          audio_checked=False, whole_video_checked=False)
            with core.transaction(root) as current:
                core._find(current, "episode", ep_id)["reviews"].append(review)
            if review["verdict"] != "pass":
                # Batch diagnosis may not locate the faulty shot reliably. Stop
                # for review instead of regenerating unrelated accepted clips.
                _save_shot(root, original["id"], status="review_required", review=review)
                _set(root, status="review_required")
                return
    fresh = core._find(core.load_project(root), "episode", ep_id)
    _gate(root)
    music = _soundtrack(root, fresh)
    captions, subtitle_status = _captions(root, fresh)
    version = fresh["version"] + 1
    out = Path(root) / "exports" / f"{ep_id}-v{version}.mp4"
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
        _set(root, status="paused" if core.load_project(root)["paused"] else "blocked", last_error=str(exc))
    except Exception as exc:
        # Adapters sanitize provider errors; avoid leaking credentials in reprs.
        message = str(exc) if getattr(exc, "safe_message", False) or isinstance(exc, (ValueError, FileNotFoundError)) else type(exc).__name__ + "：执行失败，请检查配置和本地文件"
        _set(root, status="blocked", last_error=message)
    return core.load_project(root)
