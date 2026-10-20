"""Episode postproduction: actual audio, approved music, immutable final exports.

Only local ASR/FFmpeg run automatically. Music submissions require a separate
human request; an approved library is reused without another paid submission.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from . import core, media


def _hash(value):
    return core._json_digest(value)


def _rel(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _episode(root, eid):
    return core._find(core.load_project(root), "episode", eid)


def _save(root, eid, **values):
    with core.transaction(root) as state:
        pp = core._find(state, "episode", eid).setdefault("postproduction", {})
        pp.update(copy.deepcopy(values))


def _wait(root, eid, status, **values):
    _save(root, eid, status=status, enabled=True, ready=True, final_ready=False, **values)
    with core.transaction(root) as state:
        state["status"] = "awaiting_postproduction"
        state["last_error"] = None


def music_configuration(state):
    from .music import MusicProvider
    config = state["settings"].get("music", {})
    if config.get("mode") == "library":
        return False, "使用本地曲库，无需音乐API凭据。请添加音乐到配置文件夹、扫描并试听确认。"
    cap = config.get("call_cap", 0)
    if config.get("mode") != "volc" or type(cap) not in {int, float} or not math.isfinite(cap) or cap <= 0:
        return False, "自动生成音乐需配置火山纯音乐服务（独立AK/SK）及单次费用上限；也可直接导入本地音乐。"
    try:
        MusicProvider(config)._credentials()
    except Exception:
        return False, "未找到火山音乐AK/SK；方舟API Key不能代替。可在本机配置，或导入本地音乐。"
    return True, ""


def _assert_action(state, ep, version):
    core._check_version(ep, version)
    if not state["settings"].get("postproduction", {}).get("enabled"):
        raise core.GateError("本项目尚未开启本集后期")
    if ep["status"] == "approved":
        idx = state["episodes"].index(ep)
        if any(e.get("file") or any(sh.get("file") for sh in e["shots"]) for e in state["episodes"][idx + 1:]):
            raise core.GateError("后续集已开始制作，请先暂停并处理依赖，不能直接修改已批准前集后期")


def _retire(ep):
    if ep.get("file"):
        ep.setdefault("history", []).append({k: copy.deepcopy(ep[k]) for k in
            ("file", "sha256", "version", "render_manifest", "postproduction", "approved_at", "approved_sha256") if k in ep})
    for k in ("approved_at", "approved_sha256"):
        ep.pop(k, None)
    if all(core.human_approved_shot(sh) for sh in ep["shots"]):
        ep["status"] = "postproduction"
    ep.setdefault("postproduction", {})["final_ready"] = False


def request(root, eid, version, target, feedback):
    if target not in {"continue", "subtitles", "music", "all"}:
        raise ValueError("后期修改范围只能是字幕、配乐或全部后期")
    if not isinstance(feedback, str) or len(feedback) > 4000 or (target != "continue" and not feedback.strip()):
        raise ValueError("请填写4000字以内的后期调整意见")
    with core.run_lock(root), core.transaction(root) as state:
        ep = core._find(state, "episode", eid)
        _assert_action(state, ep, version)
        if not all(core.human_approved_shot(sh) for sh in ep["shots"]):
            raise core.GateError("先逐段批准本集视频，再开始本集后期；音乐可提前导入试听")
        if target != "continue":
            _retire(ep)
            pp = ep["postproduction"]
            pp["revision"] = pp.get("revision", 0) + 1
            pp["revision_target"] = target
            if target in {"subtitles", "all"}:
                pp.pop("manual_subtitles", None)
                pp["subtitle_feedback"] = feedback.strip()
                pp["subtitle_feedback_source"] = copy.deepcopy(pp.get("subtitles", {}).get("segments"))
                pp.pop("subtitle_key", None)
                pp.pop("subtitle_proofread_key", None)
                pp.get("subtitles", {}).pop("sync_check", None)
                pp.get("subtitles", {}).pop("proofread_segments", None)
            if target in {"music", "all"}:
                pp.pop("manual_music_plan", None)
                pp["music_feedback"] = feedback.strip()
                pp["music_feedback_source"] = copy.deepcopy(pp.get("music", {}).get("cues", []))
                pp.pop("music_key", None)
            core.event(state, "postproduction_revision_requested", id=eid, target=target, feedback=feedback.strip())
        state.update(status="ready", paused=False, last_error=None)
    return core.load_project(root)


def import_music(root, eid, version, data_url, title):
    from .music_library import import_data_url
    with core.run_lock(root):
        state = core.load_project(root)
        ep = core._find(state, "episode", eid)
        _assert_action(state, ep, version)
        track = import_data_url(Path(root), data_url, title, provenance={"episode_id": eid})
        track["file"] = _rel(root, track["file"])
        with core.transaction(root) as state:
            ep = core._find(state, "episode", eid)
            _retire(ep)
            pp = ep["postproduction"]
            pp.setdefault("preferred_music_ids", []).append(track["id"])
            pp.pop("music_key", None)
            state.setdefault("music_library", {"version": 1, "tracks": []})["tracks"].append(track)
            core.event(state, "music_candidate_imported", id=track["id"], version=track["version"], sha256=track["sha256"])
    return core.load_project(root)


def decide_music(root, tid, version, sha256, feedback=None):
    from .music_library import approve_track, reject_track
    with core.run_lock(root), core.transaction(root) as state:
        library = state.setdefault("music_library", {"version": 1, "tracks": []})
        candidate = next((t for t in library["tracks"] if t.get("id") == tid and t.get("version") == version), None)
        if not candidate or candidate.get("status") != "awaiting_approval":
            raise core.GateError("该音乐候选当前不等待审批，请刷新后试听当前版本")
        if feedback is None:
            approve_track(library, tid, version, sha256, project=root)
        else:
            reject_track(library, tid, version, sha256, feedback, project=root)
        core.event(state, "music_approved" if feedback is None else "music_rejected", id=tid, version=version, sha256=sha256)
        state.update(status="ready", last_error=None)
    return core.load_project(root)


def request_generation(root, eid, version):
    with core.run_lock(root), core.transaction(root) as state:
        ep = core._find(state, "episode", eid)
        _assert_action(state, ep, version)
        ready, reason = music_configuration(state)
        if not ready:
            raise core.GateError(reason)
        pp = ep.setdefault("postproduction", {})
        if pp.get("music_generation_requested"):
            raise core.GateError("已有音乐生成请求，请继续原任务，不能重复提交")
        if any(t.get("status") == "awaiting_approval" for t in state.get("music_library", {}).get("tracks", [])):
            raise core.GateError("请先试听并处理当前音乐候选，再决定是否生成下一首；没有提交新任务")
        job = state["jobs"].get(ep.get("music_job_id", ""))
        if job and core.execution_unsettled(job):
            raise core.GateError("原音乐任务未结清，请恢复或核实原任务，不能再次提交")
        if ep.get("music_file") or job:
            pp.setdefault("music_generation_history", []).append({k: ep[k] for k in
                ("music_file", "music_job_id", "pending_music_file") if k in ep})
        for key in ("music_file", "music_job_id", "pending_music_file"):
            ep.pop(key, None)
        _retire(ep)
        pp.pop("music_key", None)
        pp.update(music_generation_requested=True, music_generation_serial=pp.get("music_generation_serial", 0) + 1)
        state.update(status="ready", paused=False, last_error=None)
        core.event(state, "music_generation_requested", id=eid, version=version)
    return core.load_project(root)


def _caption_context(state, ep, feedback, previous=None):
    # Current episode only. No novel, history, reviews, images or previous prompts.
    used = {aid for sh in ep["shots"] for aid in sh.get("asset_ids", [])}
    names = [a.get("name", "")[:30] for a in state["assets"] if a["id"] in used and a.get("kind") in {"character", "voice"}][:30]
    dialogue = "\n".join(sh.get("dialogue") or "" for sh in ep["shots"])
    context = {"names": names, "planned_dialogue_reference_only": dialogue[:3000], "feedback": feedback[:1500]}
    context["script_text_policy"] = "字幕候选优先提取邻近分镜的剧本台词；实际音轨独立核验后才可使用。数字与原识别不符时不得凭剧本改写。"
    if feedback and isinstance(previous, list):
        # Prior corrections can inform spelling, never timing, cue count or order.
        text = "\n".join(s.get("text", "") for s in previous if isinstance(s, dict))[:2000]
        if text:
            context["previous_corrected_text_reference_only"] = text
    return context


def _captions(root, ep, dry, brain_call, gate):
    from .connection import production_brain_config
    from .local_transcription import LocalTranscriber
    from .local_alignment import LocalSubtitleVerifier
    from .subtitle_sync import report_matches, run_sync
    state = gate(root)
    config = dict(state["settings"].get("subtitles", {}))
    pp = _episode(root, ep["id"]).get("postproduction", {})
    manual = pp.get("manual_subtitles")
    manual_mode = (isinstance(manual, dict) and manual.get("source_key") == pp.get("source_key")
                   and type(manual.get("revision")) is int and manual["revision"] <= pp.get("revision", 0))
    stored = pp.get("subtitles", {})
    if manual_mode:
        from .subtitle_sync import make_manual_report, caption_hash
        if (not manual.get("submitted_at") or manual.get("submitted_caption_hash") != caption_hash(manual["segments"])):
            raise core.GateError("手动字幕与明确保存的内容不一致，请在表格重新确认；没有自动复检或采用改动。")
        evidence = media.extract_evidence(dry, Path(root) / "evidence" / ("post-audio-" + pp["source_key"][:16]))
        if not evidence.get("audio"):
            raise core.GateError("锁定粗剪没有音轨，无法合成字幕；原视频已保留")
        audio = Path(evidence["audio"])
        corrected, audio_hash = copy.deepcopy(manual["segments"]), core.digest(audio)
        report = make_manual_report(corrected, audio_sha256=audio_hash, source_key=pp["source_key"],
                                    revision=manual["revision"], manual_record_hash=_hash(manual))
        _save(root, ep["id"], subtitle_key=_hash({"manual": manual, "audio": audio_hash}),
            subtitles={"status": "manual", "engine": "用户手动调整（不再自动复检）", "segments": corrected,
                "audio_sha256": audio_hash, "audio_file": _rel(root, audio), "sync_check": report,
                "previous_sync_check": copy.deepcopy(stored.get("previous_sync_check")),
                "note": "采用你填写的文字和时间，只重新合成供试看；未运行自动同步检查。"})
        return corrected
    if pp.get("revision_target") == "manual_music":
        if (isinstance(stored.get("segments"), list) and stored.get("audio_file") and stored.get("audio_sha256")
                and core.digest(core.safe_path(root, stored["audio_file"])) == stored["audio_sha256"]):
            return stored["segments"]
        raise core.GateError("原字幕音轨记录已变化，请先恢复当前字幕；没有重新识别或检查。")
    if pp.get("revision_target") == "music" and stored.get("status") == "ready":
        # A music-only revision reuses the independently verified captions.
        # No ASR, GPT spelling pass or alignment repair is needed for a remix.
        if (stored.get("audio_file") and stored.get("audio_sha256")
                and core.digest(core.safe_path(root, stored["audio_file"])) == stored["audio_sha256"]
                and report_matches(stored.get("sync_check"), stored.get("segments"),
                    audio_sha256=stored["audio_sha256"], source_key=pp["source_key"])):
            return stored["segments"]
    if not manual_mode:
        brain_config = production_brain_config(state["settings"]["brain"])
        if brain_config.get("provider", "codex") != "codex" or brain_config.get("billing") != "subscription":
            _wait(root, ep["id"], "configuration_required", missing_configuration=["字幕校对要求ChatGPT订阅连接；请在终端运行 nolan setup 选择订阅。"])
            return None
    try:
        if not manual_mode:
            transcriber = LocalTranscriber(config, cache_dir=Path(root) / "cache" / "local-asr",
                                          cancel_check=lambda: core.load_project(root)["paused"])
            transcriber.check_ready()
        verifier = LocalSubtitleVerifier(config, cache_dir=Path(root) / "cache" / "subtitle-sync",
                                        cancel_check=lambda: core.load_project(root)["paused"])
        verifier.check_ready()
    except Exception as exc:
        _wait(root, ep["id"], "configuration_required", missing_configuration=[str(exc)])
        return None
    pp = _episode(root, ep["id"]).get("postproduction", {})
    context = _caption_context(state, ep, pp.get("subtitle_feedback", ""), pp.get("subtitle_feedback_source"))
    evidence = media.extract_evidence(dry, Path(root) / "evidence" / ("post-audio-" + pp["source_key"][:16]))
    if not evidence.get("audio"):
        raise core.GateError("锁定粗剪没有音轨，无法识别实际对白；原视频已保留")
    audio = Path(evidence["audio"])
    audio_hash = core.digest(audio)
    _save(root, ep["id"], status="recognizing")
    raw = transcriber.transcribe(audio)
    if not raw and any((sh.get("dialogue") or "").strip() for sh in ep["shots"]):
        raise core.GateError("本集计划含对白，但本地识别结果为空；请检查原声音轨或本地模型。不会把剧本冒充字幕。")
    # Only current audio-derived timings are authoritative. A rework must never
    # feed stale subtitle_feedback_source timestamps back into text proofreading.
    from .subtitle_script import prefer_script
    candidates, script_selection = prefer_script(raw, ep)
    key = _hash({"source": pp["source_key"], "raw": raw, "candidates": candidates, "context": context, "policy": 3})
    current = _episode(root, ep["id"]).get("postproduction", {})
    stored = current.get("subtitles", {})
    same_proofreading = (current.get("subtitle_proofread_key") == key
                         or current.get("subtitle_key") == key)  # 0.6.4 text cache migration
    if (same_proofreading and stored.get("status") == "ready"
            and stored.get("sync_check", {}).get("checker_identity") == verifier.identity
            and report_matches(stored.get("sync_check"), stored.get("segments"),
                               audio_sha256=audio_hash, source_key=pp["source_key"])):
        return stored["segments"]
    # A repair changes time locally, not GPT wording. Keep already proofread
    # text and bounded repair progress on restart rather than spending again.
    if same_proofreading and isinstance(stored.get("segments"), list):
        corrected = copy.deepcopy(stored["segments"])
        previous_report = copy.deepcopy(stored.get("sync_check"))
    else:
        corrected, previous_report = [], None
        _save(root, ep["id"], status="proofreading", subtitles={"status": "proofreading",
              "engine": "Whisper 本地识别 + GPT订阅文字校对 + 独立中文声学同步检查", "segments": raw})
        for offset in range(0, len(candidates), 60):
            gate(root)
            batch = candidates[offset:offset + 60]
            corrected.extend(brain_call(root, "proofread_subtitles", batch, context,
                _cache_key=_hash({"method": "proofread_subtitles", "batch": batch, "context": context, "policy": 3})))
    base = {"engine": "Whisper 本地识别 + GPT订阅文字校对 + 独立中文声学同步检查",
        "raw_segments": raw, "script_selection": script_selection, "audio_sha256": audio_hash, "audio_file": _rel(root, audio), "asr_metadata": transcriber.last_metadata,
        "proofread_segments": copy.deepcopy(stored.get("proofread_segments", corrected)) if same_proofreading else copy.deepcopy(corrected),
        "note": "本地独立声学模型检查同步，异常自动重对齐后复检；GPT仅校对文字，未听音频。机器通过后仍需人工最终听审。"}

    def progress(report, candidates):
        gate(root)
        phase = "repairing_subtitle_sync" if report.get("status") == "repairing" else "checking_subtitle_sync"
        _save(root, ep["id"], status=phase, subtitle_proofread_key=key,
              subtitles={**base, "status": report.get("status", "checking"),
                         "segments": candidates, "sync_check": report,
                         "sync_metadata": verifier.last_metadata})

    checked, report = run_sync(corrected, verify=lambda cues: verifier.verify(audio, cues),
        repair=verifier.repair, previous_report=previous_report, on_progress=progress,
        audio_sha256=audio_hash, source_key=pp["source_key"], checker_identity=verifier.identity,
        duration=media.probe(audio)["duration"])
    passed = report_matches(report, checked, audio_sha256=audio_hash, source_key=pp["source_key"])
    subtitles = {**base, "status": "ready" if passed else "needs_attention", "segments": checked,
                 "sync_check": report, "sync_metadata": verifier.last_metadata}
    # A script/spelling correction can change a spoken word. Moving its clock
    # cannot repair that conflict. Try the original audio-derived wording once,
    # only when independently supported; never reset the timing-repair budget.
    if not passed:
        from .subtitle_text_recovery import recover_audio_text
        recovery_key = _hash({"policy": 1, "proofread_key": key, "raw": raw,
            "audio": audio_hash, "source": pp["source_key"], "checker": verifier.identity})
        old_recovery = stored.get("text_recovery", {})
        if old_recovery.get("key") == recovery_key:
            subtitles["text_recovery"] = copy.deepcopy(old_recovery)
        else:
            recovered, audit = recover_audio_text(checked, raw,
                verify=lambda cues: verifier.verify(audio, cues), duration=media.probe(audio)["duration"],
                audio_sha256=audio_hash, source_key=pp["source_key"], checker_identity=verifier.identity)
            audit["key"] = recovery_key
            subtitles["text_recovery"] = audit
            if audit.get("status") == "recovered":
                subtitles["previous_sync_check"] = copy.deepcopy(report)
                checked = recovered
                report = copy.deepcopy(audit["final_report"])
                report.update(attempts=subtitles["sync_check"]["attempts"],
                    input_caption_hash=subtitles["sync_check"]["input_caption_hash"])
                passed = report_matches(report, checked, audio_sha256=audio_hash, source_key=pp["source_key"])
                subtitles.update(status="ready" if passed else "needs_attention", segments=checked,
                    sync_check=report, sync_metadata=verifier.last_metadata)
    elif stored.get("text_recovery"):
        subtitles["text_recovery"] = copy.deepcopy(stored["text_recovery"])
        subtitles["previous_sync_check"] = copy.deepcopy(stored.get("previous_sync_check"))
    _save(root, ep["id"], subtitle_proofread_key=key,
          subtitle_key=_hash({"proofread_key": key, "segments": checked, "sync_check": report}), subtitles=subtitles)
    if not passed:
        _wait(root, ep["id"], "subtitle_sync_needs_attention", subtitles=subtitles)
        # Evidence uncertainty blocks approval, not the ability to inspect a
        # burned diagnostic preview. The final manifest still binds the failed
        # report, and check_final remains the authoritative approval gate.
        return checked
    return checked


def _music(root, ep, duration, captions, brain_call, gate, soundtrack):
    from . import music_library as library_api
    state = gate(root)
    pp = _episode(root, ep["id"]).get("postproduction", {})
    music = pp.get("music", {})
    if pp.get("music_feedback") and not isinstance(pp.get("music_feedback_source"), list):
        pp["music_feedback_source"] = copy.deepcopy(music.get("cues", []))
        _save(root, ep["id"], music_feedback_source=pp["music_feedback_source"])
    manual = pp.get("manual_music_plan")
    if (isinstance(manual, dict) and manual.get("source_key") == pp.get("source_key")
            and type(manual.get("revision")) is int and manual["revision"] <= pp.get("revision", 0)):
        if (not manual.get("submitted_at") or manual.get("submitted_plan_hash") != _hash(
                {"summary": manual.get("summary"), "cues": manual.get("cues")})):
            raise core.GateError("手动配乐与明确保存的内容不一致，请在表格重新确认；没有自动编排或采用改动。")
        validated = library_api.validate_cue_plan(manual, state["music_library"], duration, project=root)
        normalize = music.get("render_policy", {}).get("version") == 2
        key = _hash({"manual_plan": validated, "source_key": pp["source_key"], "policy": 2,
                     "normalize": normalize})
        if pp.get("music_key") == key and music.get("status") == "ready":
            bed = core.safe_path(root, music["file"])
            if core.digest(bed) == music["sha256"] and music.get("fingerprint") == validated["fingerprint"]:
                return bed
        _save(root, ep["id"], status="mixing_manual_music")
        out = Path(root) / "exports" / f"{ep['id']}-music-{key[:16]}.wav"
        result = library_api.render_music_bed(validated, state["music_library"], duration, out,
                                            project=root, normalize=normalize)
        _save(root, ep["id"], music_key=key, music={"status": "ready", "can_generate": False,
            "summary": validated["summary"], "cues": validated["cues"], "file": _rel(root, out),
            "sha256": result["sha256"], "duration": duration, "fingerprint": result["fingerprint"], "manual": True,
            "render_policy": result.get("render_policy", {}), "analysis": copy.deepcopy(music.get("analysis", {}))})
        return out
    # A subtitle-only request keeps the existing approved arrangement, including
    # when unrelated new tracks await audition. Validate its exact source bytes
    # before any catalog scan or candidate gate can replace the current record.
    if (pp.get("revision_target") in {"subtitles", "manual_subtitles"} and pp.get("music_key")
            and music.get("status") == "ready"):
        try:
            bed = core.safe_path(root, music["file"])
            validated = library_api.validate_cue_plan(
                {"summary": music["summary"], "cues": music["cues"]},
                state["music_library"], duration, project=root)
            if (core.digest(bed) != music["sha256"] or abs(music["duration"] - duration) > 0.05
                    or validated["fingerprint"] != music["fingerprint"]):
                raise ValueError("changed music")
            return bed
        except (ValueError, OSError, KeyError, TypeError):
            raise core.GateError("原配乐文件或编排校验失败，字幕已保留；请检查原文件或明确提交配乐调整，不会自动重新编配音乐。") from None
    if state["settings"].get("music", {}).get("mode") == "library":
        from .music_folder import scan_folder_locked
        scan_folder_locked(root)
        state = gate(root)
    pp = _episode(root, ep["id"]).get("postproduction", {})
    can_generate, reason = music_configuration(state)
    if state["settings"].get("music", {}).get("mode") == "library" and pp.get("music_generation_requested"):
        old_job = state["jobs"].get(ep.get("music_job_id", ""))
        if old_job and core.execution_unsettled(old_job):
            _wait(root, ep["id"], "configuration_required", missing_configuration=[
                "原音乐调用仍在途或结果待核实，请先核实原任务；切换本地曲库不会重发或改写原费用。"])
            return None
        _save(root, ep["id"], music_generation_requested=False,
              music_generation_cancelled_for_library=core.now())
        pp = _episode(root, ep["id"]).get("postproduction", {})
    if pp.get("music_generation_requested"):
        rejected = [t for t in state.get("music_library", {}).get("tracks", [])
                    if t.get("status") == "rejected" and t.get("provenance", {}).get("episode_id") == ep["id"]]
        feedback = rejected[-1].get("feedback", "") if rejected else ""
        candidate_prompt = ep.get("music_prompt", "适合本集的无歌词纯音乐")[:2200]
        if feedback:
            candidate_prompt += "\n用户对上版音乐的试听调整意见：" + feedback[:1500]
        candidate = soundtrack(root, {**_episode(root, ep["id"]), "post_music_candidate": True,
                                     "music_prompt": candidate_prompt})
        track = library_api.import_track(Path(root), candidate, ep["title"] + " · 配乐候选",
            prompt=candidate_prompt, provenance={"episode_id": ep["id"], "job_id": _episode(root, ep["id"]).get("music_job_id")})
        track["file"] = _rel(root, track["file"])
        with core.transaction(root) as state:
            state.setdefault("music_library", {"version": 1, "tracks": []})["tracks"].append(track)
            current_pp = core._find(state, "episode", ep["id"])["postproduction"]
            current_pp["music_generation_requested"] = False
            current_pp.setdefault("preferred_music_ids", []).append(track["id"])
        state = gate(root)
    library = state.setdefault("music_library", {"version": 1, "tracks": []})
    catalog = library_api.approved_tracks(library, project=root)
    candidates = any(t.get("status") == "awaiting_approval" for t in library["tracks"])
    if not catalog or candidates:
        _wait(root, ep["id"], "awaiting_music_approval" if candidates else "awaiting_music_selection",
            music={"status": "awaiting_approval" if candidates else "unconfigured", "can_generate": can_generate,
                   "summary": "首次音乐请试听确认；后续自动复用已批准音乐。", "cues": []},
            missing_configuration=[] if candidates else [reason or "请生成或导入首批音乐并试听确认。"])
        return None
    timeline = []
    cursor = 0.0
    for sh in ep["shots"]:
        length = max(1, round(media.probe(core.safe_path(root, sh["file"]))["duration"] * 30)) / 30
        timeline.append({"id": sh["id"], "start": round(cursor, 6), "end": round(cursor + length, 6),
                         "action": sh.get("prompt", "")[:220]})
        cursor += length
    previous = None
    episodes = state["episodes"]
    idx = next(i for i, e in enumerate(episodes) if e["id"] == ep["id"])
    if idx:
        prior = episodes[idx - 1]
        previous = {"id": prior["id"], "ending_cues": prior.get("postproduction", {}).get("music", {}).get("cues", [])[-3:]}
    context = {"episode": {"id": ep["id"], "title": ep["title"], "summary": ep.get("summary", "")[:800]},
        "duration": duration, "timeline": timeline, "music_prompt": ep.get("music_prompt", "")[:2200],
        "speech_windows": [{"start": s["start"], "end": s["end"]} for s in captions],
        "catalog": catalog[:6], "previous": previous, "feedback": pp.get("music_feedback", "")}
    preferred = [tid for tid in _episode(root, ep["id"]).get("postproduction", {}).get("preferred_music_ids", [])
                 if any(t["id"] == tid for t in catalog)]
    if preferred:
        context["preferred_track_ids"] = preferred[-6:]
    previous_ids = [cue["track_id"] for cue in (previous or {}).get("ending_cues", [])]
    retained = list(dict.fromkeys(preferred[-6:] + [c["track_id"] for c in
        (pp.get("music_feedback_source") or []) if any(t["id"] == c.get("track_id") for t in catalog)]))[:6]
    context["catalog"] = library_api.select_catalog(catalog,
        ep.get("summary", "")[:1000] + "\n" + ep.get("music_prompt", "")[:2200] + "\n" + pp.get("music_feedback", "")[:2000],
        preferred_ids=retained, previous_ids=previous_ids, limit=6)
    from .music_analysis import catalog_analysis, POLICY_VERSION
    _save(root, ep["id"], status="analyzing_music")
    for track in context["catalog"]:
        gate(root)
        item = library_api._find(library, track["id"], track["version"])
        measured = catalog_analysis(
            item, library_api._track_path(item, Path(root)),
            cache_dir=Path(root) / "cache" / "music-analysis")
        track["audio_analysis"] = {"regions": measured["regions"]}
        # Keep the model input local to this episode and a compact shortlist.
        track["prompt"] = track.get("prompt", "")[:200]
    context["mixing_policy"] = {"normalization_target_lufs": -18, "duck_max_reduction_db": 6,
        "gain_reference": "gain_db是统一响度后基准的衰减；通常-3至-10，不沿用旧的-20至-12。",
        "selection": "根据实测候选区域的强度、可用长度和边界选source_start，不默认从0开始；声学特征不是已试听的情绪/乐句结论。"}
    # Feedback may ask to retain a known arrangement while improving excerpts.
    if pp.get("music_feedback"):
        reference = pp.get("music_feedback_source")
        if not isinstance(reference, list):
            # Migrate a pre-policy pending revision once, before its first GPT
            # call. Current execution outputs must never change their own key.
            reference = copy.deepcopy(music.get("cues", []))
            _save(root, ep["id"], music_feedback_source=reference)
        context["current_cues_reference"] = [{k: c[k] for k in
            ("track_id", "start", "end", "source_start") if k in c} for c in reference]
    context = _bound_music_context(context, retained)
    key = _hash({"context": context, "policy": 2, "analysis_policy": POLICY_VERSION})
    music = pp.get("music", {})
    if pp.get("music_key") == key and music.get("status") == "ready":
        try:
            bed = core.safe_path(root, music["file"])
            if core.digest(bed) == music["sha256"] and abs(music["duration"] - duration) <= 0.05:
                validated = library_api.validate_cue_plan({"summary": music["summary"], "cues": music["cues"]}, library, duration, project=root)
                if validated["fingerprint"] != music["fingerprint"]:
                    raise ValueError("配乐编排已变化")
                return bed
        except (ValueError, OSError, KeyError):
            pass
    _save(root, ep["id"], status="arranging_music", music={**music, "status": "arranging", "can_generate": can_generate})
    plan = brain_call(root, "arrange_music", context, _cache_key=key)
    if not plan.get("cues"):
        raise core.GateError("GPT未安排音乐片段，已保留字幕与视频，请调整本集配乐意见")
    if preferred and not any(c.get("track_id") in preferred for c in plan["cues"]):
        raise core.GateError("GPT未采用本集新确认的音乐，请在配乐意见中明确如何使用后重新编排；未产生新音乐费用")
    validated = library_api.validate_cue_plan(plan, library, duration, project=root)
    out = Path(root) / "exports" / f"{ep['id']}-music-{key[:16]}.wav"
    result = library_api.render_music_bed(validated, library, duration, out, project=root, normalize=True)
    _save(root, ep["id"], music_key=key, music={"status": "ready", "can_generate": can_generate,
        "summary": validated["summary"], "cues": validated["cues"], "file": _rel(root, out),
        "sha256": result["sha256"], "duration": duration, "fingerprint": result["fingerprint"],
        "render_policy": result.get("render_policy", {}),
        "analysis": {"status": "completed", "policy_version": POLICY_VERSION,
            "tracks_analyzed": len(context["catalog"]), "cues_analyzed": len(validated["cues"]),
            "summary": "已分析实际音乐的能量变化与可截区域，GPT据此选段；原曲统一到-18 LUFS参考，峰值受限时保留差异。"}})
    return out


def _bound_music_context(context, preferred_ids):
    """Keep full current feedback, a compact timeline and necessary candidates.

    Large libraries, waveform envelopes, history and prior reports never enter
    a selection request. Preserve every explicitly preferred track in the
    shortlist while trimming optional catalog entries before the 20k hard cap.
    """
    context = copy.deepcopy(context)
    required = set(preferred_ids)
    catalog = context["catalog"]
    while len(json.dumps(context, ensure_ascii=False)) > 16000 and len(catalog) > max(3, len(required)):
        removable = next((i for i in range(len(catalog) - 1, -1, -1) if catalog[i]["id"] not in required), None)
        if removable is None:
            break
        catalog.pop(removable)
    if len(json.dumps(context, ensure_ascii=False)) > 18000:
        for row in context.get("timeline", []):
            row["action"] = row.get("action", "")[:120]
        for track in catalog:
            track["prompt"] = track.get("prompt", "")[:120]
    if len(json.dumps(context, ensure_ascii=False)) > 20000:
        raise core.GateError("本集音乐必要上下文仍超出限制，已保留视频、字幕和配乐，未提交GPT。")
    return context


def finish(root, ep, *, brain_call, gate, soundtrack):
    try:
        return _finish(root, ep, brain_call=brain_call, gate=gate, soundtrack=soundtrack)
    except Exception as exc:
        message = str(exc) if isinstance(exc, (ValueError, FileNotFoundError)) or getattr(exc, "safe_message", False) else "本集后期执行失败，请检查配置和本地文件；视频已保留。"
        _save(root, ep["id"], status="error", final_ready=False, error=message)
        raise


def _finish(root, ep, *, brain_call, gate, soundtrack):
    state = gate(root)
    for shot in ep["shots"]:
        if not core.human_approved_shot(shot):
            raise core.GateError("后期只处理已逐段人工批准的本集视频")
        core._check_shot_artifact(root, state, shot)
    source = [{"id": sh["id"], "version": sh["version"], "sha256": sh["sha256"]} for sh in ep["shots"]]
    source_key = _hash(source)
    pp = ep.get("postproduction", {})
    # A diagnostic export is reusable evidence, not permission to repeat local
    # work or spend on another music arrangement when Continue is pressed.
    if (pp.get("preview_kind") == "subtitle_diagnostic"
            and pp.get("status") == "subtitle_sync_needs_attention"
            and pp.get("source_key") == source_key and pp.get("final_ready")
            and pp.get("final_file") == ep.get("file") and pp.get("final_version") == ep.get("version")
            and pp.get("final_sha256") == ep.get("sha256")
            and ep.get("render_manifest", {}).get("postproduction", {}).get("revision") == pp.get("revision", 0)
            and ep.get("render_manifest", {}).get("postproduction", {}).get("subtitle_sync_hash")
                == _hash(pp.get("subtitles", {}).get("sync_check"))
            and ep.get("render_manifest", {}).get("postproduction", {}).get("subtitle_content_hash")
                == _hash(pp.get("subtitles", {}).get("segments"))
            and core.digest(core.safe_path(root, ep["file"])) == ep["sha256"]):
        with core.transaction(root) as current:
            core._find(current, "episode", ep["id"])["status"] = "awaiting_review"
            current.update(status="awaiting_postproduction", last_error=None)
        return False
    if (pp.get("revision_target") in {"manual_music", "manual_subtitles"}
            and pp.get("source_key") != source_key):
        raise core.GateError("本集源片段已变化，请先更新后期，再编辑当前版本。")
    if ep.get("file") and not any(
            old.get("file") == ep["file"] and old.get("version") == ep["version"]
            and old.get("sha256") == ep.get("sha256") for old in ep.get("history", [])):
        # A legacy export can enter the new machine check via Continue without
        # an explicit rework request. Preserve its complete export provenance.
        with core.transaction(root) as current:
            _retire(core._find(current, "episode", ep["id"]))
    if pp.get("source_key") != source_key:
        _save(root, ep["id"], subtitle_key=None, subtitle_proofread_key=None, music_key=None, subtitle_feedback_source=None,
              manual_music_plan=None, manual_subtitles=None, music_feedback_source=None, subtitles={}, music={},
              source_key=source_key, source_manifest=source)
    _save(root, ep["id"], enabled=True, ready=True, final_ready=False, error=None,
          status="assembling", missing_configuration=[], revision=pp.get("revision", 0))
    dry = Path(root) / "exports" / f"{ep['id']}-post-dry-{source_key[:16]}.mp4"
    current = _episode(root, ep["id"]).get("postproduction", {})
    if not dry.is_file() or current.get("dry_sha256") != core.digest(dry):
        media.compose([core.safe_path(root, sh["file"]) for sh in ep["shots"]], dry)
    duration = media.probe(dry)["duration"]
    _save(root, ep["id"], dry_file=_rel(root, dry), dry_sha256=core.digest(dry), duration=duration)
    captions = _captions(root, ep, dry, brain_call, gate)
    if captions is None:
        return False
    music = _music(root, ep, duration, captions, brain_call, gate, soundtrack)
    if music is None:
        return False
    gate(root)
    _save(root, ep["id"], status="rendering")
    fresh = _episode(root, ep["id"])
    version = fresh["version"] + 1
    out = Path(root) / "exports" / f"{ep['id']}-v{version}.mp4"
    if out.exists():
        out = out.with_name(out.stem + "-" + source_key[:12] + out.suffix)
        if out.exists():
            raise core.GateError("后期输出文件已存在，请核实原导出结果；没有覆盖任何视频")
    mixing = _episode(root, ep["id"])["postproduction"]
    music_mix = None
    if mixing.get("music", {}).get("render_policy", {}).get("version") == 2:
        manual = mixing.get("revision_target") in {"manual_music", "manual_subtitles"}
        music_mix = {"policy_version": 1, "speech_windows": [
            {"start": c["start"], "end": c["end"]} for c in captions],
            "cues": mixing["music"]["cues"], "duck_db": 6, "adaptive": not manual, "audit": not manual,
            "binding": {"source_key": source_key, "dry_sha256": mixing["dry_sha256"],
                "subtitle_content_hash": _hash(captions), "music_key": mixing["music_key"],
                "source_analysis_used": True}}
        if manual:
            previous_mix = mixing.get("music_mix_check", {})
            if previous_mix.get("binding", {}).get("source_key") != source_key:
                raise core.GateError("原混音基准与本集视频不一致，不能在手动调整时重置音乐；已有成片已保留。")
            artifacts = {}
            for name, sha_key in (("prepared_music_file", "prepared_music_sha256"),
                                  ("automation_file", "automation_sha256")):
                if not previous_mix.get(name):
                    raise core.GateError("原混音基准文件缺失，不能在手动调整时改变已有音量；请先恢复本集配乐。")
                artifact = core.safe_path(root, previous_mix[name])
                if core.digest(artifact) != previous_mix.get(sha_key):
                    raise core.GateError("原混音基准文件已变化，手动调整未重新检查或改写音乐。")
                artifacts[name] = artifact
            if (mixing.get("revision_target") == "manual_subtitles"
                    and previous_mix.get("bed_sha256") == mixing["music"]["sha256"]):
                music_mix["reuse_prepared_music"] = {
                    "file": str(artifacts["prepared_music_file"]), "sha256": previous_mix["prepared_music_sha256"],
                    "automation_file": str(artifacts["automation_file"]),
                    "automation_sha256": previous_mix["automation_sha256"]}
            else:
                music_mix["baseline_automation"] = json.loads(artifacts["automation_file"].read_text(encoding="utf-8"))
        if not manual:
            previous_mix = mixing.get("music_mix_check", {})
            if (previous_mix.get("binding") == music_mix["binding"]
                    and previous_mix.get("bed_sha256") == mixing["music"]["sha256"]):
                music_mix["repair_attempt"] = previous_mix.get("repair_attempts", 0)
                music_mix["local_corrections_db"] = previous_mix.get("local_corrections_db", {})

            def mix_progress(report):
                gate(root)
                _save(root, ep["id"], status="repairing_music_mix" if report.get("status") == "repairing" else "checking_music_mix",
                      music_mix_check=copy.deepcopy(report))

            music_mix["on_progress"] = mix_progress
    result = media.compose([core.safe_path(root, sh["file"]) for sh in ep["shots"]], out,
                           subtitles=captions, music=music, music_gain=1.0, **(
                               {"music_mix": music_mix} if music_mix else {}))
    if captions and not result.get("burned_subtitles"):
        raise core.GateError("字幕未能烧录：请安装带libass字幕滤镜的FFmpeg和中文字体；SRT和视频已保留")
    post = _episode(root, ep["id"])["postproduction"]
    from .music_library import write_music_credits
    credits = write_music_credits(
        {"cues": post["music"]["cues"]}, core.load_project(root)["music_library"],
        out.with_suffix(".credits.txt"), project=root)
    with core.transaction(root) as state:
        current = core._find(state, "episode", ep["id"])
        # A late edit cannot be stamped into a manifest for another source.
        if source != [{"id": sh["id"], "version": sh["version"], "sha256": sh["sha256"]} for sh in current["shots"]] or any(not core.human_approved_shot(sh) for sh in current["shots"]):
            raise core.GateError("本集片段已变更，保留此次导出，需重新生成后期")
        pp = current["postproduction"]
        mix_report = result.get("music_mix")
        if music_mix:
            if not isinstance(mix_report, dict):
                raise core.GateError("本集混音检查记录缺失，已保留导出，请检查后再审批。")
            mix_report.update(normalization_target_lufs=-18, source_analysis_used=True,
                              checked_at=core.now())
            for key in ("prepared_music_file", "automation_file"):
                if mix_report.get(key):
                    mix_report[key] = _rel(root, mix_report[key])
            pp["music_mix_check"] = copy.deepcopy(mix_report)
        else:
            pp.pop("music_mix_check", None)
        from .subtitle_sync import report_matches, manual_report_matches
        sync_passed = (report_matches(pp["subtitles"]["sync_check"], captions,
            audio_sha256=pp["subtitles"]["audio_sha256"], source_key=source_key)
            or manual_report_matches(pp["subtitles"]["sync_check"], captions,
                audio_sha256=pp["subtitles"]["audio_sha256"], source_key=source_key))
        mix_passed = not music_mix or mix_report.get("status") in {"passed", "manual"}
        pp.update(status=("awaiting_final_approval" if mix_passed else "music_mix_needs_attention") if sync_passed else "subtitle_sync_needs_attention",
                  manual_preview=not sync_passed, preview_kind="final" if sync_passed else "subtitle_diagnostic",
                  final_ready=True, final_file=_rel(root, out),
                  final_sha256=core.digest(out), final_version=version, missing_configuration=[])
        pp["music"].update(credits_file=_rel(root, credits["file"]), credits_sha256=credits["sha256"])
        pp["subtitles"]["file"] = _rel(root, result["subtitles"]) if result.get("subtitles") else None
        current.update(version=version, file=_rel(root, out), sha256=core.digest(out),
            status="awaiting_review", duration=result["duration"],
            subtitle_status="manual_adopted" if pp["subtitles"]["sync_check"].get("status") == "manual" else "local_asr_subscription_proofread",
            subtitle_file=pp["subtitles"]["file"], subtitles_burned=result.get("burned_subtitles", False),
            music_status="approved_library_mixed", render_warnings=result.get("warnings", []),
            review_note="请完整观看带字幕配乐的本集，审批对白字幕、音乐音量与衔接；GPT未替代人工听审。",
            render_manifest={"clips": [{"id": sh["id"], "sha256": sh["sha256"]} for sh in current["shots"]],
                "music": _rel(root, music), "subtitles": captions,
                "postproduction": {"source_key": source_key, "subtitle_key": pp["subtitle_key"],
                    "music_key": pp["music_key"], "music_sha256": pp["music"]["sha256"],
                    "music_fingerprint": pp["music"]["fingerprint"],
                    **({"music_mix_hash": _hash(mix_report)} if music_mix else {}),
                    "subtitle_content_hash": _hash(captions),
                    "subtitle_sync_hash": _hash(pp["subtitles"]["sync_check"]),
                    "credits_sha256": credits["sha256"],
                    "srt_sha256": core.digest(result["subtitles"]) if result.get("subtitles") else None,
                    "revision": pp.get("revision", 0)}})
        state.update(status="awaiting_episode_review" if sync_passed and mix_passed else "awaiting_postproduction", last_error=None)
    return True


def check_final(root, state, ep):
    """Reject a stale subtitle, mix, cue plan or final-version approval."""
    from .music_library import validate_cue_plan
    from .subtitle_sync import report_matches, manual_report_matches
    pp = ep.get("postproduction", {})
    manifest = ep.get("render_manifest", {}).get("postproduction", {})
    if pp.get("preview_kind") == "subtitle_diagnostic":
        raise core.GateError("字幕同步自动检查尚未通过，待核验预览不能最终批准；请先处理字幕。")
    source = [{"id": sh["id"], "version": sh["version"], "sha256": sh["sha256"]} for sh in ep["shots"]]
    sub = pp.get("subtitles", {})
    report = sub.get("sync_check", {})
    manual_valid = (manual_report_matches(report, sub.get("segments"),
                    audio_sha256=sub.get("audio_sha256"), source_key=_hash(source))
                    and isinstance(pp.get("manual_subtitles"), dict)
                    and report.get("manual_record_hash") == _hash(pp["manual_subtitles"])
                    and pp["manual_subtitles"].get("submitted_at")
                    and pp["manual_subtitles"].get("submitted_caption_hash") == report.get("caption_hash")
                    and pp["manual_subtitles"].get("segments") == sub.get("segments"))
    if report.get("status") != "passed" and not manual_valid:
        raise core.GateError("字幕同步自动检查尚未通过，暂不能最终批准；请查看本集字幕检查报告。")
    if ((not report_matches(report, sub.get("segments"), audio_sha256=sub.get("audio_sha256"), source_key=_hash(source)) and not manual_valid)
            or not sub.get("audio_sha256") or manifest.get("subtitle_sync_hash") != _hash(report)
            or not sub.get("audio_file") or core.digest(core.safe_path(root, sub["audio_file"])) != sub["audio_sha256"]):
        raise core.GateError("字幕同步检查记录与当前音轨或字幕不一致，请重新检查后再批准。")
    if not pp.get("final_ready") or pp.get("final_sha256") != ep.get("sha256") or pp.get("final_version") != ep["version"]:
        raise core.GateError("本集后期尚未完成或导出版本已变化")
    if pp.get("music", {}).get("render_policy", {}).get("version") == 2:
        mix = pp.get("music_mix_check", {})
        if mix.get("status") not in {"passed", "manual"}:
            raise core.GateError("配乐混音数值检查仍有疑点，请查看报告后提交配乐意见或手动调整，再试听确认。")
        if (manifest.get("music_mix_hash") != _hash(mix)
                or mix.get("final_sha256") != ep.get("sha256")
                or mix.get("bed_sha256") != pp["music"].get("sha256")
                or mix.get("binding", {}).get("source_key") != _hash(source)
                or mix.get("binding", {}).get("dry_sha256") != pp.get("dry_sha256")
                or mix.get("binding", {}).get("subtitle_content_hash") != _hash(sub.get("segments"))
                or mix.get("binding", {}).get("music_key") != pp.get("music_key")):
            raise core.GateError("配乐检查记录与本版视频、音乐或字幕不一致，请重新处理本集配乐。")
        if mix.get("status") == "manual" and pp.get("revision_target") not in {"manual_music", "manual_subtitles"}:
            raise core.GateError("没有对应的手动调整记录，不能将自动混音检查改为人工采用。")
        for name, sha_key in (("prepared_music_file", "prepared_music_sha256"),
                              ("automation_file", "automation_sha256")):
            if not mix.get(name) or core.digest(core.safe_path(root, mix[name])) != mix.get(sha_key):
                raise core.GateError("本集混音基准文件缺失或已改变，请恢复当前版本后再审批。")
    if (pp.get("subtitles", {}).get("status") not in {"ready", "manual"} or pp.get("music", {}).get("status") != "ready"
            or not pp.get("music", {}).get("cues")):
        raise core.GateError("本集字幕或音乐编排尚未就绪")
    if (manifest.get("source_key") != _hash(source) or manifest.get("subtitle_key") != pp.get("subtitle_key")
            or manifest.get("music_key") != pp.get("music_key") or manifest.get("revision") != pp.get("revision", 0)):
        raise core.GateError("字幕或音乐编排已更新，请重新导出本集后审批")
    if core.digest(core.safe_path(root, pp["music"]["file"])) != manifest.get("music_sha256"):
        raise core.GateError("本集配乐文件已变化，请重新合成")
    if ep.get("render_manifest", {}).get("subtitles") != pp["subtitles"].get("segments"):
        raise core.GateError("本集字幕内容与导出记录不一致")
    if _hash(pp["subtitles"].get("segments")) != manifest.get("subtitle_content_hash"):
        raise core.GateError("本集字幕文字与导出内容不一致，请重新合成")
    if pp["subtitles"].get("segments"):
        if (not ep.get("subtitle_file") or ep["subtitle_file"] != pp["subtitles"].get("file")
                or core.digest(core.safe_path(root, ep["subtitle_file"])) != manifest.get("srt_sha256")):
            raise core.GateError("本集字幕文件已变化或缺失，请重新合成")
    if pp["subtitles"].get("segments") and not ep.get("subtitles_burned"):
        raise core.GateError("本集字幕尚未加入成片")
    plan = validate_cue_plan({"summary": pp["music"]["summary"], "cues": pp["music"]["cues"]},
                            state["music_library"], pp["duration"], project=root)
    if plan["fingerprint"] != pp["music"].get("fingerprint") or plan["fingerprint"] != manifest.get("music_fingerprint"):
        raise core.GateError("本集配乐安排与导出工作轨不一致，请重新合成")
    if manifest.get("credits_sha256") and (not pp["music"].get("credits_file")
            or core.digest(core.safe_path(root, pp["music"]["credits_file"])) != manifest.get("credits_sha256")
            or pp["music"].get("credits_sha256") != manifest.get("credits_sha256")):
        raise core.GateError("本集配乐署名文件已变化，请重新合成")
