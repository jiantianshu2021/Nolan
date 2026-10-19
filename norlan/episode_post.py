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
            if target in {"subtitles", "all"}:
                pp["subtitle_feedback"] = feedback.strip()
                pp["subtitle_feedback_source"] = copy.deepcopy(pp.get("subtitles", {}).get("segments"))
                pp.pop("subtitle_key", None)
            if target in {"music", "all"}:
                pp["music_feedback"] = feedback.strip()
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


def _caption_context(state, ep, feedback):
    # Current episode only. No novel, history, reviews, images or previous prompts.
    used = {aid for sh in ep["shots"] for aid in sh.get("asset_ids", [])}
    names = [a.get("name", "")[:30] for a in state["assets"] if a["id"] in used and a.get("kind") in {"character", "voice"}][:30]
    dialogue = "\n".join(sh.get("dialogue", "") for sh in ep["shots"])
    return {"names": names, "planned_dialogue_reference_only": dialogue[:4500], "feedback": feedback[:2000]}


def _captions(root, ep, dry, brain_call, gate):
    from .connection import production_brain_config
    from .local_transcription import LocalTranscriber
    state = gate(root)
    config = dict(state["settings"].get("subtitles", {}))
    brain_config = production_brain_config(state["settings"]["brain"])
    if brain_config.get("provider", "codex") != "codex" or brain_config.get("billing") != "subscription":
        _wait(root, ep["id"], "configuration_required", missing_configuration=["字幕校对要求ChatGPT订阅连接；请在终端运行 nolan setup 选择订阅。"])
        return None
    try:
        transcriber = LocalTranscriber(config, cache_dir=Path(root) / "cache" / "local-asr")
        transcriber.check_ready()
    except Exception as exc:
        _wait(root, ep["id"], "configuration_required", missing_configuration=[str(exc)])
        return None
    pp = _episode(root, ep["id"]).get("postproduction", {})
    context = _caption_context(state, ep, pp.get("subtitle_feedback", ""))
    evidence = media.extract_evidence(dry, Path(root) / "evidence" / ("post-audio-" + pp["source_key"][:16]))
    if not evidence.get("audio"):
        raise core.GateError("锁定粗剪没有音轨，无法识别实际对白；原视频已保留")
    audio = Path(evidence["audio"])
    _save(root, ep["id"], status="recognizing")
    raw = transcriber.transcribe(audio)
    if not raw and any(sh.get("dialogue", "").strip() for sh in ep["shots"]):
        raise core.GateError("本集计划含对白，但本地识别结果为空；请检查原声音轨或本地模型。不会把剧本冒充字幕。")
    base = pp.get("subtitle_feedback_source") if pp.get("subtitle_feedback") else None
    base = base if isinstance(base, list) else raw
    key = _hash({"source": pp["source_key"], "raw": raw, "base": base, "context": context, "policy": 1})
    # Registered results are immutable and safe to reuse on restarts.
    current = _episode(root, ep["id"]).get("postproduction", {})
    if current.get("subtitle_key") == key and current.get("subtitles", {}).get("status") == "ready":
        return current["subtitles"]["segments"]
    _save(root, ep["id"], status="proofreading", subtitles={"status": "proofreading", "engine": "whisper.cpp + GPT订阅校对", "segments": raw})
    corrected = []
    for offset in range(0, len(base), 60):
        gate(root)
        batch = base[offset:offset + 60]
        corrected.extend(brain_call(root, "proofread_subtitles", batch, context,
            _cache_key=_hash({"method": "proofread_subtitles", "batch": batch, "context": context, "policy": 1})))
    _save(root, ep["id"], subtitle_key=key, subtitles={"status": "ready", "engine": "whisper.cpp + GPT订阅校对",
        "segments": corrected, "raw_segments": raw, "audio_sha256": core.digest(audio),
        "asr_metadata": transcriber.last_metadata,
        "note": "识别实际音频；GPT仅校对文字，未听音频。最终请审听整集。"})
    return corrected


def _music(root, ep, duration, captions, brain_call, gate, soundtrack):
    from . import music_library as library_api
    state = gate(root)
    pp = _episode(root, ep["id"]).get("postproduction", {})
    can_generate, reason = music_configuration(state)
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
        "catalog": catalog[:12], "previous": previous, "feedback": pp.get("music_feedback", "")[:2000]}
    preferred = [tid for tid in _episode(root, ep["id"]).get("postproduction", {}).get("preferred_music_ids", [])
                 if any(t["id"] == tid for t in catalog)]
    if preferred:
        context["preferred_track_ids"] = preferred[-6:]
        # A deliberately imported replacement must remain within the small catalog.
        context["catalog"] = sorted(catalog, key=lambda t: t["id"] not in preferred)[:12]
    key = _hash({"context": context, "policy": 1})
    music = pp.get("music", {})
    if pp.get("music_key") == key and music.get("status") == "ready":
        try:
            bed = core.safe_path(root, music["file"])
            if core.digest(bed) == music["sha256"]:
                library_api.validate_cue_plan({"summary": music["summary"], "cues": music["cues"]}, library, duration, project=root)
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
    result = library_api.render_music_bed(validated, library, duration, out, project=root)
    _save(root, ep["id"], music_key=key, music={"status": "ready", "can_generate": can_generate,
        "summary": validated["summary"], "cues": validated["cues"], "file": _rel(root, out),
        "sha256": result["sha256"], "duration": duration, "fingerprint": result["fingerprint"]})
    return out


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
    if pp.get("source_key") != source_key:
        _save(root, ep["id"], subtitle_key=None, music_key=None, subtitle_feedback_source=None, subtitles={}, music={},
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
    result = media.compose([core.safe_path(root, sh["file"]) for sh in ep["shots"]], out,
                           subtitles=captions, music=music, music_gain=1.0)
    if captions and not result.get("burned_subtitles"):
        raise core.GateError("字幕未能烧录：请安装带libass字幕滤镜的FFmpeg和中文字体；SRT和视频已保留")
    with core.transaction(root) as state:
        current = core._find(state, "episode", ep["id"])
        # A late edit cannot be stamped into a manifest for another source.
        if source != [{"id": sh["id"], "version": sh["version"], "sha256": sh["sha256"]} for sh in current["shots"]] or any(not core.human_approved_shot(sh) for sh in current["shots"]):
            raise core.GateError("本集片段已变更，保留此次导出，需重新生成后期")
        pp = current["postproduction"]
        pp.update(status="awaiting_final_approval", final_ready=True, final_file=_rel(root, out),
                  final_sha256=core.digest(out), final_version=version, missing_configuration=[])
        pp["subtitles"]["file"] = _rel(root, result["subtitles"]) if result.get("subtitles") else None
        current.update(version=version, file=_rel(root, out), sha256=core.digest(out),
            status="awaiting_review", duration=result["duration"], subtitle_status="local_asr_subscription_proofread",
            subtitle_file=pp["subtitles"]["file"], subtitles_burned=result.get("burned_subtitles", False),
            music_status="approved_library_mixed", render_warnings=result.get("warnings", []),
            review_note="请完整观看带字幕配乐的本集，审批对白字幕、音乐音量与衔接；GPT未替代人工听审。",
            render_manifest={"clips": [{"id": sh["id"], "sha256": sh["sha256"]} for sh in current["shots"]],
                "music": _rel(root, music), "subtitles": captions,
                "postproduction": {"source_key": source_key, "subtitle_key": pp["subtitle_key"],
                    "music_key": pp["music_key"], "music_sha256": pp["music"]["sha256"],
                    "music_fingerprint": pp["music"]["fingerprint"],
                    "subtitle_content_hash": _hash(captions),
                    "srt_sha256": core.digest(result["subtitles"]) if result.get("subtitles") else None,
                    "revision": pp.get("revision", 0)}})
        state.update(status="awaiting_episode_review", last_error=None)
    return True


def check_final(root, state, ep):
    """Reject a stale subtitle, mix, cue plan or final-version approval."""
    from .music_library import validate_cue_plan
    pp = ep.get("postproduction", {})
    manifest = ep.get("render_manifest", {}).get("postproduction", {})
    source = [{"id": sh["id"], "version": sh["version"], "sha256": sh["sha256"]} for sh in ep["shots"]]
    if not pp.get("final_ready") or pp.get("final_sha256") != ep.get("sha256") or pp.get("final_version") != ep["version"]:
        raise core.GateError("本集后期尚未完成或导出版本已变化")
    if (pp.get("subtitles", {}).get("status") != "ready" or pp.get("music", {}).get("status") != "ready"
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
