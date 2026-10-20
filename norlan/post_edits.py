"""Atomic human edits to existing episode music and subtitle rows.

This module records an explicit local postproduction revision. It does not
render, call a model/provider, alter source files, select tracks or approve a
film. Clients may edit only timing, gain and subtitle text, never identities or
paths. Actual media and music approvals are checked before any state mutation.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re

from . import core


class EditError(ValueError):
    """Fixed user-facing validation copy; never interpolate submitted content."""

    safe_message = True


def _number(value, *, minimum=0., maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise EditError("时间和音量必须填写有限数字，不能留空或使用无效数值。")
    value = float(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise EditError("手动设置超出允许范围；时间须在本集内，音量须在 −60 至 0 dB 之间。")
    return value


def _duration(ep, pp):
    for value in (pp.get("duration"), ep.get("duration")):
        if type(value) in (int, float) and math.isfinite(value) and value > 0:
            return float(value)
    raise EditError("尚无本集实测时长，请先完成原声拼接后再手动编辑。")


def _pending(pp, key):
    candidate = pp.get(key)
    return (isinstance(candidate, dict) and candidate.get("source_key") == pp.get("source_key")
            and type(candidate.get("revision")) is int
            and candidate["revision"] == pp.get("revision", 0))


def _rows(pp, section):
    manual = "manual_music_plan" if section == "music" else "manual_subtitles"
    field = "cues" if section == "music" else "segments"
    holder = pp[manual] if _pending(pp, manual) else pp.get(section, {})
    current = holder.get(field) if isinstance(holder, dict) else None
    limit = 64 if section == "music" else 10_000
    if not isinstance(current, list) or not 0 < len(current) <= limit or not all(isinstance(row, dict) for row in current):
        raise EditError("本集尚无可编辑的配乐或字幕条目，请先完成对应后期步骤。")
    return current


def _check_rows(edits, base, section, duration):
    if not isinstance(edits, list) or len(edits) != len(base):
        raise EditError("请提交当前全部条目，不能增删或遗漏；页面可能已更新，请刷新。")
    required = {"index", "start", "end", "gain_db" if section == "music" else "text"}
    checked, previous_end = [], 0.
    for index, row in enumerate(edits):
        if (not isinstance(row, dict) or set(row) != required
                or type(row.get("index")) is not int or row["index"] != index):
            raise EditError("编辑条目必须按当前顺序完整提交，只能修改时间、音量或字幕文字。")
        start = round(_number(row["start"], maximum=duration), 6)
        end = round(_number(row["end"], maximum=duration), 6)
        if end <= start:
            raise EditError("每条的结束时间必须晚于开始时间。")
        if section == "music":
            checked.append({"start": start, "end": end,
                            "gain_db": _number(row["gain_db"], minimum=-60., maximum=0.)})
        else:
            text = row["text"]
            if not isinstance(text, str) or not text.strip() or len(text) > 4000 or "\x00" in text:
                raise EditError("每条字幕需填写 1–4000 字的有效文字，不能含空字符。")
            if start < previous_end:
                raise EditError("字幕时间须按顺序排列且不能重叠，请检查相邻字幕的起止时间。")
            checked.append({"start": start, "end": end, "text": text.strip()})
            previous_end = end
    return checked


def _music_plan(root, state, pp, base, edits, duration):
    from .music_library import validate_cue_plan, MusicLibraryError
    from .media import MediaError
    cues = []
    for original, change in zip(base, edits):
        if not all(key in original for key in ("track_id", "track_version", "track_sha256")):
            raise EditError("原配乐缺少已确认的音乐版本信息，请先恢复配乐记录。")
        # Never copy unknown stored parameters or accept identities from HTTP.
        cue = {key: deepcopy(original[key]) for key in (
            "track_id", "track_version", "track_sha256", "source_start", "fade_in", "fade_out", "reason")
               if key in original}
        cue.update(change)
        span = cue["end"] - cue["start"]
        fade_in = min(_number(cue.get("fade_in", min(.6, span / 4))), span)
        fade_out = min(_number(cue.get("fade_out", min(.6, span / 4))), span)
        if fade_in + fade_out > span:
            scale = span / (fade_in + fade_out)
            fade_in, fade_out = fade_in * scale, fade_out * scale
        cue.update(fade_in=fade_in, fade_out=fade_out)
        cues.append(cue)
    old = pp.get("manual_music_plan") if _pending(pp, "manual_music_plan") else pp.get("music", {})
    summary = old.get("summary", "用户手动调整配乐时间和音量")
    if not isinstance(summary, str):
        summary = "用户手动调整配乐时间和音量"
    try:
        plan = validate_cue_plan({"summary": summary, "cues": cues}, state.get("music_library", {}),
                                 duration, project=Path(root))
    except (MusicLibraryError, MediaError, ValueError, OSError, KeyError, TypeError):
        raise EditError("配乐安排校验失败；请检查音乐审批、来源文件、素材长度及重叠处的淡入淡出。") from None
    return {"summary": plan["summary"], "cues": plan["cues"]}


def apply_edits(root, eid, version, revision, section, edits):
    """Record one all-or-nothing manual revision, bound to version and source.

    Music rows: ``[{index, start, end, gain_db}]``; subtitle rows:
    ``[{index, start, end, text}]``. All current rows must be present in order.
    The caller separately schedules a local render; this function never does.
    """
    from .episode_post import _assert_action, _retire
    if not isinstance(section, str) or section not in {"music", "subtitles"}:
        raise EditError("手动编辑范围只能是配乐或字幕。")
    if type(revision) is not int or revision < 0:
        raise EditError("后期编辑版本无效，请刷新页面后重新提交。")
    with core.run_lock(root), core.transaction(root) as state:
        ep = core._find(state, "episode", eid)
        _assert_action(state, ep, version)
        pp = ep.get("postproduction", {})
        if not isinstance(pp, dict):
            raise EditError("本集后期记录无效，请先恢复当前后期步骤。")
        if type(pp.get("revision", 0)) is not int or pp.get("revision", 0) != revision:
            raise core.GateError("后期编辑版本已变化，请刷新后重新检查当前条目。")
        if not ep.get("shots") or not all(core.human_approved_shot(sh) for sh in ep["shots"]):
            raise core.GateError("先逐段批准本集视频，再手动调整本集字幕或配乐。")
        try:
            for shot in ep["shots"]:
                core._check_shot_artifact(root, state, shot)
        except core.GateError:
            raise
        except (OSError, ValueError, TypeError):
            raise EditError("原视频校验失败；请恢复当前已批准的素材，再编辑本集后期。") from None
        source = [{"id": sh["id"], "version": sh["version"], "sha256": sh["sha256"]} for sh in ep["shots"]]
        source_key = core._json_digest(source)
        if (not isinstance(pp.get("source_key"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", pp["source_key"])
                or pp["source_key"] != source_key):
            raise core.GateError("本集源片段已变化，请先更新后期，再编辑当前版本。")
        duration = _duration(ep, pp)
        base = _rows(pp, section)
        checked = _check_rows(edits, base, section, duration)
        value = _music_plan(root, state, pp, base, checked, duration) if section == "music" else {"segments": checked}
        # All validation, including approved music bytes and actual source
        # duration, has completed. No exception above can retire a good export.
        other_key = "manual_subtitles" if section == "music" else "manual_music_plan"
        carry_other = _pending(pp, other_key)
        _retire(ep)
        pp = ep["postproduction"]
        new_revision = revision + 1
        if carry_other:
            pp[other_key]["revision"] = new_revision
        value.update(source_key=source_key, revision=new_revision)
        if section == "music":
            value["submitted_plan_hash"] = core._json_digest({"summary": value["summary"], "cues": value["cues"]})
        else:
            from .subtitle_sync import caption_hash
            value["submitted_caption_hash"] = caption_hash(value["segments"])
        value["submitted_at"] = core.now()
        pp["manual_music_plan" if section == "music" else "manual_subtitles"] = value
        pp.update(revision=new_revision, revision_target="manual_" + section,
                  status="pending_manual_edits", enabled=True, ready=True, final_ready=False, error=None)
        submission = {"section": section, "revision": new_revision, "source_key": source_key,
                      "submitted_at": core.now()}
        if section == "music":
            pp.pop("music_key", None)
        else:
            from .subtitle_sync import caption_hash
            submission["caption_hash"] = caption_hash(checked)
            for key in ("subtitle_key", "subtitle_proofread_key"):
                pp.pop(key, None)
            sub = pp.get("subtitles", {})
            if isinstance(sub, dict):
                previous = sub.pop("sync_check", None)
                if isinstance(previous, dict) and (previous.get("status") != "manual"
                                                  or not isinstance(sub.get("previous_sync_check"), dict)):
                    sub["previous_sync_check"] = deepcopy(previous)
                sub.pop("proofread_segments", None)
        # This explicit human submission is not a machine-check result. The
        # renderer adopts these times without silently rechecking or changing
        # them; its current-audio adoption record is bound separately.
        pp["manual_edit_submission"] = submission
        state.update(status="ready", paused=False, last_error=None)
        core.event(state, "postproduction_manual_edits_saved", id=eid, section=section,
                   version=version, input_revision=revision, revision=new_revision, rows=len(checked))
    return core.load_project(root)
