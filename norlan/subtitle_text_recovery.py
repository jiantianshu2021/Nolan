"""Recover unsupported automatic spelling changes from actual ASR evidence.

This is a bounded local text repair, not a new speech recognizer. The caller
must provide original recognition from the same bound audio and must never use
this route for explicitly adopted manual captions. Script text stays preferred
when the independent checker supports it. Nothing here changes timestamps,
deletes dialogue, resets timing-repair attempts or approves an episode.
"""
from __future__ import annotations

from copy import deepcopy
import re
import unicodedata

from .subtitle_sync import canonical_captions, caption_hash, evaluate_alignment


SCHEMA = "nolan-subtitle-audio-text-recovery-v1"
ELIGIBLE_CODES = frozenset({"insufficient_coverage", "inconsistent_evidence",
                            "unsupported_boundary", "evidence_text_mismatch"})
LIMITATIONS = ("仅将未获声学支持的自动改字恢复为同音轨原识别，并以独立声学证据复检；"
               "机器识别仍可能听错，不代表人工听验或整集审批。")


def _numbers(text):
    return re.findall(r"[0-9零〇一二三四五六七八九十百千万亿两]+",
                      unicodedata.normalize("NFKC", text))


def _strong_overlap(left, right):
    """Require substantial overlap on both clocks, not nearest-index guessing."""
    overlap = min(left["end"], right["end"]) - max(left["start"], right["start"])
    shorter = min(left["end"] - left["start"], right["end"] - right["start"])
    longer = max(left["end"] - left["start"], right["end"] - right["start"])
    return overlap > 0 and overlap / shorter >= .6 and overlap / longer >= .35


def _issue_key(issue):
    return tuple(issue.get(key) for key in ("code", "cue_index", "start", "end"))


def _fully_supported(index, observations, report):
    if not report.get("independent_evidence") or not report.get("full_audio_checked"):
        return False
    if any(issue.get("cue_index") == index for issue in report["issues"]):
        return False
    # Global identity/coverage errors invalidate every individual cue, even if
    # the recognizer also happened to emit a plausible local row.
    if any(issue["code"] in {"independent_evidence_missing", "audio_coverage_unchecked",
                            "evidence_identity_mismatch", "invalid_evidence", "invalid_contradictions"}
           for issue in report["issues"]):
        return False
    rows = [row for row in observations.get("cues", [])
            if isinstance(row, dict) and row.get("cue_index") == index]
    return (len(rows) == 1 and rows[0].get("coverage") == 1.
            and rows[0].get("boundary_supported") is True
            and rows[0].get("consistent") is True)


def recover_audio_text(captions, raw_segments, *, verify, audio_sha256,
                       source_key, checker_identity, duration):
    """Return ``(captions, audit)`` after at most three independent checks.

    Check current automatic captions once, propose only uniquely time-matched
    original ASR wording for acoustically unsupported cues, then check the batch
    and the exact selected final group. These are at most two additional
    verification calls, with no alignment-repair loop. A new error elsewhere or
    unverifiable replacement rolls back the proposal. Unresolved old problems
    remain explicit in ``audit['final_report']``; recovery itself is never an
    episode-approval decision.
    """
    current = canonical_captions(captions, duration)
    original_hash = caption_hash(current)
    audit = {"schema": SCHEMA, "policy": 1, "status": "unchanged",
             "source_audio_sha256": audio_sha256, "source_key": source_key,
             "checker_identity": deepcopy(checker_identity),
             "input_caption_hash": original_hash, "output_caption_hash": original_hash,
             "raw_caption_hash": "", "verification_calls": 0,
             "changes": [], "proposals": [], "rejections": [],
             "initial_report": None, "proposal_report": None, "final_report": None,
             "rejected_final_report": None,
             "limitations": LIMITATIONS}

    def check(cues):
        audit["verification_calls"] += 1
        observations = verify(deepcopy(cues))
        report = evaluate_alignment(cues, observations, duration,
            audio_sha256=audio_sha256, source_key=source_key,
            checker_identity=checker_identity)
        return observations, report

    def failed(reason):
        audit.update(status="verification_failed", reason=reason)
        return current, audit

    try:
        _, initial = check(current)
    except Exception:
        return failed("独立音轨检查未完成，未采用任何文字恢复。")
    audit["initial_report"] = deepcopy(initial)
    audit["final_report"] = deepcopy(initial)
    eligible = {issue["cue_index"] for issue in initial["issues"]
                if issue.get("code") in ELIGIBLE_CODES and type(issue.get("cue_index")) is int}
    if not eligible:
        return current, audit
    if (not initial.get("independent_evidence") or not initial.get("full_audio_checked")
            or any(issue["code"] in {"evidence_identity_mismatch", "invalid_evidence",
                                     "invalid_contradictions"} for issue in initial["issues"])):
        return failed("独立证据与当前音轨的绑定不完整，未采用原识别文字。")
    try:
        raw = canonical_captions(raw_segments, duration)
    except (ValueError, TypeError):
        audit["rejections"].append({"reason": "原始音频识别记录无效，未猜测替代文字。"})
        return current, audit
    audit["raw_caption_hash"] = caption_hash(raw)

    proposed = deepcopy(current)
    for index in sorted(eligible):
        matches = [i for i, row in enumerate(raw) if _strong_overlap(current[index], row)]
        if len(matches) != 1:
            audit["rejections"].append({"cue_index": index, "reason": "没有唯一时间重合的原识别，未替换。"})
            continue
        raw_index = matches[0]
        row = raw[raw_index]
        # A joined original cue must not be copied into both split captions.
        if sum(_strong_overlap(cue, row) for cue in current) != 1:
            audit["rejections"].append({"cue_index": index, "raw_cue_index": raw_index,
                                        "reason": "原识别同时对应多条字幕，未拆分或合并文字。"})
            continue
        if row["text"] == current[index]["text"]:
            continue
        if _numbers(row["text"]) != _numbers(current[index]["text"]):
            audit["rejections"].append({"cue_index": index, "raw_cue_index": raw_index,
                                        "reason": "原识别与现字幕的数字不同，未自动改写。"})
            continue
        proposed[index]["text"] = row["text"]
        audit["proposals"].append({"cue_index": index, "raw_cue_index": raw_index,
                                   "before": deepcopy(current[index]),
                                   "after": deepcopy(proposed[index])})
    if not audit["proposals"]:
        return current, audit
    try:
        observations, proposed_report = check(proposed)
    except Exception:
        return failed("替代文字的独立检查未完成，未采用任何文字恢复。")
    audit["proposal_report"] = deepcopy(proposed_report)
    selected = deepcopy(current)
    accepted = []
    for proposal in audit["proposals"]:
        index = proposal["cue_index"]
        if _fully_supported(index, observations, proposed_report):
            selected[index]["text"] = proposed[index]["text"]
            accepted.append(proposal)
        else:
            audit["rejections"].append({"cue_index": index, "reason": "替代文字未获得完整独立声学支持。"})
    if not accepted:
        return current, audit
    try:
        final_observations, final_report = check(selected)
    except Exception:
        return failed("最终文字组的独立复检未完成，未采用任何文字恢复。")
    recovered_indices = {row["cue_index"] for row in accepted}
    old_other_issues = {_issue_key(issue) for issue in initial["issues"]
                        if issue.get("cue_index") not in recovered_indices}
    final_other_issues = {_issue_key(issue) for issue in final_report["issues"]
                          if issue.get("cue_index") not in recovered_indices}
    if (not all(_fully_supported(index, final_observations, final_report) for index in recovered_indices)
            or not final_other_issues.issubset(old_other_issues)):
        audit["rejected_final_report"] = deepcopy(final_report)
        audit["rejections"].append({"reason": "最终复检出现新增问题或完整声学支持丢失，已保留原字幕。"})
        return current, audit
    audit.update(status="recovered", output_caption_hash=caption_hash(selected),
                 changes=deepcopy(accepted), final_report=deepcopy(final_report))
    return selected, audit
