"""Pure, bounded subtitle synchronization checks on independent audio evidence.

The adapter supplies original-clock forced alignment and a whole-audio speech
coverage check. This module never treats the timestamps used to generate the
captions, a second spelling pass, or the same VAD boundaries as verification.
It neither recognizes audio nor calls a model, network service or renderer.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
import unicodedata
from typing import Callable


SCHEMA = "nolan-subtitle-sync-v1"
POLICY = {"version": 1, "start_tolerance": 0.25, "end_tolerance": 0.45,
          "min_coverage": 0.8, "min_alignment_span": 0.08,
          "max_attempts": 2, "independent_methods": ["forced_alignment", "independent_ctc"]}
MAX_ATTEMPTS = 2
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _number(value) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("invalid finite number")
    return float(value)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def canonical_captions(captions: list[dict], duration: float | None = None) -> list[dict]:
    """Validate text and exact cue order; never shift, join or invent captions."""
    if not isinstance(captions, list) or len(captions) > 10_000:
        raise ValueError("invalid captions")
    limit = _number(duration) if duration is not None else None
    if limit is not None and limit <= 0:
        raise ValueError("invalid audio duration")
    result = []
    previous_end = 0.0
    for row in captions:
        if not isinstance(row, dict):
            raise ValueError("invalid caption")
        start, end = _number(row.get("start")), _number(row.get("end"))
        text = row.get("text")
        if (not isinstance(text, str) or not text.strip() or len(text) > 4000
                or "\x00" in text or start < 0 or end <= start
                or start < previous_end - 0.000001
                or (limit is not None and end > limit + 0.000001)):
            raise ValueError("invalid caption text or timestamps")
        start, end = round(start, 6), round(end, 6)
        if end <= start:
            raise ValueError("caption collapsed after rounding")
        result.append({"start": start, "end": end, "text": text.strip()})
        previous_end = end
    return result


def caption_hash(captions: list[dict]) -> str:
    """Text changes invalidate verification just as timing changes do."""
    return _digest(canonical_captions(captions))


def _identity(value) -> bool:
    if not isinstance(value, (str, dict)) or not value:
        return False
    try:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        return bool(serialized.strip('" ')) and len(serialized) <= 20_000
    except (TypeError, ValueError):
        return False


def _policy_matches(value) -> bool:
    try:
        # Python considers True == 1; persisted policy types are part of the
        # contract, so compare their canonical JSON identities instead.
        return isinstance(value, dict) and _digest(value) == _digest(POLICY)
    except (TypeError, ValueError):
        return False


def _binding(audio_sha256: str, source_key: str, checker_identity) -> None:
    if (not isinstance(audio_sha256, str) or not _SHA256.fullmatch(audio_sha256)
            or not isinstance(source_key, str) or not source_key or len(source_key) > 256
            or not _identity(checker_identity)):
        raise ValueError("missing audio, source or checker identity")


def _text(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", value)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def _base(captions: list[dict], *, audio_sha256: str, source_key: str,
          checker_identity, input_caption_hash: str | None = None) -> dict:
    digest = caption_hash(captions)
    return {"schema": SCHEMA, "policy": deepcopy(POLICY), "status": "checking",
            "summary": "正在按实际音轨独立检查字幕时间和人声覆盖。", "issues": [],
            "attempts": 0, "max_attempts": MAX_ATTEMPTS, "checked_cues": 0,
            "total_cues": len(captions), "source_audio_sha256": audio_sha256,
            "source_key": source_key, "caption_hash": digest,
            "input_caption_hash": input_caption_hash or digest,
            "checker_identity": deepcopy(checker_identity), "evidence_hash": "",
            "evidence_method": "", "independent_evidence": False,
            "full_audio_checked": False, "no_speech_confirmed": False, "aligned_cues": []}


def _issue(report: dict, code: str, reason: str, *, start: float = 0,
           end: float = 0, cue_index: int | None = None) -> None:
    row = {"start": round(start, 6), "end": round(max(start, end), 6),
           "reason": reason, "code": code}
    if cue_index is not None:
        row["cue_index"] = cue_index
    report["issues"].append(row)


def evaluate_alignment(captions: list[dict], observations: dict, duration: float, *,
                       audio_sha256: str, source_key: str, checker_identity) -> dict:
    """Compare every cue with independent original-clock acoustic evidence.

    ``observations`` must contain ``independent=True``,
    ``method='independent_ctc'`` (or an actual acoustic forced aligner),
    ``full_audio_checked=True``, the matching
    ``audio_sha256`` and ``checker_identity``, and ``cues``. Each evidence cue
    contains a unique zero-based ``cue_index``, ``start``, ``end``,
    ``coverage`` in [0, 1] and ``consistent=True``. Optional ``text`` must match
    the current caption (punctuation/spacing may differ). ``contradictions``
    lists uncovered speech, contradictory recognition or other acoustic
    failures discovered by the whole-audio checker. No evidence is not a pass.
    """
    current = canonical_captions(captions, duration)
    duration = _number(duration)
    _binding(audio_sha256, source_key, checker_identity)
    report = _base(current, audio_sha256=audio_sha256, source_key=source_key,
                   checker_identity=checker_identity)
    if not isinstance(observations, dict):
        observations = {}
    report["no_speech_confirmed"] = observations.get("no_speech_confirmed") is True
    if not current and not report["no_speech_confirmed"]:
        _issue(report, "empty_captions", "没有可供同步验收的字幕，也缺少整条音轨无对白的独立证据。", end=duration)
    if current and report["no_speech_confirmed"]:
        _issue(report, "no_speech_conflict", "独立音频检查报告无对白，与现有字幕矛盾。", end=duration)
    independent = (observations.get("independent") is True
                   and observations.get("method") in POLICY["independent_methods"])
    report["independent_evidence"] = independent
    report["evidence_method"] = observations.get("method", "")
    report["full_audio_checked"] = observations.get("full_audio_checked") is True
    if not independent:
        _issue(report, "independent_evidence_missing", "缺少独立音频对齐证据；原字幕或同一次语音边界不能自证。", end=duration)
    if not report["full_audio_checked"]:
        _issue(report, "audio_coverage_unchecked", "尚未检查整条音轨是否存在漏字幕的人声。", end=duration)
    if (observations.get("audio_sha256") != audio_sha256
            or observations.get("checker_identity") != checker_identity
            or ("caption_hash" in observations and observations["caption_hash"] != report["caption_hash"])):
        _issue(report, "evidence_identity_mismatch", "对齐证据与当前音轨、字幕或检查器不匹配。", end=duration)
        independent = False
        report["independent_evidence"] = False
    try:
        report["evidence_hash"] = _digest(observations) if observations else ""
    except (TypeError, ValueError):
        _issue(report, "invalid_evidence", "本地检查返回的证据无效。", end=duration)

    rows = observations.get("cues")
    if not isinstance(rows, list) or len(rows) > 10_000:
        rows = []
        _issue(report, "alignment_missing", "没有完整的逐条音频对齐结果。", end=duration)
    evidence = {}
    duplicates = set()
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("cue_index")) is not int
                or not 0 <= row["cue_index"] < len(current)):
            _issue(report, "invalid_cue_evidence", "音频对齐结果的字幕编号无效。", end=duration)
            continue
        index = row["cue_index"]
        if index in evidence:
            duplicates.add(index)
        else:
            evidence[index] = row

    previous_alignment_end = 0.0
    for index, cue in enumerate(current):
        kwargs = {"start": cue["start"], "end": cue["end"], "cue_index": index}
        row = evidence.get(index)
        if row is None:
            _issue(report, "cue_unchecked", "本条字幕缺少独立对齐结果。", **kwargs)
            continue
        if index in duplicates:
            _issue(report, "duplicate_cue_evidence", "本条字幕有重复或矛盾的对齐结果。", **kwargs)
            continue
        try:
            start, end, coverage = _number(row.get("start")), _number(row.get("end")), _number(row.get("coverage"))
            if start < 0 or end > duration + 0.000001 or end <= start or not 0 <= coverage <= 1:
                raise ValueError("invalid alignment extent")
        except (ValueError, TypeError):
            _issue(report, "invalid_alignment", "独立对齐时间或覆盖率无效。", **kwargs)
            continue
        report["checked_cues"] += 1
        valid = independent
        if end - start < POLICY["min_alignment_span"]:
            _issue(report, "alignment_collapsed", "音频对齐塌缩，不能将集中在一个时刻的结果当作真实对白时间。", **kwargs)
            valid = False
        if start < previous_alignment_end - 0.05:
            _issue(report, "alignment_order", "独立对齐结果相互重叠或顺序矛盾。", **kwargs)
            valid = False
        previous_alignment_end = max(previous_alignment_end, end)
        if coverage < POLICY["min_coverage"]:
            _issue(report, "insufficient_coverage", "本条字幕与实际音频的对齐覆盖不足。", **kwargs)
            valid = False
        if row.get("consistent") is not True:
            _issue(report, "inconsistent_evidence", "本条字幕的独立音频证据不一致。", **kwargs)
            valid = False
        if row.get("boundary_supported") is False:
            _issue(report, "unsupported_boundary", "本条字幕的首末字没有独立音频边界支持。", **kwargs)
            valid = False
        if "text" in row and (not isinstance(row["text"], str) or _text(row["text"]) != _text(cue["text"])):
            _issue(report, "evidence_text_mismatch", "对齐证据对应的文字与当前字幕不同。", **kwargs)
            valid = False
        if valid:
            report["aligned_cues"].append({"cue_index": index, "start": round(start, 6), "end": round(end, 6)})
        if abs(cue["start"] - start) > POLICY["start_tolerance"] + 0.000001:
            _issue(report, "start_mismatch", "字幕出现时间与独立音频对齐结果不符，需要重新对齐。", **kwargs)
        if abs(cue["end"] - end) > POLICY["end_tolerance"] + 0.000001:
            _issue(report, "end_mismatch", "字幕结束时间与独立音频对齐结果不符，需要重新对齐。", **kwargs)

    contradictions = observations.get("contradictions", [])
    if not isinstance(contradictions, list) or len(contradictions) > 10_000:
        _issue(report, "invalid_contradictions", "整条音轨的覆盖检查结果无效。", end=duration)
    else:
        for row in contradictions:
            try:
                if not isinstance(row, dict):
                    raise ValueError("invalid contradiction")
                start, end = _number(row.get("start")), _number(row.get("end"))
                if start < 0 or end < start or end > duration + 0.000001:
                    raise ValueError("invalid contradiction extent")
                reason = row.get("reason")
                if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
                    raise ValueError("invalid contradiction reason")
                code = row.get("code", "acoustic_contradiction")
                if not isinstance(code, str) or not re.fullmatch(r"[a-z_]{1,60}", code):
                    code = "acoustic_contradiction"
                _issue(report, code, reason, start=start, end=end)
            except (TypeError, ValueError):
                _issue(report, "invalid_contradictions", "整条音轨含有无效或无法核对的反证。", end=duration)
    if not report["issues"] and report["checked_cues"] == report["total_cues"]:
        report.update(status="passed", summary="字幕与实际人声的机器同步检查通过；完成后仍由你最终审片。")
    else:
        report.update(status="needs_attention", summary="字幕同步检查发现问题，尚未通过机器验收。")
    return report


def _report_bound(report, captions, *, audio_sha256=None, source_key=None,
                  allow_initial: bool = False) -> bool:
    if not isinstance(report, dict):
        return False
    try:
        digest = caption_hash(captions)
    except (TypeError, ValueError):
        return False
    return (report.get("schema") == SCHEMA and _policy_matches(report.get("policy"))
            and type(report.get("attempts")) is int and 0 <= report["attempts"] <= MAX_ATTEMPTS
            and type(report.get("max_attempts")) is int and report["max_attempts"] == MAX_ATTEMPTS
            and isinstance(report.get("caption_hash"), str)
            and bool(_SHA256.fullmatch(report["caption_hash"]))
            and (digest == report["caption_hash"]
                 or (allow_initial and digest == report.get("input_caption_hash")))
            and isinstance(report.get("source_audio_sha256"), str)
            and bool(_SHA256.fullmatch(report["source_audio_sha256"]))
            and isinstance(report.get("source_key"), str) and 0 < len(report["source_key"]) <= 256
            and (audio_sha256 is None or report["source_audio_sha256"] == audio_sha256)
            and (source_key is None or report["source_key"] == source_key)
            and _identity(report.get("checker_identity")))


def report_matches(report: dict, captions: list[dict], *, audio_sha256: str | None = None,
                   source_key: str | None = None) -> bool:
    """Only a complete current-policy pass unlocks final human review."""
    return (_report_bound(report, captions, audio_sha256=audio_sha256, source_key=source_key)
            and report.get("status") == "passed" and report.get("issues") == []
            and type(report.get("checked_cues")) is int and type(report.get("total_cues")) is int
            and report["checked_cues"] == report["total_cues"] == len(captions)
            and (len(captions) > 0 or report.get("no_speech_confirmed") is True)
            and report.get("evidence_method") in POLICY["independent_methods"]
            and report.get("independent_evidence") is True and report.get("full_audio_checked") is True
            and isinstance(report.get("evidence_hash"), str) and bool(_SHA256.fullmatch(report["evidence_hash"])))


def run_sync(captions: list[dict], *, verify: Callable[[list[dict]], dict],
             repair: Callable[[list[dict], dict, int], list[dict]], duration: float,
             audio_sha256: str, source_key: str, checker_identity,
             previous_report: dict | None = None,
             on_progress: Callable[[dict, list[dict]], None] | None = None) -> tuple[list[dict], dict]:
    """Check, repair at most twice, and recheck; resumes never reset the budget.

    Persist progress before each repair. The caller persists returned captions
    and report together. A new explicit subtitle-feedback revision can discard
    ``previous_report``; merely pressing Continue must pass it back unchanged.
    No repair can approve an episode or change video/music.
    """
    current = canonical_captions(captions, duration)
    _binding(audio_sha256, source_key, checker_identity)
    bound = (_report_bound(previous_report, current, audio_sha256=audio_sha256,
                           source_key=source_key, allow_initial=True))
    same_checker = bound and previous_report.get("checker_identity") == checker_identity
    if same_checker and report_matches(previous_report, current, audio_sha256=audio_sha256, source_key=source_key):
        return current, deepcopy(previous_report)
    if same_checker and previous_report.get("status") == "needs_attention" and previous_report["attempts"] >= MAX_ATTEMPTS:
        return current, deepcopy(previous_report)
    attempts = previous_report["attempts"] if bound else 0
    initial_hash = previous_report.get("input_caption_hash", caption_hash(current)) if bound else caption_hash(current)

    def emit(report):
        if on_progress:
            on_progress(deepcopy(report), deepcopy(current))

    while True:
        report = _base(current, audio_sha256=audio_sha256, source_key=source_key,
                       checker_identity=checker_identity, input_caption_hash=initial_hash)
        report["attempts"] = attempts
        emit(report)
        try:
            observations = verify(deepcopy(current))
            report = evaluate_alignment(current, observations, duration, audio_sha256=audio_sha256,
                                        source_key=source_key, checker_identity=checker_identity)
        except Exception as exc:
            # Exception text can include provider paths or credentials; do not
            # copy it into a project report or let an unavailable checker pass.
            report.update(status="needs_attention", summary="本地字幕同步检查未完成，未标记通过。")
            _issue(report, "checker_error", "独立音频检查器未能完成，请检查本地配置。", end=duration)
            report["error_type"] = type(exc).__name__
            report.update(attempts=attempts, input_caption_hash=initial_hash)
            emit(report)
            return current, report
        report.update(attempts=attempts, input_caption_hash=initial_hash)
        if report["status"] == "passed":
            emit(report)
            return current, report
        if attempts >= MAX_ATTEMPTS:
            report["summary"] = "自动重新对齐已达两次上限，复检仍未通过；已停止重复处理并保留原素材。"
            emit(report)
            return current, report
        attempts += 1
        report.update(status="repairing", attempts=attempts,
                      summary=f"检测到字幕时间或覆盖问题，正在进行第 {attempts}/{MAX_ATTEMPTS} 次本地重新对齐。")
        emit(report)
        try:
            current = canonical_captions(repair(deepcopy(current), deepcopy(report), attempts), duration)
        except Exception as exc:
            report.update(status="needs_attention", summary="本地重新对齐未完成，未标记通过；已保留现有字幕和素材。")
            _issue(report, "repair_error", "自动重新对齐未能返回有效字幕。", end=duration)
            report["error_type"] = type(exc).__name__
            emit(report)
            return current, report


bounded_run = run_sync


def make_manual_report(captions, *, audio_sha256, source_key, revision, manual_record_hash):
    """Bind explicit user edits without calling them an automatic check."""
    current = canonical_captions(captions)
    if (not current or type(revision) is not int or revision < 1
            or not isinstance(manual_record_hash, str) or not _SHA256.fullmatch(manual_record_hash)):
        raise ValueError('invalid explicit manual subtitle revision')
    _binding(audio_sha256, source_key, {'method': 'human_timing_adopted'})
    return {'schema': 'nolan-subtitle-manual-v1', 'status': 'manual',
            'summary': '手动调整版：采用你输入的字幕文字和时间，不再自动复检；请试看后最终确认。',
            'evidence_method': 'human_timing_adopted', 'independent_evidence': False,
            'source_audio_sha256': audio_sha256, 'source_key': source_key,
            'caption_hash': caption_hash(current), 'revision': revision,
            'manual_record_hash': manual_record_hash, 'issues': []}


def manual_report_matches(report, captions, *, audio_sha256=None, source_key=None):
    """An explicit manual adoption is distinct from a machine pass."""
    try:
        digest = caption_hash(captions)
    except (ValueError, TypeError):
        return False
    return (isinstance(report, dict) and report.get('schema') == 'nolan-subtitle-manual-v1'
            and report.get('status') == 'manual' and report.get('evidence_method') == 'human_timing_adopted'
            and report.get('independent_evidence') is False and report.get('issues') == []
            and bool(captions) and report.get('caption_hash') == digest
            and type(report.get('revision')) is int and report['revision'] >= 1
            and isinstance(report.get('manual_record_hash'), str) and bool(_SHA256.fullmatch(report['manual_record_hash']))
            and isinstance(report.get('source_audio_sha256'), str) and bool(_SHA256.fullmatch(report['source_audio_sha256']))
            and isinstance(report.get('source_key'), str) and 0 < len(report['source_key']) <= 256
            and (audio_sha256 is None or report['source_audio_sha256'] == audio_sha256)
            and (source_key is None or report['source_key'] == source_key))
