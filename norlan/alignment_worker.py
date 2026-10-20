"""Offline independent Chinese CTC checker, run in an isolated Python runtime.

No script, novel, GPT, VAD boundary or old ASR timestamp is supplied to the CTC
recognizer. Byte-BPE mapping follows icefall/byte_utils.py (MIT, Facebook),
https://github.com/k2-fsa/icefall/blob/master/icefall/byte_utils.py .
The model is https://huggingface.co/csukuangfj/icefall-streaming-zipformer-small-ctc-zh-2025-04-01
(Apache-2.0). A CTC emission is a time anchor, not an exact phoneme endpoint.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import statistics
import sys
import unicodedata
import wave


POLICY = {"version": 1, "start_pad": .08, "end_pad": .25,
          "crop_context": 1.2, "second_origin_shift": 1., "max_contexts": 4,
          "cross_crop_tolerance": .28, "min_coverage": .8}


def decode_anchors(tokens, timestamps, duration):
    """Keep first byte's original CTC clock when UTF-8 spans several tokens."""
    bases = (list(range(256, 288)) + list(range(32, 127))
             + [c for c in range(288, 450)
                if unicodedata.normalize("NFKC", chr(c)) == chr(c)])[:256]
    mapping = {chr(c): i for i, c in enumerate(bases)}
    if len(tokens) != len(timestamps) or len(tokens) > 200_000:
        raise ValueError("invalid CTC tokens")
    buffer, clocks, result, previous = bytearray(), [], [], 0.
    for token, clock in zip(tokens, timestamps):
        if (not isinstance(token, str) or type(clock) not in (int, float)
                or not math.isfinite(clock) or clock < previous - .001
                or clock < 0 or clock > duration + .15):
            raise ValueError(f"invalid CTC clock {clock} / {duration} after {previous}")
        previous = clock
        if token.startswith("<"):
            continue
        for c in token.replace("▁", ""):
            if c not in mapping:
                raise ValueError("unknown byte-BPE symbol")
            buffer.append(mapping[c]); clocks.append(float(clock))
    # As in icefall's smart_byte_decode, retain only valid UTF-8 characters.
    # Short streaming hypotheses can contain orphan continuation bytes; these
    # do not create replacement characters or alignment evidence.
    index = 0
    while index < len(buffer):
        first = buffer[index]
        width = 1 if first < 128 else (2 if 0xc2 <= first <= 0xdf else
                (3 if 0xe0 <= first <= 0xef else (4 if 0xf0 <= first <= 0xf4 else 0)))
        try:
            text = bytes(buffer[index:index + width]).decode("utf-8") if width else ""
        except UnicodeDecodeError:
            text = ""
        if not text:
            index += 1
            continue
        if not text.isspace():
            result.append({"char": text, "time": round(clocks[index], 6)})
        index += width
    return result


def _integer_words(value):
    value = int(value)
    digits = "零一二三四五六七八九"
    if value == 0:
        return digits[0]
    if value >= 100_000_000:
        return "".join(digits[int(c)] for c in str(value))
    if value >= 10_000:
        rest = value % 10_000
        return (_integer_words(value // 10_000) + "万"
                + (("零" if rest < 1000 else "") + _integer_words(rest) if rest else ""))
    output, pending_zero = "", False
    for power, label in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        digit = value // power % 10
        if digit:
            if pending_zero:
                output += "零"
            output += digits[digit] + label
            pending_zero = False
        elif output and value % power:
            pending_zero = True
    return output[1:] if output.startswith("一十") else output


def units(text, anchors=None):
    """Phonetic comparison permits homophones; it never rewrites shown text."""
    from pypinyin import lazy_pinyin, Style
    normalized = unicodedata.normalize("NFKC", text).lower()
    expanded = []
    index = 0
    while index < len(normalized):
        char = normalized[index]
        if char.isdecimal() and char.isascii():
            end = index + 1
            while end < len(normalized) and normalized[end].isdigit() and normalized[end].isascii():
                end += 1
            words = _integer_words(normalized[index:end])
            expanded.extend((c, index) for c in words)
            index = end
        else:
            if not char.isspace() and not unicodedata.category(char).startswith(("P", "S")):
                expanded.append(("二" if char == "两" else char, index))
            index += 1
    chars = "".join(c for c, _ in expanded)
    phonemes = lazy_pinyin(chars, style=Style.NORMAL, errors=lambda s: list(s), strict=False)
    if len(phonemes) != len(expanded):
        raise ValueError("phonetic unit mapping mismatch")
    result = []
    for phoneme, (char, original) in zip(phonemes, expanded):
        row = {"key": phoneme, "char": char, "index": original}
        if anchors is not None:
            row["time"] = anchors[original]["time"]
        result.append(row)
    return result


def match_units(target, hypothesis):
    mapping = {}
    matcher = SequenceMatcher(None, [u["key"] for u in target],
                              [u["key"] for u in hypothesis], autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset
    return mapping


def _extent(target, hypothesis, mapping, duration):
    matched = sorted(mapping)
    coverage = len(matched) / len(target) if target else 0.
    boundary = bool(target and 0 in mapping and len(target) - 1 in mapping)
    clocks = [hypothesis[mapping[i]]["time"] for i in matched]
    valid = bool(clocks and all(b >= a for a, b in zip(clocks, clocks[1:])))
    distinct = len({round(c, 2) for c in clocks})
    anchors = len(matched)
    strong = (valid and boundary and coverage >= POLICY["min_coverage"]
              and (anchors == len(target) if len(target) <= 3 else anchors >= 3)
              and (len(target) == 1 or distinct >= 2))
    if not clocks:
        return {"start": 0., "end": 0., "coverage": 0., "consistent": False,
                "matched_anchors": 0, "target_units": len(target), "boundary_supported": False}
    positive = [b - a for a, b in zip(clocks, clocks[1:]) if .05 <= b - a <= .5]
    end_pad = max(.18, min(.35, statistics.median(positive))) if positive else POLICY["end_pad"]
    return {"start": round(max(0., clocks[0] - POLICY["start_pad"]), 6),
            "end": round(min(duration, clocks[-1] + end_pad), 6),
            "coverage": round(coverage, 6), "consistent": strong,
            "matched_anchors": anchors, "target_units": len(target),
            "boundary_supported": boundary}


class Checker:
    def __init__(self, model_dir, samples, rate=16000, threads=4):
        import sherpa_onnx
        self.samples, self.rate = samples, rate
        self.duration = len(samples) / rate
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=str(Path(model_dir) / "tokens.txt"),
            model=str(Path(model_dir) / "model.int8.onnx"),
            num_threads=threads, provider="cpu", enable_endpoint_detection=False)
        self.cache = {}

    def recognize(self, start=0., end=None):
        import numpy as np
        end = self.duration if end is None else min(end, self.duration)
        start = max(0., start)
        first, last = round(start * self.rate), round(end * self.rate)
        key = (first, last)
        if key in self.cache:
            return self.cache[key]
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.rate, self.samples[first:last])
        stream.accept_waveform(self.rate, np.zeros(self.rate, dtype="float32"))
        stream.input_finished()
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        raw = self.recognizer.get_result_all(stream)
        anchors = decode_anchors(raw.tokens, raw.timestamps, end - start + 1.)
        if "".join(a["char"] for a in anchors).replace(" ", "") != raw.text.replace(" ", ""):
            raise ValueError("decoded CTC text differs from recognizer")
        # Decoder flush padding is outside observed audio. Never turn emissions
        # in that artificial second into subtitle or coverage evidence.
        anchors = [a for a in anchors if a["time"] < (last - first) / self.rate]
        for anchor in anchors:
            anchor["time"] = round(anchor["time"] + first / self.rate, 6)
        text = "".join(a["char"] for a in anchors)
        result = {"origin": round(first / self.rate, 6), "end": round(last / self.rate, 6),
                  "text": text, "anchors": anchors, "units": units(text, anchors)}
        self.cache[key] = result
        return result

    def verify(self, captions):
        full = self.recognize()
        targets, ranges = [], []
        for cue in captions:
            current = units(cue["text"])
            ranges.append((len(targets), len(targets) + len(current)))
            targets.extend(current)
        global_mapping = match_units(targets, full["units"])
        rows, contradictions, used_full, unsupported = [], [], set(), []
        for index, (cue, (first, last)) in enumerate(zip(captions, ranges)):
            target = targets[first:last]
            mapping = {i - first: j for i, j in global_mapping.items() if first <= i < last}
            extent = _extent(target, full["units"], mapping, self.duration)
            tentative_start = extent["start"] if mapping else cue["start"]
            tentative_end = extent["end"] if mapping else cue["end"]
            # Two changed crop origins expose context-dependent false anchors.
            start_a = max(0., tentative_start - POLICY["crop_context"])
            start_b = max(0., start_a - POLICY["second_origin_shift"])
            if abs(start_a - start_b) < .4 and tentative_start > 1.:
                start_a = min(tentative_start - .4, start_a + .6)
            end = min(self.duration, max(tentative_end, cue["end"] if not mapping else tentative_end) + 1.2)
            end = min(end, min(start_a, start_b) + 35.)
            hypotheses, local_extents = [], []
            variant_ends = (end, min(self.duration, end + 1.)) if abs(start_a - start_b) < .4 else (end, end)
            distinct_context = (abs(start_a - start_b) >= .4 or abs(variant_ends[0] - variant_ends[1]) >= .4)
            for origin, variant_end in zip((start_a, start_b), variant_ends):
                hypothesis = self.recognize(origin, max(origin + .1, variant_end))
                # Exclude neighboring sentences while retaining generous onset context.
                local = [u for u in hypothesis["units"]
                         if tentative_start - .8 <= u["time"] <= tentative_end + .8]
                local_mapping = match_units(target, local)
                local_extent = _extent(target, local, local_mapping, self.duration)
                hypotheses.append({"origin": hypothesis["origin"], "end": hypothesis["end"],
                                   "text": hypothesis["text"], "matched_anchors": len(local_mapping),
                                   "coverage": local_extent["coverage"]})
                local_extents.append(local_extent)
            a, b = local_extents
            consistent = (distinct_context and extent["consistent"] and a["consistent"] and b["consistent"]
                          and abs(a["start"] - b["start"]) <= POLICY["cross_crop_tolerance"]
                          and abs(a["end"] - b["end"]) <= POLICY["cross_crop_tolerance"]
                          and abs(extent["start"] - a["start"]) <= POLICY["cross_crop_tolerance"]
                          and abs(extent["end"] - a["end"]) <= POLICY["cross_crop_tolerance"])
            attempted = list(hypotheses)
            # A crop may start amid a preceding word. Try at most two longer
            # contexts; still require two changed contexts supporting the same
            # original-clock boundaries. This is bounded local checking, never
            # an unbounded retry until a convenient answer appears.
            if not consistent and (extent["consistent"] or getattr(self, "allow_local_consensus", False)):
                for extra in (2., 3.):
                    origin = max(0., start_a - extra)
                    if any(abs(origin - h["origin"]) < .4 for h in attempted):
                        continue
                    hypothesis = self.recognize(origin, end)
                    local = [u for u in hypothesis["units"]
                             if tentative_start - .8 <= u["time"] <= tentative_end + .8]
                    local_extent = _extent(target, local, match_units(target, local), self.duration)
                    check = {"origin": hypothesis["origin"], "end": hypothesis["end"],
                             "text": hypothesis["text"], "matched_anchors": local_extent["matched_anchors"],
                             "coverage": local_extent["coverage"]}
                    attempted.append(check); local_extents.append(local_extent)
                good = [j for j, e in enumerate(local_extents)
                        if extent["consistent"] and e["consistent"]
                        and abs(e["start"] - extent["start"]) <= POLICY["cross_crop_tolerance"]
                        and abs(e["end"] - extent["end"]) <= POLICY["cross_crop_tolerance"]]
                for left in good:
                    for right in good:
                        if left >= right:
                            continue
                        left_check, right_check = attempted[left], attempted[right]
                        if (abs(left_check["origin"] - right_check["origin"]) >= .4
                                or abs(left_check["end"] - right_check["end"]) >= .4) and (
                                abs(local_extents[left]["start"] - local_extents[right]["start"]) <= POLICY["cross_crop_tolerance"]
                                and abs(local_extents[left]["end"] - local_extents[right]["end"]) <= POLICY["cross_crop_tolerance"]):
                            hypotheses = [left_check, right_check]
                            consistent = True
                            break
                    if consistent:
                        break
            # A stronger offline recognizer can miss one word in the complete
            # fixed-window scan but recover it in both short audio contexts.
            # Only actual endpoint-supported local acoustic matches qualify;
            # this never changes the coverage threshold or candidate text.
            if not consistent and getattr(self, "allow_local_consensus", False):
                good = [j for j, e in enumerate(local_extents) if e["consistent"]]
                for left in good:
                    for right in good:
                        if left >= right:
                            continue
                        x, y = local_extents[left], local_extents[right]
                        a_check, b_check = attempted[left], attempted[right]
                        changed = (abs(a_check["origin"] - b_check["origin"]) >= .4
                                   or abs(a_check["end"] - b_check["end"]) >= .4)
                        if changed and abs(x["start"] - y["start"]) <= POLICY["cross_crop_tolerance"] and (
                                abs(x["end"] - y["end"]) <= POLICY["cross_crop_tolerance"]):
                            extent.update(start=round((x["start"] + y["start"]) / 2, 6),
                                          end=round((x["end"] + y["end"]) / 2, 6),
                                          coverage=min(x["coverage"], y["coverage"]),
                                          matched_anchors=min(x["matched_anchors"], y["matched_anchors"]),
                                          boundary_supported=True,
                                          boundary_evidence="two_changed_acoustic_contexts")
                            hypotheses, consistent = [a_check, b_check], True
                            break
                    if consistent:
                        break
            extent.update(cue_index=index, text=cue["text"], consistent=consistent,
                          local_checks=hypotheses, attempted_contexts=attempted)
            if extent["end"] <= extent["start"]:
                extent.update(start=cue["start"], end=cue["end"])
            rows.append(extent)
            if consistent:
                used_full.update(mapping.values())
            if not consistent:
                unsupported.append({"cue_index": index, "text": cue["text"],
                                    "start": cue["start"], "end": cue["end"],
                                    "checks": hypotheses,
                                    "independent_text_in_interval": "".join(
                                        u["char"] for u in full["units"]
                                        if cue["start"] - .15 <= u["time"] <= cue["end"] + .15)})
        # Complete independent recognition is also scanned for missing captions.
        # Tiny one/two-character additions are reported only if a second crop
        # independently reproduces them, rather than treating every CTC typo as speech.
        unmatched = []
        for j, unit in enumerate(full["units"]):
            if j not in used_full:
                unmatched.append(unit)
        groups = []
        for unit in unmatched:
            if groups and unit["time"] - groups[-1][-1]["time"] < .65:
                groups[-1].append(unit)
            else:
                groups.append([unit])
        for group in groups:
            # Interior substitutions/insertions already belong to an existing cue;
            # coverage of its spoken time is separately checked above.
            start, end = group[0]["time"], group[-1]["time"] + .2
            if any(c.get("consistent") is True and c["start"] - .3 <= start and end <= c["end"] + .3 for c in rows):
                continue
            hypothesis = self.recognize(max(0., start - 1.), min(self.duration, end + 1.))
            local = [u for u in hypothesis["units"] if start - .25 <= u["time"] <= end + .25]
            mapping = match_units(group, local)
            if (len(mapping) >= max(2, math.ceil(len(group) * .8))
                    and group[-1]["time"] - group[0]["time"] >= .12):
                contradictions.append({"start": max(0., start - .08),
                                       "end": min(self.duration, end),
                                       "code": "uncovered_speech",
                                       "observed_text": "".join(u["char"] for u in group),
                                       "observed_anchors": [{"char": u["char"], "time": u["time"]} for u in group],
                                       "reason": "独立中文识别在此处重复检测到未被字幕覆盖的人声："
                                                 + "".join(u["char"] for u in group)})
        return {"cues": rows, "contradictions": contradictions,
                "unsupported_cues": unsupported, "full_audio_checked": True,
                "full_transcript": full["text"],
                "anchors": full["anchors"], "policy": POLICY,
                "no_speech_confirmed": False,
                "limitations": "中文 CTC 识别及保守时间锚点，不等于人工逐字听验；不能核实的内容不会自动通过。"}


def main():
    import numpy as np
    request = json.loads(Path(sys.argv[1]).read_text())
    audio = Path(request["audio"])
    with wave.open(str(audio), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 16000:
            raise ValueError("requires original-clock 16 kHz mono PCM")
        frames = stream.readframes(stream.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.
    checker = Checker(request["model_dir"], samples, threads=request.get("threads", 4))
    result = checker.verify(request["captions"])
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
