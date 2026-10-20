"""Independent SenseVoice CTC evidence, without script-conditioned recognition.

https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html
Token times are original-crop acoustic anchors in seconds, not phoneme ends.
Long audio is scanned in fixed overlapping windows, independent of VAD.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import statistics
import sys
import wave

if __package__:
    from .alignment_worker import Checker, POLICY, _extent, match_units, units
else:
    from alignment_worker import Checker, POLICY, _extent, match_units, units


SCAN_POLICY = {"window_seconds": 20., "step_seconds": 15., "ownership_overlap": 2.5,
               "local_consensus": "two_changed_contexts_acoustic_boundaries_v1"}


def decode_anchors(tokens, timestamps, duration):
    """SenseVoice's Unicode pieces do not use Zipformer's byte-BPE mapping."""
    if (not isinstance(tokens, (list, tuple)) or not isinstance(timestamps, (list, tuple))
            or len(tokens) != len(timestamps) or len(tokens) > 200_000):
        raise ValueError("invalid SenseVoice tokens")
    anchors, previous = [], -.061
    for token, clock in zip(tokens, timestamps):
        if (not isinstance(token, str) or len(token) > 4000 or "\x00" in token
                or type(clock) not in (int, float) or not math.isfinite(clock)
                or clock < -.061 or clock < previous - .001 or clock > duration + .15):
            raise ValueError("invalid SenseVoice clock or token")
        previous = clock
        # sherpa already removes metadata tokens and accounts for their prefix
        # clock. Do not subtract another four frames from these timestamps.
        if token.startswith("<"):
            continue
        for char in token.replace("▁", " "):
            if not char.isspace() and clock < duration:
                anchors.append({"char": char, "time": round(max(0., float(clock)), 6)})
    return anchors


class SenseVoiceChecker(Checker):
    allow_local_consensus = True

    def __init__(self, model_dir, samples, rate=16000, threads=4, coverage_model_dir=None):
        import sherpa_onnx
        self.samples, self.rate = samples, rate
        self.duration = len(samples) / rate
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(Path(model_dir) / "model.int8.onnx"), tokens=str(Path(model_dir) / "tokens.txt"),
            num_threads=threads, provider="cpu", language="zh", use_itn=False, debug=False)
        self.cache = {}
        self.coverage_model_dir, self.coverage_checker = coverage_model_dir, None
        self.threads = threads

    def recognize(self, start=0., end=None):
        # A 130 s single offline decode can discard early speech. Fixed-clock
        # overlap scanning provides context at each ownership boundary instead.
        if end is None and start == 0. and self.duration > SCAN_POLICY["window_seconds"]:
            key = ("full", len(self.samples))
            if key in self.cache:
                return self.cache[key]
            anchors, windows, origin = [], [], 0.
            while origin < self.duration:
                stop = min(self.duration, origin + SCAN_POLICY["window_seconds"])
                observed = self.recognize(origin, stop)
                first_owned = 0. if origin == 0. else origin + SCAN_POLICY["ownership_overlap"]
                last_owned = self.duration if stop == self.duration else stop - SCAN_POLICY["ownership_overlap"]
                anchors.extend(a for a in observed["anchors"] if first_owned <= a["time"] < last_owned)
                windows.append({"start": origin, "end": stop,
                                "owned_start": first_owned, "owned_end": last_owned})
                if stop == self.duration:
                    break
                origin += SCAN_POLICY["step_seconds"]
            text = "".join(a["char"] for a in anchors)
            result = {"origin": 0., "end": self.duration, "text": text, "anchors": anchors,
                      "units": units(text, anchors), "scan_windows": windows}
            self.cache[key] = result
            return result
        end = self.duration if end is None else min(end, self.duration)
        start = max(0., start)
        first, last = round(start * self.rate), round(end * self.rate)
        if not 0 <= first < last <= len(self.samples):
            raise ValueError("invalid original-clock SenseVoice crop")
        key = (first, last)
        if key in self.cache:
            return self.cache[key]
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.rate, self.samples[first:last])
        self.recognizer.decode_stream(stream)
        raw = stream.result
        anchors = decode_anchors(raw.tokens, raw.timestamps, (last - first) / self.rate)
        for anchor in anchors:
            anchor["time"] = round(anchor["time"] + first / self.rate, 6)
        text = "".join(a["char"] for a in anchors)
        result = {"origin": round(first / self.rate, 6), "end": round(last / self.rate, 6),
                  "text": text, "anchors": anchors, "units": units(text, anchors)}
        self.cache[key] = result
        return result

    def verify(self, captions):
        result = super().verify(captions)
        additions = []
        if self.coverage_model_dir and result.get("contradictions"):
            if self.coverage_checker is None:
                self.coverage_checker = Checker(self.coverage_model_dir, self.samples, self.rate, self.threads)
            for issue in result["contradictions"]:
                text = issue.get("observed_text", "")
                # A Mandarin-only model's failure cannot veto another language.
                language_supported = bool(text and all(re.fullmatch(r"[\u3400-\u9fff0-9]", c) for c in text))
                if issue.get("code") != "uncovered_speech" or not language_supported:
                    continue
                target, start, end = units(text), issue["start"], issue["end"]
                checks = []
                full = self.coverage_checker.recognize()
                interval = [u for u in full["units"] if start - .2 <= u["time"] <= end + .2]
                checks.append({"kind": "whole_audio", "text": "".join(u["char"] for u in interval),
                               "matched_anchors": len(match_units(target, interval))})
                for before, after in ((2.5, 1.5), (3.5, 2.0)):
                    observed = self.coverage_checker.recognize(max(0., start - before), min(self.duration, end + after))
                    local = [u for u in observed["units"] if start - .2 <= u["time"] <= end + .2]
                    checks.append({"origin": observed["origin"], "end": observed["end"],
                                   "text": observed["text"], "matched_anchors": len(match_units(target, local))})
                changed = len(checks) == 3 and abs(checks[1]["origin"] - checks[2]["origin"]) >= .4
                issue["secondary_ctc"] = {"backend": "zipformer_ctc", "language_supported": True,
                                          "checks": checks,
                                          "zero_support": changed and all(c["matched_anchors"] == 0 for c in checks)}
                if changed:
                    additions.extend(self._supported_additions(issue, interval, result["cues"]))
        result.update(model_backend="sense_voice", scan_policy=SCAN_POLICY,
                      supported_additions=additions,
                      scan_windows=self.recognize().get("scan_windows", []),
                      limitations="独立 SenseVoice 中文 CTC 声学锚点及多上下文复检；时间为保守估计，"
                                  "不是人工逐字听验。文字或边界证据不一致时不标记通过。")
        return result

    def _supported_additions(self, issue, secondary_interval, existing_cues):
        """Add only shared words at shared clocks, never a desired script line."""
        observed = issue.get("observed_anchors", [])
        if not observed:
            return []
        primary = units("".join(a["char"] for a in observed), observed)
        mapping = match_units(primary, secondary_interval)
        agreed = sorted(i for i, j in mapping.items()
                        if abs(primary[i]["time"] - secondary_interval[j]["time"]) <= POLICY["cross_crop_tolerance"])
        runs = []
        for index in agreed:
            if runs and index == runs[-1][-1] + 1:
                runs[-1].append(index)
            else:
                runs.append([index])
        proposals = []
        for run in runs:
            if len(run) < 2:
                continue  # One emission alone does not establish a cue span.
            selected = [primary[i] for i in run]
            text = "".join(u["char"] for u in selected)
            target = units(text)
            initial = _extent(target, selected, match_units(target, selected), self.duration)
            if not initial["consistent"]:
                continue
            checks, extents = [], []
            for backend, checker in (("sense_voice", self), ("zipformer_ctc", self.coverage_checker)):
                for before, after in ((2.5, 1.5), (3.5, 2.0)):
                    # Reuse the original uncovered region's fixed contexts.
                    # Re-anchoring the crop to a subword changes ASR evidence
                    # unnecessarily and can truncate useful preceding speech.
                    hypothesis = checker.recognize(max(0., issue["start"] - before),
                                                   min(self.duration, issue["end"] + after))
                    local = [u for u in hypothesis["units"]
                             if initial["start"] - .35 <= u["time"] <= initial["end"] + .35]
                    extent = _extent(target, local, match_units(target, local), self.duration)
                    checks.append({"backend": backend, "origin": hypothesis["origin"],
                                   "end": hypothesis["end"], "text": hypothesis["text"], "alignment": extent})
                    extents.append(extent)
            if (not all(e["consistent"] and e["coverage"] == 1. for e in extents)
                    or any(abs(checks[i]["origin"] - checks[i + 1]["origin"]) < .4 for i in (0, 2))
                    or max(e["start"] for e in extents) - min(e["start"] for e in extents) > POLICY["cross_crop_tolerance"]
                    or max(e["end"] for e in extents) - min(e["end"] for e in extents) > POLICY["cross_crop_tolerance"]):
                continue
            start, end = (round(statistics.median(e[key] for e in extents), 6) for key in ("start", "end"))
            if any(c.get("consistent") is True and start < c["end"] and end > c["start"] for c in existing_cues):
                continue
            proposals.append({"start": start, "end": end, "text": text,
                              "method": "two_models_two_contexts_original_clock",
                              "models": ["sense_voice", "zipformer_ctc"], "checks": checks,
                              "source_issue": {"start": issue["start"], "end": issue["end"],
                                               "observed_text": issue["observed_text"]}})
        return proposals


def main():
    import numpy as np
    request = json.loads(Path(sys.argv[1]).read_text())
    with wave.open(str(Path(request["audio"])), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 16000:
            raise ValueError("requires original-clock 16 kHz mono PCM")
        frames = stream.readframes(stream.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.
    checker = SenseVoiceChecker(request["model_dir"], samples, threads=request.get("threads", 4),
                                coverage_model_dir=request.get("coverage_model_dir"))
    result = checker.verify(request["captions"])
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
