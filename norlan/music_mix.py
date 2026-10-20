"""Bounded local music automation and numerical mix checks.

These checks measure signal levels and whether the exported file actually
contains the prepared music. They do not establish musical taste, intelligibility
or human approval. No recognition, model, provider or network is used here.
"""

from __future__ import annotations

import array
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from . import media

POLICY_VERSION = 1
NORMALIZATION_TARGET_LUFS = -18
MAX_DUCK_DB = 6.0
MAX_ADAPTIVE_DB = 9.0
MAX_REPAIR_DB = 6.0
MAX_REPAIRS = 2
_QUIET_NATIVE_DB = -30.0
_NON_SPEECH_TARGET_MIN_DB = -32.0
_NON_SPEECH_TARGET_MAX_DB = -22.0
_NON_SPEECH_LOUD_DB = -18.0
_RATE = 16000
_CHANNELS = 2
_STEP = 0.1


def _finite(value, name: str, *, low=None, high=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise media.MediaError(f"配乐混音参数 {name} 必须为有限数字。")
    number = float(value)
    if (low is not None and number < low) or (high is not None and number > high):
        raise media.MediaError(f"配乐混音参数 {name} 超出允许范围。")
    return number


def _db(power: float) -> float:
    return 10 * math.log10(max(power, 1e-16))


def _decode(path: Path, duration: float) -> array.array:
    path = media._local(path)
    args = ["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe",
            "-i", str(path), "-map", "0:a:0", "-t", f"{duration:.9f}",
            "-af", f"apad,atrim=duration={duration:.9f}",
            "-f", "f32le", "-ac", str(_CHANNELS), "-ar", str(_RATE), "-"]
    try:
        result = subprocess.run(args, capture_output=True, check=False, timeout=180,
                                stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise media.MediaError("无法读取本地配乐混音测量样本；原素材未改变。") from exc
    if result.returncode:
        raise media.MediaError("配乐混音测量失败：" + result.stderr.decode(errors="replace")[-1200:])
    values = array.array("f")
    values.frombytes(result.stdout)
    if sys.byteorder != "little":
        values.byteswap()
    required = round(duration * _RATE) * _CHANNELS
    if len(values) < required:
        values.extend([0.0] * (required - len(values)))
    return values[:required]


def _powers(samples: array.array, duration: float) -> list[float]:
    size = round(_RATE * _CHANNELS * _STEP)
    count = math.ceil(duration / _STEP)
    powers = []
    for index in range(count):
        region = samples[index * size:min((index + 1) * size, len(samples))]
        powers.append(sum(value * value for value in region if math.isfinite(value)) / max(1, len(region)))
    return powers


def _windows(windows: list[dict], duration: float) -> list[dict]:
    if not isinstance(windows, list) or len(windows) > 2000:
        raise media.MediaError("对白窗口必须是有限的时间区间列表。")
    rows = []
    for row in windows:
        if not isinstance(row, dict):
            raise media.MediaError("对白窗口格式无效。")
        start = _finite(row.get("start"), "speech.start", low=0, high=duration + 0.1)
        end = _finite(row.get("end"), "speech.end", low=0, high=duration + 0.1)
        if end <= start:
            raise media.MediaError("对白窗口结束时间必须晚于开始时间。")
        rows.append({"start": min(start, duration), "end": min(end, duration)})
    rows.sort(key=lambda row: row["start"])
    merged = []
    for row in rows:
        if merged and row["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
        else:
            merged.append(row.copy())
    return merged


def _cue_rows(cues: list[dict], duration: float) -> list[dict]:
    if not isinstance(cues, list) or len(cues) > 2000:
        raise media.MediaError("配乐计划必须是有限的片段列表。")
    result = []
    for index, cue in enumerate(cues):
        if not isinstance(cue, dict):
            raise media.MediaError("配乐片段格式无效。")
        start = _finite(cue.get("start"), "cue.start", low=0, high=duration + 0.1)
        end = _finite(cue.get("end"), "cue.end", low=0, high=duration + 0.1)
        if end <= start:
            raise media.MediaError("配乐片段结束时间必须晚于开始时间。")
        fade_in = _finite(cue.get("fade_in", 0), "fade_in", low=0, high=end - start)
        fade_out = _finite(cue.get("fade_out", 0), "fade_out", low=0, high=end - start)
        result.append({"index": index, "start": min(start, duration), "end": min(end, duration),
                       "fade_in": fade_in, "fade_out": fade_out,
                       "track_id": str(cue.get("track_id", "")),
                       "source_start": cue.get("source_start", 0),
                       "track_version": cue.get("track_version"),
                       "source_sha256": cue.get("source_sha256", cue.get("track_sha256", cue.get("sha256", "")))})
    return result


def _speech_at(time: float, windows: list[dict]) -> bool:
    return any(window["start"] <= time < window["end"] for window in windows)


def _duck_at(time: float, windows: list[dict], maximum: float) -> float:
    # Explicit, bounded envelope: a loud workshop cannot trigger this control.
    amount = 0.0
    for window in windows:
        start, end = window["start"], window["end"]
        if start - 0.15 <= time < start:
            amount = max(amount, (time - start + 0.15) / 0.15)
        elif start <= time <= end:
            amount = 1.0
        elif end < time < end + 0.4:
            amount = max(amount, (end + 0.4 - time) / 0.4)
    return maximum * min(1.0, max(0.0, amount))


def _core(cue: dict, time: float) -> bool:
    # Intentional boundary fades and planned gaps must not be "repaired".
    left = min(cue["fade_in"] + 0.1, (cue["end"] - cue["start"]) / 3)
    right = min(cue["fade_out"] + 0.1, (cue["end"] - cue["start"]) / 3)
    return cue["start"] + left <= time < cue["end"] - right


def _baseline(automation: dict, cues: list[dict]) -> dict:
    if not isinstance(automation, dict) or automation.get("policy_version") != POLICY_VERSION:
        raise media.MediaError("已保存的配乐自动化基准版本无效。")
    duration = _finite(automation.get("duration"), "baseline.duration", low=0.01, high=21600)
    step = _finite(automation.get("step"), "baseline.step", low=0.01, high=1)
    values = automation.get("gain_db")
    if not isinstance(values, list) or len(values) != math.ceil(duration / step):
        raise media.MediaError("已保存的配乐自动化基准长度无效。")
    checked = [_finite(value, "baseline.gain_db", low=-15, high=15) for value in values]
    previous = _cue_rows(automation.get("cues"), duration)
    if len(previous) != len(cues) or any(
            old["track_id"] != new["track_id"] or old["track_version"] != new["track_version"]
            or old["source_sha256"] != new["source_sha256"] for old, new in zip(previous, cues)):
        raise media.MediaError("手动配乐与已保存自动化基准的音乐资产或版本不一致。")
    return {"duration": duration, "step": step, "gain_db": checked, "cues": previous}


def _baseline_at(time: float, scheduled: list[dict], baseline: dict) -> float:
    values, step = baseline["gain_db"], baseline["step"]
    mapped = []
    for cue in scheduled:
        old = baseline["cues"][cue["index"]]
        old_time = min(max(old["start"] + time - cue["start"], old["start"]),
                       max(old["start"], old["end"] - step / 2))
        position = min(max(old_time / step - 0.5, 0), len(values) - 1)
        left, right = int(position), min(int(position) + 1, len(values) - 1)
        mapped.append(values[left] * (1 - position + left) + values[right] * (position - left))
    return sum(mapped) / len(mapped) if mapped else 0.0


def _measure(native: list[float], music: list[float], bed: list[float], *,
             cues: list[dict], speech: list[dict], duration: float) -> dict:
    issues, rows, suggestions = [], [], {}
    planned = active = excluded_fades = source_silence = 0.0
    for index in range(len(music)):
        t = min(duration, (index + 0.5) * _STEP)
        interval = min(_STEP, max(0, duration - index * _STEP))
        scheduled = [cue for cue in cues if cue["start"] <= t < cue["end"]]
        if not scheduled:
            continue
        planned += interval
        if _db(music[index]) >= -50:
            active += interval
        if not any(_core(cue, t) for cue in scheduled):
            excluded_fades += interval
        elif _db(bed[index]) < -65:
            source_silence += interval
    for cue in cues:
        selected = [i for i in range(len(music)) if _core(cue, (i + 0.5) * _STEP)]
        # Check 1 s windows to avoid treating a transient/percussion rest as a
        # continuous inaudible bed. The two stereo channels are never downmixed.
        blocks = [selected[offset:offset + 10] for offset in range(0, len(selected), 10)]
        weak = loud = source_empty = measured = 0.0
        groups = {"speech": [], "non_speech": []}
        for block in blocks:
            if len(block) < 3:
                continue
            start, end = block[0] * _STEP, min(duration, (block[-1] + 1) * _STEP)
            span = end - start
            speech_present = sum(_speech_at((i + 0.5) * _STEP, speech) for i in block) / len(block) >= 0.3
            pn = sum(native[i] for i in block) / len(block)
            pm = sum(music[i] for i in block) / len(block)
            pb = sum(bed[i] for i in block) / len(block)
            music_db, native_db = _db(pm), _db(pn)
            relative = music_db - native_db if native_db > -55 else None
            category = "speech" if speech_present else "non_speech"
            groups[category].append({"start": round(start, 3), "end": round(end, 3),
                                     "music_rms_dbfs": round(music_db, 2),
                                     "native_rms_dbfs": round(native_db, 2),
                                     "music_relative_db": round(relative, 2) if relative is not None else None})
            measured += span
            if _db(pb) < -65:
                source_empty += span
            elif music_db < -45 or (relative is not None and relative < (-24 if speech_present else -21)
                    and (speech_present or native_db >= _QUIET_NATIVE_DB)):
                weak += span
            elif ((speech_present and relative is not None and relative > -7)
                    or (not speech_present and (music_db > _NON_SPEECH_LOUD_DB
                        or (native_db >= _QUIET_NATIVE_DB and relative is not None and relative > -1)))):
                loud += span
        record = {**cue, "measured_seconds": round(measured, 3),
                  "weak_seconds": round(weak, 3), "loud_seconds": round(loud, 3),
                  "source_silence_seconds": round(source_empty, 3), "windows": groups}
        rows.append(record)
        threshold = max(0.8, measured * 0.25)
        if source_empty >= threshold:
            issues.append({"start": cue["start"], "end": cue["end"], "cue_index": cue["index"],
                           "code": "source_silence", "message": "安排配乐的原曲片段存在较长静音，增益不能恢复声音。"})
        if weak >= threshold:
            suggestions[str(cue["index"])] = 3.0
            issues.append({"start": cue["start"], "end": cue["end"], "cue_index": cue["index"],
                           "code": "music_too_quiet", "message": "此配乐片段的数值电平过低，可能难以听见。"})
        if loud >= threshold:
            suggestions[str(cue["index"])] = -3.0
            issues.append({"start": cue["start"], "end": cue["end"], "cue_index": cue["index"],
                           "code": "music_too_loud", "message": "此配乐片段相对原声偏强，需要试听确认对白。"})
    return {"status": "needs_attention" if issues else "passed", "issues": issues,
            "repair_suggestions": suggestions, "cues": rows,
            "planned_music_seconds": round(planned, 3),
            "planned_silence_seconds": round(max(0, duration - planned), 3),
            "effective_music_seconds": round(active, 3),
            "excluded_fade_seconds": round(excluded_fades, 3),
            "source_silence_seconds": round(source_silence, 3),
            "measurement_note": "双声道局部 RMS 数值检查，排除计划留白及边界淡化；不代表主观听感或人工批准。"}


def prepare_mix(native: Path, bed: Path, output: Path, *, duration: float,
                cues: list[dict], speech_windows: list[dict], duck_db: float = MAX_DUCK_DB,
                adaptive: bool = True, binding: dict | None = None,
                repair_attempt: int = 0, local_corrections_db: dict | None = None,
                input_gain: float = 1.0, baseline_automation: dict | None = None) -> dict:
    """Render a local music stem with speech-only, bounded ducking.

    ``native`` is the exact native stem used by compose, already normalized once.
    ``bed`` is the approved-cue bed; the source music is never overwritten.
    Adaptive automation changes only the mix, never cue/source identities.
    Manual edits pass ``adaptive=False`` and must not call :func:`audit_mix`.
    """
    duration = _finite(duration, "duration", low=0.01, high=21600)
    duck_db = _finite(duck_db, "duck_db", low=0, high=MAX_DUCK_DB)
    input_gain = _finite(input_gain, "input_gain", low=0, high=1)
    if not isinstance(adaptive, bool):
        raise media.MediaError("配乐自动调整开关必须为布尔值。")
    if isinstance(repair_attempt, bool) or not isinstance(repair_attempt, int) or not 0 <= repair_attempt <= MAX_REPAIRS:
        raise media.MediaError("本地配乐修正最多允许两轮。")
    native, bed, output = media._local(native), media._local(bed), media._destination(output)
    if output in {native, bed}:
        raise media.MediaError("配乐混音不能覆盖原声或原曲。")
    rows, speech = _cue_rows(cues, duration), _windows(speech_windows, duration)
    if baseline_automation is not None and adaptive:
        raise media.MediaError("自动混音不能同时套用手动保留基准。")
    baseline = _baseline(baseline_automation, rows) if baseline_automation is not None else None
    corrections = local_corrections_db or {}
    if not isinstance(corrections, dict) or len(corrections) > len(rows):
        raise media.MediaError("本地配乐修正参数无效。")
    for key, value in corrections.items():
        if str(key) not in {str(cue["index"]) for cue in rows}:
            raise media.MediaError("本地配乐修正引用了不存在的片段。")
        _finite(value, "local correction", low=-MAX_REPAIR_DB, high=MAX_REPAIR_DB)
    bed_power = (_powers(_decode(bed, duration), duration) if adaptive
                 else [0.0] * math.ceil(duration / _STEP))
    # Manual mode does no native analysis, checks, correction or model call.
    native_power = _powers(_decode(native, duration), duration) if adaptive else []
    gain_values, automatic, ducks, base_gains = [], [], [], []
    for index, source_power in enumerate(bed_power):
        time = (index + 0.5) * _STEP
        scheduled = [cue for cue in rows if cue["start"] <= time < cue["end"]]
        duck = _duck_at(time, speech, duck_db)
        ducks.append(duck)
        change = 0.0
        if baseline is not None:
            change = _baseline_at(time, scheduled, baseline)
        if adaptive and scheduled and _db(source_power) > -65 and any(_core(cue, time) for cue in scheduled):
            speech_present = _speech_at(time, speech)
            pn = _db(native_power[index])
            pm = _db(source_power) + (20 * math.log10(input_gain) if input_gain else -160)
            if speech_present:
                target = pn - 16 if pn > -55 else -29
            else:
                # A quiet establishing shot is not quiet dialogue. Chasing its
                # tiny ambient signal made the old relative target inaudible,
                # and also contradicted the absolute audibility check.
                target = (max(_NON_SPEECH_TARGET_MIN_DB,
                              min(_NON_SPEECH_TARGET_MAX_DB, pn - 11)) if pn > -55 else -29)
            change = max(-MAX_ADAPTIVE_DB, min(MAX_ADAPTIVE_DB, target - pm + duck))
        automatic.append(change)
    # A median removes short effects/transients; a symmetric 1.1 s smoothing
    # avoids sudden per-cue or speech-boundary gain jumps.
    if adaptive:
        smooth = [statistics.median(automatic[max(0, i - 5):i + 6]) for i in range(len(automatic))]
        smooth = [sum(smooth[max(0, i - 5):i + 6]) / len(smooth[max(0, i - 5):i + 6])
                  for i in range(len(smooth))]
    else:
        smooth = automatic
    for index, adjustment in enumerate(smooth):
        time = (index + 0.5) * _STEP
        scheduled = [cue for cue in rows if cue["start"] <= time < cue["end"]]
        correction = max((float(corrections.get(str(cue["index"]), 0)) for cue in scheduled), default=0) if adaptive else 0
        base_gain = max(-15, min(15, adjustment + correction))
        base_gains.append(base_gain)
        final_db = base_gain - ducks[index]
        gain_values.append(input_gain * 10 ** (final_db / 20))
    with tempfile.TemporaryDirectory(prefix=".nolan-music-envelope-", dir=output.parent) as temporary:
        envelope = Path(temporary) / "gain.f32"
        # 1 kHz control with linear interpolation, resampled by FFmpeg; values
        # above 1 are kept as float, rather than clipped by integer WAV encoding.
        control = array.array("f")
        for sample in range(math.ceil(duration * 1000)):
            position = max(0.0, sample / 1000 / _STEP - 0.5)
            left = min(int(position), len(gain_values) - 1)
            right = min(left + 1, len(gain_values) - 1)
            fraction = position - left if right > left else 0
            control.append(gain_values[left] * (1 - fraction) + gain_values[right] * fraction)
        if sys.byteorder != "little":
            control.byteswap()
        envelope.write_bytes(control.tobytes())
        target = Path(temporary) / "prepared.wav"
        media._ffmpeg(["-protocol_whitelist", "file,pipe", "-i", str(bed),
                       "-f", "f32le", "-ar", "1000", "-ac", "1", "-i", str(envelope),
                       "-filter_complex", "[0:a]aresample=48000,aformat=channel_layouts=stereo[bed];"
                       "[1:a]aresample=48000,pan=stereo|c0=c0|c1=c0[control];"
                       "[bed][control]amultiply,alimiter=limit=0.95:level=false:latency=1[out]",
                       "-map", "[out]", "-t", f"{duration:.9f}", "-c:a", "pcm_f32le", str(target)])
        target.replace(output)
    measurement = (_measure(native_power, _powers(_decode(output, duration), duration), bed_power,
                            cues=rows, speech=speech, duration=duration) if adaptive else None)
    plan = {"output": str(output), "duration": duration, "policy_version": POLICY_VERSION,
            "normalization_target_lufs": NORMALIZATION_TARGET_LUFS,
            "duck_max_reduction_db": duck_db, "adaptive": adaptive,
            "repair_attempts": repair_attempt, "local_corrections_db": corrections.copy(),
            "cues": rows, "speech_windows": speech, "binding": binding or {},
            "native_sha256": media._sha256(native), "bed_sha256": media._sha256(bed),
            "prepared_music_sha256": media._sha256(output), "measurement": measurement,
            "automatic_gain_min_db": round(min(base_gains, default=0), 3),
            "automatic_gain_max_db": round(max(base_gains, default=0), 3),
            "speech_windows_sha256": hashlib.sha256(json.dumps(speech, sort_keys=True).encode()).hexdigest(),
            "cue_plan_sha256": hashlib.sha256(json.dumps(cues, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "automation": {"policy_version": POLICY_VERSION, "duration": duration, "step": _STEP,
                           "gain_db": base_gains, "cues": rows,
                           "bed_sha256": media._sha256(bed),
                           "speech_windows": speech,
                           "speech_windows_sha256": hashlib.sha256(json.dumps(speech, sort_keys=True).encode()).hexdigest(),
                           "duck_max_reduction_db": duck_db}}
    return plan


def reuse_prepared(native: Path, bed: Path, reuse: dict, *, duration: float,
                   cues: list[dict], binding: dict | None = None) -> dict:
    """Adopt a hash-bound prepared stem for subtitle-only manual rendering.

    Its old duck envelope is preserved byte-for-byte. New caption times do not
    become a request to change music. This performs file/hash validation only,
    with no audio decoding, level analysis or automatic QA.
    """
    if not isinstance(reuse, dict):
        raise media.MediaError("已保存配乐工作轨信息无效。")
    try:
        prepared = media._local(Path(reuse["file"]))
        automation_file = media._local(Path(reuse["automation_file"]))
        if (media._sha256(prepared) != reuse["sha256"]
                or media._sha256(automation_file) != reuse["automation_sha256"]
                or automation_file.stat().st_size > 8 * 1024 * 1024):
            raise ValueError("hash mismatch")
        automation = json.loads(automation_file.read_text(encoding="utf-8"))
        rows = _cue_rows(cues, duration)
        baseline = _baseline(automation, rows)
        if (abs(baseline["duration"] - duration) > 0.000001
                or automation.get("bed_sha256") != media._sha256(bed)
                or any(abs(old[key] - new[key]) > 0.000001
                       for old, new in zip(baseline["cues"], rows)
                       for key in ("start", "end", "source_start"))):
            raise ValueError("arrangement changed")
    except (KeyError, TypeError, OSError, ValueError) as exc:
        raise media.MediaError("已保存配乐工作轨或自动化基准与当前版本不匹配；未重新混音。") from exc
    return {"output": str(prepared), "duration": duration, "policy_version": POLICY_VERSION,
            "normalization_target_lufs": NORMALIZATION_TARGET_LUFS,
            "duck_max_reduction_db": automation.get("duck_max_reduction_db", MAX_DUCK_DB),
            "adaptive": False, "repair_attempts": 0, "local_corrections_db": {},
            "cues": rows, "speech_windows": automation.get("speech_windows", []),
            "binding": binding or {}, "native_sha256": media._sha256(native),
            "bed_sha256": media._sha256(bed), "prepared_music_sha256": media._sha256(prepared),
            "measurement": None, "automatic_gain_min_db": min(baseline["gain_db"], default=0),
            "automatic_gain_max_db": max(baseline["gain_db"], default=0),
            "speech_windows_sha256": automation.get("speech_windows_sha256", ""),
            "cue_plan_sha256": hashlib.sha256(json.dumps(cues, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "automation": automation, "reused_prepared_music": True}


def audit_mix(native: Path, prepared: Path, final: Path, *, plan: dict) -> dict:
    """Check the stems and independently verify their contribution to export.

    A 2-variable projection in local 1 s windows estimates native/music gains
    after final loudness processing. AAC and time-varying loudness can introduce
    residual error; correlated stems are explicitly inconclusive, never passed.
    """
    duration = plan["duration"]
    native, prepared, final = media._local(native), media._local(prepared), media._local(final)
    if media._sha256(native) != plan["native_sha256"] or media._sha256(prepared) != plan["prepared_music_sha256"]:
        raise media.MediaError("配乐检查所用原声或音乐已改变，无法绑定当前成片。")
    ns, ms, fs = _decode(native, duration), _decode(prepared, duration), _decode(final, duration)
    measurement = dict(plan.get("measurement") or {})
    issues = list(measurement.get("issues", []))
    finite_count = sum(math.isfinite(value) for value in fs)
    nonfinite_count = len(fs) - finite_count
    finite_values = [value for value in fs if math.isfinite(value)]
    peak = max((abs(value) for value in finite_values), default=0.0)
    clipped_samples = sum(abs(value) >= 1 for value in finite_values)
    if nonfinite_count:
        issues.append({"start": 0, "end": duration, "code": "export_nonfinite_audio",
                       "message": "成片采样发现无效音频数值，不能通过检查。"})
        fs = array.array("f", (value if math.isfinite(value) else 0.0 for value in fs))
    if peak > 1.02 or clipped_samples / max(1, len(fs)) > 0.0001:
        issues.append({"start": 0, "end": duration, "code": "export_clipping",
                       "message": "成片采样发现超出数字满刻度的峰值，需检查失真。"})
    del finite_values
    projections = []
    size = _RATE * _CHANNELS
    core = [cue for cue in plan["cues"]]
    for offset in range(0, min(len(ns), len(ms), len(fs)) - size // 2, size):
        start, end = offset / size, min(duration, (offset + size) / size)
        if not any(_core(cue, (start + end) / 2) for cue in core):
            continue
        n, m, f = ns[offset:offset + size], ms[offset:offset + size], fs[offset:offset + size]
        nn = sum(x * x for x in n)
        mm = sum(x * x for x in m)
        nm = sum(x * y for x, y in zip(n, m))
        nf = sum(x * y for x, y in zip(n, f))
        mf = sum(x * y for x, y in zip(m, f))
        ff = sum(x * x for x in f)
        if _db(mm / max(1, len(m))) < -50:
            continue
        determinant = nn * mm - nm * nm
        if nn / max(1, len(n)) < 1e-11:
            music_gain, native_gain = mf / max(mm, 1e-20), None
            fit = music_gain * mf / max(ff, 1e-20)
        elif determinant > max(1e-20, nn * mm * 0.01):
            native_gain = (nf * mm - mf * nm) / determinant
            music_gain = (mf * nn - nf * nm) / determinant
            fit = (native_gain * nf + music_gain * mf) / max(ff, 1e-20)
        else:
            projections.append({"start": start, "end": end, "status": "inconclusive",
                                "reason": "原声与配乐相关性过高，无法独立估计贡献。"})
            continue
        relative = (20 * math.log10(music_gain / native_gain)
                    if native_gain is not None and native_gain > 0 and music_gain > 0 else None)
        contribution_db = _db(music_gain * music_gain * mm / max(1, len(m))) if music_gain > 0 else -160
        status = "verified" if (fit >= 0.8 and music_gain > 0 and contribution_db >= -48
            and (relative is None or -4 <= relative <= 4)) else "unverified"
        projections.append({"start": start, "end": end, "status": status,
                            "fit_r2": round(fit, 5),
                            "music_gain": round(music_gain, 5),
                            "estimated_final_music_rms_dbfs": round(contribution_db, 2),
                            "native_gain": round(native_gain, 5) if native_gain is not None else None,
                            "relative_gain_change_db": round(relative, 3) if relative is not None else None})
    verified = sum(row["end"] - row["start"] for row in projections if row["status"] == "verified")
    sampled = sum(row["end"] - row["start"] for row in projections)
    if measurement.get("planned_music_seconds", 0) > 0 and (sampled == 0 or verified / sampled < 0.7):
        issues.append({"start": 0, "end": duration, "code": "export_music_unverified",
                       "message": "无法在足够多的成片区间独立确认音乐混入；请试听，不能标记数值检查通过。"})
    report = {**measurement, "status": "needs_attention" if issues else "passed", "issues": issues,
              "policy_version": POLICY_VERSION, "checked_at": datetime.now(timezone.utc).isoformat(),
              "repair_attempts": plan["repair_attempts"], "binding": plan["binding"],
              "local_corrections_db": plan["local_corrections_db"],
              "normalization_target_lufs": NORMALIZATION_TARGET_LUFS,
              "duck_max_reduction_db": plan["duck_max_reduction_db"],
              "source_analysis_used": bool(plan["binding"].get("source_analysis_used", False)),
              "native_sha256": plan["native_sha256"], "bed_sha256": plan["bed_sha256"],
              "prepared_music_sha256": plan["prepared_music_sha256"],
              "final_sha256": media._sha256(final), "speech_windows_sha256": plan["speech_windows_sha256"],
              "cue_plan_sha256": plan["cue_plan_sha256"],
              "automatic_gain_min_db": plan["automatic_gain_min_db"],
              "automatic_gain_max_db": plan["automatic_gain_max_db"],
              "export_projection": projections,
              "final_sample_measurement": {"sample_rate": _RATE, "channels": _CHANNELS,
                  "nonfinite_samples": nonfinite_count, "peak_linear": round(peak, 6),
                  "samples_at_or_above_full_scale": clipped_samples},
              "summary": ("配乐局部电平与成片混入数值检查通过，仍需你试听并审批。" if not issues
                          else "配乐数值检查发现需试听确认的区间；已保留可试听版本，未自动批准。")}
    return report
