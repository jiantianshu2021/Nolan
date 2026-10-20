"""Local, source-bound acoustic evidence for music selection and level matching.

These measurements describe energy, silence and loudness. They do not identify
beats, musical phrases, mood, a chorus or a dramatic climax. No audio is uploaded,
and original music is never changed. Detailed envelopes stay in the local cache;
``catalog_analysis`` exposes only a small set of measurable excerpt candidates.
"""
from __future__ import annotations

import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from . import media


POLICY_VERSION = "music-acoustic-v1"
_RATE = 8000
_STEP = 0.5
_FLOOR = -120.0
_MAX_DURATION = 7200.0
_LIMITATIONS = "仅声学能量与低能量边界，不代表试听、节拍、乐句或高潮识别。"


class MusicAnalysisError(media.MediaError):
    """A local music analysis failure suitable for the review room."""

    safe_message = True


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _finite(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MusicAnalysisError(f"{name}必须是有限数值。")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise MusicAnalysisError(f"{name}超出允许范围。")
    return result


def _source(path: Path) -> tuple[Path, str]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise MusicAnalysisError("音乐原件不存在，不能分析或归一化。")
    return path, _digest(path)


def _metadata(path: Path) -> dict:
    info = media.probe(path)
    _finite(info.get("duration"), "音乐实测时长", 0.01, _MAX_DURATION)
    if not info.get("has_audio") or info.get("has_video"):
        raise MusicAnalysisError("音乐分析需要有效的独立音频文件。")
    return info


def _cache_path(cache_dir: Path | None, kind: str, identity: dict) -> Path | None:
    if cache_dir is None:
        return None
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    key = hashlib.sha256(raw.encode()).hexdigest()
    return Path(cache_dir).expanduser().resolve() / f"{kind}-{key}.json"


def _read_cache(path: Path | None, identity: dict) -> dict | None:
    if path is None or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"),
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(result, dict) or result.get("identity") != identity:
            return None
        data = result["data"]
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(serialized.encode()).hexdigest() != result.get("data_sha256"):
            return None
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(path: Path | None, identity: dict, result: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    data = {"identity": identity, "data": result,
            "data_sha256": hashlib.sha256(serialized.encode()).hexdigest()}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".music-analysis-", delete=False) as source:
        temporary = Path(source.name)
        try:
            json.dump(data, source, ensure_ascii=False, allow_nan=False)
            source.flush()
            os.fsync(source.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _db(power: float) -> float:
    return max(_FLOOR, 10 * math.log10(power)) if power > 0 else _FLOOR


def _envelope(path: Path, duration: float) -> list[dict]:
    """Keep stereo channel energy, avoiding cancellation from a mono downmix."""
    with tempfile.TemporaryDirectory(prefix="nolan-music-energy-") as raw:
        decoded = Path(raw) / "energy.f32"
        media._ffmpeg(["-protocol_whitelist", "file,pipe", "-i", str(path), "-vn",
                       "-t", f"{duration:.9f}", "-ar", str(_RATE), "-ac", "2",
                       "-c:a", "pcm_f32le", "-f", "f32le", str(decoded)])
        if decoded.stat().st_size > round((_MAX_DURATION + 1) * _RATE * 2 * 4):
            raise MusicAnalysisError("音乐解码长度异常，没有采用分析结果。")
        result = []
        frames_per_step = round(_RATE * _STEP)
        position = 0
        with decoded.open("rb") as source:
            for raw_samples in iter(lambda: source.read(frames_per_step * 2 * 4), b""):
                samples = array.array("f")
                samples.frombytes(raw_samples)
                if sys.byteorder != "little":
                    samples.byteswap()
                if not samples or len(samples) % 2 or any(not math.isfinite(x) for x in samples):
                    raise MusicAnalysisError("音乐包含非有限或损坏的采样，未放大或混入。")
                power = sum(float(x) * float(x) for x in samples) / len(samples)
                peak = max(abs(x) for x in samples)
                length = len(samples) / (2 * _RATE)
                result.append({"start": round(position / _RATE, 6),
                               "end": round(position / _RATE + length, 6),
                               "rms_dbfs": round(_db(power), 3),
                               "peak_dbfs": round(_db(peak * peak), 3)})
                position += len(samples) // 2
        return result


def _average(frames: list[dict]) -> float:
    duration = sum(frame["end"] - frame["start"] for frame in frames)
    power = sum(10 ** (frame["rms_dbfs"] / 10) * (frame["end"] - frame["start"])
                for frame in frames) / max(duration, 0.000001)
    return _db(power)


def _regions(envelope: list[dict], threshold: float) -> list[dict]:
    active = [i for i, frame in enumerate(envelope) if frame["rms_dbfs"] > threshold]
    if not active:
        raise MusicAnalysisError("音乐为静音或电平过低，不能自动放大噪声。")
    first, last = active[0], active[-1]
    length = envelope[last]["end"] - envelope[first]["start"]
    window = min(30.0, max(4.0, length / 4))
    width = max(1, min(last - first + 1, round(window / _STEP)))
    # Evaluate measured windows, not title/prompt metadata. Windows with too much
    # silence are not proposed even if they have the lowest mean energy.
    windows = []
    stride = max(1, round(1.0 / _STEP))
    starts = list(range(first, max(first + 1, last - width + 2), stride))
    if last - width + 1 not in starts:
        starts.append(max(first, last - width + 1))
    for index in sorted(set(starts)):
        frames = envelope[index:min(last + 1, index + width)]
        if sum(frame["rms_dbfs"] <= threshold for frame in frames) > len(frames) * 0.2:
            continue
        windows.append((index, _average(frames)))
    if not windows:
        raise MusicAnalysisError("音乐有效声音过于零散，不能可靠提供连续选段。")
    sorted_windows = sorted(windows, key=lambda row: row[1])
    targets = [("low", sorted_windows[0]),
               ("medium", sorted_windows[len(sorted_windows) // 2]),
               ("high", sorted_windows[-1]),
               ("ending", windows[-1])]
    result = []
    seen = set()
    for energy, (index, _) in targets:
        # Search +/- 1 second for a lower-energy cut, with an audible frame just
        # after it. This is a low-energy hint, explicitly not a phrase boundary.
        nearby = [i for i in range(max(first, index - 2), min(last, index + 2) + 1)
                  if envelope[i]["rms_dbfs"] > threshold]
        begin = min(nearby, key=lambda i: (envelope[i]["rms_dbfs"], abs(i - index)))
        end_index = min(last, begin + width - 1)
        if energy == "ending":
            end_index = last
            begin = max(first, last - width + 1)
        if (begin, end_index) in seen:
            continue
        seen.add((begin, end_index))
        frames = envelope[begin:end_index + 1]
        start, end = frames[0]["start"], frames[-1]["end"]
        result.append({"start": start, "end": end,
                       "available_duration": round(end - start, 6), "energy": energy,
                       "mean_dbfs": round(_average(frames), 2),
                       "quiet_start_dbfs": frames[0]["rms_dbfs"],
                       "quiet_end_dbfs": frames[-1]["rms_dbfs"]})
    return result


def analyze_track(path: Path, cache_dir: Path | None = None) -> dict:
    """Measure a complete local track, reusing only the same bytes and policy."""
    path, digest = _source(path)
    identity = {"policy_version": POLICY_VERSION, "sha256": digest, "kind": "track"}
    cached = _read_cache(_cache_path(cache_dir, "track", identity), identity)
    if cached is not None:
        return cached
    info = _metadata(path)
    envelope = _envelope(path, info["duration"])
    if not envelope:
        raise MusicAnalysisError("音乐未解码出有效采样。")
    loudest = max(frame["rms_dbfs"] for frame in envelope)
    if loudest <= -60:
        raise MusicAnalysisError("音乐为静音或电平过低，不能自动放大噪声。")
    threshold = max(-60.0, loudest - 40)
    result = {"policy_version": POLICY_VERSION, "sha256": digest,
              "duration": info["duration"], "sample_step": _STEP,
              "mean_dbfs": round(_average(envelope), 2),
              "silence_threshold_dbfs": round(threshold, 3),
              "envelope": envelope, "regions": _regions(envelope, threshold),
              "limitations": _LIMITATIONS}
    if _digest(path) != digest:
        raise MusicAnalysisError("音乐原件在分析期间改变，结果没有采用。")
    _write_cache(_cache_path(cache_dir, "track", identity), identity, result)
    return result


def catalog_analysis(track: dict, path: Path, cache_dir: Path | None = None) -> dict:
    """Compact measured evidence for GPT; the full envelope is never returned."""
    report = analyze_track(path, cache_dir)
    if track.get("sha256") and report["sha256"] != track["sha256"]:
        raise MusicAnalysisError("音乐分析与已批准原件不一致，不能选段。")
    return {"policy_version": report["policy_version"], "sha256": report["sha256"],
            "duration": report["duration"], "regions": report["regions"][:4],
            "limitations": _LIMITATIONS}


def normalization_filter(path: Path, start: float, duration: float, *,
                         cache_dir: Path | None = None, target_lufs: float = -18,
                         peak_dbfs: float = -2) -> dict:
    """Measure true peak/LUFS, then return a bounded static gain filter.

    Static gain preserves the track's dynamics. The true-peak ceiling and a
    maximum +20 dB gain take precedence over reaching the target LUFS. Callers
    may measure the whole track once to keep consecutive cues at the same gain.
    """
    start = _finite(start, "音乐素材入点", 0, _MAX_DURATION)
    duration = _finite(duration, "音乐测量时长", 0.4, _MAX_DURATION)
    target_lufs = _finite(target_lufs, "音乐目标响度", -30, -10)
    peak_dbfs = _finite(peak_dbfs, "音乐峰值上限", -9, -1)
    path, digest = _source(path)
    identity = {"policy_version": POLICY_VERSION, "sha256": digest, "kind": "normalization",
                "start": start, "duration": duration, "target_lufs": target_lufs,
                "peak_dbfs": peak_dbfs}
    cache = _cache_path(cache_dir, "normalization", identity)
    cached = _read_cache(cache, identity)
    if cached is not None:
        return cached
    info = _metadata(path)
    if start + duration > info["duration"] + 0.000001:
        raise MusicAnalysisError("音乐测量范围超过原件时长。")
    filters = (f"atrim=start={start:.9f}:duration={duration:.9f},asetpts=PTS-STARTPTS,"
               "aresample=48000,aformat=channel_layouts=stereo,"
               f"loudnorm=I={target_lufs:.6f}:TP={peak_dbfs:.6f}:LRA=11:print_format=json")
    try:
        process = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-nostats",
                                  "-protocol_whitelist", "file,pipe", "-i", str(path),
                                  "-vn", "-af", filters, "-f", "null", "-"],
                                 capture_output=True, text=True, stdin=subprocess.DEVNULL,
                                 timeout=600, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise MusicAnalysisError("音乐响度测量未完成，请检查本地 FFmpeg。") from exc
    if process.returncode:
        raise MusicAnalysisError("本地音乐响度测量失败，原件未改变。")
    matches = re.findall(r'\{\s*"input_i".*?\}', process.stderr, re.DOTALL)
    try:
        values = json.loads(matches[-1])
        loudness, peak = float(values["input_i"]), float(values["input_tp"])
    except (ValueError, KeyError, IndexError, TypeError):
        raise MusicAnalysisError("音乐响度没有有效实测值，不能猜测增益。") from None
    if not math.isfinite(loudness) or not math.isfinite(peak) or loudness < -55:
        raise MusicAnalysisError("音乐为静音或电平过低，不能自动放大噪声。")
    requested = target_lufs - loudness
    gain = min(requested, peak_dbfs - peak, 20.0)
    if not math.isfinite(gain) or gain < -90:
        raise MusicAnalysisError("音乐响度异常，不能自动调整。")
    warnings = []
    peak_limited = gain < requested - 0.01 and peak_dbfs - peak <= 20
    if peak_limited:
        warnings.append("为保留动态并限制真实峰值，本曲未精确达到目标响度。")
    if requested > 20 and gain == 20:
        warnings.append("音乐最多提升 20 dB，避免过度放大安静素材与底噪。")
    result = {"policy_version": POLICY_VERSION, "sha256": digest,
              "start": start, "duration": duration, "target_lufs": target_lufs,
              "peak_ceiling_dbfs": peak_dbfs, "input_lufs": loudness,
              "input_true_peak_dbfs": peak, "gain_db": round(gain, 6),
              "output_lufs_estimate": round(loudness + gain, 3),
              "output_true_peak_estimate": round(peak + gain, 3),
              "peak_limited": peak_limited,
              "filter": f"volume={gain:.6f}dB", "warnings": warnings}
    if _digest(path) != digest:
        raise MusicAnalysisError("音乐原件在响度测量期间改变，结果没有采用。")
    _write_cache(cache, identity, result)
    return result
