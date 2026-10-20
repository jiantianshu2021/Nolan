"""Local, deterministic media operations. No model or network calls are made here.

Every media argument is a local file, subprocesses use argv (never a shell), and
completed outputs are published only after successful encoding.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any


class MediaError(RuntimeError):
    """A local dependency, input, or media operation failed."""


# FFmpeg 7.1 loudnorm can emit NaN samples for completely silent input. Keep
# silence exactly silent after normalization and before sidechain/encoding.
_LOUDNORM = (
    "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,"
    "aeval=exprs='if(isnan(val(0))+isinf(val(0)),0,val(0))|"
    "if(isnan(val(1))+isinf(val(1)),0,val(1))'"
)


def _run(args: list[str], *, cwd: Path | None = None, timeout: int = 600) -> str:
    try:
        result = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL, check=False,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"缺少本地工具：{args[0]}。请安装 FFmpeg（包含 ffprobe）。") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"本地媒体处理超时（{timeout} 秒）；原素材未改变。") from exc
    if result.returncode:
        raise MediaError(f"{args[0]} 处理失败：{result.stderr.strip()[-2400:]}")
    return result.stdout


def _ffmpeg(args: list[str], *, cwd: Path | None = None) -> None:
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args], cwd=cwd)


def _local(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise MediaError(f"本地媒体文件不存在：{path}")
    return path


def _destination(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    """Describe actual local streams, including audio-only files.

    ``duration`` is video duration when present, otherwise container duration;
    this avoids extending a picture because a provider leaves trailing audio.
    """
    path = _local(path)
    data = json.loads(_run([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-show_format", "-show_streams", "-of", "json", str(path),
    ], timeout=60))
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    format_duration = _number(data.get("format", {}).get("duration"))
    video_duration = _number((video or {}).get("duration"))
    duration = video_duration or format_duration
    return {
        "path": str(path), "duration": duration, "format_duration": format_duration,
        "width": int((video or {}).get("width", 0)),
        "height": int((video or {}).get("height", 0)),
        "has_video": video is not None, "has_audio": audio is not None,
        "video_codec": (video or {}).get("codec_name"),
        "audio_codec": (audio or {}).get("codec_name"),
        "sample_rate": int((audio or {}).get("sample_rate", 0)),
        "channels": int((audio or {}).get("channels", 0)),
        "frame_rate": (video or {}).get("avg_frame_rate"),
        "size": path.stat().st_size,
    }


def prepare_voice_references(
    paths: list[Path], output_dir: Path,
) -> tuple[list[Path], list[dict]]:
    """Fit already-approved local voice samples into the provider's 15 s limit.

    The caller must verify approval and source hashes first. This only takes a
    beginning excerpt: it does not regenerate, speed up, join, or approve voices.
    Original samples and their approval files remain unchanged. Cropping can cut
    off a sentence and is only for reference conditioning, not a dialogue edit.
    """
    if len(paths) > 3:
        raise ValueError("单次最多使用 3 条已批准的声音参考。")
    if not paths:
        return [], []
    sources = []
    for raw_path in paths:
        try:
            path = _local(raw_path)
            info = probe(path)
        except (MediaError, OSError, ValueError) as exc:
            raise ValueError(f"声音参考缺失或无法读取：{Path(raw_path).name}。") from exc
        if (not info["has_audio"] or info["has_video"] or info["duration"] < 2
                or info["sample_rate"] <= 0):
            raise ValueError(f"声音参考必须是至少 2 秒的有效音频文件：{path.name}。")
        sources.append((path, info, _sha256(path)))
    duration_sum = sum(info["duration"] for _, info, _ in sources)
    limit = (15.0 if duration_sum <= 15.0
             else math.floor(14.9 / len(sources) * 1000) / 1000)
    prepared: list[Path] = []
    metadata: list[dict] = []
    for source, info, source_sha in sources:
        if info["duration"] <= limit:
            prepared.append(source)
            metadata.append({
                "source_sha256": source_sha, "sha256": source_sha,
                "trim_start": 0.0, "trim_duration": info["duration"], "derived": False,
            })
            continue
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"voice-ref-{source_sha}-0-{round(limit * 1000)}ms.wav"
        manifest = target.with_suffix(".json")
        # One lock covers both immutable files, so concurrent callers never
        # observe an unfinished pair or replace an existing cached derivative.
        with (directory / ".voice-references.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.exists() or manifest.exists() or target.is_symlink() or manifest.is_symlink():
                try:
                    if (target.is_symlink() or manifest.is_symlink()
                            or not target.is_file() or not manifest.is_file()):
                        raise ValueError("缓存文件不完整")
                    cached = json.loads(manifest.read_text(encoding="utf-8"))
                    actual = probe(target)
                    if (cached.get("version") != 1 or cached.get("source_sha256") != source_sha
                            or cached.get("requested_duration") != limit
                            or cached.get("trim_start") != 0.0 or cached.get("derived") is not True
                            or cached.get("sha256") != _sha256(target)
                            or not actual["has_audio"] or actual["has_video"]
                            or not 2 <= actual["duration"] <= limit
                            or cached.get("trim_duration") != actual["duration"]):
                        raise ValueError("缓存校验不通过")
                except (OSError, MediaError, ValueError, AttributeError) as exc:
                    raise ValueError("声音参考裁切缓存已改变或不完整；停止复用，原始审批音频未改变。") from exc
            else:
                with tempfile.TemporaryDirectory(prefix=".norlan-voice-ref-", dir=directory) as raw:
                    temporary = Path(raw)
                    candidate = temporary / "reference.wav"
                    # Sample-based trimming never rounds past the allocated
                    # duration, including inputs with non-24/48 kHz sample rates.
                    end_sample = math.floor(limit * info["sample_rate"])
                    _ffmpeg([
                        "-protocol_whitelist", "file,pipe", "-i", str(source),
                        "-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1",
                        "-af", f"atrim=start_sample=0:end_sample={end_sample},asetpts=PTS-STARTPTS",
                        "-c:a", "pcm_s16le", str(candidate),
                    ])
                    actual = probe(candidate)
                    if (not actual["has_audio"] or actual["has_video"]
                            or not 2 <= actual["duration"] <= limit):
                        raise ValueError("声音参考裁切结果时长无效；原始审批音频未改变。")
                    if _sha256(source) != source_sha:
                        raise ValueError("声音参考原件在裁切期间发生改变；请重新核对审批。")
                    cached = {
                        "version": 1, "source_sha256": source_sha, "sha256": _sha256(candidate),
                        "requested_duration": limit, "trim_start": 0.0,
                        "trim_duration": actual["duration"], "derived": True,
                    }
                    candidate_manifest = temporary / "reference.json"
                    candidate_manifest.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
                    # Exclusive publication refuses to overwrite even if another
                    # process ignores the lock and creates the same destination.
                    os.link(candidate, target)
                    os.link(candidate_manifest, manifest)
        prepared.append(target)
        metadata.append({key: cached[key] for key in (
            "source_sha256", "sha256", "trim_start", "trim_duration", "derived",
        )})
    if sum(item["trim_duration"] for item in metadata) > 15.0:
        raise ValueError("声音参考总时长仍超过 15 秒，停止提交。")
    return prepared, metadata


@functools.lru_cache(maxsize=1)
def _filters() -> str:
    return _run(["ffmpeg", "-hide_banner", "-filters"], timeout=30)


def _has_filter(name: str) -> bool:
    return any(len(parts := line.split()) > 1 and parts[1] == name
               for line in _filters().splitlines())


def _font_path() -> Path | None:
    # Prefer CJK fonts when present. No font is downloaded at runtime.
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    return next((Path(p) for p in candidates if Path(p).is_file()), None)


def _demo_drawtext(directory: Path, label: str) -> str:
    if not _has_filter("drawtext"):
        raise MediaError("演示素材需要带 drawtext 的 FFmpeg，以明确标记占位画面。")
    # Text is read as data; expansion=none prevents %{...} interpretation.
    text = "NOLAN DEMO / SYNTHETIC PLACEHOLDER\n" + str(label)[:500]
    (directory / "label.txt").write_text(text, encoding="utf-8")
    font = _font_path()
    if font:
        shutil.copyfile(font, directory / "font.dat")
    font_option = "fontfile=font.dat:" if font else ""
    return (f"drawtext={font_option}textfile=label.txt:expansion=none:"
            "fontsize=24:fontcolor=white:line_spacing=10:x=24:y=28:"
            "box=1:boxcolor=black@0.65:boxborderw=12")


def synthetic_asset(path: Path, label: str, color: str = "#34495e") -> Path:
    """Make an obviously labelled placeholder image, never a character candidate."""
    path = _destination(path)
    colors = {"blue", "red", "green", "black", "white", "gray", "purple", "orange", "navy", "teal"}
    if color not in colors and not re.fullmatch(r"#?[0-9a-fA-F]{6}", color):
        raise MediaError("演示颜色必须为六位十六进制值或常用颜色名。")
    if re.fullmatch(r"[0-9a-fA-F]{6}", color):
        color = "#" + color
    with tempfile.TemporaryDirectory(prefix=".norlan-asset-", dir=path.parent) as raw:
        temporary = Path(raw)
        _ffmpeg([
            "-f", "lavfi", "-i", f"color=c={color}:s=960x540:r=1",
            "-vf", _demo_drawtext(temporary, label), "-frames:v", "1",
            "-update", "1", "-c:v", "png", str(temporary / "image.png"),
        ], cwd=temporary)
        os.replace(temporary / "image.png", path)
    return path


def synthetic_clip(path: Path, label: str, duration: float, index: int = 0) -> Path:
    """Create an explicitly synthetic animated clip with a quiet test tone."""
    duration = _number(duration)
    if not 0.04 <= duration <= 120:
        raise MediaError("演示镜头时长必须在 0.04 至 120 秒之间。")
    path = _destination(path)
    palette = ["#284b63", "#3c6e71", "#6b3e56", "#68502d"]
    color = palette[int(index) % len(palette)]
    frequency = 220 + int(index) % 12 * 30
    with tempfile.TemporaryDirectory(prefix=".norlan-clip-", dir=path.parent) as raw:
        temporary = Path(raw)
        text_filter = _demo_drawtext(temporary, label + "\nTEST TONE / NO DIALOGUE")
        _ffmpeg([
            "-f", "lavfi", "-i", f"color=c={color}:s=960x540:r=30:d={duration:.6f}",
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000:duration={duration:.6f}",
            "-vf", "hue=h=5*sin(t),drawgrid=w=80:h=60:t=1:c=white@0.08," + text_filter,
            "-af", "volume=0.2,aformat=channel_layouts=stereo",
            "-map", "0:v:0", "-map", "1:a:0", "-t", f"{duration:.6f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(temporary / "clip.mp4"),
        ], cwd=temporary)
        os.replace(temporary / "clip.mp4", path)
    return path


def _subshot_frame_samples(video, metadata, subshots):
    """Map planned sample points to decoded frame indices and actual times."""
    from .segments import validate_subshots
    try:
        planned_duration = subshots[-1]["end"]
        validate_subshots({"duration": planned_duration, "subshots": subshots})
    except (ValueError, TypeError, KeyError, IndexError):
        raise MediaError("子镜采样计划无效；请检查连续时间范围与总时长。") from None
    data = json.loads(_run([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-select_streams", "v:0", "-show_frames", "-show_entries",
        "frame=best_effort_timestamp_time", "-of", "json", str(video),
    ], timeout=60))
    try:
        timestamps = [float(frame["best_effort_timestamp_time"]) for frame in data["frames"]]
        if not timestamps or any(not math.isfinite(value) for value in timestamps):
            raise ValueError
        timestamps = [value - timestamps[0] for value in timestamps]
        if timestamps != sorted(timestamps) or timestamps[-1] > metadata["duration"] + 0.1:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise MediaError("无法核实视频帧时间，不能用计划时间冒充实测尾帧。") from None
    tail_time = timestamps[-1]
    desired = [0.0, tail_time]
    desired += [(item["start"] + item["end"]) / 2 for item in subshots]
    cuts = [item["end"] for item in subshots[:-1]]
    for index, cut in enumerate(cuts):
        # Keep samples on each planned side, even for a short subshot. These
        # positions are not a claim that the generated video cuts here.
        left = subshots[index]["end"] - subshots[index]["start"]
        right = subshots[index + 1]["end"] - subshots[index + 1]["start"]
        margin = min(0.12, left / 4, right / 4)
        desired.extend((cut - margin, cut + margin))
    desired = sorted({max(0.0, min(float(value), tail_time)) for value in desired})
    # Exact frame-index selection avoids an output seek returning a different
    # frame than the time reported here, including variable-frame-rate video.
    indices = {0, len(timestamps) - 1}
    for instant in desired:
        indices.add(min(range(len(timestamps)), key=lambda index: abs(timestamps[index] - instant)))
    indices = sorted(indices)
    return [(index, timestamps[index]) for index in indices], cuts, planned_duration


def extract_evidence(video: Path, outdir: Path, *, subshots=None) -> dict:
    """Extract bounded sampled frames and audio; this is evidence, not a review.

    Legacy two-argument calls retain three samples. A subshot plan samples each
    planned middle and both sides of planned cuts, plus the actual first/last
    decoded frames. Three subshots produce at most nine images.
    """
    video = _local(video)
    metadata = probe(video)
    if not metadata["has_video"] or metadata["duration"] <= 0:
        raise MediaError("抽帧需要具有有效时长的视频。")
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    duration = metadata["duration"]
    if subshots is None:
        # Preserve existing projects' first/middle/near-tail sampling behavior.
        times = [0.0, duration / 2, max(0.0, duration - min(0.10, duration / 4))]
        frame_indices = [None] * 3
        names = ["first", "middle", "tail"]
    else:
        samples, planned_cuts, planned_duration = _subshot_frame_samples(video, metadata, subshots)
        frame_indices = [index for index, _ in samples]
        times = [instant for _, instant in samples]
        names = ["first" if index == 0 else f"sample-{index:02}" for index in range(len(times))]
        names[-1] = "tail"
    frames: list[str] = []
    for name, instant, frame_index in zip(names, times, frame_indices, strict=True):
        target = outdir / f"{name}.png"
        args = ["-protocol_whitelist", "file,pipe", "-i", str(video)]
        scaling = "scale=1280:1280:force_original_aspect_ratio=decrease"
        if frame_index is None:
            args += ["-ss", f"{instant:.6f}", "-vf", scaling]
        else:
            args += ["-vf", f"select=eq(n\\,{frame_index})," + scaling]
        _ffmpeg(args + ["-frames:v", "1", "-update", "1", str(target)])
        if not target.is_file():
            raise MediaError(f"未能提取 {name} 帧；不能把缺失证据当作通过。")
        frames.append(str(target))
    audio_path: str | None = None
    if metadata["has_audio"]:
        audio = outdir / "audio.wav"
        _ffmpeg([
            "-protocol_whitelist", "file,pipe", "-i", str(video),
            "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "24000",
            "-c:a", "pcm_s16le", str(audio),
        ])
        audio_path = str(audio)
    result = {
        "frames": frames, "tail_frame": frames[-1], "audio": audio_path,
        "metadata": metadata, "sha256": _sha256(video), "sample_times": times,
        "sampling_note": "仅抽取首、中、尾三个画面；尾帧为片尾附近采样。抽帧不能证明完整动作、口型或声音连续性；音轨须另行听审。",
    }
    if subshots is not None:
        result.update(sampling_mode="planned_subshot_samples", planned_duration=planned_duration,
                      planned_cut_points=planned_cuts, frame_indices=frame_indices,
                      sampling_note="按计划子镜中点及计划切点前后抽样，并保留实测首帧和最后一帧；时间已截取到实际视频范围。"
                                    "这些只是计划点采样，不代表已检测或确认实际切镜。抽帧不能证明完整动作、口型或声音连续性；音轨须另行听审。")
    return result


def _srt_time(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    seconds, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def _write_subtitles(entries: list[dict], target: Path, duration: float) -> None:
    cues = []
    for entry in entries:
        start = _number(entry.get("start"), -1)
        end = _number(entry.get("end"), -1)
        if start < 0 or end <= start or end > duration + 0.05:
            raise MediaError("字幕必须提供有效的全片时间戳，且不能超出实测画面时长。")
        text = str(entry.get("text", "")).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()
        if not text:
            raise MediaError("字幕文本不能为空。")
        # Blank lines delimit SRT entries; literal markup must not become styling.
        text = html.escape(text, quote=False)
        text = re.sub(r"\n\s*\n", "\n", text)
        cues.append((start, end, text))
    cues.sort(key=lambda cue: (cue[0], cue[1]))
    target.write_text("".join(
        f"{number}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n\n"
        for number, (start, end, text) in enumerate(cues, 1)
    ), encoding="utf-8")


def compose(
    clips: list[Path], output: Path, subtitles: list[dict] | None = None,
    music: Path | None = None, *, burn_subtitles: bool = True, music_gain: float = 0.18,
    music_mix: dict | None = None,
) -> dict:
    """Normalize and join clips in order, preserving sound and filling silence.

    Music is a single supplied local cue, padded with silence when shorter than
    the episode (never an arbitrary loop). It fades in/out and is ducked by the
    existing soundtrack. If that soundtrack contains music, it cannot isolate
    dialogue; the result requires a human listening pass. Global subtitle times
    must already have been aligned to speech; this function does no recognition.
    An explicit ``music_mix`` policy instead uses the supplied verified speech
    windows for bounded ducking. Automatic policies measure and locally repair
    the music stem at most twice; manual policies only render the user's edits.
    """
    if not clips:
        raise MediaError("至少需要一个已生成镜头才能合成。")
    if isinstance(music_gain, bool) or not isinstance(music_gain, (int, float)) or not math.isfinite(music_gain) or not 0 <= music_gain <= 1:
        raise MediaError("配乐增益必须为0至1的有限数字。")
    paths = [_local(path) for path in clips]
    output = _destination(output)
    if output in paths:
        raise MediaError("合成输出不能覆盖原始镜头。")
    music_path = _local(music) if music is not None else None
    if music_path == output:
        raise MediaError("合成输出不能覆盖配乐原件。")
    if music_mix is not None and (not isinstance(music_mix, dict) or music_path is None):
        raise MediaError("对白配乐混音策略必须附带本地配乐文件。")
    metadata = [probe(path) for path in paths]
    if any(not item["has_video"] or item["duration"] <= 0 for item in metadata):
        raise MediaError("每个镜头都必须包含有效的视频流和时长。")
    if music_path is not None and not probe(music_path)["has_audio"]:
        raise MediaError("配乐文件不包含音频流。")
    first = metadata[0]
    factor = min(1.0, 1280 / max(first["width"], first["height"]))
    width = max(2, int(first["width"] * factor) // 2 * 2)
    height = max(2, int(first["height"] * factor) // 2 * 2)
    timeline: list[dict] = []
    srt_output: Path | None = output.with_suffix(".srt") if subtitles else None
    burned = bool(subtitles and burn_subtitles and _has_filter("subtitles") and _font_path())
    warnings: list[str] = []
    if subtitles and burn_subtitles and not burned:
        warnings.append("未检测到字幕滤镜或字体；已保留 SRT，视频未烧录字幕。")
    if music_path is None:
        warnings.append("没有导入配乐；保留镜头原声，静音镜头填充静音。")
    elif music_mix is not None:
        warnings.append("配乐只按给定对白时间有限压低，未自动分离视频原声；最终听感仍须本人试听。")
    elif any(item["has_audio"] for item in metadata):
        warnings.append("配乐按镜头原生混合音轨压低；无法自动分离其中已有的音乐与对白。")
    mix_report = None
    with tempfile.TemporaryDirectory(prefix=".norlan-compose-", dir=output.parent) as raw:
        temporary = Path(raw)
        normalized: list[Path] = []
        cursor = 0.0
        for index, (path, item) in enumerate(zip(paths, metadata)):
            # Frame rounding is explicit and recorded; at most half a 30 fps frame.
            duration = max(1, round(item["duration"] * 30)) / 30
            target = temporary / f"clip-{index:04d}.mp4"
            args = ["-protocol_whitelist", "file,pipe", "-i", str(path)]
            if not item["has_audio"]:
                args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            audio_input = "0:a:0" if item["has_audio"] else "1:a:0"
            video_filter = (
                f"fps=30,scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                f"tpad=stop_mode=clone:stop_duration=0.1,trim=duration={duration:.9f},setpts=PTS-STARTPTS"
            )
            _ffmpeg([
                *args, "-map", "0:v:0", "-map", audio_input, "-vf", video_filter,
                "-af", f"aresample=48000,aformat=channel_layouts=stereo,apad,atrim=duration={duration:.9f},asetpts=PTS-STARTPTS",
                "-t", f"{duration:.9f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-r", "30", "-fps_mode", "cfr",
                "-c:a", "aac", "-b:a", "192k", str(target),
            ])
            actual = probe(target)["duration"]
            timeline.append({
                "index": index, "source": str(path), "source_sha256": _sha256(path),
                "start": round(cursor, 6), "end": round(cursor + actual, 6),
                "duration": actual, "source_duration": item["duration"],
                "source_has_audio": item["has_audio"],
            })
            cursor += actual
            normalized.append(target)
        duration = cursor
        mix_plan = None
        prepared_music = music_path
        native_reference = None
        if music_mix is not None:
            from . import music_mix as mixer
            if music_mix.get("policy_version", mixer.POLICY_VERSION) != mixer.POLICY_VERSION:
                raise MediaError("当前配乐混音策略版本不受支持。")
            adaptive = music_mix.get("adaptive", True)
            audit = music_mix.get("audit", adaptive)
            if not isinstance(adaptive, bool) or not isinstance(audit, bool) or (audit and not adaptive):
                raise MediaError("配乐手动模式不能运行自动检查或修改。")
            native_reference = temporary / "native-reference.wav"
            native_args, native_graph, native_labels = [], [], []
            for index, (target, segment) in enumerate(zip(normalized, timeline)):
                native_args += ["-i", str(target)]
                native_graph.append(
                    f"[{index}:a:0]atrim=duration={segment['duration']:.9f},asetpts=PTS-STARTPTS[n{index}]")
                native_labels.append(f"[n{index}]")
            native_graph.append("".join(native_labels) +
                                f"concat=n={len(normalized)}:v=0:a=1," + _LOUDNORM + "[reference]")
            _ffmpeg([*native_args, "-filter_complex", ";".join(native_graph), "-map", "[reference]",
                     "-t", f"{duration:.9f}", "-c:a", "pcm_f32le", str(native_reference)])
            prepared_music = temporary / "prepared-music.wav"
            corrections = dict(music_mix.get("local_corrections_db", {}))
            attempt = music_mix.get("repair_attempt", 0)
            on_progress = music_mix.get("on_progress")
            if on_progress is not None and not callable(on_progress):
                raise MediaError("配乐混音进度接收器无效。")
            def notify_mix(status, *, issues=None, summary=None):
                if on_progress is not None:
                    on_progress({"status": status, "repair_attempts": attempt,
                        "local_corrections_db": corrections.copy(),
                        "binding": music_mix.get("binding", {}),
                        "bed_sha256": _sha256(music_path), "issues": issues or [],
                        "summary": summary or "正在本地测量配乐与原声，不调用生成服务。"})
            if adaptive and audit:
                notify_mix("checking")
            while True:
                reuse = music_mix.get("reuse_prepared_music")
                if reuse is not None:
                    if adaptive or audit:
                        raise MediaError("直接复用配乐工作轨只用于手动字幕修改，不运行自动检查。")
                    mix_plan = mixer.reuse_prepared(native_reference, music_path, reuse,
                        duration=duration, cues=music_mix.get("cues", []), binding=music_mix.get("binding", {}))
                    prepared_music = Path(mix_plan["output"])
                else:
                    mix_plan = mixer.prepare_mix(
                        native_reference, music_path, prepared_music, duration=duration,
                        cues=music_mix.get("cues", []), speech_windows=music_mix.get("speech_windows", []),
                        duck_db=music_mix.get("duck_db", mixer.MAX_DUCK_DB), adaptive=adaptive,
                        binding=music_mix.get("binding", {}), repair_attempt=attempt,
                        local_corrections_db=corrections, input_gain=music_gain,
                        baseline_automation=music_mix.get("baseline_automation"))
                measurement = mix_plan.get("measurement") or {}
                if adaptive and audit:
                    notify_mix("checking", issues=measurement.get("issues", []))
                suggestions = measurement.get("repair_suggestions", {})
                if not (adaptive and audit and suggestions and attempt < mixer.MAX_REPAIRS):
                    break
                new_corrections = {**corrections}
                for key, correction in suggestions.items():
                    new_corrections[key] = max(-mixer.MAX_REPAIR_DB, min(mixer.MAX_REPAIR_DB,
                        new_corrections.get(key, 0) + correction))
                if new_corrections == corrections:
                    break
                corrections, attempt = new_corrections, attempt + 1
                notify_mix("repairing", issues=measurement.get("issues", []),
                           summary=f"正在进行第 {attempt}/2 轮有界本地配乐修正；保留原视频和原曲。")
        args: list[str] = []
        graph: list[str] = []
        inputs = ""
        for index, (target, segment) in enumerate(zip(normalized, timeline)):
            args += ["-i", str(target)]
            graph.append(f"[{index}:v:0]setpts=PTS-STARTPTS[v{index}]")
            if music_mix is None:
                graph.append(f"[{index}:a:0]atrim=duration={segment['duration']:.9f},asetpts=PTS-STARTPTS[a{index}]")
                inputs += f"[v{index}][a{index}]"
            else:
                inputs += f"[v{index}]"
        if music_mix is not None:
            # Use the exact reference measured above; never normalize the dry
            # cut a second time or let unrelated environment energy drive duck.
            graph.append(f"{inputs}concat=n={len(normalized)}:v=1:a=0[video]")
        else:
            graph.append(f"{inputs}concat=n={len(normalized)}:v=1:a=1[video][native]")
            graph.append(f"[native]{_LOUDNORM}[norm]")
        if music_path and music_mix is not None:
            args += ["-protocol_whitelist", "file,pipe", "-i", str(prepared_music),
                     "-protocol_whitelist", "file,pipe", "-i", str(native_reference)]
            graph += [f"[{len(normalized) + 1}:a:0]anull[voice]",
                      f"[{len(normalized)}:a:0]apad,atrim=duration={duration:.9f},asetpts=PTS-STARTPTS[ducked]",
                      "[voice][ducked]amix=inputs=2:duration=first:normalize=0," + _LOUDNORM + "[audio]"]
        elif music_path:
            args += ["-protocol_whitelist", "file,pipe", "-i", str(music_path)]
            fade = min(0.6, duration / 4)
            graph += [
                "[norm]asplit=2[voice][sidechain]",
                f"[{len(normalized)}:a:0]aresample=48000,aformat=channel_layouts=stereo,"
                f"apad,atrim=duration={duration:.9f},asetpts=PTS-STARTPTS,volume={music_gain:.6f},"
                f"afade=t=in:st=0:d={fade:.6f},afade=t=out:st={duration-fade:.6f}:d={fade:.6f}[music]",
                "[music][sidechain]sidechaincompress=threshold=0.03:ratio=8:attack=15:release=300[ducked]",
                "[voice][ducked]amix=inputs=2:duration=first:normalize=0," + _LOUDNORM + "[audio]",
            ]
        else:
            graph.append("[norm]anull[audio]")
        if subtitles:
            _write_subtitles(subtitles, temporary / "captions.srt", duration)
        if burned:
            font = _font_path()
            if font:
                shutil.copyfile(font, temporary / font.name)
            # Fixed relative filenames avoid ffmpeg filter escaping of user paths.
            graph.append("[video]subtitles=filename=captions.srt:fontsdir=.[picture]")
        else:
            graph.append("[video]null[picture]")
        _ffmpeg([
            *args, "-filter_complex", ";".join(graph), "-map", "[picture]", "-map", "[audio]",
            "-t", f"{duration:.9f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", "30", "-fps_mode", "cfr",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-ac", "2", "-movflags", "+faststart", str(temporary / "episode.mp4"),
        ], cwd=temporary)
        result_metadata = probe(temporary / "episode.mp4")
        if not result_metadata["has_video"] or not result_metadata["has_audio"]:
            raise MediaError("合成结果缺少预期的视频或音频流。")
        if mix_plan is not None:
            from . import music_mix as mixer
            if music_mix.get("audit", music_mix.get("adaptive", True)):
                mix_report = mixer.audit_mix(native_reference, prepared_music,
                                             temporary / "episode.mp4", plan=mix_plan)
            else:
                from datetime import datetime, timezone
                mix_report = {key: mix_plan[key] for key in
                              ("policy_version", "normalization_target_lufs", "duck_max_reduction_db",
                               "native_sha256", "bed_sha256", "prepared_music_sha256",
                               "speech_windows_sha256", "cue_plan_sha256", "binding")}
                mix_report.update(status="manual", repair_attempts=0, issues=[],
                    checked_at=datetime.now(timezone.utc).isoformat(),
                    source_analysis_used=bool(mix_plan["binding"].get("source_analysis_used", False)),
                    final_sha256=_sha256(temporary / "episode.mp4"),
                    summary="按你的配乐参数直接生成试听版本，未运行自动检查或调整；等待本人最终审批。")
            # Keep the exact heard stem and the pre-duck auto/repair baseline.
            # A later subtitle-only edit can reuse it, and a manual music edit
            # changes only its explicit parameters rather than losing +9 dB of
            # previous automation. Original clips and approved music stay intact.
            stem_output = output.with_suffix(".music.wav")
            automation_output = output.with_suffix(".music-automation.json")
            if any(path in {*paths, music_path, output} for path in (stem_output, automation_output)):
                raise MediaError("配乐工作轨或自动化侧车不能覆盖原素材。")
            staged_stem, staged_automation = temporary / "published-music.wav", temporary / "music-automation.json"
            shutil.copyfile(prepared_music, staged_stem)
            if _sha256(staged_stem) != mix_plan["prepared_music_sha256"]:
                raise MediaError("配乐工作轨在保存过程中发生改变，未发布结果。")
            staged_automation.write_text(json.dumps(mix_plan["automation"], ensure_ascii=False,
                sort_keys=True, allow_nan=False, separators=(",", ":")), encoding="utf-8")
            os.replace(staged_stem, stem_output)
            os.replace(staged_automation, automation_output)
            mix_report.update(prepared_music_file=str(stem_output),
                automation_file=str(automation_output), automation_sha256=_sha256(automation_output),
                local_corrections_db=mix_plan["local_corrections_db"],
                reused_prepared_music=bool(mix_plan.get("reused_prepared_music")))
        os.replace(temporary / "episode.mp4", output)
        if srt_output:
            os.replace(temporary / "captions.srt", srt_output)
    result = {
        "output": str(output), "duration": result_metadata["duration"],
        "subtitles": str(srt_output) if srt_output else None,
        "burned_subtitles": burned, "sha256": _sha256(output), "timeline": timeline,
        "width": width, "height": height, "has_audio": True,
        "music_present": music_path is not None,
        "music_source": str(music_path) if music_path else None,
        "warnings": warnings, "frame_rate": 30,
        "audio_note": "单次响度归一化目标为 -16 LUFS、-1.5 dBTP；最终听感、台词清晰度及声线一致性须听审。",
        "subtitle_note": "字幕时间戳来自调用方；本模块不识别语音，也不证明字幕与台词一致。" if subtitles else "无字幕。",
    }
    if mix_report is not None:
        result["music_mix"] = mix_report
    return result
