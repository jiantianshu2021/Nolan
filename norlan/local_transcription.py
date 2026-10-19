"""Offline timestamped ASR through whisper.cpp; never calls a paid service.

Only observed audio is passed to ASR. Script/character vocabulary belongs in the
separate text-proofreading step, so it cannot become invented spoken dialogue.
CLI/JSON conventions: https://github.com/ggml-org/whisper.cpp/tree/master/examples/cli
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import wave
from typing import Callable

from .media import MediaError, probe
from .providers import ProviderError
from .transcription import _number, _segments


DEFAULT_MODEL_PATH = Path.home() / ".local/share/norlan/models/ggml-small.bin"
DEFAULT_VAD_MODEL_PATH = Path.home() / ".local/share/norlan/models/ggml-silero-v6.2.0.bin"
MAX_AUDIO_BYTES = 512_000_000
MAX_RESPONSE_BYTES = 8_000_000
CACHE_SCHEMA = "nolan-whisper-cpp-v3-windowed-vad-utterances"
TIMING_METHOD = "silero-windowed-speech-original-clock-v2"
VAD_PARAMETERS = {"threshold": 0.5, "min_speech_ms": 250, "min_silence_ms": 100,
                  "speech_pad_ms": 100, "max_speech_seconds": 12,
                  "window_seconds": 20, "window_step_seconds": 10, "recognition_context_seconds": 0.25}
MAX_SPEECH_WINDOWS = 2000


def _vad_windows(output: str, duration: float) -> list[tuple[float, float]]:
    """The official standalone VAD tool prints centiseconds, not milliseconds.

    Recognition uses each original-clock speech window separately. This avoids
    whisper.cpp's internally compressed VAD timeline, whose segment timestamps
    can still span a long silent gap even when VAD is enabled.
    """
    if not isinstance(output, str) or len(output) > 1_000_000:
        raise ValueError("invalid VAD response")
    counts = re.findall(r"^Detected (\d+) speech segments:\s*$", output, re.MULTILINE)
    rows = re.findall(r"^Speech segment (\d+): start = ([\d.]+), end = ([\d.]+)\s*$", output, re.MULTILINE)
    if len(counts) != 1 or int(counts[0]) != len(rows) or len(rows) > MAX_SPEECH_WINDOWS:
        raise ValueError("incomplete VAD response")
    result = []
    previous_end = 0.0
    for index, (number, start, end) in enumerate(rows):
        start, end = float(start) / 100, float(end) / 100
        if (int(number) != index or not math.isfinite(start) or not math.isfinite(end)
                or start < previous_end - 0.02 or start < 0 or end <= start or end > duration + 0.05):
            raise ValueError("invalid VAD timestamps")
        start, end = max(start, previous_end), min(end, duration)
        if end <= start:
            raise ValueError("empty VAD window")
        result.append((start, end))
        previous_end = end
    return result


def _merge_speech_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union duplicate detections from overlapping VAD chunks, not silent gaps."""
    output = []
    for start, end in sorted(windows):
        if output and start <= output[-1][1] + 0.02:
            output[-1] = (output[-1][0], max(output[-1][1], end))
        else:
            output.append((start, end))
    return output


def _recognition_window(windows: list[tuple[float, float]], index: int, duration: float) -> tuple[float, float]:
    """Give ASR a little onset context without crossing another detected voice."""
    start, end = windows[index]
    previous = windows[index - 1][1] if index else 0.0
    following = windows[index + 1][0] if index + 1 < len(windows) else duration
    context = VAD_PARAMETERS["recognition_context_seconds"]
    return (max(0.0, start - min(context, (start - previous) / 2)),
            min(duration, end + min(context, (following - end) / 2)))


def _crop_pcm(source: Path, output: Path, start: float, end: float) -> float:
    """Cut normalized PCM by sample index; source media is never rewritten."""
    with wave.open(str(source), "rb") as incoming:
        if incoming.getframerate() != 16000 or incoming.getnchannels() != 1 or incoming.getsampwidth() != 2:
            raise ValueError("unexpected normalized PCM")
        first = round(start * 16000)
        last = min(incoming.getnframes(), round(end * 16000))
        if not 0 <= first < last <= incoming.getnframes():
            raise ValueError("invalid speech crop")
        incoming.setpos(first)
        samples = incoming.readframes(last - first)
        if len(samples) != (last - first) * 2:
            raise ValueError("incomplete speech crop")
        with wave.open(str(output), "wb") as outgoing:
            outgoing.setnchannels(1)
            outgoing.setsampwidth(2)
            outgoing.setframerate(16000)
            outgoing.writeframes(samples)
    return (last - first) / 16000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpp_segments(result: dict, duration: float, *, speech_window: bool = False) -> list[dict]:
    """whisper.cpp JSON offsets are milliseconds, not Whisper's 10 ms ticks."""
    if not isinstance(result, dict) or not isinstance(result.get("transcription"), list):
        raise ValueError("missing transcription")
    source = result["transcription"]
    if len(source) > 10_000:
        raise ValueError("too many segments")
    converted = []
    for index, row in enumerate(source):
        if not isinstance(row, dict) or not isinstance(row.get("offsets"), dict):
            raise ValueError("missing offsets")
        offsets = row["offsets"]
        start, end = _number(offsets.get("from")) / 1000, _number(offsets.get("to")) / 1000
        # Whisper may emit a minimum 2 s final timestamp for a short crop.
        # The independent VAD window is a stricter physical bound; never extend
        # the subtitle into silence just to accommodate the decoder's tick.
        if speech_window and index == len(source) - 1 and duration < 2 and 0 <= start < duration and duration < end <= 2.05:
            end = duration
        converted.append({"start": start, "end": end, "text": row.get("text")})
    if not source and str(result.get("text", "")).strip():
        raise ValueError("untimed text")
    return _segments({"segments": converted}, duration)


class LocalTranscriber:
    """Content-addressed local ASR, independent of Nolan's Python runtime.

    ``transcribe`` returns the same ``[{start, end, text}]`` format as the cloud
    adapter. ``last_metadata`` records provenance; it is never an accuracy claim.
    A cache miss detects speech and recognizes each original-clock speech window.
    No network fallback is available.
    """

    def __init__(self, config: dict, *, cache_dir: Path | None = None,
                 cancel_check: Callable[[], bool] | None = None):
        if not isinstance(config, dict) or config.get("mode", "local") != "local":
            raise ProviderError("本地字幕需要 subtitles.mode=local。")
        if config.get("backend", "whisper_cpp") != "whisper_cpp":
            raise ProviderError("当前本地字幕使用 whisper_cpp 后端。")
        self.config = dict(config)
        self.language = config.get("language", "zh")
        if not isinstance(self.language, str) or not re.fullmatch(r"(?:[a-z]{2}|auto)", self.language):
            raise ProviderError("本地字幕语言须为 zh 等两位小写代码，或 auto。")
        self.timeout = config.get("timeout", 600)
        if type(self.timeout) not in {int, float} or not math.isfinite(self.timeout) or not 1 <= self.timeout <= 3600:
            raise ProviderError("本地字幕超时必须在 1–3600 秒之间。")
        self.threads = config.get("threads", 4)
        if type(self.threads) is not int or not 1 <= self.threads <= 32:
            raise ProviderError("本地字幕线程数必须在 1–32 之间。")
        executable = config.get("executable", "whisper-cli")
        model_path = config.get("model_path", str(DEFAULT_MODEL_PATH))
        if not isinstance(executable, str) or not executable or "\x00" in executable:
            raise ProviderError("本地字幕识别程序路径无效。")
        if not isinstance(model_path, str) or not model_path or "\x00" in model_path:
            raise ProviderError("本地字幕模型路径无效。")
        vad_executable = config.get("vad_executable", "whisper-vad-speech-segments")
        vad_model_path = config.get("vad_model_path", str(DEFAULT_VAD_MODEL_PATH))
        if not isinstance(vad_executable, str) or not vad_executable or "\x00" in vad_executable:
            raise ProviderError("本地语音检测程序路径无效。")
        if not isinstance(vad_model_path, str) or not vad_model_path or "\x00" in vad_model_path:
            raise ProviderError("本地语音检测模型路径无效。")
        self.executable = executable
        self.model_path = Path(model_path).expanduser().resolve()
        self.vad_executable = vad_executable
        self.vad_model_path = Path(vad_model_path).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.last_metadata: dict = {}
        self.last_request_id = ""
        self.cancel_check = cancel_check

    def _check_cancelled(self) -> None:
        if self.cancel_check and self.cancel_check():
            raise ProviderError("本地字幕识别已暂停；原视频保留，继续后会重新核对实际音轨。")

    def check_ready(self) -> dict:
        """Read-only dependency check with actionable errors; no model download."""
        executable = shutil.which(self.executable)
        if not executable:
            raise ProviderError("尚未安装本地字幕识别程序。macOS 请运行 brew install whisper-cpp，"
                                "再运行项目中的 scripts/install-local-asr.sh 下载多语言模型。")
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise ProviderError("本地字幕需要 FFmpeg。macOS 请运行 brew install ffmpeg。")
        try:
            if not self.model_path.is_file() or self.model_path.stat().st_size == 0:
                raise OSError("missing model")
        except OSError:
            raise ProviderError("尚未配置本地 Whisper 多语言模型。请运行 scripts/install-local-asr.sh，"
                                "或设置 subtitles.model_path 指向已有 ggml 模型文件。") from None
        vad_executable = shutil.which(self.vad_executable)
        if not vad_executable:
            raise ProviderError("本地字幕缺少语音检测程序 whisper-vad-speech-segments。"
                                "请运行 brew upgrade whisper-cpp，再运行 scripts/install-local-asr.sh；未退回旧时间轴。")
        if not self.vad_model_path.is_file() or not 0 < self.vad_model_path.stat().st_size <= 10_000_000:
            raise ProviderError("本地字幕缺少 Silero 语音检测模型。请运行 scripts/install-local-asr.sh，"
                                "或设置 subtitles.vad_model_path；未退回旧时间轴。")
        return {"backend": "whisper_cpp", "executable": str(Path(executable).resolve()),
                "model_path": str(self.model_path), "language": self.language,
                "vad_executable": str(Path(vad_executable).resolve()),
                "vad_model_path": str(self.vad_model_path), "timing_method": TIMING_METHOD}

    def _run(self, args: list[str], *, timeout: float, capture: bool = False) -> str:
        try:
            result = subprocess.run(args, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, timeout=timeout, check=False,
                                    text=capture)
        except FileNotFoundError:
            raise ProviderError("本地字幕依赖不可用，请检查 whisper-cli、FFmpeg 和模型路径。") from None
        except subprocess.TimeoutExpired:
            raise ProviderError("本地字幕识别超时；原视频保留，可调整 subtitles.timeout 后重试。") from None
        except OSError:
            raise ProviderError("无法启动本地字幕识别，请检查程序权限与安装。") from None
        if result.returncode:
            raise ProviderError("本地字幕识别失败，请检查 whisper-cli、语音检测程序与模型是否兼容；未调用云端转录。")
        return result.stdout if capture else ""

    def transcribe(self, audio: Path) -> list[dict]:
        self.last_metadata = {}
        self._check_cancelled()
        ready = self.check_ready()
        path = Path(audio).expanduser().resolve()
        try:
            if not path.is_file() or not 0 < path.stat().st_size <= MAX_AUDIO_BYTES:
                raise ValueError("audio missing or too large")
            metadata = probe(path)
            duration = _number(metadata.get("duration"))
            if not metadata.get("has_audio") or not 0 < duration <= 3600:
                raise ValueError("invalid duration")
            audio_hash = _sha256(path)
            model_hash = _sha256(self.model_path)
            executable_hash = _sha256(Path(ready["executable"]))
            vad_model_hash = _sha256(self.vad_model_path)
            vad_executable_hash = _sha256(Path(ready["vad_executable"]))
        except (OSError, ValueError, TypeError, MediaError):
            raise ProviderError("无法读取实际音轨或本地模型。音轨须小于 512 MB、时长不超过一小时；原素材未改变。") from None
        identity = {"schema": CACHE_SCHEMA, "audio_sha256": audio_hash, "model_sha256": model_hash,
                    "executable_sha256": executable_hash, "language": self.language,
                    "threads": self.threads, "max_segment_length": 18,
                    "timing_method": TIMING_METHOD, "vad_parameters": VAD_PARAMETERS,
                    "vad_model_sha256": vad_model_hash, "vad_executable_sha256": vad_executable_hash}
        cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache_dir = self.cache_dir or path.parent / ".nolan-local-asr"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # The lock covers recognition as well as publication: concurrent resumes
            # cannot launch duplicate ASR for unchanged media.
            with (cache_dir / (cache_key + ".lock")).open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                target = cache_dir / (cache_key + ".json")
                cached = self._read_cache(target, identity, duration)
                if cached is not None:
                    self.last_metadata = {**identity, "cache_key": cache_key, "cache_hit": True,
                                          "speech_window_count": len(json.loads(target.read_text(encoding="utf-8")).get("speech_windows", []))}
                    return cached
                with tempfile.TemporaryDirectory(prefix=".asr-", dir=cache_dir) as directory:
                    temporary = Path(directory)
                    normalized = temporary / "input.wav"
                    self._check_cancelled()
                    self._run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                               "-protocol_whitelist", "file,pipe", "-i", str(path), "-vn", "-map", "0:a:0",
                               "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(normalized)],
                              timeout=min(self.timeout, 600))
                    if _sha256(path) != audio_hash:
                        raise ProviderError("待识别音轨在处理期间发生变化，请重新核对后继续。")
                    deadline = time.monotonic() + self.timeout
                    def remaining() -> float:
                        self._check_cancelled()
                        value = deadline - time.monotonic()
                        if value <= 0:
                            raise ProviderError("本地字幕识别超时；原视频保留，可调整 subtitles.timeout 后重试。")
                        return value
                    detections = []
                    offset = 0.0
                    while offset < duration:
                        remaining()
                        window_end = min(duration, offset + VAD_PARAMETERS["window_seconds"])
                        if offset == 0 and window_end == duration:
                            vad_audio = normalized
                            vad_duration = duration
                        else:
                            vad_audio = temporary / f"vad-window-{round(offset * 1000)}.wav"
                            vad_duration = _crop_pcm(normalized, vad_audio, offset, window_end)
                        # Reset the recurrent detector in overlapping chunks: a
                        # long run of scene noise can suppress a later quiet line.
                        vad_output = self._run([ready["vad_executable"], "-f", str(vad_audio),
                                                "-vm", str(self.vad_model_path), "-t", str(self.threads),
                                                "-vt", "0.5", "-vspd", "250", "-vsd", "100",
                                                "-vp", "100", "-vmsd", "12", "-np"],
                                               timeout=remaining(), capture=True)
                        detections.extend((round(offset + start, 6), round(offset + end, 6))
                                          for start, end in _vad_windows(vad_output, vad_duration))
                        if window_end == duration:
                            break
                        offset += VAD_PARAMETERS["window_step_seconds"]
                    windows = _merge_speech_windows(detections)
                    if len(windows) > MAX_SPEECH_WINDOWS:
                        raise ValueError("too many speech windows")
                    segments = []
                    for index, (start, end) in enumerate(windows):
                        self._check_cancelled()
                        cropped = temporary / f"speech-{index}.wav"
                        recognition_start, recognition_end = _recognition_window(windows, index, duration)
                        speech_duration = _crop_pcm(normalized, cropped, recognition_start, recognition_end)
                        output = temporary / f"recognized-{index}"
                        self._run([ready["executable"], "-m", str(self.model_path), "-f", str(cropped),
                                   "-l", self.language, "-t", str(self.threads), "-oj", "-of", str(output),
                                   "-ml", "18", "-sns", "-np"], timeout=remaining())
                        json_path = output.with_suffix(".json")
                        if json_path.stat().st_size > MAX_RESPONSE_BYTES:
                            raise ValueError("oversized local response")
                        observed = _cpp_segments(json.loads(json_path.read_text(encoding="utf-8")), speech_duration,
                                                 speech_window=True)
                        for row in observed:
                            caption_start = max(start, recognition_start + row["start"])
                            caption_end = min(end, recognition_start + row["end"])
                            if caption_end > caption_start:
                                segments.append({"start": caption_start, "end": caption_end, "text": row["text"]})
                    segments = _segments({"segments": segments}, duration)
                    if (_sha256(self.model_path) != model_hash or _sha256(path) != audio_hash
                            or _sha256(self.vad_model_path) != vad_model_hash
                            or _sha256(Path(ready["vad_executable"])) != vad_executable_hash):
                        raise ProviderError("音轨或识别模型在处理期间发生变化，未保存本次识别缓存。")
                    record = {"identity": identity, "duration": duration, "segments": segments,
                              "speech_windows": [[round(s, 6), round(e, 6)] for s, e in windows]}
                    published = temporary / "cache.json"
                    published.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                    os.replace(published, target)
                self.last_metadata = {**identity, "cache_key": cache_key, "cache_hit": False,
                                      "speech_window_count": len(windows)}
                return segments
        except ProviderError:
            raise
        except (OSError, ValueError, TypeError, KeyError, OverflowError, wave.Error):
            raise ProviderError("本地字幕结果或时间戳无效，未采用结果；请检查识别程序和音频，原视频保留。") from None

    @staticmethod
    def _read_cache(path: Path, identity: dict, duration: float) -> list[dict] | None:
        try:
            if not path.is_file() or path.stat().st_size > MAX_RESPONSE_BYTES:
                return None
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("identity") != identity or record.get("duration") != duration:
                return None
            return _segments({"segments": record["segments"]}, duration)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None
