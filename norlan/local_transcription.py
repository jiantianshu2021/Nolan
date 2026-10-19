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

from .media import MediaError, probe
from .providers import ProviderError
from .transcription import _number, _segments


DEFAULT_MODEL_PATH = Path.home() / ".local/share/norlan/models/ggml-small.bin"
MAX_AUDIO_BYTES = 512_000_000
MAX_RESPONSE_BYTES = 8_000_000
CACHE_SCHEMA = "nolan-whisper-cpp-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpp_segments(result: dict, duration: float) -> list[dict]:
    """whisper.cpp JSON offsets are milliseconds, not Whisper's 10 ms ticks."""
    if not isinstance(result, dict) or not isinstance(result.get("transcription"), list):
        raise ValueError("missing transcription")
    source = result["transcription"]
    if len(source) > 10_000:
        raise ValueError("too many segments")
    converted = []
    for row in source:
        if not isinstance(row, dict) or not isinstance(row.get("offsets"), dict):
            raise ValueError("missing offsets")
        offsets = row["offsets"]
        converted.append({"start": _number(offsets.get("from")) / 1000,
                          "end": _number(offsets.get("to")) / 1000,
                          "text": row.get("text")})
    if not source and str(result.get("text", "")).strip():
        raise ValueError("untimed text")
    return _segments({"segments": converted}, duration)


class LocalTranscriber:
    """Content-addressed local ASR, independent of Nolan's Python runtime.

    ``transcribe`` returns the same ``[{start, end, text}]`` format as the cloud
    adapter. ``last_metadata`` records provenance; it is never an accuracy claim.
    A cache miss runs one local recognition. No network fallback is available.
    """

    def __init__(self, config: dict, *, cache_dir: Path | None = None):
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
        self.executable = executable
        self.model_path = Path(model_path).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.last_metadata: dict = {}
        self.last_request_id = ""

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
        return {"backend": "whisper_cpp", "executable": str(Path(executable).resolve()),
                "model_path": str(self.model_path), "language": self.language}

    def _run(self, args: list[str], *, timeout: float) -> None:
        try:
            result = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, timeout=timeout, check=False)
        except FileNotFoundError:
            raise ProviderError("本地字幕依赖不可用，请检查 whisper-cli、FFmpeg 和模型路径。") from None
        except subprocess.TimeoutExpired:
            raise ProviderError("本地字幕识别超时；原视频保留，可调整 subtitles.timeout 后重试。") from None
        except OSError:
            raise ProviderError("无法启动本地字幕识别，请检查程序权限与安装。") from None
        if result.returncode:
            raise ProviderError("本地字幕识别失败，请检查 whisper-cli 与多语言模型是否兼容；未调用云端转录。")

    def transcribe(self, audio: Path) -> list[dict]:
        self.last_metadata = {}
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
        except (OSError, ValueError, TypeError, MediaError):
            raise ProviderError("无法读取实际音轨或本地模型。音轨须小于 512 MB、时长不超过一小时；原素材未改变。") from None
        identity = {"schema": CACHE_SCHEMA, "audio_sha256": audio_hash, "model_sha256": model_hash,
                    "executable_sha256": executable_hash, "language": self.language,
                    "threads": self.threads, "max_segment_length": 18}
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
                    self.last_metadata = {**identity, "cache_key": cache_key, "cache_hit": True}
                    return cached
                with tempfile.TemporaryDirectory(prefix=".asr-", dir=cache_dir) as directory:
                    temporary = Path(directory)
                    normalized = temporary / "input.wav"
                    self._run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                               "-protocol_whitelist", "file,pipe", "-i", str(path), "-vn", "-map", "0:a:0",
                               "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(normalized)],
                              timeout=min(self.timeout, 600))
                    if _sha256(path) != audio_hash:
                        raise ProviderError("待识别音轨在处理期间发生变化，请重新核对后继续。")
                    output = temporary / "recognized"
                    self._run([ready["executable"], "-m", str(self.model_path), "-f", str(normalized),
                               "-l", self.language, "-t", str(self.threads), "-oj", "-of", str(output),
                               "-ml", "18", "-sns", "-np"], timeout=self.timeout)
                    json_path = output.with_suffix(".json")
                    if json_path.stat().st_size > MAX_RESPONSE_BYTES:
                        raise ValueError("oversized local response")
                    segments = _cpp_segments(json.loads(json_path.read_text(encoding="utf-8")), duration)
                    if _sha256(self.model_path) != model_hash or _sha256(path) != audio_hash:
                        raise ProviderError("音轨或识别模型在处理期间发生变化，未保存本次识别缓存。")
                    record = {"identity": identity, "duration": duration, "segments": segments}
                    published = temporary / "cache.json"
                    published.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                    os.replace(published, target)
                self.last_metadata = {**identity, "cache_key": cache_key, "cache_hit": False}
                return segments
        except ProviderError:
            raise
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
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
