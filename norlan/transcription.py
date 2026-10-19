"""Timestamped ASR of actual generated audio using the official OpenAI API.

Sources checked 2026-10-19:
https://developers.openai.com/api/docs/guides/speech-to-text
https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create

The caller owns consent, budget reservation, persistence, and human review.
This adapter never substitutes planned dialogue for observed speech.
"""
from __future__ import annotations

from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .media import MediaError, probe
from .providers import ProviderError


ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
MAX_AUDIO_BYTES = 25_000_000
MAX_RESPONSE_BYTES = 4_000_000
MIME_TYPES = {
    ".flac": "audio/flac", ".mp3": "audio/mpeg", ".mp4": "video/mp4",
    ".mpeg": "audio/mpeg", ".mpga": "audio/mpeg", ".m4a": "audio/mp4",
    ".ogg": "audio/ogg", ".wav": "audio/wav", ".webm": "audio/webm",
}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        # Never forward audio or Authorization to a redirected host.
        return None


def _open(request: Request, timeout: float):
    return build_opener(_NoRedirect).open(request, timeout=timeout)


def _request_id(headers) -> str:
    value = str(headers.get("X-Request-Id", ""))
    return value if re.fullmatch(r"[A-Za-z0-9_.:-]{0,100}", value) else ""


def _number(value) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("invalid timestamp")
    return float(value)


def _segments(result: dict, duration: float) -> list[dict]:
    """Validate against local media duration, not a model's proposed timeline."""
    if not isinstance(result, dict) or not isinstance(result.get("segments"), list):
        raise ValueError("missing segments")
    segments = result["segments"]
    if len(segments) > 10_000:
        raise ValueError("too many segments")
    if not segments and str(result.get("text", "")).strip():
        raise ValueError("text lacks timestamps")
    output = []
    previous_end = 0.0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("invalid segment")
        start, end = _number(segment.get("start")), _number(segment.get("end"))
        text = segment.get("text")
        if not isinstance(text, str) or len(text) > 4000 or "\x00" in text:
            raise ValueError("invalid text")
        # Accommodate only a small numeric/audio-frame rounding discrepancy.
        if start < 0 or start < previous_end - 0.05 or end <= start or end > duration + 0.05:
            raise ValueError("timestamp out of bounds or overlapping")
        start, end = max(start, previous_end), min(end, duration)
        if end <= start:
            raise ValueError("empty segment after boundary rounding")
        previous_end = end
        if text.strip():
            output.append({"start": round(start, 6), "end": round(end, 6), "text": text.strip()})
    if not output and str(result.get("text", "")).strip():
        raise ValueError("nonempty transcript has no timed speech")
    return output


class Transcriber:
    """OpenAI whisper-1 segment timestamps with an explicit no-retry policy."""

    def __init__(self, config: dict):
        self.config = dict(config)
        if config.get("mode", "openai") != "openai":
            raise ProviderError("自动字幕转录仅支持 subtitles.mode=openai。")
        self.model = str(config.get("model", "whisper-1"))
        if self.model != "whisper-1":
            raise ProviderError("当前自动字幕需要 whisper-1 的分段时间戳，请勿直接替换为不兼容模型。")
        if str(config.get("base_url", "https://api.openai.com/v1")).rstrip("/") != "https://api.openai.com/v1":
            raise ProviderError("自动字幕仅连接官方 OpenAI API 地址。")
        self.api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ProviderError("字幕 API 环境变量名称无效。")
        try:
            self.timeout = float(config.get("timeout", 120))
        except (TypeError, ValueError):
            raise ProviderError("字幕请求超时必须在 1–600 秒之间。") from None
        if type(config.get("timeout")) is bool or not math.isfinite(self.timeout) or not 1 <= self.timeout <= 600:
            raise ProviderError("字幕请求超时必须在 1–600 秒之间。")
        self.language = config.get("language")
        if self.language is not None and (not isinstance(self.language, str) or not re.fullmatch(r"[a-z]{2}", self.language)):
            raise ProviderError("字幕语言使用两个小写字母的 ISO 代码，例如 zh。")
        self.last_request_id = ""

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key or any(char in key for char in "\r\n"):
            raise ProviderError("字幕转录尚未配置有效的 OpenAI API 环境变量凭据。")
        return key

    def _multipart(self, suffix: str, audio_bytes: bytes) -> tuple[bytes, str]:
        boundary = "norlan-" + secrets.token_hex(24)
        parts = []
        fields = [("model", self.model), ("response_format", "verbose_json"),
                  ("timestamp_granularities[]", "segment")]
        if self.language:
            fields.append(("language", self.language))
        for name, value in fields:
            parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode("utf-8"))
        # A synthetic filename preserves extension metadata without exposing local names.
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio{suffix}"\r\nContent-Type: {MIME_TYPES[suffix]}\r\n\r\n').encode("ascii"))
        parts.extend((audio_bytes, f"\r\n--{boundary}--\r\n".encode("ascii")))
        return b"".join(parts), "multipart/form-data; boundary=" + boundary

    def transcribe(self, audio: Path) -> list[dict]:
        """Return observed ASR segments; callers must still review their accuracy.

        Network errors and malformed successful responses are uncertain, because
        the request may already have consumed credits. No POST is retried here.
        """
        self.last_request_id = ""
        key = self._key()
        path = Path(audio).expanduser()
        suffix = path.suffix.lower()
        if suffix not in MIME_TYPES:
            raise ProviderError("字幕音频格式不受支持，请先导出 WAV 或 MP3。")
        try:
            size = path.stat().st_size
            if not path.is_file() or not 0 < size <= MAX_AUDIO_BYTES:
                raise ProviderError("字幕音频为空、缺失或超过 25 MB，请先按镜头切分。")
            metadata = probe(path)
            duration = float(metadata["duration"])
            if not metadata.get("has_audio") or not math.isfinite(duration) or duration <= 0:
                raise ProviderError("字幕需要具有有效时长的实际音轨。")
            with path.open("rb") as stream:
                audio_bytes = stream.read(MAX_AUDIO_BYTES + 1)
            if len(audio_bytes) != size or len(audio_bytes) > MAX_AUDIO_BYTES:
                raise ProviderError("音频文件在读取期间发生变化，请重新核对素材。")
        except (OSError, MediaError, ValueError, KeyError, TypeError):
            raise ProviderError("无法读取或检查待转录音频，请确认文件及 FFmpeg 环境。") from None
        body, content_type = self._multipart(suffix, audio_bytes)
        request = Request(ENDPOINT, data=body, method="POST", headers={
            "Authorization": "Bearer " + key,
            "Content-Type": content_type,
            "Accept": "application/json",
        })
        try:
            with _open(request, self.timeout) as response:
                self.last_request_id = _request_id(response.headers)
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ValueError("oversized response")
                result = json.loads(raw)
            return _segments(result, duration)
        except HTTPError as exc:
            self.last_request_id = _request_id(exc.headers)
            code = exc.code
            exc.close()
            raise ProviderError(f"OpenAI 字幕转录请求失败，HTTP {code}；未自动重试。",
                                uncertain=code >= 500 or code == 408 or 300 <= code < 400,
                                request_id=self.last_request_id) from None
        except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError, HTTPException):
            raise ProviderError("OpenAI 字幕转录连接中断或超时，请先核对账单；未自动重试。",
                                uncertain=True, request_id=self.last_request_id) from None
        except (ValueError, TypeError, KeyError, OverflowError):
            raise ProviderError("OpenAI 已返回字幕响应，但内容或时间戳无效；请核对后处理，未自动重试。",
                                uncertain=True, request_id=self.last_request_id) from None
