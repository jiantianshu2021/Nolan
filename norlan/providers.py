"""Explicit, non-retrying adapters for the official Volcengine Ark API.

References (checked 2026-10-20):
https://console.volcengine.com/ark/region:cn-beijing/docs/create-video-generation-task-api
https://console.volcengine.com/ark/region:cn-beijing/docs/get-video-generation-task-api
https://docs.volcengine.com/docs/ark/image-generation-api?lang=zh

API POST uncertainty is intentionally surfaced to the caller's durable ledger.
Never automatically repeat a generation request after a timeout or invalid reply.
"""
from __future__ import annotations

import base64
import hashlib
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


class ProviderError(RuntimeError):
    safe_message = True

    def __init__(self, message: str, *, uncertain: bool = False, request_id: str = ""):
        super().__init__(message)
        self.uncertain = uncertain
        self.request_id = request_id


class _NoApiRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        # urllib otherwise forwards the Authorization header to a new host.
        raise HTTPError(request.full_url, code, "API redirect refused", headers, fp)


def _api_open(request, timeout):
    return build_opener(_NoApiRedirect()).open(request, timeout=timeout)


def _safe_code(value: object) -> str:
    """Do not expose remote messages, signed URLs, prompts or credentials."""
    value = str(value or "unknown")
    return value if re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", value) else "unknown"


def _https_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ProviderError("仅允许不包含账号密码的 HTTPS 服务或下载地址。")
    return url


def _data_uri(path: Path, media: str, max_bytes: int) -> str:
    path = Path(path)
    formats = {"image": {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp",
                         ".bmp": "bmp", ".tiff": "tiff", ".gif": "gif", ".heic": "heic", ".heif": "heif"},
               "audio": {".wav": "wav", ".mp3": "mp3"}}
    subtype = formats.get(media, {}).get(path.suffix.lower())
    if not subtype:
        raise ProviderError(f"不支持的 {media} 参考文件格式。")
    try:
        if not path.is_file() or not 0 < path.stat().st_size < max_bytes:
            raise ProviderError(f"参考文件为空、缺失或超过 {max_bytes // 1000000} MB 限制。")
        data = path.read_bytes()
    except OSError as exc:
        raise ProviderError("无法读取参考文件。") from exc
    return f"data:{media}/{subtype};base64," + base64.b64encode(data).decode("ascii")


def _atomic_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".norlan-", delete=False) as file:
            tmp = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    finally:
        if tmp and tmp.exists():
            tmp.unlink()


class ArkProvider:
    """Use Seedream and Seedance 2.x multimodal references, never web cookies.

    Credentials are read from ARK_API_KEY (or api_key_env), not config files.
    Model IDs must be selected by the operator; endpoint IDs are also accepted.
    Limits are conservative for Seedance 2.0 and also work with 2.5.
    """
    capabilities = {"image_references": True, "max_video_images": 9,
                    "max_audio_references": 3, "max_audio_seconds": 15,
                    "local_video_upload": False, "video_reference_url": True,
                    "native_audio": True, "stable_voice_guarantee": False,
                    "first_frame_plus_references": False,
                    "min_duration": 4, "max_duration": 15}

    def __init__(self, config: dict):
        self.config = dict(config)
        self.base_url = _https_url(str(config.get("base_url", "https://ark.cn-beijing.volces.com/api/v3")).rstrip("/"))
        self.timeout = float(config.get("timeout", 120))
        if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 600:
            raise ProviderError("请求超时必须在 1–600 秒之间。")
        self._key()

    def preflight(self, kind: str) -> None:
        """Validate local configuration before the orchestrator reserves money."""
        if kind not in {"image", "video"}:
            raise ProviderError("未知的 Ark 请求种类。")
        self._key()
        self._model(kind)

    def _key(self) -> str:
        key = os.environ.get(str(self.config.get("api_key_env", "ARK_API_KEY")), "")
        if not key:
            raise ProviderError("未配置 ARK_API_KEY；请在本机环境变量中配置，勿写入项目。")
        return key

    def _model(self, kind: str) -> str:
        model = str(self.config.get(f"{kind}_model", "")).strip()
        if not model:
            raise ProviderError(f"请先配置已开通服务的 {kind}_model。")
        family = model.lower()
        if kind == "image" and "seedance-" in family:
            raise ProviderError("Seedance 用于视频生成，不能配置为 image_model；请另选已开通的 Seedream 图片模型。")
        if kind == "video" and "seedream-" in family:
            raise ProviderError("Seedream 用于图片生成，不能配置为 video_model；请另选已开通的 Seedance 视频模型。")
        if kind == "video" and "seedance-1-" in family:
            raise ProviderError("连续性流程需要 Seedance 2.x 全模态参考能力，不能使用 Seedance 1.x。")
        return model

    def _request(self, method: str, path: str, payload: dict | None = None) -> tuple[dict, str]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        if body is not None and len(body) >= 64_000_000:
            raise ProviderError("参考素材编码后的请求超过 64 MB，请减少或压缩参考图和音频。")
        request = Request(self.base_url + path, data=body, method=method,
                          headers={"Authorization": "Bearer " + self._key(), "Content-Type": "application/json"})
        request_id = ""
        try:
            with _api_open(request, timeout=self.timeout) as response:
                request_id = _safe_code(response.headers.get("X-Request-Id", ""))
                raw = response.read(70_000_001)
                if len(raw) > 70_000_000:
                    raise ValueError("oversized response")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ValueError("non-object response")
                return result, request_id
        except HTTPError as exc:
            # HTTP 5xx/408 may follow an accepted job. Preserve the reservation.
            uncertain = method == "POST" and (exc.code >= 500 or exc.code == 408)
            request_id = _safe_code(exc.headers.get("X-Request-Id", ""))
            exc.close()
            raise ProviderError(f"Ark 请求失败，HTTP {exc.code}；请在服务控制台核对。",
                                uncertain=uncertain, request_id=request_id) from None
        except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError, ValueError, HTTPException):
            raise ProviderError("Ark 请求超时、连接中断或响应无法解析；未自动重发。",
                                uncertain=method == "POST", request_id=request_id) from None

    def generate_image(self, prompt: str, output: Path, references: list[Path] | None = None) -> dict:
        refs = list(references or [])
        if not prompt.strip() or len(refs) > 9:
            raise ProviderError("图片提示词不能为空，参考图最多 9 张。")
        model = self._model("image")
        payload = {"model": model, "prompt": prompt,
                   "size": self.config.get("image_size", "2K"), "response_format": "b64_json",
                   "watermark": True, "stream": False}
        # Seedream 5.0 pro/flash generate one image and reject this option.
        # Preserve explicit single-image mode for 4.x/5.0 lite and existing endpoints.
        if re.search(r"seedream-5-0-(?:pro|flash)(?:-|$)", model.lower()):
            # Pipeline assets are saved as .png and later uploaded as image/png.
            payload["output_format"] = "png"
        else:
            payload["sequential_image_generation"] = "disabled"
        if refs:
            payload["image"] = [_data_uri(p, "image", 30_000_000) for p in refs]
        result, request_id = self._request("POST", "/images/generations", payload)
        try:
            item = result["data"][0]
            if item.get("error"):
                raise ProviderError("图片生成服务返回失败：" + _safe_code(item["error"].get("code")), request_id=request_id)
            if item.get("b64_json"):
                data = base64.b64decode(item["b64_json"], validate=True)
                if not data:
                    raise ValueError("empty image")
                _atomic_bytes(Path(output), data)
            elif item.get("url"):
                self.download(item["url"], Path(output))
            else:
                raise ValueError("missing image")
        except ProviderError as exc:
            # Generation has completed even if local download failed; don't regenerate blindly.
            exc.uncertain = True
            exc.request_id = request_id
            raise
        except (KeyError, IndexError, TypeError, ValueError, OSError):
            raise ProviderError("图片响应或本地保存失败；需核对已生成结果，禁止直接重发。",
                                uncertain=True, request_id=request_id) from None
        return {"output": str(Path(output)), "request_id": request_id, "usage": result.get("usage", {})}

    def _audio_duration(self, path: Path) -> float:
        try:
            done = subprocess.run([str(self.config.get("ffprobe", "ffprobe")), "-v", "error", "-show_entries",
                                   "format=duration", "-of", "json", str(Path(path).resolve())],
                                  capture_output=True, text=True, timeout=30, check=True)
            duration = float(json.loads(done.stdout)["format"]["duration"])
            if not math.isfinite(duration) or not 2 <= duration <= 15:
                raise ValueError("duration")
            return duration
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            raise ProviderError("声音参考须为可读取的 2–15 秒 WAV/MP3，且本机已安装 ffprobe。") from None

    def submit_video(self, prompt: str, duration: float, references: list[Path],
                     previous_video: Path | str | None = None,
                     audio_references: list[Path] | None = None) -> dict:
        refs, audio = list(references), list(audio_references or [])
        if not prompt.strip():
            raise ProviderError("视频提示词不能为空。")
        if isinstance(duration, bool) or not math.isfinite(float(duration)) or not 4 <= duration <= 15 or int(duration) != duration:
            raise ProviderError("本适配器视频时长须为 4–15 之间的整数秒。")
        if len(refs) > 9 or len(audio) > 3:
            raise ProviderError("每次最多 9 张参考图、3 段声音参考；不会静默丢弃素材。")
        if audio and not (refs or previous_video):
            raise ProviderError("Seedance 2.0 的声音参考须搭配图片或视频。")
        if sum(self._audio_duration(p) for p in audio) > 15.01:
            raise ProviderError("全部声音参考的总时长不能超过 15 秒。")
        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": _data_uri(p, "image", 30_000_000)},
                        "role": "reference_image"} for p in refs)
        if previous_video is not None:
            if isinstance(previous_video, str) and previous_video.startswith("https://"):
                video_url = _https_url(previous_video)
            else:
                try:
                    digest = hashlib.sha256(Path(previous_video).read_bytes()).hexdigest()
                except OSError:
                    raise ProviderError("无法读取上一镜头视频。") from None
                video_url = self.config.get("reference_video_urls", {}).get(digest)
                if not video_url:
                    raise ProviderError("视频参考不支持本地 Base64。请提供此文件哈希对应的有效 HTTPS 来源，或显式改用实际尾帧参考。")
                video_url = _https_url(video_url)
            content.append({"type": "video_url", "video_url": {"url": video_url}, "role": "reference_video"})
        content.extend({"type": "audio_url", "audio_url": {"url": _data_uri(p, "audio", 15_000_000)},
                        "role": "reference_audio"} for p in audio)
        payload = {"model": self._model("video"), "content": content, "duration": int(duration),
                   "ratio": self.config.get("ratio", "16:9"), "resolution": self.config.get("resolution", "720p"),
                   "generate_audio": bool(self.config.get("generate_audio", True)), "watermark": True}
        result, request_id = self._request("POST", "/contents/generations/tasks", payload)
        if not isinstance(result.get("id"), str) or not result["id"]:
            raise ProviderError("视频任务响应缺少 ID；需人工核对，未自动重发。", uncertain=True, request_id=request_id)
        return {"id": result["id"], "request_id": request_id}

    def poll_video(self, task_id: str) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", task_id):
            raise ProviderError("无效的视频任务 ID。")
        result, request_id = self._request("GET", "/contents/generations/tasks/" + quote(task_id, safe=""))
        status = result.get("status")
        base = {"id": task_id, "request_id": request_id, "provider_status": status, "usage": result.get("usage", {})}
        if status == "succeeded":
            content = result.get("content")
            url = content.get("video_url") if isinstance(content, dict) else None
            if not url:
                raise ProviderError("已完成任务缺少下载地址；请核对原任务，不要重新生成。")
            return {**base, "status": "completed", "url": _https_url(url)}
        if status in {"failed", "cancelled", "expired"}:
            error = result.get("error")
            return {**base, "status": "failed", "error": _safe_code(error.get("code", status) if isinstance(error, dict) else status)}
        if status not in {"queued", "running"}:
            raise ProviderError("未知的视频任务状态；保留任务 ID 等待核对。")
        return {**base, "status": "pending"}

    def download(self, url: str, output: Path) -> None:
        """Download without credentials; original remains intact on partial failure."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            with urlopen(Request(_https_url(url), headers={"User-Agent": "Norlan/0.1"}), timeout=self.timeout) as response:
                _https_url(response.geturl())
                with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".norlan-download-", delete=False) as file:
                    tmp, count = Path(file.name), 0
                    while chunk := response.read(1024 * 1024):
                        count += len(chunk)
                        if count > int(self.config.get("max_download_bytes", 1_000_000_000)):
                            raise ProviderError("下载文件超过本机限制。")
                        file.write(chunk)
                    expected = response.headers.get("Content-Length")
                    if not count or (expected is not None and count != int(expected)):
                        raise ProviderError("下载不完整；请重试下载原任务，勿重新生成。")
                    file.flush()
                    os.fsync(file.fileno())
            os.replace(tmp, output)
        except (OSError, URLError, ValueError, TimeoutError, HTTPException):
            raise ProviderError("下载失败；原文件保留，请重试下载已完成任务。") from None
        finally:
            if tmp and tmp.exists():
                tmp.unlink()
