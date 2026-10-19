"""Optional Volcengine pure-music API, independent of Jimeng web login.

Protocol sources checked 2026-10-19:
https://docs.volcengine.com/docs/Audiovideounderstandingandprocessing/Generatepuremusic?lang=zh
https://docs.volcengine.com/docs/Audiovideounderstandingandprocessing/Querytask-1?lang=zh
https://docs.volcengine.com/docs/Audiovideounderstandingandprocessing/Publicfields?lang=zh
https://docs.volcengine.com/docs/APISignatureCallGuide/Signaturemethod-9?lang=zh

No automatic resubmission: a lost submission response leaves a possibly paid
job. The orchestrator must persist its reservation before calling ``submit``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import socket
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .providers import ProviderError


_HOST = "open.volcengineapi.com"
_REGION = "cn-beijing"
_SERVICE = "imagination"
_VERSION = "2024-08-12"


def _safe_code(value: object) -> str:
    text = str(value or "unknown")
    return text if re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", text) else "unknown"


def _signing_key(secret: str, date: str, region: str, service: str) -> bytes:
    key = secret.encode("utf-8")
    for part in (date, region, service, "request"):
        key = hmac.new(key, part.encode("utf-8"), hashlib.sha256).digest()
    return key


def _signed_headers(body: bytes, action: str, access_key: str, secret_key: str,
                    now: datetime | None = None) -> dict[str, str]:
    instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    stamp = instant.strftime("%Y%m%dT%H%M%SZ")
    date = stamp[:8]
    digest = hashlib.sha256(body).hexdigest()
    headers = {
        "content-type": "application/json", "host": _HOST,
        "x-content-sha256": digest, "x-date": stamp,
    }
    signed = ";".join(sorted(headers))
    canonical_headers = "".join(f"{key}:{headers[key]}\n" for key in sorted(headers))
    query = urlencode({"Action": action, "Version": _VERSION})
    canonical = "\n".join(("POST", "/", query, canonical_headers, signed, digest))
    scope = f"{date}/{_REGION}/{_SERVICE}/request"
    string_to_sign = "\n".join(("HMAC-SHA256", stamp, scope,
                                hashlib.sha256(canonical.encode("utf-8")).hexdigest()))
    signature = hmac.new(_signing_key(secret_key, date, _REGION, _SERVICE),
                         string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    headers["authorization"] = (f"HMAC-SHA256 Credential={access_key}/{scope}, "
                                f"SignedHeaders={signed}, Signature={signature}")
    return headers


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward signed API credentials to a redirect destination.
        raise HTTPError(req.full_url, code, "API redirects are not permitted", headers, fp)


def _audio_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        port = parts.port
    except (TypeError, ValueError):
        raise ProviderError("音乐结果地址格式无效。") from None
    # The QuerySong documentation explicitly specifies this output CDN range.
    allowed = re.fullmatch(r"v(?:[1-9]|[1-4][0-9]|50)-default\.douyinvod\.com", parts.hostname or "")
    if (parts.scheme != "https" or not allowed or parts.username or parts.password
            or port not in (None, 443)):
        raise ProviderError("音乐结果地址不符合官方音频 CDN 规则；请核对服务文档，未自动下载。")
    return url


class _AudioRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _audio_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class MusicProvider:
    """Submit v5.0 instrumental music; poll existing tasks; download finished audio.

    ``config`` accepts action (GenBGMForTime or GenBGM), timeout (1..300 seconds),
    and optional segments [{Name, Duration}]. Credentials are read only from
    VOLC_ACCESS_KEY_ID / VOLC_SECRET_ACCESS_KEY in the process environment.
    Billing/approval limits belong to the caller's durable project ledger.
    """

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.action = self.config.get("action", "GenBGMForTime")
        if self.action not in ("GenBGMForTime", "GenBGM"):
            raise ProviderError("音乐接口 action 必须是 GenBGMForTime 或 GenBGM。")
        try:
            self.timeout = float(self.config.get("timeout", 90))
        except (ValueError, TypeError):
            raise ProviderError("音乐接口超时必须在 1–300 秒之间。") from None
        if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 300:
            raise ProviderError("音乐接口超时必须在 1–300 秒之间。")

    @staticmethod
    def _credentials() -> tuple[str, str]:
        access_key = os.environ.get("VOLC_ACCESS_KEY_ID", "").strip()
        secret_key = os.environ.get("VOLC_SECRET_ACCESS_KEY", "").strip()
        if not access_key or not secret_key:
            raise ProviderError("音乐生成需要单独开通火山音乐服务，并在本机设置 VOLC_ACCESS_KEY_ID / VOLC_SECRET_ACCESS_KEY。")
        if any(c in access_key + secret_key for c in "\r\n"):
            raise ProviderError("音乐服务凭据格式无效。")
        return access_key, secret_key

    def _call(self, action: str, payload: dict, *, generating: bool) -> tuple[dict, str]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        access_key, secret_key = self._credentials()
        headers = _signed_headers(body, action, access_key, secret_key)
        request = Request(f"https://{_HOST}/?" + urlencode({"Action": action, "Version": _VERSION}),
                          data=body, headers=headers, method="POST")
        try:
            with build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("oversized response")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("non-object response")
        except HTTPError as exc:
            raise ProviderError(f"火山音乐接口返回 HTTP {exc.code}；未自动重发。",
                                uncertain=generating and (exc.code >= 500 or exc.code in (301, 302, 303, 307, 308, 408))) from None
        except (URLError, OSError, TimeoutError, socket.timeout, ValueError):
            raise ProviderError("火山音乐接口连接中断或响应无法解析；未自动重发。",
                                uncertain=generating) from None
        metadata = data.get("ResponseMetadata") or {}
        if not isinstance(metadata, dict):
            raise ProviderError("音乐服务响应元数据无效。", uncertain=generating)
        request_id = _safe_code(metadata.get("RequestId"))
        if data.get("Code") != 0 or metadata.get("Error"):
            error = metadata.get("Error") or {}
            code = _safe_code(error.get("Code") if isinstance(error, dict) else data.get("Code"))
            if code == "unknown":
                code = _safe_code(data.get("Code"))
            # Service error payloads are deliberately not echoed (may contain a prompt).
            raise ProviderError("火山音乐服务拒绝或未完成请求，错误码：" + code,
                                uncertain=generating and data.get("Code") is None,
                                request_id=request_id)
        result = data.get("Result")
        if not isinstance(result, dict):
            raise ProviderError("音乐服务响应缺少结果对象。", uncertain=generating, request_id=request_id)
        return result, request_id

    def submit(self, prompt: str, duration: int | float) -> dict:
        if not isinstance(prompt, str) or not prompt.strip() or not re.search(r"[\u3400-\u9fff]", prompt):
            raise ProviderError("纯音乐提示词需要中文描述。")
        if len(prompt) > 4000:
            raise ProviderError("Norlan 将音乐提示词限制为 4000 字，请先精简。")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration):
            raise ProviderError("音乐时长必须是 30–120 秒内的整数。")
        if int(duration) != duration or not 30 <= duration <= 120:
            raise ProviderError("音乐时长必须是 30–120 秒内的整数。")
        duration = int(duration)
        supplied = self.config.get("segments")
        segments = supplied if supplied is not None else [{"Name": "inst", "Duration": duration}]
        if not isinstance(segments, list) or not segments:
            raise ProviderError("音乐分段结构不能为空。")
        checked = []
        for segment in segments:
            if not isinstance(segment, dict) or segment.get("Name") not in {"intro", "verse", "chorus", "inst", "bridge", "outro"}:
                raise ProviderError("音乐分段名称不符合 v5.0 接口规范。")
            seconds = segment.get("Duration")
            if isinstance(seconds, bool) or not isinstance(seconds, int) or not 5 <= seconds <= 120:
                raise ProviderError("每段音乐时长必须是 5–120 秒的整数。")
            checked.append({"Name": segment["Name"], "Duration": seconds})
        if sum(segment["Duration"] for segment in checked) != duration:
            raise ProviderError("音乐分段时长之和必须等于已预算的生成时长。")
        # Segments have higher duration priority than instructions inside Text.
        # This prevents a model-written prompt from silently increasing runtime.
        payload = {"Text": prompt.strip(), "Duration": duration, "Version": "v5.0",
                   "Segments": checked, "EnableInputRewrite": False, "AigcWatermark": True}
        result, request_id = self._call(self.action, payload, generating=True)
        task_id = str(result.get("TaskID") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", task_id):
            raise ProviderError("音乐提交结果没有可保存的任务 ID；禁止自动重发。",
                                uncertain=True, request_id=request_id)
        return {"id": task_id, "status": "pending", "request_id": request_id,
                "duration_requested": duration, "provider": "volc-music", "action": self.action}

    def poll(self, task_id: str) -> dict:
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", task_id):
            raise ProviderError("音乐任务 ID 无效。")
        result, request_id = self._call("QuerySong", {"TaskID": task_id}, generating=False)
        if result.get("TaskID") and str(result["TaskID"]) != task_id:
            raise ProviderError("查询返回的音乐任务 ID 不一致。", request_id=request_id)
        status = result.get("Status")
        if isinstance(status, bool):
            raise ProviderError("音乐服务返回无效的任务状态。", request_id=request_id)
        reply = {"id": task_id, "request_id": request_id, "provider_status": status}
        if status in (0, 1):
            return {**reply, "status": "pending"}
        if status == 3:
            reason = result.get("FailureReason") or {}
            if not isinstance(reason, dict):
                reason = {}
            return {**reply, "status": "failed", "error_code": _safe_code(reason.get("Code"))}
        if status != 2:
            raise ProviderError("音乐服务返回未识别的任务状态；保留原任务继续核对。", request_id=request_id)
        song = result.get("SongDetail") or {}
        if not isinstance(song, dict):
            raise ProviderError("音乐任务已完成但结果格式无效；请保留任务 ID。", request_id=request_id)
        url = song.get("AudioUrl")
        if not isinstance(url, str):
            raise ProviderError("音乐任务已完成但未返回音频地址；请保留任务 ID。", request_id=request_id)
        _audio_url(url)
        duration = song.get("Duration")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ProviderError("音乐任务已完成但实测时长无效；请核对账单和任务。", request_id=request_id)
        return {**reply, "status": "completed", "url": url, "duration": duration}

    def download(self, url: str, path: Path) -> dict:
        _audio_url(url)
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        size = 0
        digest = hashlib.sha256()
        try:
            # No API credentials are attached to a media request.
            with build_opener(_AudioRedirect()).open(Request(url, headers={"User-Agent": "Norlan/0.1"}),
                                                     timeout=self.timeout) as response:
                _audio_url(response.geturl())
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".norlan-music-", delete=False) as target:
                    temporary = Path(target.name)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > 100_000_000:
                            raise ProviderError("音乐下载超过 100 MB，已停止。")
                        digest.update(chunk)
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
            if size == 0:
                raise ProviderError("音乐下载返回空文件。")
            os.replace(temporary, path)
        except (HTTPError, URLError, OSError, TimeoutError, socket.timeout):
            raise ProviderError("音乐已生成，但下载失败；请用原任务 ID 重试下载，勿重新生成。") from None
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        return {"output": str(path), "sha256": digest.hexdigest(), "size": size}
