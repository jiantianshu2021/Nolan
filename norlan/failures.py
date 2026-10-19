"""Small, allowlisted error records safe to persist and show in the dashboard.

Never persist exception reprs, HTTP bodies, request payloads, or credentials.
Provider adapters can additionally supply exact prompt/key strings for removal.
"""
from __future__ import annotations

import os
import re
from typing import Iterable


STAGES = {"local_validation", "submission", "generation", "poll", "download", "unknown"}
_DEFAULT_MESSAGES = {
    "local_validation": "本机参数或素材校验失败，尚未提交生成。",
    "submission": "提交生成请求失败，请根据错误详情核对后再决定是否重试。",
    "generation": "供应商未完成本次生成，没有可供审批的视频。",
    "poll": "查询已有生成任务失败，原任务不会因此重新提交。",
    "download": "下载已有生成结果失败，请重试下载原任务。",
    "unknown": "操作失败，未记录到可安全展示的具体原因。",
}


def _secrets(extra: Iterable[str]) -> list[str]:
    values = [value for key, value in os.environ.items()
              if re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", key, re.I)
              and isinstance(value, str) and len(value) >= 4]
    values.extend(value for value in extra if isinstance(value, str) and value)
    return sorted(set(values), key=len, reverse=True)


def _text(value: object, limit: int, secrets: list[str]) -> str:
    if not isinstance(value, str):
        return ""
    # Cap work before pattern matching; no remote response is ever returned whole.
    value = value[:16_384]
    for secret in secrets:
        value = value.replace(secret, "[已隐藏]")
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    value = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", value)
    value = re.sub(r"data:[^\s\"'<>]+", "[素材数据已隐藏]", value, flags=re.I)
    value = re.sub(r"https?://[^\s\"'<>]+", "[地址已隐藏]", value, flags=re.I)
    value = re.sub(r"\bBearer\s+[^\s,;\"'<>]+", "Bearer [已隐藏]", value, flags=re.I)
    value = re.sub(r"\b(?:sk|ak|pk)[-_][A-Za-z0-9_-]{8,}", "[凭据已隐藏]", value)
    value = re.sub(
        r"(?i)\b((?:api[-_ ]?key|access[-_ ]?key|secret|token|password|authorization|signature)\s*[=:]\s*)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)", r"\1[已隐藏]", value)
    # Error messages sometimes echo input values. Preserve field names and limits,
    # but remove quoted or assignment-style input text instead of logging scripts.
    value = re.sub(
        r"(?i)((?:prompt|input[-_ ]?text|text|提示词|输入文本)\s*(?:is|was|=|:|为|是)\s*)"
        r"(?:\"[^\"]*\"|'[^']*')", r"\1[输入内容已隐藏]", value)
    value = re.sub(
        r"(?i)((?:prompt|input[-_ ]?text|text|提示词|输入文本)\s*[=:]\s*)[^\n;]+",
        r"\1[输入内容已隐藏]", value)
    value = re.sub(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{80,}(?![A-Za-z0-9+/=_-])",
                   "[长数据已隐藏]", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit] + ("…" if len(value) > limit else "")


def sanitize_failure(value: object, *, redact: Iterable[str] = ()) -> dict:
    """Return only bounded safe fields; safe to call again on persisted records."""
    source = value if isinstance(value, dict) else {}
    stage = source.get("stage")
    stage = stage if isinstance(stage, str) and stage in STAGES else "unknown"
    secrets = _secrets(redact)
    result = {"stage": stage,
              "message": _text(source.get("message"), 400, secrets) or _DEFAULT_MESSAGES[stage],
              "uncertain": source.get("uncertain") is True}
    provider_message = _text(source.get("provider_message"), 800, secrets)
    if provider_message:
        result["provider_message"] = provider_message
    for name, pattern, limit in (("provider_code", r"[A-Za-z0-9_.:-]+", 120),
                                 ("request_id", r"[A-Za-z0-9_.:-]+", 200),
                                 ("parameter", r"[A-Za-z0-9_.\[\]:-]+", 160)):
        raw = source.get(name)
        if isinstance(raw, str) and len(raw) <= limit and re.fullmatch(pattern, raw):
            # A short prompt such as "test" is not a secret embedded in a
            # request ID like "test-request-1". Exact values and realistic
            # credential-length substrings remain protected.
            identifier_secrets = [secret for secret in secrets if len(secret) >= 16 or secret == raw]
            if _text(raw, limit, identifier_secrets) == raw:
                result[name] = raw
    status = source.get("http_status")
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
        result["http_status"] = status
    return result


def exception_failure(exc: BaseException, stage: str = "unknown") -> dict:
    """Extract an adapter's safe record; unknown errors reveal type, never text."""
    existing = getattr(exc, "failure", None)
    if isinstance(existing, dict):
        result = dict(existing)
        if result.get("stage") in {None, "unknown"} or (result.get("stage") == "local_validation"
                                                       and stage in {"poll", "download"}):
            result["stage"] = stage
        # Older callers may update these compatibility attributes after decoding.
        result["uncertain"] = getattr(exc, "uncertain", result.get("uncertain")) is True
        if getattr(exc, "request_id", None):
            result["request_id"] = exc.request_id
        return sanitize_failure(result)
    kind = type(exc).__name__
    kind = kind if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", kind) else "Exception"
    safe_stage = stage if stage in STAGES else "unknown"
    return sanitize_failure({"stage": safe_stage,
                             "message": _DEFAULT_MESSAGES[safe_stage] + f"（错误类型：{kind}）",
                             "uncertain": getattr(exc, "uncertain", False) is True})
