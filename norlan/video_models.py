"""Verified Ark video choices and local-only compatibility checks.

Checked 2026-10-20 against the official model list, creation API and tutorials:
https://docs.volcengine.com/docs/ark/model-list?lang=zh
https://docs.volcengine.com/docs/ark/create-video-generation-task-api?lang=zh
https://docs.volcengine.com/docs/ark/seedance-2-5?lang=zh
https://docs.volcengine.com/docs/ark/seedance-2-0?lang=zh

This selector never probes accounts or silently changes references or duration.
Reference limits intentionally retain Nolan's existing 9-image/3-audio subset.
"""

from collections.abc import Mapping
import copy
import math
import re


CHECKED_AT = "2026-10-20"
DEFAULT_MODEL = "doubao-seedance-2-0-fast-260128"
_MODELS = (
    {"id": DEFAULT_MODEL, "label": "Seedance 2.0 Fast", "max_duration": 15,
     "resolutions": ("480p", "720p"), "available": True},
    {"id": "doubao-seedance-2-0-260128", "label": "Seedance 2.0", "max_duration": 15,
     "resolutions": ("480p", "720p", "1080p", "4k"), "available": True},
    {"id": "doubao-seedance-2-0-mini-260615", "label": "Seedance 2.0 Mini", "max_duration": 15,
     "resolutions": ("480p", "720p"), "available": True},
    {"id": "doubao-seedance-2-5-260628", "label": "Seedance 2.5", "max_duration": 30,
     "resolutions": ("480p", "720p", "1080p"), "available": True},
    {"id": "doubao-seedance-1-0-pro-250528", "label": "Seedance 1.0 Pro", "max_duration": 12,
     "resolutions": ("480p", "720p", "1080p"), "available": False},
    {"id": "doubao-seedance-1-0-pro-fast-251015", "label": "Seedance 1.0 Pro Fast", "max_duration": 12,
     "resolutions": ("480p", "720p", "1080p"), "available": False},
)
_BY_ID = {item["id"]: item for item in _MODELS}
_LEGACY_REASON = "此模型不支持当前连续性流程所需的多资产参考和原生声音，请选择 Seedance 2.x。"
_BASE_CAPABILITIES = {
    "image_references": True, "max_video_images": 9,
    "max_audio_references": 3, "max_audio_seconds": 15,
    "local_video_upload": False, "video_reference_url": True,
    "official_last_frame": True, "official_video_artifacts": True,
    "max_video_reference_seconds": 15, "max_video_reference_bytes": 200_000_000,
    "native_audio": True, "stable_voice_guarantee": False,
    "first_frame_plus_references": False, "min_duration": 4, "max_duration": 15,
}


def effective_config(state, shot=None):
    """Copy provider settings, with only this segment's explicit model override."""
    state = state if isinstance(state, Mapping) else {}
    settings = state.get("settings", {})
    settings = settings if isinstance(settings, Mapping) else {}
    ark = settings.get("ark", {})
    config = copy.deepcopy(dict(ark)) if isinstance(ark, Mapping) else {}
    if isinstance(shot, Mapping) and shot.get("video_model"):
        config["video_model"] = shot["video_model"]
    plan = state.get("plan", {})
    plan = plan if isinstance(plan, Mapping) else {}
    config["ratio"] = plan.get("ratio", "16:9")
    return config


def capabilities(config):
    """Adapter limits; unknown configured endpoints retain the legacy subset."""
    result = dict(_BASE_CAPABILITIES)
    model = config.get("video_model") if isinstance(config, Mapping) else None
    definition = _BY_ID.get(model) if isinstance(model, str) else None
    if definition and definition["available"]:
        result["max_duration"] = definition["max_duration"]
    return result


def validate_request(config, duration, image_count, audio_count, has_video):
    """Validate known models before submission, without reading media or keys.

    Unknown endpoint aliases keep the pre-existing provider path. They do not
    become new selectable models merely by appearing in an incoming UI payload.
    """
    config = config if isinstance(config, Mapping) else {}
    model = config.get("video_model")
    definition = _BY_ID.get(model) if isinstance(model, str) else None
    if definition is None:
        return
    if not definition["available"]:
        raise ValueError(_LEGACY_REASON)
    caps = capabilities(config)
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or int(duration) != duration
            or not caps["min_duration"] <= duration <= caps["max_duration"]):
        raise ValueError(f"{definition['label']} 的视频时长须为 {caps['min_duration']}–{caps['max_duration']} 秒的整数；不会自动修改片段时长。")
    for value, maximum, label in ((image_count, caps["max_video_images"], "参考图"),
                                  (audio_count, caps["max_audio_references"], "声音参考")):
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError(f"本流程每次最多 {maximum} 个{label}；不会静默丢弃素材。")
    if type(has_video) is not bool:
        raise ValueError("视频参考状态无效。")
    if audio_count and not (image_count or has_video):
        raise ValueError("本流程的声音参考须搭配图片或视频。")
    resolution = config.get("resolution", "720p")
    if resolution not in definition["resolutions"]:
        raise ValueError(f"{definition['label']} 不支持当前 {resolution} 分辨率；请先调整制作设置。")


def _manifest(candidate):
    if not isinstance(candidate, Mapping):
        return {}
    reference_manifest = candidate.get("reference_manifest")
    if isinstance(reference_manifest, Mapping):
        return reference_manifest
    specification = candidate.get("specification")
    if isinstance(specification, Mapping) and isinstance(specification.get("references"), Mapping):
        return specification["references"]
    references = candidate.get("references")
    if isinstance(references, Mapping):
        return references
    return candidate


def validate_selection(state, shot, manifest, model):
    """Return an accepted canonical selection, or a human-readable ValueError."""
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", model):
        raise ValueError("请选择列表中的视频模型。")
    configured = effective_config(state).get("video_model")
    if model not in _BY_ID and model != configured:
        raise ValueError("该视频模型不在已核实的模型列表中。")
    if model in _BY_ID and not _BY_ID[model]["available"]:
        raise ValueError(_LEGACY_REASON)
    if "seedance-1-" in model.lower() or "seedream-" in model.lower():
        raise ValueError(_LEGACY_REASON)
    shot = shot if isinstance(shot, Mapping) else {}
    manifest = _manifest(manifest)
    context = manifest.get("context", {})
    context = context if isinstance(context, Mapping) else {}
    previous = context.get("previous_actual", {})
    previous = previous if isinstance(previous, Mapping) else {}
    images = manifest.get("images", [])
    audio = manifest.get("audio", [])
    if not isinstance(images, list) or not isinstance(audio, list):
        raise ValueError("候选提示词的参考素材清单无效。")
    has_video = bool(context.get("video_references") or previous.get("reference_mode") == "original_video")
    config = effective_config(state, {**shot, "video_model": model})
    validate_request(config, shot.get("duration", 5), len(images), len(audio), has_video)
    return model


def catalog(state):
    configured = effective_config(state).get("video_model")
    items = [{**copy.deepcopy(item), "resolutions": list(item["resolutions"]),
              "reason": "" if item["available"] else _LEGACY_REASON} for item in _MODELS]
    if isinstance(configured, str) and configured and configured not in _BY_ID:
        safe = bool(re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", configured))
        if safe and "seedance-1-5-" not in configured.lower():
            unavailable = "seedance-1-" in configured.lower() or "seedream-" in configured.lower()
            items.insert(0, {"id": configured, "label": "当前已配置模型（自定义接入点）",
                             "available": not unavailable,
                             "reason": _LEGACY_REASON if unavailable else "能力与费用依照当前接入点配置。"})
    safe_default = configured if isinstance(configured, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", configured) else ""
    return {"default_model": safe_default,
            "models": items, "checked_at": CHECKED_AT}


def options(state, shot, candidate=None):
    result = []
    for item in catalog(state)["models"]:
        reason = item["reason"]
        available = item["available"]
        if available:
            try:
                validate_selection(state, shot, _manifest(candidate), item["id"])
            except ValueError as exc:
                available, reason = False, str(exc)
        result.append({"id": item["id"], "label": item["label"],
                       "available": available, "reason": reason})
    return result
