"""Explicit, per-generation asset models; no IO or project mutations.

Image model IDs/specifications checked 2026-10-20 against official Ark docs:
https://docs.volcengine.com/docs/ark/model-list?lang=zh
https://docs.volcengine.com/docs/ark/image-generation-api?lang=en
https://docs.volcengine.com/docs/ark/seedream-5-0-pro?lang=zh

Voice auditions reuse the existing video catalog, with native audio required.
The selector never drops actual references or silently alters duration/size.
"""

from collections.abc import Mapping
import copy
import math
import re

from . import video_models
from .billing import IMAGE_MODEL, estimate_reservation, price_snapshot


CHECKED_AT = "2026-10-20"
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,200}")
_IMAGES = (
    {"id": IMAGE_MODEL, "label": "Seedream 5.0 Flash", "sizes": ("1K", "1.5K", "2K"),
     "min_pixels": 921600, "max_pixels": 4624220, "retiring": False},
    {"id": "doubao-seedream-5-0-pro-260628", "label": "Seedream 5.0 Pro", "sizes": ("1K", "1.5K", "2K"),
     "min_pixels": 921600, "max_pixels": 4624220, "retiring": False},
    {"id": "doubao-seedream-5-0-260128", "label": "Seedream 5.0 Lite", "sizes": ("2K", "3K", "4K"),
     "min_pixels": 3686400, "max_pixels": 16777216, "retiring": True},
    {"id": "doubao-seedream-5-0-lite-260128", "label": "Seedream 5.0 Lite（别名）", "sizes": ("2K", "3K", "4K"),
     "min_pixels": 3686400, "max_pixels": 16777216, "retiring": True},
    {"id": "doubao-seedream-4-5-251128", "label": "Seedream 4.5", "sizes": ("2K", "4K"),
     "min_pixels": 3686400, "max_pixels": 16777216, "retiring": True},
    {"id": "doubao-seedream-4-0-250828", "label": "Seedream 4.0", "sizes": ("1K", "2K", "4K"),
     "min_pixels": 921600, "max_pixels": 16777216, "retiring": True},
)
_BY_ID = {item["id"]: item for item in _IMAGES}


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _voice(asset):
    return _mapping(asset).get("kind") == "voice"


def effective_config(state, asset, model=None):
    """Detached provider settings for one asset; project defaults stay intact."""
    state, asset = _mapping(state), _mapping(asset)
    config = copy.deepcopy(dict(_mapping(_mapping(state.get("settings")).get("ark"))))
    key = "video_model" if _voice(asset) else "image_model"
    selected = model if model is not None else asset.get("generation_model")
    if selected:
        config[key] = selected
    config.setdefault(key, video_models.DEFAULT_MODEL if _voice(asset) else IMAGE_MODEL)
    if _voice(asset):
        config["ratio"] = _mapping(state.get("plan")).get("ratio", "16:9")
        config.setdefault("resolution", "720p")
        config.setdefault("generate_audio", True)
        config.setdefault("video_watermark", False)
    else:
        config.setdefault("image_size", "2K")
    return config


def _references(state, asset, candidate=None):
    """Use the prepared manifest if supplied, otherwise current actual inputs."""
    candidate = _mapping(candidate)
    manifest = candidate.get("reference_manifest")
    if not isinstance(manifest, Mapping):
        manifest = candidate.get("references")
    if not isinstance(manifest, Mapping) and ("images" in candidate or "audio" in candidate):
        manifest = candidate
    if isinstance(manifest, Mapping):
        images, audio = manifest.get("images", []), manifest.get("audio", [])
        if not isinstance(images, list) or not isinstance(audio, list):
            raise ValueError("候选资产提示词的参考素材清单无效。")
        return images, audio
    if _voice(asset):
        characters = [item for item in _mapping(state).get("assets", [])
                      if isinstance(item, Mapping) and item.get("id") == asset.get("character_id")
                      and item.get("kind") == "character"]
        if not characters:
            raise ValueError("声音试片需要绑定已确认的角色或内部画外声源。")
        character = characters[0]
        if character.get("voice_only") is True:
            return [], []
        if character.get("status") != "approved" or not character.get("file"):
            raise ValueError("请先批准角色形象，再确认声音试片提示词与模型。")
        return [character["file"]], []
    refs = _mapping(asset.get("creation_input")).get("references", [])
    if not isinstance(refs, list):
        raise ValueError("资产参考图片清单无效。")
    return refs, []


def _duration(asset, duration=None, candidate=None):
    if duration is not None:
        return duration
    if "duration" in _mapping(candidate):
        return candidate["duration"]
    if "generation_duration" in _mapping(asset):
        return asset["generation_duration"]
    from .voice import audition_duration
    return audition_duration(asset)


def _validate_image_size(config, definition):
    size = config.get("image_size", "2K")
    if size in definition["sizes"]:
        return
    match = re.fullmatch(r"([1-9][0-9]{0,5})x([1-9][0-9]{0,5})", size) if isinstance(size, str) else None
    if match:
        width, height = map(int, match.groups())
        if (definition["min_pixels"] <= width * height <= definition["max_pixels"]
                and 1 / 16 <= width / height <= 16):
            return
    raise ValueError(f"{definition['label']} 不支持当前图片尺寸；请选择兼容模型或先调整图片尺寸，不会自动改尺寸。")


def catalog(state, asset):
    if _voice(asset):
        return video_models.catalog({**_mapping(state), "settings": {
            **_mapping(_mapping(state).get("settings")), "ark": effective_config(state, asset)}})
    configured = effective_config(state, asset).get("image_model")
    items = [{"id": item["id"], "label": item["label"], "available": True,
              "reason": "官方标注即将下线；是否可用以账号服务状态为准。" if item["retiring"] else ""}
             for item in _IMAGES]
    if isinstance(configured, str) and _ID.fullmatch(configured) and configured not in _BY_ID:
        invalid = "seedance-" in configured.lower()
        items.insert(0, {"id": configured, "label": "当前已配置图片模型（自定义接入点）",
                         "available": not invalid,
                         "reason": "视频模型不能生成图片资产。" if invalid else "能力与费用依照当前接入点配置。"})
    safe = configured if isinstance(configured, str) and _ID.fullmatch(configured) else ""
    return {"default_model": safe, "models": items, "checked_at": CHECKED_AT}


def validate_choice(state, asset, model, duration=None, *, candidate=None):
    """Validate a single selected model against its exact candidate inputs."""
    state, asset = _mapping(state), _mapping(asset)
    if not isinstance(model, str) or not _ID.fullmatch(model):
        raise ValueError("请选择列表中的资产生成模型。")
    config = effective_config(state, asset, model)
    images, audio = _references(state, asset, candidate)
    if _voice(asset):
        if config.get("generate_audio") is not True:
            raise ValueError("声音试片必须启用原生声音 generate_audio；不会静默改动已有设置。")
        video_models.validate_selection(
            {**state, "settings": {**_mapping(state.get("settings")), "ark": effective_config(state, asset)}},
            {"duration": _duration(asset, duration, candidate)},
            {"images": images, "audio": audio, "context": {}}, model)
        # Existing endpoint aliases retain conservative provider duration/count
        # limits too; video_models only checks official release IDs itself.
        caps = video_models.capabilities(config)
        seconds = _duration(asset, duration, candidate)
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or int(seconds) != seconds
                or not caps["min_duration"] <= seconds <= caps["max_duration"]):
            raise ValueError("声音试片时长不符合所选模型限制；不会自动缩短或加速试音。")
        if len(images) > caps["max_video_images"] or len(audio) > caps["max_audio_references"]:
            raise ValueError("声音试片参考素材超过模型限制；不会静默丢弃素材。")
        if audio and not images:
            raise ValueError("声音参考须搭配图片；不会虚构或静默丢弃素材。")
        return config
    configured = effective_config(state, asset).get("image_model")
    if model not in _BY_ID and model != configured:
        raise ValueError("该图片模型不在已核实的模型列表或当前配置中。")
    if "seedance-" in model.lower():
        raise ValueError("Seedance 是视频模型，请为图片资产选择 Seedream。")
    if audio:
        raise ValueError("图片资产不接受声音参考；不会静默丢弃素材。")
    if len(images) > 9:
        raise ValueError("当前资产流程每次最多 9 张参考图片；不会静默丢弃素材。")
    if model in _BY_ID:
        _validate_image_size(config, _BY_ID[model])
    else:
        cap = _mapping(_mapping(state.get("settings")).get("caps")).get("image")
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
            raise ValueError("自定义图片接入点费用未知，请先配置正数 image 预算占用上限。")
    return config


def validate_selection(state, asset, candidate, model):
    validate_choice(state, asset, model, candidate=candidate)
    return model


def options(state, asset, candidate=None, *, duration=None):
    result = []
    for item in catalog(state, asset)["models"]:
        available, reason = item["available"], item["reason"]
        if available:
            try:
                validate_choice(state, asset, item["id"], duration, candidate=candidate)
            except ValueError as exc:
                available, reason = False, str(exc)
        result.append({"id": item["id"], "label": item["label"],
                       "available": available, "reason": reason})
    return result


def reserve_amount(state, asset, model=None, duration=None, *, candidate=None):
    """Conservative budget occupancy, never an assertion of charged cost."""
    config = effective_config(state, asset, model)
    selected = config["video_model" if _voice(asset) else "image_model"]
    validate_choice(state, asset, selected, duration, candidate=candidate)
    settings = _mapping(_mapping(state).get("settings"))
    caps = _mapping(settings.get("caps"))
    currency = _mapping(_mapping(state).get("budget")).get("currency", "CNY")
    if _voice(asset):
        seconds = _duration(asset, duration, candidate)
        unit = caps.get("video_second", 0)
        if isinstance(unit, bool) or not isinstance(unit, (int, float)) or not math.isfinite(unit) or unit < 0:
            raise ValueError("声音试片的 video_second 预算占用上限无效。")
        estimate = estimate_reservation({**config, "currency": currency}, seconds)
        if estimate is None and not unit:
            raise ValueError("所选声音试片模型费用未知，请先配置正数 video_second 预算占用上限。")
        return round(max(unit * seconds, (estimate or 0) * 1.15), 6)
    cap = caps.get("image", 0)
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 0:
        raise ValueError("图片预算占用上限无效。")
    images, _ = _references(state, asset, candidate)
    snapshot = price_snapshot({**config, "currency": currency}, "image")
    public_cap = snapshot.get("rate", 0) + max(0, len(images) - snapshot.get("input_images_free", 0)) * snapshot.get("input_rate", 0)
    if not public_cap and not cap:
        raise ValueError("所选模型费用未知，请先配置正数 image 预算占用上限。")
    return round(max(cap, public_cap), 6)
