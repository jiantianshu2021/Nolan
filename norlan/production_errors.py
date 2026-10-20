"""Small durable production diagnostics, independent of transient last_error.

Only a caller-selected safe message and bounded phase identifiers are kept.
This module never records exceptions, request data, prompts or configuration.
"""
from __future__ import annotations

import re

from . import core
from .failures import sanitize_failure


HISTORY_LIMIT = 20
_STAGES = {"planning", "assets", "episode_planning", "shot_preparation",
           "shot_generation", "shot_review", "postproduction", "production"}
_ASSET_PRIORITY = {"generating": 0, "asset_prompt_approved": 1}
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})")


def _identifier(value):
    if (isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value)
            and sanitize_failure({"request_id": value}).get("request_id") == value):
        return value
    return None


def _phase(state):
    if state.get("plan", {}).get("status") != "approved":
        return {"stage": "planning"}
    assets = [asset for asset in state.get("assets", [])
              if isinstance(asset, dict) and not core.asset_ready(asset)]
    if assets:
        # The active generation is the best target even if earlier unrelated
        # assets are still waiting for their first input or human decision.
        asset = min(assets, key=lambda item: _ASSET_PRIORITY.get(item.get("status"), 2))
        result = {"stage": "assets"}
        identifier, version = _identifier(asset.get("id")), asset.get("version")
        if identifier:
            result["asset_id"] = identifier
        if type(version) is int and version >= 0:
            result["asset_version"] = version
        return result
    for episode in state.get("episodes", []):
        if not isinstance(episode, dict) or episode.get("status") == "approved":
            continue
        shots = episode.get("shots", [])
        if not shots:
            return {"stage": "episode_planning"}
        shot = next((item for item in shots if isinstance(item, dict)
                     and not core.human_approved_shot(item)), None)
        if shot is None:
            return {"stage": "postproduction"}
        status = shot.get("status")
        if status in {"generating", "generation_failed"} or shot.get("job_id"):
            return {"stage": "shot_review" if status == "review_required" else "shot_generation"}
        if status == "review_required" or shot.get("file"):
            return {"stage": "shot_review"}
        return {"stage": "shot_preparation"}
    return {"stage": "production"}


def _safe_record(value):
    """Re-project history to the same allowlist rather than copying old blobs."""
    if not isinstance(value, dict) or type(value.get("schema")) is not int or value.get("schema") != 1:
        return None
    message = sanitize_failure({"provider_message": value.get("message")}).get("provider_message")
    stamp = value.get("time")
    if not message or not isinstance(stamp, str) or not _TIME.fullmatch(stamp):
        return None
    stage = value.get("stage")
    result = {"schema": 1, "time": stamp, "message": message,
              "stage": stage if isinstance(stage, str) and stage in _STAGES else "production"}
    identifier, version = _identifier(value.get("asset_id")), value.get("asset_version")
    if identifier:
        result["asset_id"] = identifier
    if type(version) is int and version >= 0:
        result["asset_version"] = version
    return result


def _signature(value):
    return tuple(value.get(key) for key in ("stage", "asset_id", "asset_version", "message"))


def record(root, message, *, status="blocked") -> None:
    """Persist one safe current failure and at most 20 distinct phase records."""
    safe = sanitize_failure({"provider_message": message})
    text = safe.get("provider_message") or safe["message"]
    with core.transaction(root) as state:
        current = {"schema": 1, "time": core.now(), "message": text, **_phase(state)}
        history = []
        previous = state.get("production_failure_history", [])
        if not isinstance(previous, list):
            previous = []
        # Include a previous current record if a legacy interrupted write left
        # it outside history, keeping only allowlisted safe fields.
        for entry in [*previous[-HISTORY_LIMIT:], state.get("production_failure"), current]:
            item = _safe_record(entry)
            if item is None:
                continue
            signature = _signature(item)
            history = [old for old in history if _signature(old) != signature]
            history.append(item)
        state.update(status=status, last_error=text, production_failure=current,
                     production_failure_history=history[-HISTORY_LIMIT:])
