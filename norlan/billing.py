"""Provider usage pricing and bill reconciliation, separate from budget caps.

Public-price calculations are not account charges: discounts, credits and other
bill adjustments are deliberately not inferred here. This module performs no IO.
"""

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import math
import re


ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
PRICE_SOURCE = "https://docs.volcengine.com/docs/ark/model-pricing?lang=zh&redirect=1"
PRICE_CHECKED_AT = "2026-10-20"
IMAGE_MODEL = "doubao-seedream-5-0-flash-260915"
VIDEO_MODEL = "doubao-seedance-2-0-fast-260128"
USAGE_FIELDS = ("completion_tokens", "generated_images", "input_images", "output_tokens", "total_tokens")
_SIX_PLACES = Decimal("0.000001")


def _identifier(value):
    """Keep bounded identifiers, never response bodies, URLs or auth text."""
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value) else ""


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() and result >= 0 else None


def _integer(value):
    number = _number(value)
    return int(number) if number is not None and number == number.to_integral_value() else None


def _amount(value):
    """Round decimal arithmetic once at the JSON boundary, not per token."""
    with localcontext() as context:
        context.prec = max(28, len(value.as_tuple().digits), value.adjusted() + 8)
        result = float(value.quantize(_SIX_PLACES, rounding=ROUND_HALF_UP))
    return result if math.isfinite(result) else None


def price_snapshot(config, kind, *, has_video_input=False):
    """Lock a verified public price for the exact request configuration.

    Missing/unsupported prices have no ``rate`` field. Endpoint aliases and
    third-party gateways must be reconciled separately; their prices are unknown.
    ``rate`` is CNY/image or CNY/million completion tokens, according to ``unit``.
    """
    config = config if isinstance(config, Mapping) else {}
    video = kind in {"voice", "video"}
    model = _identifier(config.get("video_model" if video else "image_model", ""))
    resolution = _identifier(config.get("resolution", "720p") if video else config.get("image_size", "2K"))
    currency = _identifier(config.get("currency", "CNY"))
    tier = _identifier(config.get("service_tier", "default"))
    provider = _identifier(config.get("provider", "ark"))
    snapshot = {"provider": provider, "model": model, "unit": "", "currency": currency,
                "price_checked_at": PRICE_CHECKED_AT, "source": PRICE_SOURCE,
                "has_video_input": has_video_input if isinstance(has_video_input, bool) else None,
                "resolution": resolution, "service_tier": tier}
    base_url = config.get("base_url", ARK_BASE_URL)
    if (provider != "ark" or currency != "CNY" or tier != "default"
            or not isinstance(base_url, str) or base_url.rstrip("/") != ARK_BASE_URL
            or not isinstance(has_video_input, bool)):
        return snapshot
    if kind == "image" and model == IMAGE_MODEL and not has_video_input:
        snapshot.update(unit="generated_images", rate=0.12)
    elif video and model == VIDEO_MODEL and resolution in {"480p", "720p"}:
        snapshot.update(unit="million_completion_tokens", rate=22 if has_video_input else 37)
    return snapshot


def usage_record(result, snapshot):
    """Keep safe reported usage and price it only against a matching snapshot."""
    result = result if isinstance(result, Mapping) else {}
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    raw_usage = result.get("usage")
    raw_usage = raw_usage if isinstance(raw_usage, Mapping) else {}
    usage = {}
    for field in USAGE_FIELDS:
        value = _integer(raw_usage.get(field))
        if value is not None:
            usage[field] = value
    model = _identifier(result.get("model"))
    provider_status = _identifier(result.get("provider_status", result.get("status", "")))
    currency = _identifier(snapshot.get("currency"))
    record = {"usage": usage, "usage_cost": None, "currency": currency,
              "status": "pricing_missing", "model": model,
              "request_id": _identifier(result.get("request_id")), "provider_status": provider_status}
    if "resolution" in result:
        record["resolution"] = _identifier(result["resolution"])
    if provider_status not in {"succeeded", "completed", "failed", "cancelled", "expired"}:
        record["status"] = "usage_missing"
        return record
    returned_tier = result.get("service_tier", "default")
    if "service_tier" in result:
        record["service_tier"] = _identifier(returned_tier)
    rate = _number(snapshot.get("rate"))
    unit = snapshot.get("unit")
    if (not model or model != snapshot.get("model") or snapshot.get("provider") != "ark"
            or currency != "CNY" or snapshot.get("service_tier") != "default"
            or returned_tier != "default" or rate is None):
        return record
    if unit == "generated_images" and model == IMAGE_MODEL:
        field, divisor = "generated_images", Decimal(1)
    elif (unit == "million_completion_tokens" and model == VIDEO_MODEL
          and snapshot.get("resolution") in {"480p", "720p"}):
        returned_resolution = result.get("resolution", snapshot["resolution"])
        if (not isinstance(returned_resolution, str) or returned_resolution not in {"480p", "720p"}
                or returned_resolution != snapshot["resolution"]):
            return record
        field, divisor = "completion_tokens", Decimal(1_000_000)
    else:
        return record
    record["status"] = "usage_missing"
    quantity = usage.get(field)
    if quantity is None:
        return record
    # Only successful video generation is billable under this public schedule.
    # Positive tokens on a failed task are contradictory evidence, not a charge.
    if field == "completion_tokens" and quantity > 0 and provider_status in {"failed", "cancelled", "expired"}:
        return record
    # A successful image response claiming zero generated images is contradictory.
    # Do not turn that missing evidence into a zero-cost successful generation.
    if field == "generated_images" and quantity == 0 and provider_status not in {"failed", "cancelled", "expired"}:
        return record
    with localcontext() as context:
        context.prec = max(28, len(str(quantity)) + len(rate.as_tuple().digits) + 8)
        cost = _amount(Decimal(quantity) * rate / divisor)
    if cost is not None:
        record.update(usage_cost=cost, status="usage_priced")
    return record


def summarize_costs(jobs, currency):
    """Summarize unique ledger entries without interpreting caps as paid costs.

    A verified bill amount takes precedence over usage pricing for that job, even
    when the explicit amount is zero. Missing monetary evidence stays unknown.
    """
    jobs = jobs if isinstance(jobs, Mapping) else {}
    actual_total, usage_total = Decimal(0), Decimal(0)
    cost_jobs = actual_jobs = usage_jobs = 0
    for job in jobs.values():
        if not isinstance(job, Mapping):
            continue
        amount = _number(job.get("amount"))
        if amount is None or amount <= 0 or job.get("accounting") not in {"spent", "unknown"}:
            continue
        cost_jobs += 1
        billing = job.get("billing")
        if currency not in {"CNY", "USD"} or not isinstance(billing, Mapping) or billing.get("currency") != currency:
            continue
        actual = _number(billing.get("actual_cost"))
        reference = billing.get("bill_reference")
        if actual is not None and isinstance(reference, str) and reference.strip():
            actual_total += actual
            actual_jobs += 1
            continue
        usage = _number(billing.get("usage_cost"))
        if usage is not None and billing.get("status") == "usage_priced" and (usage > 0 or _zero_usage_evidence(billing)):
            usage_total += usage
            usage_jobs += 1
    return {"actual_cost": _amount(actual_total) if actual_jobs else None,
            "actual_cost_jobs": actual_jobs, "cost_pending_jobs": cost_jobs - actual_jobs,
            "usage_cost": _amount(usage_total) if usage_jobs else None,
            "usage_cost_jobs": usage_jobs, "cost_jobs": cost_jobs}


def _zero_usage_evidence(billing):
    """An absent usage field is not evidence that a request cost zero."""
    usage = billing.get("usage")
    if not isinstance(usage, Mapping):
        return False
    if billing.get("model") == VIDEO_MODEL:
        return _integer(usage.get("completion_tokens")) == 0
    return (billing.get("model") == IMAGE_MODEL and _integer(usage.get("generated_images")) == 0
            and billing.get("provider_status") in {"failed", "cancelled", "expired"})
