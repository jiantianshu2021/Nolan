"""Read-only provider usage lookup and safe local cost reporting.

Usage lookups issue GET requests for existing tasks only. Bill amounts are
imported through core.import_bill_costs, never inferred from public prices.
"""

from collections.abc import Mapping
import math
import re

from . import core
from .billing import USAGE_FIELDS, price_snapshot, summarize_costs, usage_record
from .providers import ArkProvider, ProviderError


def _identifier(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value) else ""


def _number(value):
    return value if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None


def _usage(value):
    value = value if isinstance(value, Mapping) else {}
    return {key: int(number) for key in USAGE_FIELDS
            if (number := _number(value.get(key))) is not None and int(number) == number}


def _complete_usage(billing):
    if not isinstance(billing, Mapping) or billing.get("status") != "usage_priced":
        return False
    if billing.get("provider_status") not in {"succeeded", "completed", "failed", "cancelled", "expired"}:
        return False
    record = usage_record(billing, billing.get("price", {}))
    return (record.get("status") == "usage_priced" and _number(billing.get("usage_cost")) is not None
            and record["usage_cost"] == billing["usage_cost"])


def sync_usage(root, confirm_no_video_input=False):
    """GET existing terminal/uncertain video tasks without changing production.

    Historical tasks without a locked price only acquire a current-price
    calculation after the caller explicitly confirms there was no input video.
    This does not establish a historical bill or settle an unknown job outcome.
    """
    state = core.load_project(root)
    summary = {"synced": 0, "skipped": 0, "unavailable": 0, "errors": []}
    eligible = []
    for job_id, job in state["jobs"].items():
        if state.get("mode") != "live" or job.get("accounting") not in {"spent", "unknown"}:
            summary["skipped"] += 1
        elif _complete_usage(job.get("billing")):
            summary["skipped"] += 1
        elif job.get("kind") not in {"voice", "video"} or not isinstance(job.get("task_id"), str) or not job["task_id"]:
            summary["unavailable"] += 1
        else:
            eligible.append((job_id, job))
    if not eligible:
        return summary
    config = dict(state["settings"].get("ark", {}))
    config["currency"] = state["budget"]["currency"]
    try:
        provider = ArkProvider(config)
    except (ProviderError, ValueError, TypeError, OSError):
        raise ProviderError("无法同步用量：请先在本机配置方舟 API Key 对应的环境变量，并检查方舟连接设置；不会显示密钥。") from None
    for job_id, job in eligible:
        try:
            result = provider.video_usage(job["task_id"])
            historical = None
            billing = job.get("billing") or {}
            if not billing.get("price") and confirm_no_video_input is True:
                historical_config = {**config, "video_model": result.get("model"),
                                     "resolution": result.get("resolution"), "service_tier": result.get("service_tier")}
                candidate = price_snapshot(historical_config, job["kind"], has_video_input=False)
                if "rate" in candidate:
                    candidate["basis"] = "current_public_price_for_historical_usage"
                    historical = candidate
            updated = core.record_usage(root, job_id, result, historical_pricing=historical)
            if "completion_tokens" in _usage(updated.get("billing", {}).get("usage")):
                summary["synced"] += 1
            else:
                summary["unavailable"] += 1
        except (ProviderError, ValueError, TypeError, KeyError, OSError):
            summary["errors"].append({"job_id": _identifier(job_id),
                                      "error": "无法读取该任务用量；请核对任务 ID、权限或服务状态，原任务未重发。"})
    return summary


def cost_summary(state):
    """Public cost totals; historical budget ceilings are never active controls."""
    currency = state.get("budget", {}).get("currency", "CNY")
    jobs = state.get("jobs", {})
    result = {"currency": currency, **summarize_costs(jobs, currency)}
    entries = [job for job in jobs.values() if isinstance(job, Mapping)]
    result["unknown_jobs"] = sum(
        job.get("accounting") == "unknown" or job.get("state") == "uncertain"
        for job in entries)
    result["inflight_jobs"] = sum(
        job.get("accounting") != "unknown" and job.get("state") != "uncertain"
        and (job.get("accounting") == "reserved"
             or job.get("state") in {"submitting", "submitted", "download_pending"})
        for job in entries)
    return result


def cost_report(root):
    """Return only bounded accounting metadata, not prompts, keys or responses."""
    state = core.load_project(root)
    currency = state["budget"]["currency"]
    # Retain the JSON container name for existing clients, not budget controls.
    budget = cost_summary(state)
    calls = []
    for job_id, job in state["jobs"].items():
        billing = job.get("billing")
        billing = billing if isinstance(billing, Mapping) else {}
        price = billing.get("price")
        price = price if isinstance(price, Mapping) else {}
        single = summarize_costs({job_id: job}, currency)
        calls.append({"job_id": _identifier(job_id), "kind": _identifier(job.get("kind")),
                      "item_id": _identifier(job.get("item_id")), "task_id": _identifier(job.get("task_id")),
                      "accounting": _identifier(job.get("accounting")), "state": _identifier(job.get("state")),
                      "currency": currency,
                      "actual_cost": single["actual_cost"], "bill_verified": single["actual_cost_jobs"] > 0,
                      "usage_cost": single["usage_cost"], "usage": _usage(billing.get("usage")),
                      "usage_status": _identifier(billing.get("status")), "model": _identifier(billing.get("model")),
                      "request_id": _identifier(billing.get("request_id")),
                      "historical_public_price": bool(price.get("historical_lookup"))})
    return {"budget": budget, "calls": calls,
            "basis": {"actual_cost": "逐调用账单核实的实际费用；没有账单依据时显示待对账。",
                      "usage_cost": "供应商实际用量 × 锁定公开单价，未核对账单，不含折扣、赠额等调整，不等于账户扣款。"}}


def _money(value, currency, missing="待对账"):
    number = value if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) else None
    return f"{number:.6f}".rstrip("0").rstrip(".") + " " + currency if number is not None else missing


def format_report(report):
    budget = report["budget"]
    currency = budget["currency"]
    rows = ["Nolan · 费用明细",
            "已核对实际费用：" + _money(budget["actual_cost"], currency)
            + f"（{budget['actual_cost_jobs']} / {budget['cost_jobs']} 次调用已核对）",
            "按实际用量计费（未对账）：" + _money(budget["usage_cost"], currency, "暂无可计费用量")
            + f"；未核实调用 {budget['cost_pending_jobs']} 次",
            f"进行中调用：{budget.get('inflight_jobs', 0)} 次；结果待核实调用：{budget.get('unknown_jobs', 0)} 次", ""]
    for call in report["calls"]:
        rows.append(f"{call['job_id']} · {call['kind']} · {call['item_id']} · {call['accounting']}")
        rows.append("  核实费用 " + _money(call["actual_cost"], currency)
                    + " / 用量计费 " + _money(call["usage_cost"], currency, "—"))
        if call["task_id"]:
            rows.append("  任务 ID " + call["task_id"])
        if call["historical_public_price"]:
            rows.append("  历史用量按当前公开价折算，不是历史账单费用。")
    if not report["calls"]:
        rows.append("尚无调用记录。")
    rows.extend(("", *report["basis"].values()))
    return "\n".join(rows)
