"""Atomic project state, human gates and conservative cost accounting.

Only this module changes project.json. Long-running network/media work happens
outside its short flock transactions. A separate process lock owns a run.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path


class GateError(ValueError):
    """A resumable production gate, not permission to skip it."""


class VideoReviewDisabled(GateError):
    """The operator disabled a new advisory request before submission."""


def video_review_enabled(state):
    # Missing or malformed legacy values never opt the operator into calls.
    return state.get("settings", {}).get("gpt_video_review_enabled") is True


def set_video_review_enabled(root, enabled):
    """Persist a live preference only; do not run, cancel or approve any work."""
    if type(enabled) is not bool:
        raise ValueError("GPT 视频审查开关必须为布尔值。")
    with transaction(root) as state:
        state["settings"]["gpt_video_review_enabled"] = enabled
        event(state, "video_review_setting_changed", enabled=enabled)
    return load_project(root)


_PLAN_ERRORS = {
    "title_required": ("规划缺少标题。", "填写非空字符串 title。"),
    "ratio_invalid": ("不支持的画面比例。", "ratio 使用 16:9、9:16、1:1、4:3、3:4 或 21:9。"),
    "assets_required": ("规划必须包含资产。", "assets 使用非空资产列表。"),
    "episodes_required": ("规划必须包含分集。", "episodes 使用非空分集列表。"),
    "too_many_items": ("一次规划的条目过多。", "每次最多 200 个资产、100 集，每集最多 100 个生成片段；请拆分规划。"),
    "item_type": ("规划条目必须为对象。", "使用符合工具 schema 的对象。"),
    "id_invalid": ("规划 ID 不合法或重复。", "ID 必须全局唯一，使用 1–64 位字母、数字、下划线或连字符，并以字母或数字开头。"),
    "asset_kind_invalid": ("资产类型不合法。", "kind 使用 character、location、prop 或 voice。"),
    "asset_text_required": ("资产缺少名称或生成说明。", "name 和 prompt 均须填写非空字符串。"),
    "voice_character_required": ("声音资产未绑定有效角色。", "voice 的 character_id 必须指向已定义的 character 资产。画外声音也须定义声源角色，并在提示词中说明不入画。"),
    "nonvoice_character_forbidden": ("非声音资产不能绑定 character_id。", "仅 voice 填写 character_id；其他资产省略此字段或填空字符串，角色自身也不例外。"),
    "voice_only_invalid": ("画外声源标记不合法。", "voice_only 使用布尔值；仅 character 可以为 true，表示只定义声源、不生成可见人物。"),
    "voice_selection_invalid": ("试音选句或来源不符合要求。", "为 voice 填写原文连续台词 audition_text、选句依据 audition_reason 和含说话人归属的原文摘录 audition_source。无合适句时 text/source 留空并在 reason 说明；非 voice 填空。"),
    "episode_title_required": ("每集必须包含标题。", "填写非空字符串 title。"),
    "shots_required": ("每集必须包含生成片段。", "shots 使用非空生成片段列表，每项对应一条视频 API 任务。"),
    "duration_invalid": ("生成片段时长必须为 4–15 秒整数。", "duration 填写 4 至 15 的整数秒数；默认优先 10–15 秒。"),
    "shot_prompt_required": ("生成片段缺少提示词。", "prompt 填写非空字符串。"),
    "subshots_invalid": ("生成片段的内部镜头时间表不合法。", "subshots 可省略；若提供须为1–3项，每项仅含有限数字 start/end 和非空 prompt（最多2000字）；从0秒连续覆盖 duration，不得留空、间隙或重叠。"),
    "asset_references_invalid": ("生成片段引用的资产不存在、重复或格式不正确。", "asset_ids 使用非空列表，只引用 assets 中已定义的 ID，须包含内部镜头需要的全部实际参考，且不能重复。"),
    "scene_required": ("生成片段缺少 scene_id。", "填写 scene_id，以维护空间与剪辑连续性。"),
    "voice_character_not_referenced": ("生成片段引用声音时缺少其绑定角色。", "在本生成片段 asset_ids 中加入该 voice 的 character_id。若声音来自画外，在 prompt 中明确该角色仅作为声源、不出现在画面中。"),
}


class PlanValidationError(ValueError):
    """Public planning diagnostics contain fixed text and schema/index paths only."""

    def __init__(self, code, path):
        message, hint = _PLAN_ERRORS[code]
        self.code, self.path, self.hint = code, path, hint
        super().__init__(message)

    def diagnostic(self):
        return {"category": "invalid_plan", "code": self.code, "path": self.path,
                "message": str(self), "hint": self.hint}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("路径必须位于项目目录内")
    return path


def event(project_state, action, **details):
    project_state["events"].append({"time": now(), "action": action, **details})


def is_nonblocking_advisory_unknown(job):
    """A lost advisory result may be skipped without inventing a billing outcome.

    This narrow execution exception never settles the ledger. A recovered
    review needs an explicit durable no-retry marker; amount zero, an LLM kind,
    or an arbitrary purpose label alone cannot exempt an unknown submission.
    """
    if not isinstance(job, dict):
        return False
    recovery = job.get("advisory_recovery")
    return (job.get("kind") == "llm" and job.get("item_id") == "review"
            and job.get("task_id") is None and job.get("accounting") == "unknown"
            and job.get("state") == "uncertain" and isinstance(recovery, dict)
            and recovery.get("status") == "interrupted" and recovery.get("no_retry") is True)


def blocking_unknown_jobs(state):
    """Return unresolved jobs whose execution still blocks new production."""
    return [job for job in state.get("jobs", {}).values()
            if (job.get("accounting") == "unknown" or job.get("state") == "uncertain")
            and not is_nonblocking_advisory_unknown(job)]


def execution_unsettled(job):
    """Separate execution gates from honest, still-unresolved cost accounting."""
    return (isinstance(job, dict) and not is_nonblocking_advisory_unknown(job)
            and (job.get("accounting") in {"reserved", "unknown"}
                 or job.get("state") in {"submitting", "submitted", "download_pending", "uncertain"}))


def _totals(state):
    from .billing import summarize_costs
    b = state["budget"]
    def exposure(job):
        billing = job.get("billing", {})
        values = [job["amount"]]
        if billing.get("currency") == b["currency"]:
            for key in ("usage_cost", "actual_cost"):
                value = billing.get(key)
                if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                    values.append(value)
        return max(values)
    for status in ("spent", "reserved", "unknown"):
        b[status] = round(sum(exposure(j) for j in state["jobs"].values()
                              if j["accounting"] == status), 6)
    b["available"] = round(b["limit"] - b["spent"] - b["reserved"] - b["unknown"], 6)
    b.update(summarize_costs(state["jobs"], b["currency"]))
    b["basis"] = "历史限额不再限制制作；用量费用按公开标准价计算，实际费用须核对账单"


def _write(root, state):
    _totals(state)
    state["updated_at"] = now()
    fd, temp = tempfile.mkstemp(prefix=".state-", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, Path(root) / "project.json")
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def transaction(root):
    root = Path(root).resolve()
    with (root / ".state.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (root / "project.json").open(encoding="utf-8") as f:
            state = json.load(f)
        yield state
        _write(root, state)


@contextmanager
def run_lock(root):
    with (Path(root) / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GateError("已有制作任务运行中，请勿重复启动") from None
        yield


def load_project(root):
    # os.replace provides a whole snapshot even while another process writes.
    with (Path(root) / "project.json").open(encoding="utf-8") as f:
        state = json.load(f)
    _totals(state)
    return state


def create_project(root, novel: str, mode="demo", title="Nolan 漫剧"):
    root = Path(root).expanduser().resolve()
    if mode not in {"demo", "live"}:
        raise ValueError("模式只能是 demo 或 live")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "project.json").exists():
        raise ValueError("项目已存在，请使用 run / studio 继续")
    if any(root.iterdir()):
        raise ValueError("新项目需要空目录，不能覆盖目录中已有的小说或素材")
    if not novel.strip():
        raise ValueError("小说内容不能为空")
    for folder in ("source", "media", "evidence", "exports", "plans"):
        (root / folder).mkdir(exist_ok=True)
    (root / "source/novel.txt").write_text(novel, encoding="utf-8")
    state = {
        "schema_version": 1, "id": uuid.uuid4().hex, "title": title,
        "mode": mode, "created_at": now(), "status": "initialized", "paused": False,
        "plan": {"status": "pending", "version": 0}, "assets": [], "episodes": [],
        "settings": {
            "brain": {"provider": "codex"}, "ark": {"video_watermark": False},
            "caps": {"image": 0, "video_second": 0, "llm": 0},
            "max_attempts": 3, "max_llm_calls": 80, "batch_size": 3,
            "poll_seconds": 10, "max_poll_seconds": 1800,
            "music": {"mode": "library"} if mode == "live" else {},
            "subtitles": {"mode": "local" if mode == "live" else "none", "call_cap": 0},
            "postproduction": {"enabled": mode == "live"},
            "creative_brief": "", "require_voice_reference": True,
            "gpt_video_review_enabled": False,
        },
        "budget": {"limit": 0, "currency": "CNY", "spent": 0, "reserved": 0,
                   "unknown": 0, "approved": mode == "demo", "enforcement": "disabled",
                   "basis": "只记录费用；用量估算不是供应商实际账单"},
        "jobs": {}, "llm_calls": 0, "events": [], "last_error": None,
    }
    event(state, "project_created", mode=mode)
    _write(root, state)
    return state


def validate_plan(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get("title"), str) or not plan["title"].strip():
        raise PlanValidationError("title_required", "plan.title")
    ratio = plan.get("ratio", "16:9")
    if not isinstance(ratio, str) or ratio not in {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9"}:
        raise PlanValidationError("ratio_invalid", "plan.ratio")
    assets, episodes = plan.get("assets"), plan.get("episodes")
    if not isinstance(assets, list) or not assets:
        raise PlanValidationError("assets_required", "plan.assets")
    if not isinstance(episodes, list) or not episodes:
        raise PlanValidationError("episodes_required", "plan.episodes")
    if len(assets) > 200:
        raise PlanValidationError("too_many_items", "plan.assets")
    if len(episodes) > 100:
        raise PlanValidationError("too_many_items", "plan.episodes")
    ids = set()
    def identifier(item, path):
        if not isinstance(item, dict):
            raise PlanValidationError("item_type", path)
        value = item.get("id", "")
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value) or value in ids:
            raise PlanValidationError("id_invalid", path + ".id")
        ids.add(value)
    for index, asset in enumerate(assets):
        path = f"plan.assets[{index}]"
        identifier(asset, path)
        if not isinstance(asset.get("kind"), str) or asset["kind"] not in {"character", "location", "prop", "voice"}:
            raise PlanValidationError("asset_kind_invalid", path + ".kind")
        for field in ("name", "prompt"):
            if not isinstance(asset.get(field), str) or not asset[field].strip():
                raise PlanValidationError("asset_text_required", path + "." + field)
        if "voice_only" in asset and (type(asset["voice_only"]) is not bool or (asset["voice_only"] and asset["kind"] != "character")):
            raise PlanValidationError("voice_only_invalid", path + ".voice_only")
        if (not isinstance(asset.get("audition_speaker", ""), str) or len(asset.get("audition_speaker", "")) > 200
                or (asset["kind"] != "voice" and asset.get("audition_speaker", "").strip())):
            raise PlanValidationError("voice_selection_invalid", path + ".audition_speaker")
    asset_ids = {a["id"] for a in assets}
    character_ids = {a["id"] for a in assets if a["kind"] == "character"}
    voices = {}
    for index, asset in enumerate(assets):
        path = f"plan.assets[{index}].character_id"
        if asset["kind"] == "voice":
            character_id = asset.get("character_id")
            if not isinstance(character_id, str) or character_id not in character_ids:
                raise PlanValidationError("voice_character_required", path)
            voices[asset["id"]] = character_id
        elif asset.get("character_id", "") != "":
            raise PlanValidationError("nonvoice_character_forbidden", path)
        fields = ("audition_text", "audition_reason", "audition_source")
        if any(key in asset for key in fields):
            if any(not isinstance(asset.get(key, ""), str) or len(asset.get(key, "")) > 4000 for key in fields):
                raise PlanValidationError("voice_selection_invalid", f"plan.assets[{index}].audition_text")
            if asset["kind"] != "voice" and any(asset.get(key, "").strip() for key in fields):
                raise PlanValidationError("voice_selection_invalid", f"plan.assets[{index}].audition_text")
            if asset["kind"] == "voice":
                if not asset.get("audition_reason", "").strip() or bool(asset.get("audition_text", "").strip()) != bool(asset.get("audition_source", "").strip()):
                    raise PlanValidationError("voice_selection_invalid", f"plan.assets[{index}].audition_text")
    for ep_index, ep in enumerate(episodes):
        ep_path = f"plan.episodes[{ep_index}]"
        identifier(ep, ep_path)
        if not isinstance(ep.get("title"), str) or not ep["title"].strip():
            raise PlanValidationError("episode_title_required", ep_path + ".title")
        if not isinstance(ep.get("shots"), list) or not ep["shots"]:
            raise PlanValidationError("shots_required", ep_path + ".shots")
        if len(ep["shots"]) > 100:
            raise PlanValidationError("too_many_items", ep_path + ".shots")
        for shot_index, shot in enumerate(ep["shots"]):
            path = f"{ep_path}.shots[{shot_index}]"
            identifier(shot, path)
            d = shot.get("duration")
            if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or not 4 <= d <= 15 or int(d) != d:
                raise PlanValidationError("duration_invalid", path + ".duration")
            if not isinstance(shot.get("prompt"), str) or not shot["prompt"].strip():
                raise PlanValidationError("shot_prompt_required", path + ".prompt")
            from .segments import validate_subshots
            try:
                validate_subshots(shot)
            except ValueError:
                raise PlanValidationError("subshots_invalid", path + ".subshots") from None
            refs = shot.get("asset_ids")
            if (not isinstance(refs, list) or not refs
                    or not all(isinstance(ref, str) for ref in refs)
                    or len(set(refs)) != len(refs) or not set(refs) <= asset_ids):
                raise PlanValidationError("asset_references_invalid", path + ".asset_ids")
            if not isinstance(shot.get("scene_id"), str) or not shot["scene_id"].strip():
                raise PlanValidationError("scene_required", path + ".scene_id")
            for ref_index, ref in enumerate(refs):
                if ref in voices and voices[ref] not in refs:
                    raise PlanValidationError("voice_character_not_referenced", f"{path}.asset_ids[{ref_index}]")
    return plan


def install_plan(root, plan):
    validate_plan(plan)
    from .asset_inputs import empty_input
    from .voice import AuditionSelectionError, validate_audition
    novel = (Path(root) / "source/novel.txt").read_text(encoding="utf-8")
    characters = {a["id"]: a["name"] for a in plan["assets"] if a["kind"] == "character"}
    for index, asset in enumerate(plan["assets"]):
        if asset["kind"] == "voice" and asset.get("audition_text", "").strip():
            try:
                validate_audition(asset, novel, asset.get("audition_speaker", "").strip() or characters[asset["character_id"]])
            except AuditionSelectionError:
                raise PlanValidationError("voice_selection_invalid", f"plan.assets[{index}].audition_source") from None
    with transaction(root) as s:
        if any(execution_unsettled(job) for job in s["jobs"].values()):
            raise GateError("尚有未完成或结果不明的调用，不能覆盖规划")
        if any(a.get("file") for a in s["assets"]) or any(e.get("file") for e in s["episodes"]):
            raise GateError("已生成素材的项目不能直接覆盖规划，请新建项目或进行局部返工")
        version = s["plan"]["version"] + 1
        s["plan"] = {"status": "candidate", "version": version, "style": plan.get("style", ""),
                     "ratio": plan.get("ratio", "16:9"), "feedback": ""}
        s["title"] = plan["title"]
        s["assets"] = [{"character_id": "", **copy.deepcopy(a), "status": "planned", "version": 0,
                         "history": [], "feedback": "", "creation_input": empty_input()}
                        for a in plan["assets"]]
        for asset in s["assets"]:
            if is_internal_voice_source(asset):
                _register_voice_source(asset)
        s["episodes"] = []
        for ep in plan["episodes"]:
            ep = copy.deepcopy(ep)
            ep.update(status="planned", version=0, history=[], feedback="", reviews=[])
            for shot in ep["shots"]:
                shot.update(status="planned", version=0, attempts=[], revision=0, feedback="")
            s["episodes"].append(ep)
        s["status"] = "awaiting_plan"
        event(s, "plan_created", version=version)
    p = Path(root) / "plans" / f"plan-v{version}.json"
    p.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_project(root)


def _find(s, kind, item_id):
    if kind == "plan" and item_id == "plan":
        return s["plan"]
    if kind in {"asset", "episode"}:
        items = s["assets"] if kind == "asset" else s["episodes"]
        for item in items:
            if item["id"] == item_id:
                return item
    if kind == "shot":
        for ep in s["episodes"]:
            for shot in ep["shots"]:
                if shot["id"] == item_id:
                    return shot
    raise ValueError("未找到审批对象")


def _check_version(item, version):
    if isinstance(version, bool) or not isinstance(version, int) or item.get("version", item.get("revision", 0)) != version:
        raise GateError("页面版本已过期，请刷新后重新审阅")


def voice_only_digest(asset):
    """Bind an internal offscreen source to its exact logical definition."""
    return _json_digest({key: asset.get(key) for key in ("id", "kind", "name", "prompt", "voice_only")})


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def is_internal_voice_source(asset):
    """Only explicit offscreen character roles are internal; portraits never are."""
    return isinstance(asset, dict) and asset.get("kind") == "character" and asset.get("voice_only") is True


def asset_ready(asset):
    """Human approval applies to real media; logical source roles have no human decision."""
    if is_internal_voice_source(asset):
        return (asset.get("status") == "internal" and asset.get("internal_role") is True
                and asset.get("sha256") == voice_only_digest(asset))
    return asset.get("status") == "approved"


def _register_voice_source(asset):
    """Register bookkeeping, never synthesize or manufacture a human approval."""
    if not is_internal_voice_source(asset):
        raise ValueError("仅显式画外声源可登记为内部角色")
    expected = voice_only_digest(asset)
    if asset.get("sha256") not in {None, expected}:
        raise GateError("画外声源定义哈希已改变，请先核实原定义；不会自动登记变化后的声源")
    approval_keys = ("approved_at", "approved_sha256", "approved_version", "human_decision", "human_decisions")
    prior = {key: copy.deepcopy(asset[key]) for key in approval_keys if key in asset}
    if asset.get("status") != "internal" or prior:
        asset.setdefault("internal_role_history", []).append({"status": asset.get("status"), "version": asset.get("version", 0),
            **prior, "reason": "改为内部声源绑定；真实声音资产仍须人工试听批准", "recorded_at": now()})
    for key in approval_keys:
        asset.pop(key, None)
    version = asset.get("version", 0)
    asset.update(status="internal", internal_role=True, version=version if type(version) is int and version > 0 else 1,
                 sha256=expected)


def normalize_voice_sources(root, *, already_locked=False):
    """Idempotently migrate old offscreen definitions, preserving genuine asset gates.

    Old amendment baselines are refreshed only when they matched the complete
    original asset snapshot before this semantic migration. A previously stale
    proposal remains stale; unrelated asset changes are never blessed here.
    """
    with (nullcontext() if already_locked else run_lock(root)), transaction(root) as state:
        sources = [asset for asset in state["assets"] if is_internal_voice_source(asset) and
                   (not asset_ready(asset) or any(key in asset for key in
                    ("approved_at", "approved_sha256", "approved_version", "human_decision", "human_decisions")))]
        if not sources:
            return state
        source_ids = {asset["id"] for asset in sources}
        for job in state["jobs"].values():
            if (job.get("accounting") == "unknown" or job.get("state") == "uncertain") and not is_nonblocking_advisory_unknown(job):
                raise GateError("存在结果不明的调用，请先核实原任务；内部声源记录暂不迁移")
            if job.get("item_id") in source_ids and execution_unsettled(job):
                raise GateError("声源自身尚有未完成的旧素材任务，请先恢复原任务；内部绑定暂不迁移")
        valid_baselines = []
        for amendment in state.get("asset_amendments", []):
            if amendment.get("status") not in {"candidate", "awaiting_assets"}:
                continue
            ids = set(amendment.get("baseline_asset_ids", []))
            old_assets = [asset for asset in state["assets"] if asset["id"] in ids]
            if _json_digest(old_assets) == amendment.get("baseline_assets_sha256"):
                valid_baselines.append((amendment, ids))
        for asset in sources:
            _register_voice_source(asset)
        invalidated = []
        for episode in state["episodes"]:
            for shot in episode["shots"]:
                if shot.get("status") not in {"awaiting_prompt_approval", "prompt_approved"} or not source_ids.intersection(shot.get("asset_ids", [])):
                    continue
                job = state["jobs"].get(shot.get("job_id"), {})
                submitted = (job.get("kind") == "video" and job.get("item_id") == shot["id"] and job.get("task_id")
                             and job.get("state") in {"submitted", "download_pending", "completed"})
                if submitted:
                    continue
                candidate = shot.pop("prompt_candidate", None)
                if candidate:
                    shot.setdefault("prompt_history", []).append({**candidate, "status": "invalidated",
                        "invalidation_reason": "内部声源记录升级，须重新准备并确认本次提示词"})
                shot.pop("prompt_approval", None)
                shot["status"] = "planned"
                invalidated.append(shot["id"])
        for amendment, ids in valid_baselines:
            amendment["baseline_assets_sha256"] = _json_digest([asset for asset in state["assets"] if asset["id"] in ids])
        if invalidated and state.get("status") == "awaiting_prompt_approval":
            state["status"] = "paused" if state["paused"] else "ready"
        event(state, "voice_sources_internalized", ids=[asset["id"] for asset in sources], invalidated_prompts=invalidated)
    return load_project(root)


def _check_artifact(root, item):
    if is_internal_voice_source(item):
        if item.get("sha256") != voice_only_digest(item):
            raise GateError("内部声源定义校验失败，请核实并修复声音绑定记录")
        return
    if not item.get("file"):
        raise GateError("素材文件缺失，不能审批")
    path = safe_path(root, item["file"])
    if not path.is_file() or digest(path) != item.get("sha256"):
        raise GateError("素材缺失或内容已改变，不能批准旧版本")
    if item.get("kind") == "voice":
        if not item.get("audio_file"):
            raise GateError("声音参考文件缺失，不能审批")
        audio = safe_path(root, item["audio_file"])
        if not audio.is_file() or digest(audio) != item.get("audio_sha256"):
            raise GateError("声音参考缺失或已改变，必须重新审核试听")


def _editable_asset_input(state, asset_id, asset_version, input_revision):
    from .asset_inputs import has_started, input_state
    if state["plan"].get("status") != "approved":
        raise GateError("请先批准当前策划，再确认资产生成方式或上传参考图。")
    asset = _find(state, "asset", asset_id)
    if is_internal_voice_source(asset):
        raise GateError("画外声源是内部绑定，无需选择图片或生成方式；请设置对应声音资产")
    _check_version(asset, asset_version)
    data = input_state(asset)
    if (isinstance(input_revision, bool) or not isinstance(input_revision, int)
            or input_revision != data["revision"]):
        raise GateError("资产输入已更新，请刷新网页后重新确认。")
    if has_started(state, asset) or asset.get("status") not in {"planned", "awaiting_asset_prompt_approval", "asset_prompt_approved"}:
        raise GateError("该资产已开始生成，初始描述和参考图不能再修改；请通过资产返工提出修改意见。")
    if any(execution_unsettled(job) for job in state["jobs"].values()):
        raise GateError("尚有在途或结果不明的调用，请等待落盘或完成对账后再修改资产输入。")
    return asset, data


def set_remaining_asset_inputs_gpt(root, plan_version):
    """Confirm only untouched first-time inputs; retain every custom draft."""
    from .asset_inputs import has_started, input_state
    with run_lock(root), transaction(root) as state:
        _check_version(state["plan"], plan_version)
        if state["plan"].get("status") != "approved":
            raise GateError("请先批准当前策划，再确认资产生成方式。")
        if any(execution_unsettled(job) for job in state["jobs"].values()):
            raise GateError("尚有在途或结果不明的调用，请等待落盘或完成对账后再修改资产输入。")
        selected = []
        for asset in state["assets"]:
            if is_internal_voice_source(asset):
                continue
            if has_started(state, asset) or asset.get("status") not in {"planned", "awaiting_asset_prompt_approval", "asset_prompt_approved"}:
                continue
            data = input_state(asset)
            if (data.get("mode") not in {None, ""} or data.get("description")
                    or data.get("references") or data.get("confirmed")):
                continue
            data.update(mode="gpt", description="", confirmed=True, revision=data["revision"] + 1)
            asset["creation_input"] = data
            selected.append({"id": asset["id"], "asset_version": asset["version"],
                             "input_revision": data["revision"]})
        if selected:
            event(state, "human_remaining_asset_inputs_confirmed", plan_version=plan_version,
                  mode="gpt", assets=selected)
            # Input choices do not establish that an existing production
            # blocker is resolved. The pipeline clears it at the next gate.
            if state.get("status") != "blocked":
                state["last_error"] = None
    return load_project(root)


def set_asset_input(root, asset_id, asset_version, input_revision, mode, description):
    """A version-bound human choice; never approves or generates an asset."""
    from .asset_inputs import MAX_DESCRIPTION_CHARS, reference_paths
    if not isinstance(mode, str) or mode not in {"gpt", "custom"}:
        raise ValueError("请选择由 GPT 决定，或由你提供描述。")
    if not isinstance(description, str) or len(description) > MAX_DESCRIPTION_CHARS:
        raise ValueError("资产描述须为文本，最多 4000 字。")
    description = description.strip()
    with run_lock(root), transaction(root) as state:
        asset, data = _editable_asset_input(state, asset_id, asset_version, input_revision)
        references = data.get("references", [])
        if mode == "gpt" and (description or references):
            raise ValueError("由 GPT 决定时请清空自定义描述和参考图，或选择由你提供描述。")
        if mode == "custom" and not (description or references):
            raise ValueError("请填写资产描述，或提供至少一张参考图片。")
        if asset["kind"] == "voice" and references:
            raise ValueError("声音资产只接受稳定声线描述；视觉参考使用已批准的绑定角色图片。")
        data.update(mode=mode, description=description, confirmed=True,
                    revision=data["revision"] + 1)
        candidate = {**asset, "creation_input": data}
        reference_paths(root, candidate)
        asset["creation_input"] = data
        asset["status"] = "planned"
        event(state, "human_asset_input_confirmed", id=asset_id, asset_version=asset_version,
              input_revision=data["revision"], mode=mode,
              reference_hashes=[ref["sha256"] for ref in references])
        # Retain the stopped phase's reason until production actually advances.
        if state.get("status") != "blocked":
            state["last_error"] = None
    return load_project(root)


def add_asset_reference(root, asset_id, asset_version, input_revision, data_url):
    """Store a validated local image; uploads invalidate the prior confirmation."""
    from .asset_inputs import MAX_REFERENCES, decode_reference, validate_image_file
    content, mime, suffix = decode_reference(data_url)
    with run_lock(root):
        state = load_project(root)
        asset, data = _editable_asset_input(state, asset_id, asset_version, input_revision)
        if asset["kind"] == "voice":
            raise ValueError("声音资产不接受参考图片；请描述稳定声线，画面使用已批准的绑定角色图。")
        if len(data["references"]) >= MAX_REFERENCES:
            raise ValueError("每项资产最多上传 9 张参考图片。")
        sha256 = hashlib.sha256(content).hexdigest()
        if any(ref.get("sha256") == sha256 for ref in data["references"]):
            raise ValueError("这张参考图片已经上传，无需重复添加。")
        directory = safe_path(root, "inputs/assets/" + asset_id)
        directory.mkdir(parents=True, exist_ok=True)
        reference_id = uuid.uuid4().hex
        relative = str((directory / (reference_id + suffix)).relative_to(Path(root).resolve()))
        path = safe_path(root, relative)
        try:
            with path.open("xb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            validate_image_file(path, mime)
            with transaction(root) as state:
                asset, data = _editable_asset_input(state, asset_id, asset_version, input_revision)
                data["references"].append({"id": reference_id, "file": relative, "sha256": sha256,
                                           "mime": mime, "bytes": len(content)})
                data.update(confirmed=False, revision=data["revision"] + 1)
                asset["creation_input"] = data
                asset["status"] = "planned"
                event(state, "asset_reference_added", id=asset_id, reference_id=reference_id,
                      sha256=sha256, input_revision=data["revision"])
        except BaseException:
            # Preserve any image whose state was published before an fsync error.
            saved = _find(load_project(root), "asset", asset_id).get("creation_input", {})
            if not any(ref.get("file") == relative for ref in saved.get("references", [])):
                path.unlink(missing_ok=True)
            raise
    return load_project(root)


def remove_asset_reference(root, asset_id, asset_version, input_revision, reference_id):
    """Remove a draft selection while retaining its original immutable bytes."""
    with run_lock(root), transaction(root) as state:
        asset, data = _editable_asset_input(state, asset_id, asset_version, input_revision)
        if not isinstance(reference_id, str) or not any(ref.get("id") == reference_id for ref in data["references"]):
            raise ValueError("参考图片不存在或已经移除，请刷新后重试。")
        data["references"] = [ref for ref in data["references"] if ref["id"] != reference_id]
        data.update(confirmed=False, revision=data["revision"] + 1)
        asset["creation_input"] = data
        asset["status"] = "planned"
        event(state, "asset_reference_removed", id=asset_id, reference_id=reference_id,
              input_revision=data["revision"])
    return load_project(root)


def _shot_file_matches_version(shot):
    filename = Path(shot.get("file") or "").name
    match = re.fullmatch(re.escape(str(shot.get("id", ""))) + r"-v([1-9]\d*)\.(?:mp4|m4v|mov|webm)", filename)
    return match is None or int(match[1]) == shot.get("version")


def _check_shot_artifact(root, state, shot):
    """Old interrupted attempts can retain an earlier file under a newer version."""
    _check_artifact(root, shot)
    version = shot.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1 or not _shot_file_matches_version(shot):
        raise GateError("视频文件不属于当前版本，请先恢复原生成任务并刷新后重新审阅")
    job = state["jobs"].get(shot.get("job_id", ""))
    if job and (job.get("kind") != "video" or job.get("item_id") != shot["id"] or
                job.get("state") != "completed" or
                (job.get("output") and safe_path(root, job["output"]) != safe_path(root, shot["file"]))):
        raise GateError("视频文件不属于当前版本，请先恢复原生成任务并刷新后重新审阅")


def human_approved_shot(shot):
    """Recognize only a human decision bound to this exact stored video version.

    This is a state check; callers using the media must also verify its bytes.
    The legacy override remains valid only alongside the old approval metadata.
    A stale structured decision must never fall back to that legacy override.
    """
    version, sha256 = shot.get("version"), shot.get("sha256")
    if (shot.get("status") != "accepted" or not shot.get("file") or
            isinstance(version, bool) or not isinstance(version, int) or version < 1 or
            not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256) or
            shot.get("approved_sha256") != sha256 or not shot.get("approved_at") or
            not _shot_file_matches_version(shot)):
        return False
    approved_version = shot.get("approved_version", version)
    if isinstance(approved_version, bool) or approved_version != version:
        return False
    decision = shot.get("human_decision")
    if decision is not None:
        return (isinstance(decision, dict) and decision.get("decision") == "approved" and
                not isinstance(decision.get("version"), bool) and
                decision.get("version") == version and decision.get("sha256") == sha256 and
                bool(decision.get("at")))
    review = shot.get("review")
    return isinstance(review, dict) and review.get("human_override") is True


def _shot_decision(shot, decision, feedback="", *, continuity_checked=None):
    review = shot.get("review")
    report_bound = (isinstance(review, dict) and bool(shot.get("sha256")) and
                    review.get("sha256") == shot["sha256"] and
                    ("version" not in review or (type(review["version"]) is int and
                                                 review["version"] == shot["version"])))
    report_unavailable = isinstance(review, dict) and review.get("report_unavailable") is True
    report_available = report_bound and not report_unavailable
    report = ({key: copy.deepcopy(review[key]) for key in ("verdict", "issues") if key in review}
              if report_available else {})
    if report_bound and report_unavailable:
        report = {key: copy.deepcopy(review[key]) for key in
                  ("report_unavailable", "review_interrupted", "scope", "version", "sha256") if key in review}
    note = ("自动审查报告未完成或不可用；这里仅保留缺失状态和版本元数据，人工决定针对实际观看的这个版本视频。"
            if report_unavailable else
            "报告保留 GPT 的原始检查意见，未经用户逐条核实；人工决定针对这个版本的整体成片。"
            if report_bound else
            "现有检查报告未绑定当前视频版本，未作为本次人工决定的报告依据；人工决定针对实际观看的这个版本。")
    record = {"decision": decision, "version": shot["version"], "sha256": shot.get("sha256"),
              "at": now(), "feedback": feedback,
              "report": report, "report_bound": report_bound, "report_available": report_available,
              "report_note": note}
    if not shot.get("file"):
        record.update(job_id=shot.get("job_id"), revision=shot.get("revision", 0),
                      outcome="generation_failed")
    if continuity_checked:
        record["continuity_checked"] = copy.deepcopy(continuity_checked)
    shot["human_decision"] = record
    shot.setdefault("human_decisions", []).append(copy.deepcopy(record))


def _check_shot_continuity(root, state, shot):
    pending = shot.get("continuity_review") or {}
    if not pending.get("required"):
        return []
    checked = []
    for source in pending.get("source_shots", []):
        current = _find(state, "shot", source["id"])
        if not human_approved_shot(current) or current["version"] <= source["version"]:
            raise GateError("请先完成人工要求返工的上游片段并批准新版本，再确认这段视频的衔接")
        _check_shot_artifact(root, state, current)
        checked.append({"id": current["id"], "version": current["version"], "sha256": current["sha256"]})
    if not checked:
        raise GateError("衔接复核来源缺失，请先检查上游片段记录")
    return checked


def approve(root, kind, item_id, version):
    with transaction(root) as s:
        item = _find(s, kind, item_id)
        _check_version(item, version)
        if kind == "plan":
            if item["status"] != "candidate":
                raise GateError("规划当前不可审批")
        elif kind == "asset":
            if is_internal_voice_source(item):
                raise GateError("画外声源是内部绑定，无需单独审批；请试听并批准对应声音资产")
            if item["status"] != "candidate":
                raise GateError("只有已生成候选的资产可以审批")
            _check_artifact(root, item)
        elif kind == "episode":
            if item["status"] != "awaiting_review":
                raise GateError("本集尚未完成或已被修改")
            idx = s["episodes"].index(item)
            if any(e["status"] != "approved" for e in s["episodes"][:idx]):
                raise GateError("请按顺序审阅每一集")
            _check_artifact(root, item)
            if any(not human_approved_shot(sh) for sh in item["shots"]):
                raise GateError("本集仍有未经人工批准当前版本的视频片段")
            for sh in item["shots"]:
                _check_shot_artifact(root, s, sh)
            used_assets = {ref for sh in item["shots"] for ref in sh["asset_ids"]}
            for asset in s["assets"]:
                if asset["id"] in used_assets:
                    if not asset_ready(asset):
                        raise GateError("本集依赖的资产不再处于已批准状态")
                    _check_artifact(root, asset)
            rendered = item.get("render_manifest", {}).get("clips", [])
            if rendered != [{"id": sh["id"], "sha256": sh["sha256"]} for sh in item["shots"]]:
                raise GateError("本集镜头已更新或缺少合成记录，请重新合成后审批")
            if s["settings"].get("postproduction", {}).get("enabled"):
                from .episode_post import check_final
                check_final(root, s, item)
        elif kind == "shot":
            if item.get("pending_asset_amendment"):
                raise GateError("本片段正在等待资产增补确认，请先完成新资产审批再返修")
            if item["status"] != "review_required":
                raise GateError("此镜头不等待人工复核")
            _check_shot_artifact(root, s, item)
            continuity_checked = _check_shot_continuity(root, s, item)
        item["status"] = "accepted" if kind == "shot" else "approved"
        item["approved_at"] = now()
        item["approved_sha256"] = item.get("sha256")
        if kind == "shot":
            item["approved_version"] = version
            _shot_decision(item, "approved", continuity_checked=continuity_checked)
            item.pop("continuity_review", None)
            item.setdefault("review", {})["human_override"] = True
        event(s, "human_approved", kind=kind, id=item_id, version=version,
              sha256=item.get("sha256"))
        s["last_error"] = None
        s["status"] = "ready"
    return load_project(root)


def _invalidate_shot(shot, reason, *, increment_revision=True, state=None):
    from . import storyboard_revision
    if shot.get("file"):
        shot.setdefault("history", []).append({k: copy.deepcopy(shot[k]) for k in
            ("file", "sha256", "evidence", "review", "version", "human_decision", "approved_at",
             "approved_sha256", "approved_version", "continuity_review", "job_id", "prompt", "dialogue",
             "duration", "scene_id", "subshots", "continuity_notes", "asset_ids", "refined_prompt",
             "reference_manifest", "prompt_candidate", "prompt_approval", "video_model", "storyboard_revision",
             "storyboard_revision_binding", "source_shot_ids", "source_shot_instructions") if k in shot})
    if shot.get("prompt_candidate"):
        shot.setdefault("prompt_history", []).append(copy.deepcopy(shot["prompt_candidate"]))
    if shot.get("revision_analysis"):
        shot.setdefault("revision_analysis_history", []).append({**copy.deepcopy(shot["revision_analysis"]),
            "superseded_at": now(), "resolution": "human_revision"})
    storyboard_revision.discard(shot, reason=reason, stamp=now(), state=state)
    shot.update(status="planned", revision=shot.get("revision", 0) + int(increment_revision), feedback=reason)
    for key in ("file", "sha256", "evidence", "review", "job_id", "refined_prompt", "approved_at", "approved_sha256", "approved_version", "generation_error",
                "prompt_candidate", "prompt_approval", "revision_analysis"):
        shot.pop(key, None)


def _invalidate_render(ep, reason):
    """Retire a stale edit without requesting any other generated clip again."""
    if ep.get("file"):
        ep.setdefault("history", []).append({key: copy.deepcopy(ep[key]) for key in
            ("file", "sha256", "version", "render_manifest", "approved_at", "approved_sha256") if key in ep})
    ep.update(status="revision_requested", feedback=reason, subtitle_status="needs_retiming")
    for key in ("file", "sha256", "approved_at", "approved_sha256", "verified_subtitles", "subtitle_file",
                "render_manifest", "progress_stage", "progress_started_at"):
        ep.pop(key, None)


def _request_shot_revision(state, shot, feedback):
    """Only the named clip is regenerated; downstream video awaits human review."""
    source = {"id": shot["id"], "version": shot["version"], "sha256": shot.get("sha256"),
              "at": now(), "reason": feedback}
    owner = next(ep for ep in state["episodes"] if any(sh is shot for sh in ep["shots"]))
    episode_index, shot_index = state["episodes"].index(owner), owner["shots"].index(shot)
    _shot_decision(shot, "revision_requested", feedback)
    _invalidate_shot(shot, feedback, state=state)
    _invalidate_render(owner, feedback)
    for index, ep in enumerate(state["episodes"][episode_index:], start=episode_index):
        affected = []
        for current_index, current in enumerate(ep["shots"]):
            downstream = index > episode_index or (current_index > shot_index and current["scene_id"] == shot["scene_id"])
            if downstream and current.get("file"):
                current["status"] = "review_required"
                pending = current.setdefault("continuity_review", {"required": True, "source_shots": []})
                pending["required"] = True
                sources = pending.setdefault("source_shots", [])
                if not any(existing["id"] == source["id"] and existing["version"] == source["version"] for existing in sources):
                    sources.append(copy.deepcopy(source))
                for key in ("approved_at", "approved_sha256", "approved_version"):
                    current.pop(key, None)
                affected.append(current["id"])
        if index > episode_index and (affected or ep.get("file")):
            _invalidate_render(ep, "上游片段已申请返工，请人工复核已有视频衔接：" + feedback)


def _check_shot_rejection(root, state, shot):
    if shot.get("file"):
        _check_shot_artifact(root, state, shot)
        return
    failed = state["jobs"].get(shot.get("job_id", ""), {})
    if (not shot.get("generation_error") or failed.get("state") != "failed" or
            failed.get("accounting") not in {"spent", "released"} or
            failed.get("item_id") != shot["id"] or failed.get("kind") != "video" or
            type(shot.get("version")) is not int or shot["version"] < 1):
        raise GateError("生成失败记录尚未核实，不能再次提交；请先核实原调用结果")


def can_edit_shot_revision(state, shot):
    """A queued human rework may receive new feedback without a media POST."""
    if (shot.get("status") != "planned" or type(shot.get("version")) is not int or shot["version"] < 1
            or shot.get("file") or shot.get("job_id") or shot.get("pending_asset_amendment")
            or any(execution_unsettled(job) for job in state.get("jobs", {}).values())
            or any(amendment.get("status") in {"candidate", "awaiting_assets"}
                   and (amendment.get("target_shot_id") == shot.get("id")
                        or any(update.get("id") == shot.get("id") for update in amendment.get("shot_updates", [])))
                   for amendment in state.get("asset_amendments", []))):
        return False
    if any(isinstance(old, dict) and old.get("file") and old.get("sha256")
           and type(old.get("version")) is int and 1 <= old["version"] <= shot["version"]
           for old in shot.get("history", [])):
        return True
    return any(state.get("jobs", {}).get(attempt.get("job_id"), {}).get("kind") == "video"
               and state["jobs"][attempt["job_id"]].get("item_id") == shot.get("id")
               and state["jobs"][attempt["job_id"]].get("state") == "failed"
               and state["jobs"][attempt["job_id"]].get("accounting") in {"spent", "released"}
               for attempt in shot.get("attempts", []) if isinstance(attempt, dict))


def _invalidate_episode(ep, reason, shot_ids=None, *, state=None):
    if ep.get("file"):
        ep["history"].append({k: ep[k] for k in ("file", "sha256", "version") if k in ep})
    selected = set(shot_ids or [sh["id"] for sh in ep["shots"]])
    changed_scenes = set()
    for shot in ep["shots"]:
        if shot["id"] in selected or shot["scene_id"] in changed_scenes:
            changed_scenes.add(shot["scene_id"])
            _invalidate_shot(shot, reason, state=state)
    ep.update(status="revision_requested", feedback=reason)
    for key in ("file", "sha256", "approved_at", "approved_sha256", "verified_subtitles", "render_manifest",
                "progress_stage", "progress_started_at"):
        ep.pop(key, None)
    ep["subtitle_status"] = "needs_retiming"


def _reject_asset(state, item, version, feedback):
    """Apply an already validated human asset revision in the current transaction."""
    s, item_id = state, item["id"]
    if is_internal_voice_source(item):
        raise GateError("画外声源是内部绑定；请在对应声音资产中提出音色修改要求")
    if item["status"] not in {"candidate", "approved", "rejected", "generation_failed"}:
        raise GateError("资产尚未生成")
    if item["status"] == "generation_failed":
        failed = s["jobs"].get(item.get("job_id", ""), {})
        expected_kind = "voice" if item.get("kind") == "voice" else "image"
        if (not item.get("generation_error") or failed.get("state") != "failed" or
                failed.get("accounting") not in {"spent", "released"} or
                failed.get("kind") != expected_kind or failed.get("item_id") != item_id or
                item.get("voice_only") is True):
            raise GateError("资产失败记录尚未核实，不能再次提交；请先核实原调用结果")
        from .failures import sanitize_failure
        detail = sanitize_failure(failed.get("failure") or {
            "stage": "unknown", "message": "原资产生成失败，旧版本未保存具体原因。", "uncertain": False})
        history = item.setdefault("failure_history", [])
        if not any(record.get("job_id") == item["job_id"] and record.get("version") == version
                   for record in history):
            history.append({"job_id": item["job_id"], "version": version, "failure": detail,
                            "recorded_at": now()})
    item.update(status="rejected", feedback=feedback)
    # The completed task belongs to the rejected candidate, not its replacement.
    for key in ("job_id", "approved_at", "approved_sha256", "generation_error"):
        item.pop(key, None)
    affected = False
    for ep in s["episodes"]:
        selected = [sh["id"] for sh in ep["shots"] if item_id in sh["asset_ids"]]
        if selected or affected:
            _invalidate_episode(ep, "资产变更：" + feedback, None if affected else selected, state=s)
            affected = True


def reject(root, kind, item_id, version, feedback, shot_ids=None):
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("返工意见不能为空")
    with run_lock(root), transaction(root) as s:
        if any(execution_unsettled(job) for job in s["jobs"].values()):
            raise GateError("尚有调用在执行，请先暂停并等待在途任务落盘后再返工")
        item = _find(s, kind, item_id)
        _check_version(item, version)
        if kind == "plan":
            if any(a.get("file") for a in s["assets"]):
                raise GateError("已有生成素材，请使用资产或分集返工")
            item.update(status="rejected", feedback=feedback)
        elif kind == "asset":
            _reject_asset(s, item, version, feedback)
        elif kind == "episode":
            if item["status"] not in {"awaiting_review", "approved", "revision_requested"}:
                raise GateError("本集尚未进入审片阶段")
            valid_ids = {sh["id"] for sh in item["shots"]}
            if shot_ids is not None and (not isinstance(shot_ids, list) or
                                        any(not isinstance(value, str) for value in shot_ids) or
                                        not set(shot_ids) <= valid_ids):
                raise ValueError("返工镜头 ID 不属于本集")
            selected = set(shot_ids or valid_ids)
            targets = [sh for sh in item["shots"] if sh["id"] in selected]
            for shot in targets:
                # A clip already queued by a human decision needs no new attempt
                # authorization. New instructions still apply to its next render.
                if shot.get("status") == "planned" and not shot.get("file") and not shot.get("job_id"):
                    continue
                _check_shot_rejection(root, s, shot)
            for shot in targets:
                if shot.get("status") == "planned" and not shot.get("file") and not shot.get("job_id"):
                    # Supplemental episode feedback edits the already queued
                    # attempt; it does not create another generation authority.
                    _invalidate_shot(shot, feedback, increment_revision=False, state=s)
                else:
                    _request_shot_revision(s, shot, feedback)
            _invalidate_render(item, feedback)
            shot_ids = [shot["id"] for shot in targets]
        elif kind == "shot":
            if item.get("pending_asset_amendment"):
                raise GateError("本片段已有资产增补方案，请先完成或拒绝该方案")
            queued = can_edit_shot_revision(s, item)
            if item["status"] not in {"review_required", "accepted", "generation_failed"} and not queued:
                raise GateError("镜头不在人工复核阶段")
            if not queued:
                _check_shot_rejection(root, s, item)
            _request_shot_revision(s, item, feedback)
        event(s, "human_rejected", kind=kind, id=item_id, version=version,
              feedback=feedback, shot_ids=shot_ids)
        s["status"] = "ready"
    return load_project(root)


def resolve_completed_asset_feedback(root, shot_id):
    """Consume only a proven, completed machine asset-rejection cascade.

    The production runner already owns run_lock. This bookkeeping operation
    never approves media, changes an attempted video, or interprets human prose.
    Unknown, mixed or newly submitted shot instructions remain untouched.
    """
    with transaction(root) as state:
        shot = _find(state, "shot", shot_id)
        if (shot.get("status") != "planned" or type(shot.get("version", 0)) is not int
                or shot.get("version", 0) != 0 or shot.get("file") or shot.get("job_id")
                or shot.get("attempts") or shot.get("prompt_approval")
                or any(job.get("kind") == "video" and job.get("item_id") == shot_id
                       for job in state.get("jobs", {}).values())):
            return False
        feedback = shot.get("feedback")
        if not isinstance(feedback, str) or not feedback:
            return False
        events = state.get("events", [])
        assets = {asset["id"]: asset for asset in state["assets"]}
        latest = {}
        for index, record in enumerate(events):
            if (record.get("action") == "human_rejected" and record.get("kind") == "asset"
                    and isinstance(record.get("feedback"), str) and record["feedback"]):
                latest[(record.get("id"), record["feedback"])] = (index, record)
        proofs = {}
        for (asset_id, original), (index, rejected) in latest.items():
            machine_text = "资产变更：" + original
            asset = assets.get(asset_id, {})
            version, rejected_version = asset.get("version"), rejected.get("version")
            sha = asset.get("sha256")
            approved = None
            if (asset.get("status") == "approved" and type(version) is int
                    and type(rejected_version) is int and version > rejected_version
                    and isinstance(sha, str) and sha and sha == asset.get("approved_sha256")):
                approved = next(((position, record) for position, record in reversed(list(enumerate(events)))
                    if position > index and record.get("action") == "human_approved"
                    and record.get("kind") == "asset" and record.get("id") == asset_id
                    and record.get("version") == version and record.get("sha256") == sha), None)
                if approved:
                    try:
                        _check_artifact(root, asset)
                    except (GateError, ValueError, OSError):
                        approved = None
            # Identical prose can belong to several assets. Every matching
            # latest rejection must be resolved; never guess which one it was.
            proofs.setdefault(machine_text, []).append(None if not approved else {
                "asset_id": asset_id, "version": version, "sha256": sha,
                "rejected_event_index": index, "rejected_event": copy.deepcopy(rejected),
                "approved_event_index": approved[0], "approved_event": copy.deepcopy(approved[1])})
        resolved = {text: proof for text, proof in proofs.items() if proof and all(proof)}
        instructions = shot.get("source_shot_instructions", [])
        if not isinstance(instructions, list) or any(not isinstance(value, dict) for value in instructions):
            return False
        source_ids = set(value for value in shot.get("source_shot_ids", []) if isinstance(value, str))
        source_ids.update(value["id"] for value in instructions if isinstance(value.get("id"), str))
        variants = {text: (None, text) for text in resolved}
        variants.update({f"{source_id}：{text}": (source_id, text)
                         for source_id in source_ids for text in resolved})
        # Match complete original records, including any embedded newlines.
        paths, pending = {0: []}, [0]
        while pending and len(feedback) not in paths:
            start = pending.pop()
            for text, token in variants.items():
                end = start + len(text)
                if feedback.startswith(text, start) and (end == len(feedback) or feedback[end:end + 1] == "\n"):
                    next_start = end if end == len(feedback) else end + 1
                    if next_start not in paths:
                        paths[next_start] = paths[start] + [token]
                        pending.append(next_start)
        matched = paths.get(len(feedback))
        if not matched:
            return False
        proof = [record for _, text in matched for record in resolved[text]]
        first_rejection = min(record["rejected_event_index"] for record in proof)
        protected_ids = source_ids | {shot_id}
        owner = next(ep for ep in state["episodes"] if shot in ep["shots"])
        for record in events[first_rejection + 1:]:
            direct = record.get("id") in protected_ids
            if ((direct and record.get("action") in {"human_planning_feedback", "human_rejected_prompt"})
                    or (record.get("action") == "human_rejected" and (
                        (record.get("kind") == "shot" and direct)
                        or (record.get("kind") == "episode" and record.get("id") == owner["id"]
                            and (not record.get("shot_ids") or protected_ids.intersection(record["shot_ids"])))))):
                return False
        subshots = copy.deepcopy(shot.get("subshots", []))
        for instruction in instructions:
            original = instruction.get("feedback", "")
            if not original:
                continue
            source_id = instruction.get("id")
            if original not in resolved or (source_id, original) not in matched:
                return False
            pattern = re.compile(r"\n原生成片段 " + re.escape(source_id)
                + r" 的返工意见（仅适用于本生成片段 (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?) 秒的来源段；"
                  r"覆盖该来源段的内部镜头，保留各句对白的归属，不作为每镜新增对白）：\n"
                + re.escape(original) + r"\Z")
            found = []
            for index, subshot in enumerate(subshots):
                match = pattern.search(subshot.get("prompt", ""))
                if match and 0 <= float(match[1]) < float(match[2]) <= shot["duration"]:
                    found.append((index, match.start()))
            if len(found) != 1:
                return False
            index, boundary = found[0]
            if not subshots[index]["prompt"][:boundary].strip():
                return False
            subshots[index]["prompt"] = subshots[index]["prompt"][:boundary]
        stamp = now()
        archive = {"resolved_at": stamp, "feedback": feedback, "proof": proof,
            "subshots": copy.deepcopy(shot.get("subshots")),
            "source_shot_instructions": copy.deepcopy(shot.get("source_shot_instructions")),
            "revision_analysis": copy.deepcopy(shot.get("revision_analysis"))}
        shot.setdefault("resolved_asset_feedback_history", []).append(archive)
        if "subshots" in shot:
            shot["subshots"] = subshots
        for instruction in instructions:
            instruction["feedback"] = ""
        analysis = shot.pop("revision_analysis", None)
        if analysis:
            shot.setdefault("revision_analysis_history", []).append({**copy.deepcopy(analysis),
                "superseded_at": stamp, "resolution": "completed_asset_feedback"})
        candidate = shot.pop("prompt_candidate", None)
        if candidate:
            shot.setdefault("prompt_history", []).append({**copy.deepcopy(candidate), "status": "invalidated",
                "invalidation_reason": "已批准的新资产完成原资产返工；重新准备完整提示词"})
        shot.pop("refined_prompt", None)
        shot["feedback"] = ""
        event(state, "completed_asset_feedback_resolved", id=shot_id,
              asset_ids=list(dict.fromkeys(record["asset_id"] for record in proof)))
    return True


def request_caption_fix(root, item_id, version, feedback):
    """A subtitle-only edit preserves the generated picture and sound."""
    if load_project(root)["settings"].get("postproduction", {}).get("enabled"):
        return request_postproduction(root, item_id, version, "subtitles", feedback)
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("请说明要修改的字幕文字")
    with run_lock(root), transaction(root) as s:
        ep = _find(s, "episode", item_id)
        _check_version(ep, version)
        if ep["status"] not in {"awaiting_review", "approved"}:
            raise GateError("只有已合成的集数可以修改字幕")
        _check_artifact(root, ep)
        segments = ep.get("render_manifest", {}).get("subtitles")
        if not segments:
            raise GateError("本集没有已对齐的字幕，请先开启自动字幕后重新合成")
        ep["history"].append({k: ep[k] for k in ("file", "sha256", "version")})
        ep.update(caption_feedback=feedback, caption_source=copy.deepcopy(segments),
                  caption_source_clips=[sh["sha256"] for sh in ep["shots"]], status="revision_requested")
        for key in ("approved_at", "approved_sha256", "progress_stage", "progress_started_at"):
            ep.pop(key, None)
        s["status"] = "ready"
        event(s, "caption_fix_requested", id=item_id, version=version, feedback=feedback)
    return load_project(root)


def request_postproduction(root, episode_id, version, target="continue", feedback=""):
    from .episode_post import request
    return request(root, episode_id, version, target, feedback)


def import_music(root, episode_id, version, data_url, title):
    from .episode_post import import_music as action
    return action(root, episode_id, version, data_url, title)


def approve_music(root, track_id, version, sha256):
    from .episode_post import decide_music
    return decide_music(root, track_id, version, sha256)


def reject_music(root, track_id, version, sha256, feedback):
    from .episode_post import decide_music
    return decide_music(root, track_id, version, sha256, feedback)


def request_music_generation(root, episode_id, version):
    from .episode_post import request_generation
    return request_generation(root, episode_id, version)


def configure_music_folder(root, folder):
    """Register a local, read-only source folder without running production."""
    from .music_folder import configure_folder
    return configure_folder(root, folder)


def scan_music_folder(root, folder=None):
    """Import previously unseen audio bytes as unapproved music candidates."""
    from .music_folder import scan_folder
    return scan_folder(root, folder)


def _amendment_idle(state):
    if any(execution_unsettled(job) for job in state["jobs"].values()):
        raise GateError("尚有在途或结果不明的调用，请先恢复原任务或核实结果，再提出资产增补")


def _amendment_target(shot):
    return {key: copy.deepcopy(shot.get(key)) for key in
            ("id", "version", "revision", "sha256", "file", "status", "prompt", "dialogue", "asset_ids", "feedback",
             "duration", "subshots", "scene_id", "continuity_notes")}


def _amendment_find(state, amendment_id):
    for amendment in state.get("asset_amendments", []):
        if amendment.get("id") == amendment_id:
            return amendment
    raise GateError("未找到资产增补方案，请刷新当前项目")


def _check_amendment_baseline(state, amendment):
    shot = _find(state, "shot", amendment["target_shot_id"])
    if _amendment_target(shot) != amendment["target"]:
        raise GateError("片段版本或返修内容已改变，请重新提出资产增补方案")
    baseline_ids = set(amendment["baseline_asset_ids"])
    baseline = [asset for asset in state["assets"] if asset["id"] in baseline_ids]
    if _json_digest(baseline) != amendment["baseline_assets_sha256"]:
        raise GateError("原有资产或审批已改变，请重新提出资产增补方案")
    for saved in amendment["shot_baselines"]:
        if _amendment_target(_find(state, "shot", saved["id"])) != saved:
            raise GateError("受影响的后续片段已改变，请重新提出资产增补方案")
    return shot


def _validate_amendment(state, root, shot_id, version, feedback, additions, shot_updates, audio_policy):
    if not isinstance(feedback, str) or not feedback.strip() or len(feedback) > 100000:
        raise ValueError("请填写本次资产增补和片段返修要求")
    if state["plan"].get("status") != "approved":
        raise GateError("请先批准当前策划，再增补制作资产")
    target = _find(state, "shot", shot_id)
    _check_version(target, version)
    if target.get("status") not in {"planned", "review_required", "accepted", "generation_failed"}:
        raise GateError("此片段暂不支持资产增补，请先恢复当前任务")
    if target.get("file"):
        _check_shot_artifact(root, state, target)
    elif target.get("job_id"):
        _check_shot_rejection(root, state, target)
    if (not isinstance(additions, list) or not 0 <= len(additions) <= 30 or
            not isinstance(shot_updates, list) or not 1 <= len(shot_updates) <= 100):
        raise ValueError("资产变更须包含 0–30 项新资产和 1–100 项明确片段更新；没有新资产时须复用已有资产或改变声音规则")
    allowed = {"id", "kind", "name", "prompt", "character_id", "audition_text", "audition_reason", "audition_source", "voice_only", "audition_speaker"}
    if any(not isinstance(asset, dict) or set(asset) - allowed for asset in additions):
        raise ValueError("新资产只允许填写定义和试音选句，不能设置审批、媒体、预算或历史记录")
    existing_ids = {asset["id"] for asset in state["assets"]}
    existing_ids.update(ep["id"] for ep in state["episodes"])
    all_shots = [shot for ep in state["episodes"] for shot in ep["shots"]]
    existing_ids.update(shot["id"] for shot in all_shots)
    if any(asset.get("id") in existing_ids for asset in additions):
        raise GateError("新增资产 ID 已存在；增补方案不能覆盖原有资产或片段")
    # Validate only new definitions; old approved auditions may predate current
    # selection rules and must never be silently revalidated or reconstructed.
    old_definitions = [{"id": asset["id"], "kind": asset["kind"], "name": "已有资产", "prompt": "保留已批准定义",
                        **({"character_id": asset.get("character_id", "")} if asset["kind"] == "voice" else {})}
                       for asset in state["assets"]]
    merged_assets = old_definitions + copy.deepcopy(additions)
    new_ids = {asset.get("id") for asset in additions}
    if len(merged_assets) > 200:
        raise ValueError("项目资产总数不能超过 200 项")
    target_index = next(index for index, shot in enumerate(all_shots) if shot["id"] == shot_id)
    updates, seen, added_reference = [], set(), False
    for update in shot_updates:
        if (not isinstance(update, dict) or set(update) - {"id", "add_asset_ids", "prompt", "dialogue", "continuity_notes"}
                or not isinstance(update.get("id"), str) or update["id"] in seen):
            raise ValueError("片段更新只能包含唯一 ID、追加资产引用及提示词/对白/连续性说明")
        current = _find(state, "shot", update["id"])
        index = all_shots.index(current)
        if index < target_index or (current is not target and
                (current.get("file") or current.get("job_id") or current.get("attempts") or current.get("status") != "planned")):
            raise GateError("增补仅可修改目标片段和后续尚未开始的片段，不能改写其他已生成视频")
        refs = update.get("add_asset_ids")
        if (not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs) or
                len(refs) != len(set(refs)) or not set(refs) <= (existing_ids | new_ids)):
            raise ValueError("追加资产引用必须为不重复的有效资产 ID 列表")
        updated = copy.deepcopy(current)
        updated["asset_ids"] = list(dict.fromkeys(current["asset_ids"] + refs))
        added_reference = added_reference or bool(set(refs) - set(current["asset_ids"]))
        for key in ("prompt", "dialogue", "continuity_notes"):
            if key in update:
                if not isinstance(update[key], str) or len(update[key]) > 100000 or (key == "prompt" and not update[key].strip()):
                    raise ValueError("片段提示词须为非空文本，其他更新字段须为文本，最多 100000 字")
                updated[key] = update[key]
        updates.append(updated)
        seen.add(current["id"])
    if shot_id not in seen:
        raise ValueError("资产变更方案须明确包含目标片段更新")
    if new_ids and not any(new_ids.intersection(shot["asset_ids"]) for shot in updates):
        raise ValueError("方案须明确更新目标片段，并实际引用本次新增资产")
    if not additions and not added_reference and (audio_policy is None or audio_policy == state.get("audio_policy", {})):
        raise ValueError("没有新增资产时，方案须实际追加已有资产引用或改变声音规则，不能提交空方案")
    # Use unique synthetic episode/shot IDs only for the validation envelope.
    ep_id = next("amendment-check-" + str(index) for index in range(1000) if "amendment-check-" + str(index) not in existing_ids | new_ids)
    validate_plan({"title": state["title"], "assets": merged_assets,
                   "episodes": [{"id": ep_id, "title": "局部增补校验", "shots": updates}]})
    from .voice import AuditionSelectionError, validate_audition
    characters = {asset["id"]: asset["name"] for asset in state["assets"] + additions if asset["kind"] == "character"}
    from .season_sources import source_text
    novel, _ = source_text(root, state=state, shot_id=shot_id)
    for asset in additions:
        if asset["kind"] == "voice" and asset.get("audition_text", "").strip():
            try:
                validate_audition(asset, novel, asset.get("audition_speaker", "").strip() or characters[asset["character_id"]])
            except AuditionSelectionError as exc:
                raise GateError(str(exc)) from None
    if audio_policy is not None:
        if not isinstance(audio_policy, dict) or set(audio_policy) != {"system_broadcast"}:
            raise ValueError("声音规则仅支持 system_broadcast")
        policy = audio_policy["system_broadcast"]
        if (not isinstance(policy, dict) or set(policy) != {"enabled", "character_id", "voice_id", "instruction"}
                or type(policy["enabled"]) is not bool or not isinstance(policy["instruction"], str)
                or len(policy["instruction"]) > 4000):
            raise ValueError("系统播报规则须包含启用状态、角色、声音和不超过 4000 字的播报要求")
        assets_by_id = {asset["id"]: asset for asset in state["assets"] + additions}
        voice = assets_by_id.get(policy["voice_id"], {})
        character = assets_by_id.get(policy["character_id"], {})
        if policy["enabled"] and (voice.get("kind") != "voice" or character.get("kind") != "character" or voice.get("character_id") != character.get("id")):
            raise ValueError("系统播报声音须绑定方案中的有效声源角色")
    return target


def propose_asset_amendment(root, shot_id, version, feedback, additions, shot_updates, audio_policy=None, *, already_locked=False,
                            revision_analysis=None):
    """Save a local candidate; never replace a plan, approve an asset or call a provider."""
    with (nullcontext() if already_locked else run_lock(root)), transaction(root) as state:
        _amendment_idle(state)
        if any(item.get("status") in {"candidate", "awaiting_assets"} for item in state.get("asset_amendments", [])):
            raise GateError("已有待确认或待完成的资产增补，请先在网页处理当前方案")
        target = _validate_amendment(state, root, shot_id, version, feedback, additions, shot_updates, audio_policy)
        amendment = {"id": "amend-" + uuid.uuid4().hex, "version": 1, "status": "candidate", "created_at": now(),
                     "target_shot_id": shot_id, "target_version": version, "target": _amendment_target(target),
                     "feedback": feedback.strip(), "additions": copy.deepcopy(additions), "shot_updates": copy.deepcopy(shot_updates),
                     "baseline_asset_ids": [asset["id"] for asset in state["assets"]],
                     "baseline_assets_sha256": _json_digest(state["assets"]),
                     "shot_baselines": [_amendment_target(_find(state, "shot", update["id"])) for update in shot_updates]}
        if audio_policy is not None:
            amendment["audio_policy"] = copy.deepcopy(audio_policy)
        state.setdefault("asset_amendments", []).append(amendment)
        target["pending_asset_amendment"] = amendment["id"]
        if revision_analysis is not None:
            from .storyboard_revision import feedback_signature
            if (not isinstance(revision_analysis, dict)
                    or revision_analysis.get("signature") != feedback_signature(target)):
                raise GateError("资产增补与本次返修分析不一致，请重新提交意见。")
            old = target.get("revision_analysis")
            if isinstance(old, dict) and old and old.get("signature") != revision_analysis["signature"]:
                target.setdefault("revision_analysis_history", []).append({**copy.deepcopy(old), "superseded_at": now()})
            # Keep the camera plan in the same atomic write as its asset gate;
            # a restart must never approve assets while losing this redesign.
            target["revision_analysis"] = {**copy.deepcopy(revision_analysis), "amendment_id": amendment["id"]}
        state["status"] = "awaiting_asset_amendment"
        state["last_error"] = None
        event(state, "asset_amendment_proposed", id=amendment["id"], shot_id=shot_id, version=version,
              asset_ids=[asset["id"] for asset in additions])
    return load_project(root)


def apply_asset_amendment(root, amendment_id, decision):
    """Apply a human-approved definition proposal, leaving video untouched until assets pass review."""
    if decision not in {"approve", "reject"}:
        raise ValueError("请选择采用或拒绝本次资产增补方案")
    from .asset_inputs import empty_input
    with run_lock(root), transaction(root) as state:
        _amendment_idle(state)
        amendment = _amendment_find(state, amendment_id)
        if amendment["status"] != "candidate":
            raise GateError("此资产增补方案已处理，请刷新当前项目")
        if decision == "reject":
            target = _find(state, "shot", amendment["target_shot_id"])
            amendment.update(status="rejected", decided_at=now())
            if target.get("pending_asset_amendment") == amendment_id:
                target.pop("pending_asset_amendment", None)
            state["status"] = "ready"
        else:
            target = _check_amendment_baseline(state, amendment)
            # Revalidate new definitions, without revalidating old auditions.
            _validate_amendment(state, root, target["id"], amendment["target_version"], amendment["feedback"],
                                amendment["additions"], amendment["shot_updates"], amendment.get("audio_policy"))
            for definition in amendment["additions"]:
                asset = {"character_id": "", **copy.deepcopy(definition), "status": "planned", "version": 0,
                         "history": [], "feedback": "", "creation_input": empty_input(), "amendment_id": amendment_id}
                episode = next(ep for ep in state["episodes"] if any(shot["id"] == target["id"] for shot in ep["shots"]))
                if episode.get("source_file"):
                    asset.update(source_episode_id=episode["id"], source_file=episode["source_file"],
                                 source_sha256=episode["source_sha256"])
                if is_internal_voice_source(asset):
                    _register_voice_source(asset)
                state["assets"].append(asset)
            amendment.update(status="awaiting_assets", decided_at=now())
            state["status"] = "awaiting_assets"
        state["last_error"] = None
        event(state, "human_asset_amendment_decided", id=amendment_id, decision=decision)
    return load_project(root)


def activate_asset_amendment(root, amendment_id):
    """After asset approval, queue the named revision exactly once; no paid work occurs here.

    The production runner may already own its run lock; this short transaction
    intentionally does not reacquire it. Return False while any new asset awaits review.
    """
    with transaction(root) as state:
        amendment = _amendment_find(state, amendment_id)
        if amendment.get("status") == "applied":
            return False
        if amendment.get("status") != "awaiting_assets":
            raise GateError("请先由用户确认资产增补方案")
        _amendment_idle(state)
        target = _check_amendment_baseline(state, amendment)
        from . import storyboard_revision
        cached_analysis = copy.deepcopy(target.get("revision_analysis", {}))
        cached_redesign = cached_analysis.get("storyboard_revision")
        from . import future_continuity
        cached_future = copy.deepcopy(cached_analysis.get("future_continuity_updates", []))
        if cached_future:
            checked = future_continuity.validate_updates(state, target, [item["update"] for item in cached_future])
            if checked != cached_future:
                raise GateError("资产增补绑定的后段接续草稿已过期，请重新准备。")
        if cached_redesign is not None:
            if (cached_analysis.get("signature") != storyboard_revision.feedback_signature(target)
                    or cached_analysis.get("storyboard_baseline") != storyboard_revision.baseline(target)):
                raise GateError("资产增补绑定的局部分镜方案已过期，请重新提交返修意见。")
            storyboard_revision.validate_plan(target, cached_redesign)
        for definition in amendment["additions"]:
            asset = _find(state, "asset", definition["id"])
            if not asset_ready(asset):
                return False
            _check_artifact(root, asset)
            if not is_internal_voice_source(asset) and asset.get("approved_sha256") != asset.get("sha256"):
                raise GateError("新增资产审批与当前定义或媒体不符，请重新确认")
        required_ids = {identifier for update in amendment["shot_updates"]
                        for identifier in (_find(state, "shot", update["id"])["asset_ids"] + update["add_asset_ids"])}
        for identifier in required_ids:
            asset = _find(state, "asset", identifier)
            if not asset_ready(asset):
                return False
            _check_artifact(root, asset)
        # Retire the old media with its actual creative definition before any
        # approved amendment edits the canonical prompt/voice references.
        if target.get("status") == "planned" and not target.get("file") and not target.get("job_id"):
            target["feedback"] = amendment["feedback"]
            target.pop("refined_prompt", None)
        else:
            _request_shot_revision(state, target, amendment["feedback"])
        for update in amendment["shot_updates"]:
            shot = _find(state, "shot", update["id"])
            shot["asset_ids"] = list(dict.fromkeys(shot["asset_ids"] + update["add_asset_ids"]))
            for key in ("prompt", "dialogue", "continuity_notes"):
                if key in update:
                    shot[key] = update[key]
            # A prompt prepared against the former asset set cannot be reused.
            shot.pop("prompt_candidate", None)
            shot.pop("prompt_approval", None)
        target.pop("pending_asset_amendment", None)
        for update in amendment["shot_updates"]:
            shot = _find(state, "shot", update["id"])
            # A full prompt update was reviewed together with the exact saved
            # feedback baseline. Do not interpret that same feedback again as
            # a new amendment when production reaches a future segment. Merely
            # adding a reference/continuity note cannot resolve other feedback.
            if shot is not target and ("prompt" not in update or not shot.get("feedback", "").strip()):
                continue
            if shot.get("revision_analysis"):
                shot.setdefault("revision_analysis_history", []).append(copy.deepcopy(shot["revision_analysis"]))
            shot["revision_analysis"] = {
                "signature": hashlib.sha256(json.dumps([shot.get("revision", 0), shot.get("feedback", "")],
                                                        ensure_ascii=False).encode()).hexdigest(),
                "amendment_id": amendment_id, "blockers": [],
                "summary": "资产与片段更新方案已获人工确认，沿用已批准素材准备提示词，仍须确认后才生成。"}
            if shot is target:
                shot["revision_analysis"]["planning_policy_version"] = future_continuity.PLANNING_POLICY_VERSION
                if cached_future:
                    shot["revision_analysis"]["future_continuity_updates"] = future_continuity.validate_updates(
                        state, target, [item["update"] for item in cached_future])
            if shot is target and cached_redesign is not None:
                shot["revision_analysis"].update(storyboard_revision=cached_redesign,
                    storyboard_baseline=storyboard_revision.baseline(shot))
        if amendment.get("audio_policy"):
            state.setdefault("audio_policy", {}).update(copy.deepcopy(amendment["audio_policy"]))
        amendment.update(status="applied", applied_at=now())
        state["status"] = "ready"
        state["last_error"] = None
        event(state, "asset_amendment_activated", id=amendment_id, shot_id=target["id"], revision=target.get("revision", 0))
    return True


def approve_asset_prompt(root, asset_id, candidate_id, sha256, model=None, prompt=None):
    """A web human decision; approval authorizes only this asset's next request."""
    from .asset_prompt_gate import approve
    return approve(root, asset_id, candidate_id, sha256, model=model, prompt=prompt)


def approve_prompt(root, shot_id, candidate_id, sha256, video_model=None):
    from .prompt_gate import approve
    return approve(root, shot_id, candidate_id, sha256, video_model=video_model)


def reject_prompt(root, shot_id, candidate_id, sha256, feedback):
    from .prompt_gate import reject
    return reject(root, shot_id, candidate_id, sha256, feedback)


def edit_prompt_and_approve(root, shot_id, candidate_id, sha256, prompt, video_model=None):
    from .prompt_gate import edit_and_approve
    return edit_and_approve(root, shot_id, candidate_id, sha256, prompt, video_model=video_model)


def submit_planning_feedback(root, shot_id, feedback, expected_version, analysis_signature):
    """Resolve a visible planning question with new human instructions only.

    This does not select a timing option, approve an asset/prompt, clear a media
    failure, or submit work. The next production run must analyze the changed
    instructions and still present its exact prompt for human confirmation.
    """
    if not isinstance(feedback, str) or not feedback.strip() or len(feedback) > 8000:
        raise ValueError("请填写不超过 8000 字的规划调整意见")
    if (type(expected_version) is not int or expected_version < 0
            or not isinstance(analysis_signature, str) or not re.fullmatch(r"[0-9a-f]{64}", analysis_signature)):
        raise GateError("规划问题已更新，请刷新后针对当前问题填写意见")
    feedback = feedback.strip()
    with run_lock(root), transaction(root) as state:
        if state["plan"].get("status") != "approved":
            raise GateError("请先确认当前策划，再补充分镜制作意见")
        if any(execution_unsettled(job) for job in state["jobs"].values()):
            raise GateError("尚有调用在执行或结果待核实，请先等待落盘或核实原任务，再调整规划意见")
        shot = _find(state, "shot", shot_id)
        if (shot.get("status") != "planned" or shot.get("file") or shot.get("job_id")
                or type(shot.get("version", 0)) is not int or shot.get("version", 0) != expected_version):
            raise GateError("片段状态或版本已改变，只有尚未提交生成的待准备片段可以补充规划意见")
        if shot.get("pending_asset_amendment") or any(
                amendment.get("status") in {"candidate", "awaiting_assets"}
                and any(update.get("id") == shot_id for update in amendment.get("shot_updates", []))
                for amendment in state.get("asset_amendments", [])):
            raise GateError("此片段仍有待处理的资产变更方案，请先完成或取消该方案")
        old_feedback, revision = shot.get("feedback", ""), shot.get("revision", 0)
        if not isinstance(old_feedback, str) or type(revision) is not int or revision < 0:
            raise GateError("片段规划记录无效，请检查当前项目后再提交意见")
        current_signature = hashlib.sha256(json.dumps([revision, old_feedback], ensure_ascii=False).encode()).hexdigest()
        analysis = shot.get("revision_analysis")
        blockers = analysis.get("blockers") if isinstance(analysis, dict) else None
        if (analysis_signature != current_signature or not isinstance(analysis, dict)
                or analysis.get("signature") != current_signature or not isinstance(blockers, list)
                or not blockers or any(not isinstance(item, str) or not item.strip() for item in blockers)):
            raise GateError("规划问题已更新或已解决，请刷新后针对当前问题填写意见")
        from .future_continuity import FEEDBACK_SEPARATOR
        if feedback in [part.strip() for part in old_feedback.split(FEEDBACK_SEPARATOR)]:
            raise GateError("这条修改意见已经保存，无需重复提交。请继续准备提示词；如确有新要求，再填写新增内容。")
        combined = (old_feedback + FEEDBACK_SEPARATOR + feedback).strip()
        if len(combined) > 100000:
            raise ValueError("累计规划调整意见超过 100000 字，请精简后再提交")
        stamp = now()
        history = shot.setdefault("revision_analysis_history", [])
        if not any(item.get("signature") == current_signature for item in history if isinstance(item, dict)):
            history.append({**copy.deepcopy(analysis), "superseded_at": stamp,
                            "resolution": "human_planning_feedback", "human_feedback": feedback})
        candidate = shot.pop("prompt_candidate", None)
        if candidate:
            shot.setdefault("prompt_history", []).append({**copy.deepcopy(candidate), "status": "invalidated",
                "invalidation_reason": "用户补充规划意见，须重新准备并确认提示词"})
        for key in ("revision_analysis", "prompt_approval", "refined_prompt"):
            shot.pop(key, None)
        from .storyboard_revision import discard
        discard(shot, reason=feedback, stamp=stamp, state=state)
        shot.update(status="planned", revision=revision + 1, feedback=combined)
        state.update(status="paused" if state["paused"] else "ready", last_error=None)
        event(state, "human_planning_feedback", id=shot_id, version=expected_version, revision=shot["revision"],
              analysis_signature=current_signature, feedback=feedback)
    return load_project(root)


def set_paused(root, paused):
    with transaction(root) as s:
        s["paused"] = bool(paused)
        if paused:
            s["status"] = "paused"
        event(s, "paused" if paused else "resumed")
    return load_project(root)


def configure(root, settings=None, limit=None, currency=None):
    """Update configuration; legacy limits are retained only as historical data."""
    with transaction(root) as s:
        s["settings"].setdefault("postproduction", {"enabled": False})
        if settings:
            if set(settings) - set(s["settings"]) - {"gpt_video_review_enabled"}:
                raise ValueError("未知配置字段")
            if "gpt_video_review_enabled" in settings and type(settings["gpt_video_review_enabled"]) is not bool:
                raise ValueError("GPT 视频审查开关必须为布尔值。")
            secret_words = {"api_key", "token", "secret", "access_key", "secret_key", "password"}
            def check(v):
                if isinstance(v, dict):
                    if any(k.lower() in secret_words for k in v):
                        raise ValueError("凭证仅通过环境变量读取，不得写入项目配置")
                    for x in v.values(): check(x)
                elif isinstance(v, list):
                    for x in v: check(x)
            check(settings)
            for key, value in settings.items():
                if isinstance(value, dict):
                    s["settings"][key].update(value)
                else:
                    s["settings"][key] = value
            for key in ("max_attempts", "max_llm_calls", "batch_size", "poll_seconds", "max_poll_seconds"):
                if isinstance(s["settings"][key], bool) or not isinstance(s["settings"][key], int) or not 1 <= s["settings"][key] <= 86400:
                    raise ValueError("运行限制必须是正整数")
            for val in s["settings"]["caps"].values():
                if isinstance(val, bool) or not isinstance(val, (float, int)) or not math.isfinite(val) or val < 0:
                    raise ValueError("调用价格上限必须为非负有限数字")
            if type(s["settings"]["postproduction"].get("enabled")) is not bool:
                raise ValueError("本集后期开关必须为布尔值")
            if type(s["settings"]["ark"].get("video_watermark", False)) is not bool:
                raise ValueError("视频水印开关 video_watermark 必须为布尔值")
            if type(s["settings"].get("gpt_video_review_enabled", False)) is not bool:
                raise ValueError("GPT 视频审查开关必须为布尔值。")
        _totals(s)
        if currency:
            if currency not in {"CNY", "USD"}:
                raise ValueError("币种只支持 CNY / USD")
            if s["jobs"] and currency != s["budget"]["currency"]:
                raise ValueError("产生账本后不能改变币种")
            s["budget"]["currency"] = currency
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit < 0:
                raise ValueError("历史限额记录必须为非负有限数字")
            s["budget"].update(limit=limit, approved=True)
        s["budget"]["enforcement"] = "disabled"
        event(s, "configuration_updated", budget_limit=s["budget"]["limit"])
    return load_project(root)


def _validate_advisory_target(root, state, kind, item_id, target):
    if target is None:
        return None
    if kind != "llm" or item_id != "review" or not isinstance(target, dict):
        raise ValueError("审查目标只可登记到 llm/review 调用，且必须为对象")
    target_kind = target.get("kind")
    if not isinstance(target_kind, str) or target_kind not in {"shot", "rough_cut"}:
        raise ValueError("审查目标类型或字段不合法；仅支持片段或累计粗剪的完整绑定")
    common = {"kind", "id", "sha256", "file"}
    expected = common | ({"version", "video_job_id"} if target_kind == "shot" else {"key"})
    if set(target) != expected:
        raise ValueError("审查目标类型或字段不合法；仅支持片段或累计粗剪的完整绑定")
    if (not isinstance(target["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", target["id"])
            or not isinstance(target["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", target["sha256"])
            or not isinstance(target["file"], str) or not 1 <= len(target["file"]) <= 4096
            or Path(target["file"]).is_absolute()):
        raise ValueError("审查目标须提供有效 ID、SHA256 与项目内相对文件路径")
    safe_path(root, target["file"])
    if target["kind"] == "shot":
        if (type(target["version"]) is not int or target["version"] < 1
                or not isinstance(target["video_job_id"], str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", target["video_job_id"])):
            raise ValueError("片段审查目标须绑定整数版本和原视频任务 ID")
        shot = _find(state, "shot", target["id"])
        if any(shot.get(key) != target[key] for key in ("version", "sha256", "file")) or shot.get("job_id") != target["video_job_id"]:
            raise GateError("审查目标与当前视频版本不符，未提交审查")
        video_job = state["jobs"].get(target["video_job_id"], {})
        if (video_job.get("kind") != "video" or video_job.get("item_id") != shot["id"]
                or video_job.get("state") != "completed" or video_job.get("output") != target["file"]):
            raise GateError("审查目标缺少已完成且匹配的原视频任务，未提交审查")
        _check_shot_artifact(root, state, shot)
    else:
        if not isinstance(target["key"], str) or not 1 <= len(target["key"]) <= 128:
            raise ValueError("累计粗剪审查目标须提供非空批次标识")
        episode = _find(state, "episode", target["id"])
        same_key = [cut for cut in episode.get("rough_cuts", []) if cut.get("key") == target["key"]]
        if same_key and not any(cut.get("file") == target["file"] and cut.get("sha256") == target["sha256"] for cut in same_key):
            raise GateError("累计粗剪审查目标与已登记批次不符，未提交审查")
        # A one-shot episode can have a verified cut before it is registered in
        # rough_cuts; its exact bytes are still bound here before the LLM call.
        _check_artifact(root, target)
    return copy.deepcopy(target)


def reserve(root, job_id, kind, item_id, amount, *, pricing=None, advisory_target=None, require_video_review=False,
            cost_track=None):
    """Persist BEFORE calling provider, without a monetary spending gate.

    ``amount`` is a legacy estimate, never proof of billing. Retain duplicate,
    pause, unknown-outcome and finite call-count protections independently.
    """
    with transaction(root) as s:
        if require_video_review and kind == "llm" and item_id == "review" and not video_review_enabled(s):
            raise VideoReviewDisabled("GPT 视频审查已关闭，未提交新的审查请求。")
        if job_id in s["jobs"]:
            raise GateError("调用已登记，必须恢复原调用，不能重复提交")
        if s["paused"]:
            raise GateError("制作已暂停")
        if blocking_unknown_jobs(s):
            raise GateError("存在未核实调用，必须先对账才能提交新任务")
        checked_target = _validate_advisory_target(root, s, kind, item_id, advisory_target)
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
            raise ValueError("调用费用估算无效")
        if cost_track is not None and type(cost_track) is not bool:
            raise ValueError("费用记录标记必须为布尔值")
        if kind == "llm":
            if s["llm_calls"] >= s["settings"]["max_llm_calls"]:
                raise GateError("GPT 调用次数已达到上限")
            s["llm_calls"] += 1
        # A zero legacy budget is no longer evidence that media could not have
        # been submitted. Persist this before POST for legacy recovery checks.
        s["budget"]["enforcement"] = "disabled"
        s["jobs"][job_id] = {"id": job_id, "kind": kind, "item_id": item_id,
            "amount": round(amount, 6), "accounting": "reserved", "state": "submitting",
            "created_at": now(), "task_id": None}
        brain = s["settings"]["brain"]
        s["jobs"][job_id]["cost_track"] = (
            s["mode"] == "live" and (kind != "llm" or brain.get("provider") == "openai"
                                    or brain.get("billing") == "api")
            if cost_track is None else cost_track)
        if kind == "llm" and item_id == "review":
            s["jobs"][job_id].update(purpose="advisory_review", advisory_target=checked_target)
        if pricing is not None:
            s["jobs"][job_id]["billing"] = {"price": copy.deepcopy(pricing),
                "currency": s["budget"]["currency"], "status": "usage_missing"}
        event(s, "call_reserved", id=job_id, amount=amount, kind=kind)
    return load_project(root)["jobs"][job_id]


def update_job(root, job_id, **values):
    with transaction(root) as s:
        j = s["jobs"][job_id]
        if set(values) & {"id", "item_id", "kind", "amount", "created_at", "billing", "purpose", "advisory_target", "cost_track"}:
            raise ValueError("已登记的调用身份和费用记录属性不可更改")
        if "accounting" in values and values["accounting"] not in {"spent", "reserved", "unknown", "released"}:
            raise ValueError("账本状态无效")
        if j["accounting"] == "spent" and values.get("accounting", "spent") != "spent":
            raise GateError("已保守记账的调用不能通过任务回调退回可用额度")
        if j.get("task_id") and "task_id" in values and values["task_id"] != j["task_id"]:
            raise GateError("恢复必须使用已登记的供应商任务 ID")
        if all(j.get(key) == value for key, value in values.items()):
            return copy.deepcopy(j)
        j.update(values, updated_at=now())
        event(s, "call_updated", id=job_id, state=j["state"], accounting=j["accounting"])
    return load_project(root)["jobs"][job_id]


def record_usage(root, job_id, result, *, historical_pricing=None):
    """Persist only whitelisted provider usage; never treat it as a paid bill."""
    from .billing import usage_record
    with transaction(root) as state:
        job = state["jobs"][job_id]
        billing = job.setdefault("billing", {"currency": state["budget"]["currency"]})
        if not billing.get("price") and historical_pricing:
            billing["price"] = copy.deepcopy(historical_pricing)
            billing["price"]["historical_lookup"] = True
        record = usage_record(result, billing.get("price", {}))
        if record.get("currency") != state["budget"]["currency"]:
            record.update(currency=state["budget"]["currency"], usage_cost=None, status="pricing_missing")
        # A later incomplete poll must not erase an already recorded usage result.
        if record.get("usage_cost") is None and billing.get("usage_cost") is not None:
            return copy.deepcopy(job)
        if all(billing.get(key) == value for key, value in record.items()):
            return copy.deepcopy(job)
        billing.update(record)
        billing["usage_recorded_at"] = now()
        event(state, "provider_usage_recorded", id=job_id)
    return load_project(root)["jobs"][job_id]


def import_bill_costs(root, records):
    """Operator-supplied per-call bill amounts; aggregate bills cannot be allocated."""
    if not isinstance(records, list) or not records or len(records) > 10000:
        raise ValueError("账单应为非空调用明细列表")
    with transaction(root) as state:
        seen = set()
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("每条账单必须是调用明细对象")
            job_id, amount = row.get("job_id"), row.get("amount")
            reference = row.get("bill_reference")
            if not isinstance(job_id, str) or job_id in seen or job_id not in state["jobs"]:
                raise ValueError("账单调用 ID 缺失、重复或不存在")
            seen.add(job_id)
            if row.get("currency") != state["budget"]["currency"]:
                raise ValueError("账单币种必须与项目相同，不能自动换算")
            if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
                raise ValueError("实际费用必须是非负有限金额")
            if row.get("scope") != "per_call" or not isinstance(reference, str) or not 0 < len(reference.strip()) <= 300:
                raise ValueError("必须提供逐调用账单依据；不可把账户或日期汇总金额分摊到单次调用")
            job = state["jobs"][job_id]
            if job["accounting"] not in {"spent", "unknown"}:
                raise ValueError("仅可核实已结束或结果待核实的调用费用")
            billing = job.setdefault("billing", {})
            amount = round(amount, 6)
            if "actual_cost" in billing:
                if billing["actual_cost"] == amount and billing.get("bill_reference") == reference.strip():
                    continue
                raise GateError("已有不同核实账单，不能覆盖原记录")
            billing.update(actual_cost=amount, currency=row["currency"], bill_reference=reference.strip(),
                           bill_basis="operator_verified_per_call", bill_verified_at=now())
            event(state, "bill_cost_verified", id=job_id, amount=amount)
    return load_project(root)


def reconcile(root, job_id, outcome, note, task_id=None):
    if not note.strip():
        raise ValueError("对账必须填写来自供应商控制台的核实说明")
    with transaction(root) as s:
        j = s["jobs"][job_id]
        if j["accounting"] != "unknown":
            raise GateError("该调用不是待核实状态")
        if outcome == "resume":
            if not task_id or j["kind"] not in {"video", "voice", "music"}:
                raise ValueError("恢复支持视频或配乐调用，必须填写供应商任务 ID")
            j.update(accounting="reserved", state="submitted", task_id=task_id)
            # Recover a crash between ledger reservation and object linkage.
            kind = "asset" if j["kind"] == "voice" else "episode" if j["kind"] == "music" else "shot"
            item = _find(s, kind, j["item_id"])
            item["music_job_id" if kind == "episode" else "job_id"] = job_id
        elif outcome in {"spent", "released"}:
            j.update(accounting=outcome, state="failed")
        else:
            raise ValueError("对账结果只能是 resume / spent / released")
        event(s, "human_reconciled", id=job_id, outcome=outcome, note=note)
    return load_project(root)


def set_voice_audition(root, asset_id, text, source, reason):
    """Save a sourced selection without approval, generation or version changes."""
    from .voice import AuditionSelectionError, audition_selection_key, validate_audition
    values = {"audition_text": text, "audition_source": source, "audition_reason": reason}
    if any(not isinstance(value, str) or not value.strip() or len(value) > 4000 for value in values.values()):
        raise GateError("试音选句、原文来源和选句理由须为非空文本，每项不超过4000字；缺少原文时须先补充或选择，不能编造台词。")
    with run_lock(root), transaction(root) as state:
        asset = _find(state, "asset", asset_id)
        if asset.get("kind") != "voice":
            raise GateError("原文试音选句只能设置在声音资产上。")
        if asset.get("status") not in {"planned", "rejected"}:
            raise GateError("仅待制作或已申请返工的声音可调整选句；候选或已批准声音须先由用户明确提出返工。")
        jobs = [job for job in state["jobs"].values()
                if job.get("kind") == "voice" and job.get("item_id") == asset_id]
        if any(execution_unsettled(job) for job in jobs):
            raise GateError("该声音尚有在途或结果不明的调用；请先恢复原任务或完成对账，再调整试音选句。")
        active = state["jobs"].get(asset.get("job_id", ""), {})
        if asset["status"] != "rejected" and active.get("state") == "completed":
            raise GateError("该声音已有完成但尚待恢复的任务；请先取回并审核旧素材，再决定是否返工。")
        character = _find(state, "asset", asset.get("character_id"))
        if character.get("kind") != "character":
            raise GateError("声音试音必须绑定一个有效角色，不能猜测说话人。")
        from .season_sources import source_text
        novel, source_hash = source_text(root, state=state, asset=asset)
        candidate = {**asset, **values}
        try:
            speaker = asset.get("audition_speaker", "").strip() or character["name"]
            validate_audition(candidate, novel, speaker)
        except AuditionSelectionError as exc:
            raise GateError(str(exc)) from None
        asset.update(values)
        asset["audition_selection_key"] = audition_selection_key(asset, speaker, source_hash)
        asset.pop("audition_selection_error", None)
        event(state, "voice_audition_selected", id=asset_id)
    return load_project(root)
