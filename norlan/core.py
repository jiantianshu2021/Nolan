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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class GateError(ValueError):
    """A resumable production gate, not permission to skip it."""


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
    b["basis"] = "预算占用与费用分开；用量费用按公开标准价计算，实际费用须核对账单"


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
            "brain": {"provider": "codex"}, "ark": {},
            "caps": {"image": 0, "video_second": 0, "llm": 0},
            "max_attempts": 3, "max_llm_calls": 80, "batch_size": 3,
            "poll_seconds": 10, "max_poll_seconds": 1800,
            "music": {}, "subtitles": {"mode": "none", "call_cap": 0},
            "creative_brief": "", "require_voice_reference": True,
        },
        "budget": {"limit": 0, "currency": "CNY", "spent": 0, "reserved": 0,
                   "unknown": 0, "approved": mode == "demo",
                   "basis": "按用户填写的每次调用上限保守记账；非供应商实时账单"},
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
                validate_audition(asset, novel, characters[asset["character_id"]])
            except AuditionSelectionError:
                raise PlanValidationError("voice_selection_invalid", f"plan.assets[{index}].audition_source") from None
    with transaction(root) as s:
        if any(j["state"] in {"submitting", "submitted", "download_pending", "uncertain"} for j in s["jobs"].values()):
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
        s["episodes"] = []
        for ep in plan["episodes"]:
            ep = copy.deepcopy(ep)
            ep.update(status="planned", version=0, history=[], feedback="", reviews=[])
            for shot in ep["shots"]:
                shot.update(status="planned", attempts=[], revision=0, feedback="")
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


def _check_artifact(root, item):
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
    _check_version(asset, asset_version)
    data = input_state(asset)
    if (isinstance(input_revision, bool) or not isinstance(input_revision, int)
            or input_revision != data["revision"]):
        raise GateError("资产输入已更新，请刷新网页后重新确认。")
    if has_started(state, asset) or asset.get("status") != "planned":
        raise GateError("该资产已开始生成，初始描述和参考图不能再修改；请通过资产返工提出修改意见。")
    if any(job.get("accounting") in {"reserved", "unknown"} for job in state["jobs"].values()):
        raise GateError("尚有在途或结果不明的调用，请等待落盘或完成对账后再修改资产输入。")
    return asset, data


def set_remaining_asset_inputs_gpt(root, plan_version):
    """Confirm only untouched first-time inputs; retain every custom draft."""
    from .asset_inputs import has_started, input_state
    with run_lock(root), transaction(root) as state:
        _check_version(state["plan"], plan_version)
        if state["plan"].get("status") != "approved":
            raise GateError("请先批准当前策划，再确认资产生成方式。")
        if any(job.get("accounting") in {"reserved", "unknown"} for job in state["jobs"].values()):
            raise GateError("尚有在途或结果不明的调用，请等待落盘或完成对账后再修改资产输入。")
        selected = []
        for asset in state["assets"]:
            if has_started(state, asset) or asset.get("status") != "planned":
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
        event(state, "human_asset_input_confirmed", id=asset_id, asset_version=asset_version,
              input_revision=data["revision"], mode=mode,
              reference_hashes=[ref["sha256"] for ref in references])
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
    record = {"decision": decision, "version": shot["version"], "sha256": shot.get("sha256"),
              "at": now(), "feedback": feedback,
              "report": {key: copy.deepcopy(review[key]) for key in ("verdict", "issues") if key in review} if report_bound else {},
              "report_bound": report_bound,
              "report_note": ("报告保留 GPT 的原始检查意见，未经用户逐条核实；人工决定针对这个版本的整体成片。"
                              if report_bound else "现有检查报告未绑定当前视频版本，未作为本次人工决定的报告依据；人工决定针对实际观看的这个版本。")}
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
                    if asset["status"] != "approved":
                        raise GateError("本集依赖的资产不再处于已批准状态")
                    _check_artifact(root, asset)
            rendered = item.get("render_manifest", {}).get("clips", [])
            if rendered != [{"id": sh["id"], "sha256": sh["sha256"]} for sh in item["shots"]]:
                raise GateError("本集镜头已更新或缺少合成记录，请重新合成后审批")
        elif kind == "shot":
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


def _invalidate_shot(shot, reason):
    if shot.get("file"):
        shot.setdefault("history", []).append({k: copy.deepcopy(shot[k]) for k in
            ("file", "sha256", "evidence", "review", "version", "human_decision", "approved_at",
             "approved_sha256", "approved_version", "continuity_review") if k in shot})
    shot.update(status="planned", revision=shot.get("revision", 0) + 1, feedback=reason)
    for key in ("file", "sha256", "evidence", "review", "job_id", "refined_prompt", "approved_at", "approved_sha256", "approved_version", "generation_error"):
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
    _invalidate_shot(shot, feedback)
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


def _invalidate_episode(ep, reason, shot_ids=None):
    if ep.get("file"):
        ep["history"].append({k: ep[k] for k in ("file", "sha256", "version") if k in ep})
    selected = set(shot_ids or [sh["id"] for sh in ep["shots"]])
    changed_scenes = set()
    for shot in ep["shots"]:
        if shot["id"] in selected or shot["scene_id"] in changed_scenes:
            changed_scenes.add(shot["scene_id"])
            _invalidate_shot(shot, reason)
    ep.update(status="revision_requested", feedback=reason)
    for key in ("file", "sha256", "approved_at", "approved_sha256", "verified_subtitles", "render_manifest",
                "progress_stage", "progress_started_at"):
        ep.pop(key, None)
    ep["subtitle_status"] = "needs_retiming"


def reject(root, kind, item_id, version, feedback, shot_ids=None):
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("返工意见不能为空")
    with run_lock(root), transaction(root) as s:
        if any(j["accounting"] in {"reserved", "unknown"} or
               j.get("state") in {"submitting", "submitted", "download_pending", "uncertain"}
               for j in s["jobs"].values()):
            raise GateError("尚有调用在执行，请先暂停并等待在途任务落盘后再返工")
        item = _find(s, kind, item_id)
        _check_version(item, version)
        if kind == "plan":
            if any(a.get("file") for a in s["assets"]):
                raise GateError("已有生成素材，请使用资产或分集返工")
            item.update(status="rejected", feedback=feedback)
        elif kind == "asset":
            if item["status"] not in {"candidate", "approved", "rejected"}:
                raise GateError("资产尚未生成")
            item.update(status="rejected", feedback=feedback)
            # The completed task belongs to the rejected candidate, not its replacement.
            for key in ("job_id", "approved_at", "approved_sha256"):
                item.pop(key, None)
            affected = False
            for ep in s["episodes"]:
                selected = [sh["id"] for sh in ep["shots"] if item_id in sh["asset_ids"]]
                if selected or affected:
                    _invalidate_episode(ep, "资产变更：" + feedback, None if affected else selected)
                    affected = True
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
                    shot["feedback"] = feedback
                else:
                    _request_shot_revision(s, shot, feedback)
            _invalidate_render(item, feedback)
            shot_ids = [shot["id"] for shot in targets]
        elif kind == "shot":
            if item["status"] not in {"review_required", "accepted"}:
                raise GateError("镜头不在人工复核阶段")
            _check_shot_rejection(root, s, item)
            _request_shot_revision(s, item, feedback)
        event(s, "human_rejected", kind=kind, id=item_id, version=version,
              feedback=feedback, shot_ids=shot_ids)
        s["status"] = "ready"
    return load_project(root)


def request_caption_fix(root, item_id, version, feedback):
    """A subtitle-only edit preserves the generated picture and sound."""
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


def set_paused(root, paused):
    with transaction(root) as s:
        s["paused"] = bool(paused)
        if paused:
            s["status"] = "paused"
        event(s, "paused" if paused else "resumed")
    return load_project(root)


def configure(root, settings=None, limit=None, currency=None):
    with transaction(root) as s:
        if settings:
            if set(settings) - set(s["settings"]):
                raise ValueError("未知配置字段")
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
        _totals(s)
        if currency:
            if currency not in {"CNY", "USD"}:
                raise ValueError("币种只支持 CNY / USD")
            if s["jobs"] and currency != s["budget"]["currency"]:
                raise ValueError("产生账本后不能改变币种")
            s["budget"]["currency"] = currency
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit < 0 or limit < sum(s["budget"][x] for x in ("spent", "reserved", "unknown")):
                raise ValueError("预算不能小于已花费和已占用额度")
            s["budget"].update(limit=limit, approved=True)
        event(s, "configuration_updated", budget_limit=s["budget"]["limit"])
    return load_project(root)


def reserve(root, job_id, kind, item_id, amount, *, pricing=None):
    """Persist BEFORE calling provider. Unknown outcomes stay charged/reserved."""
    with transaction(root) as s:
        if job_id in s["jobs"]:
            raise GateError("调用已登记，必须恢复原调用，不能重复提交")
        if s["paused"]:
            raise GateError("制作已暂停")
        if any(j["accounting"] == "unknown" for j in s["jobs"].values()):
            raise GateError("存在未核实调用，必须先对账才能提交新任务")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
            raise ValueError("预算金额无效")
        if s["mode"] == "live" and kind != "llm" and (not s["budget"]["approved"] or amount <= 0):
            raise GateError("请先设置预算和媒体调用价格上限")
        if kind == "llm":
            if s["llm_calls"] >= s["settings"]["max_llm_calls"]:
                raise GateError("GPT 调用次数已达到上限")
            if s["settings"]["brain"].get("provider") == "openai" and amount <= 0:
                raise GateError("OpenAI API 模式需要填写每次调用预算上限")
            s["llm_calls"] += 1
        _totals(s)
        if amount > s["budget"]["available"] + 0.0000001:
            raise GateError("剩余预算不足，任务未提交")
        s["jobs"][job_id] = {"id": job_id, "kind": kind, "item_id": item_id,
            "amount": round(amount, 6), "accounting": "reserved", "state": "submitting",
            "created_at": now(), "task_id": None}
        if pricing is not None:
            s["jobs"][job_id]["billing"] = {"price": copy.deepcopy(pricing),
                "currency": s["budget"]["currency"], "status": "usage_missing"}
        event(s, "call_reserved", id=job_id, amount=amount, kind=kind)
    return load_project(root)["jobs"][job_id]


def update_job(root, job_id, **values):
    with transaction(root) as s:
        j = s["jobs"][job_id]
        if set(values) & {"id", "item_id", "kind", "amount", "created_at", "billing"}:
            raise ValueError("已登记的调用身份和预算金额不可更改")
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
        if any(job.get("accounting") == "unknown" or job.get("state") in
               {"submitting", "submitted", "download_pending"} for job in jobs):
            raise GateError("该声音尚有在途或结果不明的调用；请先恢复原任务或完成对账，再调整试音选句。")
        active = state["jobs"].get(asset.get("job_id", ""), {})
        if asset["status"] != "rejected" and active.get("state") == "completed":
            raise GateError("该声音已有完成但尚待恢复的任务；请先取回并审核旧素材，再决定是否返工。")
        character = _find(state, "asset", asset.get("character_id"))
        if character.get("kind") != "character":
            raise GateError("声音试音必须绑定一个有效角色，不能猜测说话人。")
        novel_path = safe_path(root, "source/novel.txt")
        novel = novel_path.read_text(encoding="utf-8")
        candidate = {**asset, **values}
        try:
            validate_audition(candidate, novel, character["name"])
        except AuditionSelectionError as exc:
            raise GateError(str(exc)) from None
        asset.update(values)
        asset["audition_selection_key"] = audition_selection_key(asset, character["name"], digest(novel_path))
        asset.pop("audition_selection_error", None)
        event(state, "voice_audition_selected", id=asset_id)
    return load_project(root)
