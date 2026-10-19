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
    b = state["budget"]
    for status in ("spent", "reserved", "unknown"):
        b[status] = round(sum(j["amount"] for j in state["jobs"].values()
                              if j["accounting"] == status), 6)
    b["available"] = round(b["limit"] - b["spent"] - b["reserved"] - b["unknown"], 6)


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


def create_project(root, novel: str, mode="demo", title="Norlan 漫剧"):
    root = Path(root).expanduser().resolve()
    if mode not in {"demo", "live"}:
        raise ValueError("模式只能是 demo 或 live")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "project.json").exists():
        raise ValueError("项目已存在，请使用 run / studio 继续")
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
    if not isinstance(plan, dict) or not str(plan.get("title", "")).strip():
        raise ValueError("规划缺少标题")
    if plan.get("ratio", "16:9") not in {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9"}:
        raise ValueError("不支持的画面比例")
    assets, episodes = plan.get("assets"), plan.get("episodes")
    if not isinstance(assets, list) or not assets or not isinstance(episodes, list) or not episodes:
        raise ValueError("规划必须包含资产和分集")
    if len(episodes) > 100 or len(assets) > 200:
        raise ValueError("一次规划过大，请拆分小说")
    ids = set()
    def identifier(item):
        if not isinstance(item, dict):
            raise ValueError("规划条目必须为对象")
        value = item.get("id", "")
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value) or value in ids:
            raise ValueError("规划 ID 不合法或重复")
        ids.add(value)
    for asset in assets:
        identifier(asset)
        if asset.get("kind") not in {"character", "location", "prop", "voice"}:
            raise ValueError("资产类型不合法")
        if not asset.get("name") or not asset.get("prompt"):
            raise ValueError("资产必须包含名称和生成说明")
    asset_ids = {a["id"] for a in assets}
    character_ids = {a["id"] for a in assets if a["kind"] == "character"}
    voices = {}
    for asset in assets:
        if asset["kind"] == "voice":
            character_id = asset.get("character_id")
            if not isinstance(character_id, str) or character_id not in character_ids:
                raise ValueError("声音资产必须通过 character_id 绑定一个角色资产")
            voices[asset["id"]] = character_id
        elif asset.get("character_id", ""):
            raise ValueError("非声音资产的 character_id 必须为空")
    for ep in episodes:
        identifier(ep)
        if not ep.get("title") or not isinstance(ep.get("shots"), list) or not ep["shots"]:
            raise ValueError("每集必须包含标题和分镜")
        if len(ep["shots"]) > 100:
            raise ValueError("单集镜头过多，请拆分")
        for shot in ep["shots"]:
            identifier(shot)
            d = shot.get("duration")
            if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or not 4 <= d <= 15 or int(d) != d:
                raise ValueError("单镜时长必须为 4–15 秒整数")
            refs = shot.get("asset_ids")
            if (not shot.get("prompt") or not isinstance(refs, list) or not refs
                    or not all(isinstance(ref, str) for ref in refs)
                    or len(set(refs)) != len(refs) or not set(refs) <= asset_ids):
                raise ValueError("每镜必须引用存在的资产并包含提示词")
            if not shot.get("scene_id"):
                raise ValueError("每镜必须包含 scene_id 以维护空间与剪辑连续性")
            if any(voices[ref] not in refs for ref in refs if ref in voices):
                raise ValueError("镜头引用声音时，必须同时引用该声音绑定的角色")
    return plan


def install_plan(root, plan):
    validate_plan(plan)
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
                         "history": [], "feedback": ""} for a in plan["assets"]]
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
            if any(sh["status"] != "accepted" for sh in item["shots"]):
                raise GateError("本集仍有未通过检查的镜头")
            for sh in item["shots"]:
                _check_artifact(root, sh)
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
            _check_artifact(root, item)
        item["status"] = "accepted" if kind == "shot" else "approved"
        item["approved_at"] = now()
        item["approved_sha256"] = item.get("sha256")
        if kind == "shot":
            item.setdefault("review", {})["human_override"] = True
        event(s, "human_approved", kind=kind, id=item_id, version=version,
              sha256=item.get("sha256"))
        s["last_error"] = None
        s["status"] = "ready"
    return load_project(root)


def _invalidate_shot(shot, reason):
    if shot.get("file"):
        shot.setdefault("history", []).append({k: copy.deepcopy(shot[k]) for k in
            ("file", "sha256", "evidence", "review", "version") if k in shot})
    shot.update(status="planned", revision=shot.get("revision", 0) + 1, feedback=reason)
    for key in ("file", "sha256", "evidence", "review", "job_id", "refined_prompt", "approved_at", "approved_sha256"):
        shot.pop(key, None)


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
    for key in ("file", "sha256", "approved_at", "approved_sha256", "verified_subtitles", "render_manifest"):
        ep.pop(key, None)
    ep["subtitle_status"] = "needs_retiming"


def reject(root, kind, item_id, version, feedback, shot_ids=None):
    if not feedback or not feedback.strip():
        raise ValueError("返工意见不能为空")
    with run_lock(root), transaction(root) as s:
        if any(j["accounting"] in {"reserved", "unknown"} for j in s["jobs"].values()):
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
            if shot_ids and not set(shot_ids) <= valid_ids:
                raise ValueError("返工镜头 ID 不属于本集")
            idx = s["episodes"].index(item)
            _invalidate_episode(item, feedback, shot_ids)
            for later in s["episodes"][idx + 1:]:
                if any(sh.get("file") for sh in later["shots"]):
                    _invalidate_episode(later, "前集已修改，连续性需重建：" + feedback)
        elif kind == "shot":
            if item["status"] != "review_required":
                raise GateError("镜头不在人工复核阶段")
            _invalidate_shot(item, feedback)
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
        for key in ("approved_at", "approved_sha256"):
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


def reserve(root, job_id, kind, item_id, amount):
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
        event(s, "call_reserved", id=job_id, amount=amount, kind=kind)
    return load_project(root)["jobs"][job_id]


def update_job(root, job_id, **values):
    with transaction(root) as s:
        j = s["jobs"][job_id]
        if set(values) & {"id", "item_id", "kind", "amount", "created_at"}:
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
