"""Immutable episode source import; no whole-season model calls or media work."""
from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from . import core


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_scripts(texts):
    """Parse numbered Markdown scripts, retaining their original durations."""
    found = {}
    for origin, text in texts:
        matches = list(re.finditer(r"(?m)^##[ \t]+第(\d{1,3})集[ \t]+([^\n]+)$", text))
        for index, match in enumerate(matches):
            number = int(match[1])
            if not 1 <= number <= 100 or number in found:
                raise core.GateError("剧本集号重复或超出范围；未导入。")
            body = text[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(text)].strip() + "\n"
            duration = re.search(r"(?m)^预计时长[：:][ \t]*约?(\d{1,3})秒[ \t]*$", body)
            if not duration or not 4 <= int(duration[1]) <= 180 or len(body) > 100000:
                raise core.GateError("每集须有明确的原稿预计时长（4–180秒），且正文不超过100000字；未导入。")
            found[number] = {"id": f"ep{number:02}", "number": number, "title": match[2].strip(),
                "text": body, "source_duration": int(duration[1]), "source_sha256": _sha(body),
                "origin": str(origin)}
    if not found or sorted(found) != list(range(1, max(found) + 1)):
        raise core.GateError("须从第1集起连续提供分集剧本；未导入。")
    return [found[number] for number in sorted(found)]


def read_scripts(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise core.GateError("分集剧本文件夹不存在。")
    canonical = folder / "分集剧本"
    files = sorted((canonical if canonical.is_dir() else folder).glob("*.md"))
    texts = []
    for file in files:
        if file.is_symlink() or not file.is_file() or file.stat().st_size > 2 * 1024 * 1024:
            raise core.GateError("剧本文件类型或大小不符合要求；未导入。")
        text = file.read_text(encoding="utf-8-sig")
        if re.search(r"(?m)^##[ \t]+第\d{1,3}集[ \t]+", text):
            texts.append((file, text))
    return parse_scripts(texts)


def import_season(root, folder):
    """Append real script episodes, preserving every existing episode verbatim."""
    root = Path(root).expanduser().resolve()
    scripts = read_scripts(folder)
    with core.run_lock(root):
        state = core.load_project(root)
        if state["plan"].get("status") != "approved":
            raise core.GateError("先批准现有项目规划，再导入后续分集。")
        if any(core.execution_unsettled(job) for job in state["jobs"].values()):
            raise core.GateError("尚有在途或结果不明的调用；请处理后再导入分集。")
        existing = state["episodes"]
        if len(existing) > len(scripts):
            raise core.GateError("导入剧本不能删除已有剧集。")
        for old, script in zip(existing, scripts):
            if old["id"] != script["id"] or old["title"].strip() != script["title"]:
                raise core.GateError("现有剧集与导入剧本的集号或标题不一致；未覆盖已有进度。")
            if old.get("source_sha256") not in (None, script["source_sha256"]):
                raise core.GateError("已导入剧本内容发生变化；未覆盖已绑定的分集来源。")
        previous = state.get("season_source", {})
        identity = core._json_digest([{key: script[key] for key in ("id", "title", "source_sha256", "source_duration")}
                                     for script in scripts])
        if previous.get("sha256") == identity and len(existing) == len(scripts):
            return state
        source_dir = core.safe_path(root, "source/episodes")
        source_dir.mkdir(parents=True, exist_ok=True)
        for script in scripts:
            relative = f"source/episodes/{script['id']}.md"
            target = core.safe_path(root, relative)
            if target.exists() and core.digest(target) != script["source_sha256"]:
                raise core.GateError("项目内分集原文已存在且内容不同；未覆盖。")
        # Writes are immutable and harmless on interruption. State commits only
        # after every expected source file is present and hash-checked.
        for script in scripts:
            target = core.safe_path(root, f"source/episodes/{script['id']}.md")
            if not target.exists():
                with target.open("x", encoding="utf-8") as stream:
                    stream.write(script["text"])
        additions = [{"id": script["id"], "title": script["title"], "summary": "待按本集原稿准备分镜",
            "status": "awaiting_episode_plan", "version": 0, "shots": [], "feedback": "", "history": [],
            "reviews": [], "source_file": f"source/episodes/{script['id']}.md",
            "source_sha256": script["source_sha256"], "source_duration": script["source_duration"]}
            for script in scripts[len(existing):]]
        manifest = {"version": 1, "sha256": identity, "duration_policy": "original_script",
            "episode_count": len(scripts), "imported_at": core.now(),
            "episodes": [{key: script[key] for key in ("id", "title", "source_sha256", "source_duration", "origin")}
                         for script in scripts]}
        with core.transaction(root) as current:
            if current["episodes"] != existing:
                raise core.GateError("导入时项目进度已改变；未覆盖，请重试。")
            current["episodes"].extend(copy.deepcopy(additions))
            current["season_source"] = manifest
            if additions and current.get("status") == "complete":
                current.update(status="ready", last_error=None)
            core.event(current, "season_source_imported", episode_count=len(scripts),
                       added_ids=[episode["id"] for episode in additions], duration_policy="original_script")
    return core.load_project(root)


def source_text(root, *, episode=None, asset=None, shot_id=None, state=None):
    """Resolve one episode source, never concatenate the whole season."""
    state = state if state is not None else core.load_project(root)
    if shot_id is not None:
        episode = next((ep for ep in state["episodes"] if any(sh["id"] == shot_id for sh in ep["shots"])), None)
        if episode is None:
            raise core.GateError("未找到片段所属分集。")
    item = asset if asset is not None else episode or {}
    relative = item.get("source_file") or "source/novel.txt"
    path = core.safe_path(root, relative)
    if item.get("source_sha256") and core.digest(path) != item["source_sha256"]:
        raise core.GateError("分集原文校验失败；请检查保存的原始文件，未调用模型。")
    text = path.read_text(encoding="utf-8")
    return text, _sha(text)
