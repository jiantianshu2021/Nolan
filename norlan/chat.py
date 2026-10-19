"""A conversational terminal controller with a deliberately small action set."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shlex
import threading
import time

from . import core, pipeline
from .cli import preview, remember, summary


HELP = """你可以直接用中文：
  开始制作 / 查看进度 / 暂停 / 继续
  查看规划 / 规划通过
  看一下人物 / hero 通过 / hero 的衣服改成白色
  播放第一集 / E001 通过 / E001 的 E001-S001 需要返工，理由是……
  E001 字幕 把“你好”改成“您好”（仅校正字幕，保留视频）
  总预算设为 30 元
演示模式不调用 GPT：用 ID（如 hero、E001）明确指定对象。
退出：/quit。退出会暂停新的提交，已提交的任务可能仍在供应商执行。
详细配置：norlan configure 项目目录 --file settings.json --budget 金额
"""


def context(s):
    items = [{"kind": "plan", "id": "plan", **s["plan"]}]
    for kind, collection in (("asset", s["assets"]), ("episode", s["episodes"])):
        for item in collection:
            items.append({"kind": kind, **{k: item.get(k) for k in ("id", "name", "title", "status", "version")}})
            if kind == "episode":
                items.extend({"kind": "shot", "episode": item["id"], **{k: sh.get(k) for k in
                    ("id", "status", "version", "duration")}} for sh in item["shots"])
    return {"title": s["title"], "mode": s["mode"], "status": s["status"], "items": items,
            "budget": s["budget"], "paused": s["paused"], "last_error": s.get("last_error")}


def _blank(action, **kwargs):
    return {"action": action, "kind": "none", "id": "", "feedback": "", "shot_ids": [],
            "limit": None, "reply": "", **kwargs}


def _is_question_or_hypothesis(message):
    return bool(re.search(r"[?？\"“”‘’]|是否|能否|可否|吗|么|如何|怎么|怎样|什么|为什么|为何|哪里|多少|多久|如果|假如|假设|要是|一旦|万一|例如|比如|举例|示例|是不是|可不可以|能不能|会不会|说过|原话|听说|据说|或者|还是", message, re.I))


def _without_quoted_content(message):
    return re.sub(r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’', "〔内容〕", message)


def _budget_update(message):
    """Only explicit setting commands authorize changing a spending limit."""
    if _is_question_or_hypothesis(message) or re.search(r"不|别|没|暂缓|等等", message):
        return None
    pattern = r"(?:总预算|预算(?:上限)?|上限|budget)\s*(?:设为|设置为|设置成|改为|改成|调整为|调整到|调到|增加到|提高到|降低到|减少到|set\s+to)\s*(\d+(?:\.\d+)?)\s*(美元|美金|USD|CNY|人民币|元)?"
    matches = list(re.finditer(pattern, message, re.I))
    if len(matches) != 1:
        return None
    value, currency = matches[0].groups()
    value = float(value)
    if not math.isfinite(value):
        return None
    unit = "USD" if currency and currency.lower() in {"美元", "美金", "usd"} else "CNY" if currency else None
    return value, unit


def _bound_item(intent, seen, message):
    """Do not let a model redirect an explicit approval to a different asset."""
    identity = intent["kind"], intent["id"]
    matches = [item for item in seen["items"] if (item["kind"], item["id"]) == identity]
    if len(matches) != 1:
        raise ValueError("请明确要审批的资产、分集或镜头 ID")
    mentions = {(item["kind"], item["id"]) for item in seen["items"]
                if re.search(r"(?<![A-Za-z0-9_-])" + re.escape(item["id"]) + r"(?![A-Za-z0-9_-])", message, re.I)}
    if not mentions:
        mentions = {(item["kind"], item["id"]) for item in seen["items"]
                    if any(isinstance(item.get(key), str) and len(item[key]) >= 2 and item[key] in message for key in ("name", "title"))}
        if re.search(r"规划|策划", message):
            mentions.add(("plan", "plan"))
        episodes = [item for item in seen["items"] if item["kind"] == "episode"]
        for index, episode in enumerate(episodes, 1):
            numbers = [str(index)] + (["一二三四五六七八九十"[index - 1]] if index <= 10 else [])
            if any("第" + number + "集" in message for number in numbers):
                mentions.add(("episode", episode["id"]))
        if re.search(r"本集|这集|这一集", message):
            mentions.update(("episode", item["id"]) for item in episodes if item["status"] == "awaiting_review")
    if mentions and identity not in mentions:
        raise ValueError("识别出的对象与你明确提到的对象不一致；请使用准确 ID 重试")
    if not mentions:
        pending = [(item["kind"], item["id"]) for item in seen["items"] if item.get("status") in {"candidate", "awaiting_review", "review_required"}]
        if pending != [identity]:
            raise ValueError("有多个可能的对象，请在修改或审批时明确填写 ID")
    return matches[0]


def _model_command(root, message):
    # Conversation and generation share one durable reservation ledger. Holding
    # the same run lock prevents recovery from treating an active GPT call as lost.
    with core.run_lock(root):
        seen = context(core.load_project(root))
        return pipeline._brain(root, "command", message, seen), seen


def quick_command(message, ctx):
    """Small offline shortcuts; richer creative dialogue is handled by GPT."""
    original = message.strip().strip("。！!")
    msg = original.lower()
    if msg in {"帮助", "/help", "help"}:
        return _blank("help")
    if msg in {"状态", "进度", "查看进度", "status", "/status", "预算", "查看预算"}:
        return _blank("status")
    if msg in {"开始", "开始制作", "开始生成", "run", "/run"}:
        return _blank("run")
    if msg in {"继续", "继续制作", "resume", "/resume"}:
        return _blank("resume")
    if msg in {"暂停", "暂停制作", "pause", "/pause"}:
        return _blank("pause")
    if msg in {"查看规划", "看规划", "规划", "/plan"}:
        return _blank("preview", kind="plan", id="plan")
    if msg in {"看一下人物", "看一下资产", "查看资产", "看人物"}:
        return _blank("preview", kind="asset")
    if msg in {"播放本集", "看本集", "看视频", "播放视频", "预览"}:
        return _blank("preview")
    if msg in {"规划通过", "批准规划", "同意规划"}:
        return _blank("approve", kind="plan", id="plan")
    budget = _budget_update(msg)
    if budget:
        return _blank("budget", limit=budget[0])
    episodes = [item for item in ctx["items"] if item["kind"] == "episode"]
    if msg in {"本集通过", "这一集通过", "这集通过"}:
        pending = [item for item in episodes if item["status"] == "awaiting_review"]
        if len(pending) == 1:
            return _blank("approve", kind="episode", id=pending[0]["id"])
    match = re.fullmatch(r"(播放|查看|看)第([一二三四五六七八九十]|\d+)集", msg)
    if match:
        index = int(match[2]) if match[2].isdigit() else "一二三四五六七八九十".index(match[2]) + 1
        if 1 <= index <= len(episodes):
            return _blank("preview", kind="episode", id=episodes[index - 1]["id"])
    for item in sorted(ctx["items"], key=lambda x: len(x["id"]), reverse=True):
        item_id = item["id"].lower()
        if msg in {item_id + " 通过", item_id + "通过", "批准 " + item_id, "同意 " + item_id}:
            return _blank("approve", kind=item["kind"], id=item["id"])
        if msg in {prefix + space + item_id for prefix in ("播放", "查看", "看") for space in ("", " ")}:
            return _blank("preview", kind=item["kind"], id=item["id"])
        if item["kind"] == "episode":
            match = re.fullmatch(re.escape(item["id"]) + r"\s+字幕\s*[:：]?\s*(.+)", original, re.I)
            if match:
                return _blank("captions", kind="episode", id=item["id"], feedback=match[1])
        match = re.fullmatch(re.escape(item["id"]) + r"\s+(?:返工|修改)\s*[:：]?\s*(.+)", original, re.I)
        if match:
            return _blank("reject", kind=item["kind"], id=item["id"], feedback=match.group(1))
    return None


def execute_intent(root, message, intent, seen):
    """Validate against the state shown before parsing, then invoke core gates."""
    action = intent["action"]
    if action == "captions":
        if intent["kind"] != "episode":
            raise ValueError("字幕校正需要明确指定分集 ID")
        item = _bound_item(intent, seen, message)
        # Quoted replacement text is valid here; questions/hypotheticals still
        # cannot authorize changing a completed episode's subtitle track.
        command_text = _without_quoted_content(message)
        if "字幕" not in message or _is_question_or_hypothesis(command_text) or re.search(r"(?:不要|不用|别)\s*(?:改|修|校|替换)", command_text) or not re.search(r"改|修正|校正|更正|替换", command_text):
            raise ValueError("请明确给出字幕更正，例如：E001 字幕 把你好改成您好")
        return core.request_caption_fix(root, intent["id"], item["version"], intent["feedback"])
    if action in {"approve", "reject"}:
        item = _bound_item(intent, seen, message)
        if action == "approve":
            # Model output itself is never human permission. Require direct,
            # affirmative language in this utterance, not a question/negation.
            if not re.search(r"通过|批准|同意|\bapprove\b", message, re.I) or _is_question_or_hypothesis(message) or re.search(r"不|别|没|未|暂缓|等等|稍后|再说|先等", message):
                raise ValueError("只有明确的同意才会批准，例如：E001 通过")
            return core.approve(root, intent["kind"], intent["id"], item["version"])
        command_text = _without_quoted_content(message)
        if _is_question_or_hypothesis(command_text) or re.search(r"(?:不要|不用|不必|无需|别)\s*(?:再|去|先|对)?\s*(?:修改|改|返工|重做|重来|重新)", command_text) or not re.search(r"改|返工|重做|重来|重新|reject|不通过", command_text, re.I):
            raise ValueError("请明确提出修改意见后再返工")
        return core.reject(root, intent["kind"], intent["id"], item["version"], intent["feedback"], intent["shot_ids"])
    if action == "budget":
        limit = intent.get("limit")
        update = _budget_update(message)
        if not update or type(limit) not in {float, int} or not math.isfinite(limit) or update[0] != limit:
            raise ValueError("请明确填写预算上限，例如：总预算设为 30 元")
        if update[1] and update[1] != seen["budget"]["currency"]:
            raise ValueError("这次预算的币种与账本不一致；请在首次调用前用 configure 明确选择币种")
        return core.configure(root, limit=limit)
    if action in {"pause", "resume", "run"}:
        verbs = {"pause": r"暂停|停下|停止|\bpause\b", "resume": r"继续|恢复|\bresume\b", "run": r"开始|执行|生成|制作|\brun\b"}
        negated = re.search(r"(?:不要|不用|不能|别|不)\s*(?:暂停|停止|停下)", message) if action == "pause" else re.search(r"不|别|没|暂缓", message)
        if _is_question_or_hypothesis(message) or negated or not re.search(verbs[action], message, re.I):
            raise ValueError("请直接说明是否开始、继续或暂停；提问不会改变制作状态")
        if action != "run":
            return core.set_paused(root, action == "pause")
    if action == "preview":
        preview(root, intent["kind"] if intent["kind"] != "none" else "", intent["id"])
    elif action == "help":
        print(HELP)
    elif action == "answer":
        print("Norlan：" + intent["reply"])
    return core.load_project(root)


def _new_from_message(message):
    # Paths are supplied explicitly; no scan of personal files is performed.
    if message.strip().lower() in {"演示", "创建演示", "demo", "/demo"}:
        target = Path.cwd() / ("norlan-demo-" + time.strftime("%Y%m%d-%H%M%S"))
        core.create_project(target, "Synthetic workflow demo", "demo")
        return target
    if message.startswith("/new "):
        parts = shlex.split(message)
        if len(parts) != 2:
            raise ValueError("用法：/new 小说文件路径")
        source = Path(parts[1]).expanduser()
    else:
        match = re.search(r"(?:\"([^\"]+\.(?:txt|md))\"|((?:/|~/)[^\s，。]+\.(?:txt|md)))", message)
        if not match:
            raise ValueError("请提供小说 TXT/MD 的本地路径，例如：用 /Users/me/novel.txt 创建漫剧；或输入“演示”。")
        command_text = message[:match.start()] + message[match.end():]
        if _is_question_or_hypothesis(command_text) or re.search(r"不要|不能|不用|别|不创建", command_text):
            raise ValueError("提问不会创建项目。确认开始时，请输入 /new 小说文件路径")
        if command_text.strip() and not re.search(r"创建|改编|做成|制作|生成", command_text):
            raise ValueError("请明确说明用这个文件创建漫剧，或输入 /new 小说文件路径")
        source = Path(match.group(1) or match.group(2)).expanduser()
    if source.suffix.lower() not in {".txt", ".md"} or not source.is_file():
        raise ValueError("请提供存在的小说 TXT/MD 文件；带空格的路径请用引号包围")
    target = Path.cwd() / ("norlan-project-" + time.strftime("%Y%m%d-%H%M%S"))
    core.create_project(target, source.read_text(encoding="utf-8"), "live")
    core.configure(target, {"creative_brief": message})
    return target


def _requests_new(message):
    if message.startswith("/new ") or message.lower() in {"演示", "创建演示", "demo", "/demo"}:
        return True
    return bool(re.search(r"(?:/|~/)[^\n]+\.(?:txt|md)", message) and re.search(r"创建|改编|做成|制作|生成", message) and not _is_question_or_hypothesis(_without_quoted_content(message)) and not re.search(r"不要|不能|不用|别|不创建", message))


def chat(project_dir=None):
    root = Path(project_dir).expanduser().resolve() if project_dir else None
    worker = None
    lock = threading.Lock()
    print("Norlan · 终端导演助手\n直接用中文交流；/help 查看用法，/quit 保存并退出。")
    if root:
        print(summary(core.load_project(root)))
    else:
        print("还没有制作项目。告诉我小说的本地 TXT/MD 路径，或输入“演示”体验无费用流程。")

    def start():
        nonlocal worker
        if worker and worker.is_alive():
            print("Norlan：正在制作，你可以查询进度或暂停。")
            return
        core.set_paused(root, False)
        def work():
            try:
                state = pipeline.run(root)
                report = "本轮已保存。" + summary(state)
            except Exception as exc:
                report = str(exc) if isinstance(exc, ValueError) else "制作已停止，请查看进度及配置。"
            with lock:
                print("\nNorlan：" + report + "\n你：", end="", flush=True)
        worker = threading.Thread(target=work, daemon=True, name="norlan-production")
        worker.start()
        print("Norlan：开始执行。你可以继续输入“进度”或“暂停”。")

    while True:
        try:
            message = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            message = "/quit"
        if not message:
            continue
        if message.lower() in {"/quit", "quit", "exit", "退出"}:
            if root:
                core.set_paused(root, True)
                remember(root)
            print("Norlan：进度已保存。下次运行 norlan 后说“继续”；在途任务不会被重复提交。" if root else "Norlan：已退出，尚未创建项目。")
            return 0
        try:
            if root is not None and _requests_new(message):
                if worker and worker.is_alive():
                    print("Norlan：当前项目正在制作。请先暂停并等本轮停止，再创建新项目；原项目会完整保留。")
                    continue
                with core.run_lock(root):
                    core.set_paused(root, True)
                created = _new_from_message(message)
                root = created
                remember(root)
                print(f"Norlan：新项目已创建于 {root}，原项目已保留。")
                start()
                continue
            if root is None:
                if message.lower() in {"帮助", "/help", "help"}:
                    print(HELP)
                    continue
                if message.lower() in {"状态", "进度", "查看进度", "status", "/status", "预算", "查看预算"}:
                    print("Norlan：还没有制作项目。提供小说 TXT/MD 路径，或输入“演示”。")
                    continue
                root = _new_from_message(message)
                remember(root)
                print(f"Norlan：项目已创建于 {root}")
                start()
                continue
            state = core.load_project(root)
            seen = context(state)
            intent = quick_command(message, seen)
            if intent is None:
                if state["mode"] == "demo":
                    print("Norlan：演示模式不会调用 GPT。请用明确的 ID 和指令，例如“hero 通过”。\n" + HELP)
                    continue
                if worker and worker.is_alive():
                    print("Norlan：制作正在进行，暂不同时发起新的 GPT 对话。可以输入“进度”“暂停”或预览指令，其他要求请在本轮停止后提出。")
                    continue
                print("Norlan：正在理解你的要求……", flush=True)
                intent, seen = _model_command(root, message)
            result = execute_intent(root, message, intent, seen)
            action = intent["action"]
            if action in {"run", "resume", "approve", "reject", "captions"}:
                start()
            elif action in {"status", "pause", "budget"}:
                print(summary(result))
        except Exception as exc:
            message = str(exc) if isinstance(exc, (ValueError, OSError)) or getattr(exc, "safe_message", False) else type(exc).__name__ + "：未执行，请检查配置"
            print("Norlan：" + message)


if __name__ == "__main__":
    chat()
