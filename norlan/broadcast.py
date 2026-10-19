"""Editorial defaults for system voices, separate from character dialogue."""
from __future__ import annotations

import copy
import re


EDITORIAL_INSTRUCTION = (
    "系统播报只读剧情必需的关键任务、结果或奖励，不逐条朗读详细规则。"
    "GPT忠实摘选或压缩本次系统原文，不能增加规则、改变数值、否定、条件或因果关系；"
    "完整原文保留在剧情任务框/系统界面上，合理分屏展示，留足阅读时间。"
    "GPT决定内容取舍、出现时点、自然语速和停顿，可跨内部切镜连续播音，"
    "不覆盖人物对白，留出动作与人物反应的空间，不要求用户另行确认具体秒数。"
    "复用已批准的同一画外声线，不生成可见播报员；人物原对白不改，"
    "不增加片段或整集时长、不新增资产、不加配乐。"
    "旧计划或旧instruction要求完整朗读系统文字时，以本条简播规则为准。"
    "内容与时间都写进尚待审批的完整提示词；用户确认后才提交本版视频。"
)


def policy_for(context):
    """Return an effective policy without mutating persisted approvals or plans."""
    raw = (context.get("audio_policy") or {}).get("system_broadcast")
    policy = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    if policy.get("enabled") is not True:
        return policy
    policy["editorial_mode"] = "key_points"
    policy["instruction"] = EDITORIAL_INSTRUCTION
    assets = context.get("assets", context.get("plan", {}).get("assets", []))
    by_id = {asset.get("id"): asset for asset in assets if isinstance(asset, dict)}
    character = by_id.get(policy.get("character_id"), {})
    voice = by_id.get(policy.get("voice_id"), {})
    labels = [character.get("name"), voice.get("audition_speaker"), voice.get("name")]
    name = character.get("name")
    if isinstance(name, str):
        labels.append(re.sub(r"[（(]仅画外声源[）)]$", "", name).strip())
    policy["speaker_labels"] = list(dict.fromkeys(label.strip() for label in labels
                                                 if isinstance(label, str) and label.strip()))
    return policy
