"""Validate and preserve a planned sequence within one video generation call.

``episode.shots`` still represents provider tasks. Optional ``subshots`` only
describe the internal camera sequence; they never create extra requests, hide
reference requirements, or claim exact timing from a generative model.
"""
from __future__ import annotations

import math


MAX_SUBSHOT_PROMPT_CHARS = 2000
SEGMENT_PLAN_RULES = (
    "shots 中每一项是一条视频 API 任务，称为生成片段，不等于单个摄影镜头。"
    "先按戏剧节拍规划因果与情绪变化，再决定镜头，而非机械均分。"
    "默认每个生成片段10–15秒，安排2–3个同一场景、动作连续的内部镜头；"
    "不硬凑15秒；复杂动作、独白、关键口型或需要单独控制的表演可用4–9秒短片段，必要时也可较长单镜。"
    "跨场景、明显时间跳跃或资产版本变化须拆成不同生成片段。"
    "用 subshots=[{start,end,prompt}] 表达内部镜头，1–3项，时间以本片段起点为0秒，"
    "按顺序无间隙、无重叠地覆盖整个 duration；每项 prompt 写景别、运镜、人物方位、动作起止和完整所属对白。"
    "最终提示词细化保持当前时间表，不自行改变镜头数量、顺序、start/end或总时长。"
    "用户在同一返修意见中要求改镜头内容、数量或时长时，先走局部重新分镜，GPT按内容拟定4–15秒时长及1–3个内部镜头；"
    "本集总长随该段变化，不自动改其他片段，完整新提示词仍须人工确认后才生成。"
    "时间点是生成目标，不承诺模型精确按秒切换。不要为凑镜头数量拆开一句对白，"
    "也不要靠加速、截断或跨生成片段重复来容纳超长对白。"
    "内部镜头保持同一场景与空间轴线、人物服装和道具、已批准声线及环境声连续性；"
    "切镜不重复对白或动作，环境底声连续，动作音效只出现一次。"
    "不在视频中生成配乐或字幕，配乐和字幕在整集后期统一处理。"
    "父生成片段 asset_ids 必须列出内部镜头实际需要的全部已定义资产；"
    "引用声音仍须同时引用绑定角色，不能用文字描述代替真实参考素材。"
    "旧计划可省略 subshots 并按原流程继续，不自动拆镜，不得自动改写正在制作或已有素材的计划。"
    "仅当用户选择返工时，重新生成整个片段；减少任务次数不等于降低费用，费用仍按实际时长、分辨率、参考输入和供应商计费核算。"
)


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def validate_subshots(shot) -> None:
    """Validate exact adjoining intervals without mutating or normalizing them."""
    if not isinstance(shot, dict):
        raise ValueError("生成片段必须为对象。")
    if "subshots" not in shot:
        return
    duration = shot.get("duration")
    if not _finite_number(duration) or not 4 <= duration <= 15 or int(duration) != duration:
        raise ValueError("生成片段时长必须为4–15秒整数。")
    subshots = shot["subshots"]
    if not isinstance(subshots, list) or not 1 <= len(subshots) <= 3:
        raise ValueError("生成片段的 subshots 必须包含1–3个内部镜头；旧计划可省略此字段，不能填写空数组。")
    previous_end = 0
    for subshot in subshots:
        if not isinstance(subshot, dict) or set(subshot) != {"start", "end", "prompt"}:
            raise ValueError("生成片段的每个内部镜头只能且必须包含 start、end、prompt。")
        start, end = subshot["start"], subshot["end"]
        if not _finite_number(start) or not _finite_number(end):
            raise ValueError("生成片段的内部镜头时间须为有限数字，不能使用布尔值。")
        if start != previous_end or not start < end <= duration:
            raise ValueError("生成片段的内部镜头须从0秒连续相接，无间隙或重叠，每项 end 大于 start。")
        prompt = subshot["prompt"]
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_SUBSHOT_PROMPT_CHARS or "\x00" in prompt:
            raise ValueError("生成片段的内部镜头 prompt 须为非空文本，最多2000字，不能包含空字符。")
        previous_end = end
    if previous_end != duration:
        raise ValueError("生成片段的内部镜头时间表必须完整覆盖 duration，最后的 end 必须等于生成片段时长。")


def _seconds(value):
    # JSON numbers are already validated; preserve decimal precision rather
    # than rounding a boundary into a different approved schedule.
    return str(int(value)) if int(value) == value else str(value)


def timeline_prompt(shot) -> str:
    """Append the approved schedule after refinement; legacy shots add nothing."""
    validate_subshots(shot)
    if "subshots" not in shot:
        return ""
    lines = [
        f"生成片段内部镜头时间表（总时长{_seconds(shot['duration'])}秒，一次连续生成）：",
        "以下秒点是构图与切镜的生成目标，不代表模型能精确到秒或逐帧执行；保持已批准的镜头数量、顺序、起止时间与剧情因果。",
    ]
    for index, subshot in enumerate(shot["subshots"], 1):
        lines.append(f"内部镜头{index}，{_seconds(subshot['start'])}–{_seconds(subshot['end'])}秒：{subshot['prompt'].strip()}")
    lines.extend([
        "各内部镜头的对白必须在所属时段内完整、自然说完，不加速、不截断、不跨生成片段重复；不擅自增加或改写原定对白。",
        "内部切镜保持同一场景与空间轴线，人物身份、服装和道具连续，使用已提供的真实人物/场景及声音参考；不能以文字替代参考素材。",
        "切镜不改变角色声线，不重复动作或对白，环境底声自然承接，动作音效只出现一次；不生成背景配乐，不添加画面字幕。配乐及字幕留到整集后期统一处理。",
    ])
    return "\n".join(lines)
