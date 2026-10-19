"""Human choices before an asset's first generation, with immutable references.

These helpers never call a model or mutate project state. The ``core`` module
owns choices and uploads; production must use the verified reference paths.
"""
from __future__ import annotations

import base64
import binascii
import copy
import json
import re
import subprocess


MAX_REFERENCE_BYTES = 8 * 1024 * 1024
MAX_REFERENCES = 9
MAX_DESCRIPTION_CHARS = 4000
_MEDIA_JOBS = {"image", "voice"}


def empty_input():
    return {"mode": None, "description": "", "references": [],
            "revision": 0, "confirmed": False}


def input_state(asset):
    """Return a detached, backwards-compatible input record for the dashboard."""
    value = empty_input()
    if isinstance(asset.get("creation_input"), dict):
        value.update(copy.deepcopy(asset["creation_input"]))
    return value


def has_started(state, asset):
    """Do not retroactively gate or change existing assets and provider calls."""
    return bool(asset.get("file") or asset.get("history") or asset.get("version", 0) > 0
                or any(job.get("kind") in _MEDIA_JOBS and job.get("item_id") == asset["id"]
                       for job in state.get("jobs", {}).values()))


def confirmed(asset):
    data = input_state(asset)
    if data.get("confirmed") is not True or data.get("mode") not in {"gpt", "custom"}:
        return False
    if asset.get("kind") == "voice" and data.get("references"):
        return False
    if data["mode"] == "gpt":
        return not data.get("description") and not data.get("references")
    return bool(isinstance(data.get("description"), str) and data["description"].strip()
                or data.get("references"))


def pending_assets(state):
    from .core import is_internal_voice_source
    return [asset for asset in state.get("assets", [])
            if not is_internal_voice_source(asset)
            and not has_started(state, asset) and not confirmed(asset)]


def compose_prompt(asset, style=""):
    """Keep the original GPT plan while adding exact human preferences.

    There is no second GPT rewrite: the original plan already establishes the
    story identity, and extra model calls could paraphrase away a user's choice.
    Voice generation still runs this through its stable-trait allowlist.
    """
    data = input_state(asset)
    original = asset.get("prompt", "")
    if asset.get("kind") == "voice":
        if confirmed(asset) and data["mode"] == "custom":
            return original + "\n用户指定的稳定声线属性：" + data["description"].strip()
        return original
    parts = [original]
    if style:
        parts.append("整体画风：" + style)
    if confirmed(asset) and data["mode"] == "custom":
        parts.append("保留原剧本中的角色身份、年龄关系、场景用途和时代设定；"
                     "具体外貌、服饰、空间视觉细节以用户下述描述和所附参考图为优先。")
        if data["description"].strip():
            parts.append("用户的资产描述：" + data["description"].strip())
        if data["references"]:
            parts.append("所附图片是用户为该资产提供的真实视觉参考，参考其相关外形与视觉细节，"
                         "结合上述剧本身份和整体画风生成该资产。")
    if asset.get("feedback", "").strip():
        parts.append("本次返工意见（优先于此前存在冲突的视觉描述）：" + asset["feedback"].strip())
    return "\n".join(parts)


def decode_reference(data_url):
    """Accept only a bounded, exact image data URL; never a filename or URL."""
    if not isinstance(data_url, str) or len(data_url) > ((MAX_REFERENCE_BYTES + 2) // 3) * 4 + 64:
        raise ValueError("参考图片每张不能超过 8 MiB。")
    match = re.fullmatch(r"data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/]*={0,2})", data_url)
    if not match:
        raise ValueError("参考图片仅支持 PNG、JPEG 或 WebP 的有效图片数据。")
    try:
        content = base64.b64decode(match[2], validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("参考图片编码无效，请重新粘贴或选择图片。") from None
    if not content or len(content) > MAX_REFERENCE_BYTES:
        raise ValueError("参考图片不能为空，每张不能超过 8 MiB。")
    mime = match[1]
    signatures = {"image/png": content.startswith(b"\x89PNG\r\n\x1a\n"),
                  "image/jpeg": content.startswith(b"\xff\xd8\xff"),
                  "image/webp": len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP"}
    if not signatures[mime]:
        raise ValueError("图片实际格式与声明类型不一致，请使用 PNG、JPEG 或 WebP 图片。")
    return content, mime, {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[mime]


def validate_image_file(path, mime):
    """Bound dimensions, then require one actual decoded local image frame."""
    def run(args):
        try:
            result = subprocess.run(args, capture_output=True, text=True,
                                    stdin=subprocess.DEVNULL, timeout=15, check=False)
        except FileNotFoundError:
            raise ValueError("验证参考图片需要 FFmpeg（包含 ffprobe），请先安装。") from None
        except subprocess.TimeoutExpired:
            raise ValueError("参考图片解码超时，请使用较小的静态图片。") from None
        if result.returncode:
            raise ValueError("参考图片无法完整解码，请重新导出为有效的 PNG、JPEG 或 WebP。")
        return result.stdout
    raw = run(["ffprobe", "-v", "error", "-max_alloc", "67108864", "-protocol_whitelist", "file,pipe",
               "-show_entries", "stream=codec_type,codec_name,width,height,nb_frames", "-of", "json", str(path)])
    try:
        streams = json.loads(raw)["streams"]
        stream = streams[0]
        width, height = int(stream["width"]), int(stream["height"])
        codec = {"image/png": "png", "image/jpeg": "mjpeg", "image/webp": "webp"}[mime]
        if (len(streams) != 1 or stream.get("codec_type") != "video" or stream.get("codec_name") != codec
                or not 1 <= width <= 8192 or not 1 <= height <= 8192 or width * height > 40000000
                or stream.get("nb_frames", "1") not in {"1", "N/A", None}):
            raise ValueError
    except (KeyError, IndexError, TypeError, ValueError):
        raise ValueError("请提供有效的静态参考图：单边不超过 8192 像素，总像素不超过 4000 万。") from None
    run(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-max_alloc", "67108864", "-err_detect", "explode",
         "-protocol_whitelist", "file,pipe", "-threads", "1", "-i", str(path), "-map", "0:v:0",
         "-frames:v", "1", "-threads", "1", "-f", "null", "-"])


def reference_paths(root, asset):
    """Return only confirmed custom references after checking file integrity."""
    from .core import GateError, digest, safe_path
    data = input_state(asset)
    if not confirmed(asset) or data["mode"] != "custom":
        return []
    references = data.get("references", [])
    if not isinstance(references, list) or len(references) > MAX_REFERENCES:
        raise GateError("资产参考图片记录无效，请重新确认生成方式。")
    paths = []
    for ref in references:
        try:
            if not isinstance(ref, dict) or ref.get("mime") not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError
            path = safe_path(root, ref["file"])
            directory = safe_path(root, "inputs/assets/" + asset["id"])
            if (not path.is_relative_to(directory) or not path.is_file()
                    or path.stat().st_size != ref["bytes"] or digest(path) != ref["sha256"]):
                raise ValueError
        except (KeyError, TypeError, ValueError, OSError):
            raise GateError("已确认的参考图片缺失或内容已改变，生成已停止；请重新上传并确认。") from None
        paths.append(path)
    return paths
