"""Byte-preserving Ark continuity assets. Signed URLs remain in memory only."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import tempfile

from . import core, media
from .failures import exception_failure, sanitize_failure


def _created(value):
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, timezone.utc)
        stamp = datetime.fromisoformat(value)
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _age(job, metadata=None):
    value = (metadata or {}).get("created_at") or job.get("origin_video", {}).get("created_at") or job.get("created_at")
    stamp = _created(value)
    if stamp is None or not 0 <= (datetime.now(timezone.utc) - stamp).total_seconds() <= 30 * 86400:
        raise core.GateError("官方参考素材已超过 30 天受信期或生成时间无法核实；未提交新任务。")
    return value


def source_job(root, previous):
    if not core.human_approved_shot(previous):
        raise core.GateError("连续性素材必须来自人工批准的前段。")
    core._check_artifact(root, previous)
    state = core.load_project(root)
    job = state["jobs"].get(previous.get("job_id"), {})
    if (job.get("kind") != "video" or job.get("item_id") != previous["id"]
            or job.get("state") != "completed" or not job.get("task_id")
            or job.get("output") != previous.get("file")):
        raise core.GateError("前段缺少匹配的官方生成任务来源；未使用本地截图替代。")
    original = job.get("origin_video")
    if original and (original.get("sha256") != previous["sha256"]
                     or original.get("task_id") != job["task_id"]):
        raise core.GateError("前段原始视频来源与批准版本不一致；未提交新任务。")
    _age(job)
    return job


def _checked_tail(root, job):
    record = job.get("official_tail")
    if not record:
        return None
    original = job.get("origin_video", {})
    if (record.get("source") != "ark_official_tail" or record.get("task_id") != job.get("task_id")
            or record.get("video_sha256") != original.get("sha256")):
        raise core.GateError("官方尾帧与原视频任务不匹配；未使用其他图片替代。")
    path = core.safe_path(root, record["file"])
    if not path.is_file() or core.digest(path) != record.get("sha256"):
        raise core.GateError("已保存的官方尾帧缺失或内容改变；原记录保留，未覆盖或重新生成。")
    _age(job, record)
    return path


def cache_tail(root, job_id, result, provider):
    """Save an original official tail without changing/re-encoding any media."""
    job = core.load_project(root)["jobs"][job_id]
    if _checked_tail(root, job):
        return job["official_tail"]
    url = result.get("last_frame_url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise core.GateError("原任务未提供官方尾帧；视频原件保留，下一段不会改用本地抽帧。")
    output = core.safe_path(root, job["output"])
    origin = job.get("origin_video", {})
    if not output.is_file() or core.digest(output) != origin.get("sha256"):
        raise core.GateError("原视频哈希无法核实，不能绑定官方尾帧。")
    _age(job, result)
    directory = Path(root) / "media" / "official-tails"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tail-", dir=directory) as temporary:
        pending = Path(temporary) / "original"
        provider.download(url, pending)
        if pending.stat().st_size >= 30_000_000:
            raise core.GateError("官方尾帧原件超过参考图大小限制；未压缩或转换。")
        with pending.open("rb") as stream:
            header = stream.read(8)
        if header.startswith(b"\xff\xd8\xff"):
            suffix = ".jpg"
        elif header == b"\x89PNG\r\n\x1a\n":
            suffix = ".png"
        else:
            raise core.GateError("官方尾帧下载结果不是 JPEG/PNG 原图；未转换或提交替代图片。")
        digest = core.digest(pending)
        destination = directory / (job_id + suffix)
        if destination.exists():
            if core.digest(destination) != digest:
                raise core.GateError("官方尾帧目标文件已有不同内容；未覆盖。")
        else:
            pending.rename(destination)
    record = {"file": str(destination.relative_to(Path(root))), "sha256": digest,
              "task_id": job["task_id"], "video_sha256": origin["sha256"],
              "source": "ark_official_tail", "created_at": _age(job, result)}
    core.update_job(root, job_id, official_tail=record, official_tail_error=None)
    return record


def tail_error(root, job_id, exc):
    detail = (sanitize_failure({"stage": "download", "message": str(exc), "uncertain": False})
              if isinstance(exc, core.GateError) else exception_failure(exc, "download"))
    core.update_job(root, job_id, official_tail_error=detail)


def get_tail(root, previous, provider):
    job = source_job(root, previous)
    existing = _checked_tail(root, job)
    if existing:
        return existing
    try:
        result = provider.video_artifacts(job["task_id"])
        if result.get("status") != "completed":
            raise core.GateError("无法从原任务取回官方尾帧；不会自动重新生成视频。")
        # Legacy originals were downloaded without changing bytes. Bind to the
        # approved immutable video before registering any official tail.
        if not job.get("origin_video"):
            core.update_job(root, job["id"], origin_video={"file": previous["file"],
                "sha256": previous["sha256"], "task_id": job["task_id"],
                "created_at": _age(job, result)})
        cache_tail(root, job["id"], result, provider)
    except Exception as exc:
        tail_error(root, job["id"], exc)
        raise core.GateError("前段官方尾帧暂不可用：" +
                            core.load_project(root)["jobs"][job["id"]]["official_tail_error"]["message"] +
                            " 尚未提交新视频任务。") from None
    return _checked_tail(root, core.load_project(root)["jobs"][job["id"]])


def resolve_video(root, previous, provider):
    """Resolve and verify an existing original video, never create a task."""
    job = source_job(root, previous)
    original = core.safe_path(root, previous["file"])
    duration = media.probe(original)["duration"]
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or not 2 <= duration <= 15
            or original.stat().st_size > 200_000_000):
        raise core.GateError("原始视频参考须为 2–15 秒、至多 200 MB；不会裁剪或压缩原片。")
    result = provider.video_artifacts(job["task_id"])
    url = result.get("video_url") or result.get("url")
    if result.get("status") != "completed" or not isinstance(url, str) or not url.startswith("https://"):
        raise core.GateError("原任务未返回可用的原始视频地址；未提交新任务。")
    _age(job, result)
    expected_model = job.get("billing", {}).get("price", {}).get("model")
    if expected_model and result.get("model") != expected_model:
        raise core.GateError("原视频任务的模型来源与已登记记录不一致。")
    # GET may return an expired URL; querying never renews it. Verify both URL
    # availability and exact bytes before any GPT/video submission.
    with tempfile.TemporaryDirectory(prefix=".verify-original-", dir=Path(root)) as temporary:
        downloaded = Path(temporary) / "original.mp4"
        provider.download(url, downloaded)
        if core.digest(downloaded) != previous["sha256"]:
            raise core.GateError("远程原视频与人工批准版本哈希不同；未提交新任务。")
    return url, duration, {"index": 1, "role": "previous_original_video", "shot_id": previous["id"],
                           "sha256": previous["sha256"], "task_id": job["task_id"],
                           "duration": duration, "file": previous["file"]}
