"""A private, localhost-only review room. No remote assets or dependencies."""
from __future__ import annotations

import fcntl
import hashlib
import json
import mimetypes
import re
import secrets
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .regroup import PUBLIC_ERRORS as SEGMENT_ERRORS


MAX_BODY = 32_768
MAX_REFERENCE_BODY = 12 * 1024 * 1024
INPUT_ERRORS = frozenset({
    "这张参考图片已经上传，无需重复添加。",
    "每项资产最多上传 9 张参考图片。",
    "参考图片每张不能超过 8 MiB。",
    "参考图片不能为空，每张不能超过 8 MiB。",
    "参考图片仅支持 PNG、JPEG 或 WebP 的有效图片数据。",
    "图片实际格式与声明类型不一致，请使用 PNG、JPEG 或 WebP 图片。",
    "参考图片无法完整解码，请重新导出为有效的 PNG、JPEG 或 WebP。",
    "请提供有效的静态参考图：单边不超过 8192 像素，总像素不超过 4000 万。",
    "验证参考图片需要 FFmpeg（包含 ffprobe），请先安装。",
    "由 GPT 决定时请清空自定义描述和参考图，或选择由你提供描述。",
    "请填写资产描述，或提供至少一张参考图片。",
    "资产输入已更新，请刷新网页后重新确认。",
    "尚有在途或结果不明的调用，请等待落盘或完成对账后再修改资产输入。",
    "请先批准新的分段方案，或选择保留原分段，再继续制作。",
})
MEDIA_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".m4v", ".mov", ".webm", ".wav", ".mp3", ".m4a", ".ogg", ".srt", ".vtt"})


def _core():
    from . import core
    return core


def _run(project_dir: Path) -> dict:
    from .pipeline import run
    return run(project_dir)


def _pick(value: dict, keys: tuple[str, ...]) -> dict:
    return {key: value[key] for key in keys if key in value}


def _public_cost_sync_result(value: dict) -> dict:
    """Expose counts and bounded job identifiers, never arbitrary exceptions."""
    if not isinstance(value, dict) or not isinstance(value.get("errors"), list):
        raise ValueError("费用同步摘要无效")
    result = {}
    for key in ("synced", "skipped", "unavailable"):
        count = value.get(key)
        if type(count) is not int or count < 0:
            raise ValueError("费用同步摘要无效")
        result[key] = count
    result["error_count"] = len(value["errors"])
    errors = []
    for item in value["errors"][:100]:
        job_id = item.get("job_id", "") if isinstance(item, dict) else ""
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", job_id):
            job_id = ""
        errors.append({"job_id": job_id, "error": "该调用用量暂时无法同步，请检查任务 ID、权限或服务状态。"})
    result["errors"] = errors
    return result


def _version(value):
    return value if type(value) is int and value >= 0 else 0


def _valid_sha(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _public_timestamps(record):
    from .progress import _date
    return {key: record[key] for key in ("created_at", "completed_at")
            if isinstance(record.get(key), str) and len(record[key]) <= 64 and _date(record[key])}


def _public_video_path(value, prefix):
    if not isinstance(value, str) or len(value) > 512 or not value.startswith(prefix + "/") or "\\" in value or "\x00" in value:
        return False
    parts = value.split("/")
    return (not any(part in {"", ".", ".."} for part in parts)
            and Path(value).suffix.lower() in {".mp4", ".m4v", ".mov", ".webm"})


def _file_version(shot, file):
    match = re.fullmatch(re.escape(str(shot.get("id", ""))) + r"-v([1-9]\d*)\.(?:mp4|m4v|mov|webm)",
                         Path(file).name) if isinstance(file, str) else None
    return int(match[1]) if match else None


def _current_shot_artifact(shot):
    if not shot.get("file") or not shot.get("sha256"):
        return False
    file_version = _file_version(shot, shot["file"])
    return file_version is None or file_version == _version(shot.get("version"))


def _public_shot_history(shot):
    history, seen = [], set()
    for record in shot.get("history", []):
        if (not isinstance(record, dict) or not _public_video_path(record.get("file"), "media")
                or not _valid_sha(record.get("sha256"))):
            continue
        identity = (record["file"], record["sha256"])
        if identity in seen:
            continue
        seen.add(identity)
        item = {"file": record["file"], **_public_timestamps(record)}
        item["version"] = _version(record.get("version")) or _file_version(shot, record["file"]) or 0
        history.append(item)
    return history


def _public_rough_cuts(episode):
    """Expose explicit cut files and distinguish current from obsolete footage."""
    output, seen = [], set()
    shots = episode.get("shots", [])
    sources = [(record, False) for record in episode.get("rough_cuts", [])]
    sources += [(record, True) for record in episode.get("reviews", [])]
    for record, legacy in sources:
        if (not isinstance(record, dict) or not _public_video_path(record.get("file"), "exports")
                or not _valid_sha(record.get("sha256")) or not _valid_sha(record.get("key"))):
            continue
        identity = (record["file"], record["sha256"])
        if identity in seen:
            continue
        seen.add(identity)
        count = record.get("shot_count")
        manifest = record.get("shots")
        current = False
        public_manifest = []
        if not legacy:
            valid = (type(count) is int and 0 < count <= len(shots) and isinstance(manifest, list)
                     and len(manifest) == count and all(isinstance(row, dict) and isinstance(row.get("id"), str)
                         and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", row["id"])
                         and type(row.get("version")) is int and row["version"] > 0
                         and _valid_sha(row.get("sha256")) for row in manifest))
            if valid:
                public_manifest = [_pick(row, ("id", "version")) for row in manifest]
                current = all(row["id"] == shot.get("id") and row["version"] == _version(shot.get("version"))
                              and row["sha256"] == shot.get("sha256") and _current_shot_artifact(shot)
                              for row, shot in zip(manifest, shots[:count]))
        else:
            match = re.fullmatch(re.escape(str(episode.get("id", ""))) + r"-rough-([1-9]\d*)-([0-9a-f]{16})\.mp4",
                                 Path(record["file"]).name)
            count = int(match[1]) if match else 0
            if match and 0 < count <= len(shots) and match[2] == record["key"][:16]:
                originals = shots[:count]
                if all(_valid_sha(shot.get("sha256")) and _current_shot_artifact(shot) for shot in originals):
                    calculated = hashlib.sha256("-".join(shot["sha256"] for shot in originals).encode()).hexdigest()
                    current = calculated == record["key"]
                    if current:
                        public_manifest = [{"id": shot["id"], "version": _version(shot.get("version"))} for shot in originals]
        item = {"key": record["key"], "file": record["file"], **_public_timestamps(record)}
        item.update(shot_count=count if type(count) is int and 0 < count <= len(shots) else 0,
                    current=current, legacy=legacy, shots=public_manifest)
        output.append(item)
    return output


def _shot_progress(state, episode, shot, episode_status, stamp):
    """Show durable phases and elapsed time, never invented provider percent."""
    from .progress import _elapsed
    version = _version(shot.get("version"))
    artifact = _current_shot_artifact(shot)
    status = shot.get("status")
    job = state.get("jobs", {}).get(shot.get("job_id"), {})
    job = job if isinstance(job, dict) and job.get("kind") == "video" and job.get("item_id") == shot.get("id") else {}
    hint = episode.get("progress_stage", {})
    hint = hint if isinstance(hint, dict) and hint.get("shot_id") == shot.get("id") else {}
    phase = hint.get("stage")
    job_phase = {"submitting": "shot_submission", "submitted": "shot_generation",
                 "download_pending": "shot_download", "completed": "shot_checks"}.get(job.get("state"))
    labels = {"shot_preparation": "正在整理参考与编写提示词", "shot_submission": "正在提交生成请求",
              "shot_generation": "视频生成中", "shot_download": "视频已生成，正在下载",
              "shot_checks": "已生成，正在检查技术参数", "shot_review": "已生成，正在检查画面与衔接"}
    active = job.get("state") in {"submitting", "submitted", "download_pending"}
    result = {"stage": "queued", "label": "待返工" if version else "待生成", "version": version,
              "target_version": version, "elapsed_seconds": None, "stage_elapsed_seconds": None,
              "elapsed_basis": "生成任务用时", "completed": artifact, "accepted": status == "accepted" and artifact,
              "is_working": False, "waiting_for_human": False, "has_active_task": active}
    if status == "accepted" and artifact:
        result.update(stage="accepted", label="已生成，检查通过")
    elif status == "review_required" and artifact:
        result.update(stage="review_required", label="已生成，等待人工复核", waiting_for_human=True)
    elif job.get("state") == "uncertain" or job.get("accounting") == "unknown":
        result.update(stage="unknown", label="调用结果待核实", waiting_for_human=True)
    elif job.get("state") == "failed":
        result.update(stage="failed", label="生成失败，等待处理", waiting_for_human=True)
    else:
        if status == "reviewing" and artifact:
            phase = "shot_review"
        elif job_phase and not (job.get("state") == "completed" and phase in {"shot_checks", "shot_review"}):
            phase = job_phase
        elif phase not in labels and episode_status.get("current_shot") == shot.get("id"):
            phase = episode_status.get("stage")
        if phase in labels:
            result.update(stage=phase, label=labels[phase], is_working=True)
            if phase == "shot_preparation":
                result["target_version"] = version + 1
        if state.get("paused") or state.get("status") == "paused":
            if result["is_working"] or active:
                result.update(stage="paused", label="已暂停跟踪；原任务可能仍在生成" if active else "已暂停", is_working=False)
        elif state.get("status") == "blocked" and result["is_working"]:
            result.update(stage="blocked", label="制作已停止，请查看项目提示", is_working=False, waiting_for_human=True)
    # A completed provider job is frozen at its own update time. Subsequent
    # episode work must not keep increasing an earlier shot's displayed cost in time.
    if job.get("created_at"):
        end = job.get("updated_at") if job.get("state") in {"completed", "failed", "uncertain"} else stamp
        result["elapsed_seconds"] = _elapsed(job["created_at"], end)
    elif phase == "shot_preparation" and result["is_working"]:
        result["elapsed_seconds"] = _elapsed(hint.get("started_at"), stamp)
        result["elapsed_basis"] = "准备阶段用时"
    if result["is_working"]:
        stage_start = hint.get("started_at") if hint.get("stage") == result["stage"] else job.get("updated_at") or job.get("created_at")
        result["stage_elapsed_seconds"] = _elapsed(stage_start, stamp)
    return result


def public_state(state: dict) -> dict:
    """Keep credentials, source novel, provider responses and settings off the UI API."""
    result = _pick(state, ("id", "title", "mode", "status", "paused", "created_at", "updated_at"))
    result["plan"] = _pick(state.get("plan", {}), ("status", "version", "style", "ratio", "feedback"))
    if state.get("last_error"):
        message = str(state["last_error"])[:1500]
        message = re.sub(r"(?i)\b(?:sk|ark)[-_][A-Za-z0-9_-]{12,}", "[已隐藏凭据]", message)
        message = re.sub(r"(?i)(api[_ -]?key|authorization|secret|token)\s*[:=]\s*[^\s,;]+", r"\1=[已隐藏]", message)
        result["last_error"] = message
    from .billing import summarize_costs
    budget = state.get("budget", {})
    result["budget"] = _pick(budget, ("limit", "spent", "reserved", "unknown", "currency", "available"))
    result["budget"].update(summarize_costs(state.get("jobs", {}), budget.get("currency", "CNY")))
    from .asset_inputs import input_state, has_started, confirmed
    from .progress import episode_progress
    from .core import now
    result["assets"] = []
    for asset in state.get("assets", []):
        item = _pick(asset, ("id", "kind", "name", "prompt", "status", "version", "file", "audio_file", "feedback", "audition_profile"))
        data = input_state(asset)
        item["creation_input"] = _pick(data, ("mode", "description", "revision", "confirmed"))
        item["creation_input"]["references"] = [_pick(ref, ("id", "file", "mime", "bytes")) for ref in data.get("references", [])]
        item["input_editable"] = not has_started(state, asset)
        item["input_required"] = item["input_editable"] and not confirmed(asset)
        result["assets"].append(item)
    episodes = []
    stamp = now()
    for episode in state.get("episodes", []):
        item = _pick(episode, ("id", "title", "summary", "status", "version", "file", "feedback", "review", "subtitle_status", "music_status", "subtitles_burned", "render_warnings", "review_note", "subtitle_file"))
        item["has_subtitles"] = bool(episode.get("render_manifest", {}).get("subtitles"))
        item["progress"] = episode_progress(state, episode, now=stamp)
        item["rough_cuts"] = _public_rough_cuts(episode)
        proposal = episode.get("segment_proposal")
        if isinstance(proposal, dict) and proposal.get("status") == "candidate":
            item["segment_proposal"] = _pick(proposal, ("id", "status", "source_ids", "created_at"))
            item["segment_proposal"]["shots"] = []
            for proposed in proposal.get("shots", []):
                entry = _pick(proposed, ("id", "duration", "prompt", "dialogue", "source_shot_ids"))
                if isinstance(proposed.get("subshots"), list):
                    entry["subshots"] = [_pick(part, ("start", "end", "prompt")) for part in proposed["subshots"]]
                item["segment_proposal"]["shots"].append(entry)
        item["shots"] = []
        for shot in episode.get("shots", []):
            entry = _pick(shot, ("id", "status", "file", "prompt", "duration", "review"))
            if isinstance(shot.get("subshots"), list):
                entry["subshots"] = [_pick(part, ("start", "end", "prompt"))
                                     for part in shot["subshots"] if isinstance(part, dict)]
            entry["version"] = _version(shot.get("version"))
            if not _current_shot_artifact(shot):
                entry.pop("file", None)
            entry["progress"] = _shot_progress(state, episode, shot, item["progress"], stamp)
            entry["history"] = _public_shot_history(shot)
            item["shots"].append(entry)
        episodes.append(item)
    result["episodes"] = episodes
    return result


def _media_paths(state: dict) -> set[str]:
    paths = set()
    items = list(state.get("assets", [])) + list(state.get("episodes", []))
    for episode in state.get("episodes", []):
        items.extend(episode.get("shots", []))
        paths.update(cut["file"] for cut in _public_rough_cuts(episode))
        for shot in episode.get("shots", []):
            paths.update(record["file"] for record in _public_shot_history(shot))
    for item in items:
        for ref in item.get("creation_input", {}).get("references", []):
            if isinstance(ref, dict) and isinstance(ref.get("file"), str):
                paths.add(ref["file"])
        for key in ("file", "audio_file", "subtitle_file"):
            path = item.get(key)
            if isinstance(path, str):
                paths.add(path)
    return paths


def resolve_media(project_dir: Path, requested: str, state: dict) -> Path | None:
    """Only explicitly referenced media can be served; never follow a symlink."""
    if not requested or "\\" in requested or "\x00" in requested:
        return None
    parts = requested.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    if requested not in _media_paths(state) or Path(requested).suffix.lower() not in MEDIA_SUFFIXES:
        return None
    root = project_dir.resolve()
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            return None
    try:
        if not path.is_file() or not path.resolve().is_relative_to(root):
            return None
    except OSError:
        return None
    return path


@dataclass
class _Context:
    project_dir: Path
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    lock: threading.Lock = field(default_factory=threading.Lock)
    mutation_lock: threading.Lock = field(default_factory=threading.Lock)
    cost_sync_lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    worker_error: str | None = None
    cost_sync_running: bool = False
    cost_sync_result: dict | None = None
    cost_sync_error: str | None = None

    def external_run_active(self) -> bool:
        """Probe the existing production lock without creating or changing it."""
        path = self.project_dir / ".run.lock"
        if not path.is_file():
            return False
        try:
            with path.open("r") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                fcntl.flock(lock, fcntl.LOCK_UN)
        except OSError:
            return False
        return False

    def state(self) -> dict:
        state = public_state(_core().load_project(self.project_dir))
        with self.lock:
            state["dashboard"] = {"running": self.running or self.external_run_active(), "error": self.worker_error}
        with self.cost_sync_lock:
            state["dashboard"]["cost_sync"] = {"running": self.cost_sync_running,
                                              "result": self.cost_sync_result, "error": self.cost_sync_error}
        return state

    def launch_cost_sync(self) -> bool:
        """Query usage in a separate worker; never resume or run production."""
        with self.cost_sync_lock:
            if self.cost_sync_running:
                return False
            self.cost_sync_running = True
            self.cost_sync_result = None
            self.cost_sync_error = None

        def worker():
            try:
                from .costs import sync_usage
                result = _public_cost_sync_result(sync_usage(self.project_dir, confirm_no_video_input=False))
                with self.cost_sync_lock:
                    self.cost_sync_result = result
            except Exception:
                with self.cost_sync_lock:
                    self.cost_sync_error = "费用同步未完成，请检查本机方舟凭据、网络和任务权限。"
            finally:
                with self.cost_sync_lock:
                    self.cost_sync_running = False

        try:
            threading.Thread(target=worker, name="norlan-cost-sync", daemon=True).start()
        except Exception:
            with self.cost_sync_lock:
                self.cost_sync_running = False
                self.cost_sync_error = "费用同步未能启动，请稍后重试。"
            raise RuntimeError("费用同步未能启动，请稍后重试。") from None
        return True

    def launch(self) -> bool:
        with self.lock:
            if self.running:
                return False
            state = _core().load_project(self.project_dir)
            if any(isinstance(episode.get("segment_proposal"), dict)
                   and episode["segment_proposal"].get("status") == "candidate"
                   for episode in state.get("episodes", [])):
                raise ValueError("请先批准新的分段方案，或选择保留原分段，再继续制作。")
            self.running = True
            self.worker_error = None
        try:
            _core().set_paused(self.project_dir, False)
        except Exception:
            with self.lock:
                self.running = False
            raise

        def worker():
            try:
                _run(self.project_dir)
            except Exception:
                # Never put raw provider errors, credentials or tracebacks in HTML.
                with self.lock:
                    self.worker_error = "执行已停止。请查看项目状态，或在终端运行 nolan run 检查原因。"
            finally:
                with self.lock:
                    self.running = False
        threading.Thread(target=worker, name="norlan-pipeline", daemon=True).start()
        return True


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    context: _Context


class _Handler(BaseHTTPRequestHandler):
    server: ReviewServer
    server_version = "Nolan"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, *_args):
        pass

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _headers(self, status: int, content_type: str, length: int, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        nonce = self.server.context.nonce
        self.send_header("Content-Security-Policy", f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; img-src 'self'; media-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        for key, value in (extra or {}).items():
            self.send_header(key, str(value))
        self.end_headers()

    def _json(self, status: int, value: dict):
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(payload))
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _error(self, status: int, message: str):
        self._json(status, {"error": message})

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._host_ok():
            return self._error(403, "仅允许本机访问。")
        path = urlsplit(self.path).path
        context = self.server.context
        try:
            if path == "/":
                html = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
                payload = html.replace("{{TOKEN}}", context.token).replace("{{NONCE}}", context.nonce).encode("utf-8")
                self._headers(200, "text/html; charset=utf-8", len(payload))
                if self.command != "HEAD":
                    self.wfile.write(payload)
            elif path == "/api/state":
                self._json(200, context.state())
            elif path.startswith("/media/"):
                state = _core().load_project(context.project_dir)
                media = resolve_media(context.project_dir, unquote(path[len("/media/"):]), state)
                if media is None:
                    return self._error(404, "媒体不存在或不可访问。")
                self._media(media)
            else:
                self._error(404, "未找到此页面。")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self._error(500, "项目状态暂时无法读取。")

    def _media(self, path: Path):
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        headers = {"Accept-Ranges": "bytes"}
        if path.suffix.lower() in {".srt", ".vtt"}:
            headers["Content-Disposition"] = 'attachment; filename="subtitles' + path.suffix.lower() + '"'
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if not match or not size or not any(match.groups()):
                self._headers(416, "application/octet-stream", 0, {"Content-Range": f"bytes */{size}"})
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                start = max(size - int(last), 0)
            if start > end or start >= size:
                self._headers(416, "application/octet-stream", 0, {"Content-Range": f"bytes */{size}"})
                return
            status = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        length = max(0, end - start + 1)
        self._headers(status, mimetypes.guess_type(path.name)[0] or "application/octet-stream", length, headers)
        if self.command == "HEAD":
            return
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                block = stream.read(min(65_536, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def do_POST(self):
        if not self._host_ok():
            return self._error(403, "仅允许本机访问。")
        path = urlsplit(self.path).path
        if path not in {"/api/action", "/api/asset-reference"}:
            return self._error(404, "未找到此操作。")
        context = self.server.context
        if not secrets.compare_digest(self.headers.get("X-Norlan-Token", "").encode("utf-8"), context.token.encode("ascii")):
            return self._error(403, "会话已失效，请刷新页面。")
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + self.headers.get("Host", ""):
            return self._error(403, "不允许跨站操作。")
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            return self._error(415, "操作需要 JSON 数据。")
        if self.headers.get("Transfer-Encoding"):
            return self._error(400, "不支持此请求格式。")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._error(400, "请求长度无效。")
        limit = MAX_REFERENCE_BODY if path == "/api/asset-reference" else MAX_BODY
        if length <= 0 or length > limit:
            return self._error(413, "请求过大或为空。")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError
            action = data.get("action")
            with context.mutation_lock:
                if path == "/api/asset-reference" or action in {"asset_input", "remove_asset_reference"}:
                    item_id, version, revision = data.get("id"), data.get("version"), data.get("input_revision")
                    if (not isinstance(item_id, str) or not 0 < len(item_id) <= 128
                            or type(version) is not int or version < 0
                            or type(revision) is not int or revision < 0):
                        raise ValueError
                    if path == "/api/asset-reference":
                        _core().add_asset_reference(context.project_dir, item_id, version, revision, data.get("data_url"))
                    elif action == "asset_input":
                        _core().set_asset_input(context.project_dir, item_id, version, revision, data.get("mode"), data.get("description", ""))
                    else:
                        _core().remove_asset_reference(context.project_dir, item_id, version, revision, data.get("reference_id"))
                    return self._json(200, context.state())
                if action == "asset_inputs_gpt_remaining":
                    version = data.get("plan_version")
                    if type(version) is not int or version < 1:
                        raise ValueError
                    _core().set_remaining_asset_inputs_gpt(context.project_dir, version)
                    return self._json(200, context.state())
                if action == "segment_plan":
                    from .regroup import decide_segments
                    item_id, proposal_id, approve = data.get("id"), data.get("proposal_id"), data.get("approve")
                    if (not isinstance(item_id, str) or not 0 < len(item_id) <= 64
                            or not isinstance(proposal_id, str) or not 0 < len(proposal_id) <= 64
                            or type(approve) is not bool):
                        raise ValueError
                    decide_segments(context.project_dir, item_id, proposal_id, approve)
                    return self._json(200, context.state())
                if action == "sync_costs":
                    if not context.launch_cost_sync():
                        return self._error(409, "费用正在同步，请稍候，无需重复启动。")
                    return self._json(202, context.state())
                if action == "run":
                    if not context.launch():
                        return self._error(409, "制作正在进行，无需重复启动。")
                    return self._json(202, context.state())
                if action == "pause":
                    if type(data.get("paused")) is not bool:
                        raise ValueError
                    _core().set_paused(context.project_dir, data["paused"])
                elif action in {"approve", "reject", "captions"}:
                    kind, item_id, version = data.get("kind"), data.get("id"), data.get("version")
                    if kind not in {"plan", "asset", "episode", "shot"} or not isinstance(item_id, str) or not 0 < len(item_id) <= 128:
                        raise ValueError
                    if type(version) is not int or version < 1:
                        raise ValueError
                    if action == "approve":
                        _core().approve(context.project_dir, kind, item_id, version)
                    else:
                        feedback, shot_ids = data.get("feedback"), data.get("shot_ids")
                        if not isinstance(feedback, str) or not feedback.strip() or len(feedback) > 8000:
                            raise ValueError
                        if shot_ids is not None and (not isinstance(shot_ids, list) or len(shot_ids) > 100 or any(not isinstance(shot, str) or not 0 < len(shot) <= 128 for shot in shot_ids)):
                            raise ValueError
                        if action == "captions":
                            if kind != "episode":
                                raise ValueError
                            _core().request_caption_fix(context.project_dir, item_id, version, feedback.strip())
                        else:
                            _core().reject(context.project_dir, kind, item_id, version, feedback.strip(), shot_ids)
                else:
                    raise ValueError
            self._json(200, context.state())
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            message = str(exc)
            self._error(409, message if message in INPUT_ERRORS or message in SEGMENT_ERRORS else "状态已变化或参数无效，请刷新后重试。")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self._error(500, "操作未完成，请刷新后检查项目状态。")


def make_server(project_dir: Path, port: int = 8765) -> ReviewServer:
    """Create an unstarted server, useful for embedding and isolated tests."""
    root = Path(project_dir).expanduser().resolve()
    _core().load_project(root)
    server = ReviewServer(("127.0.0.1", port), _Handler)
    server.context = _Context(root)
    return server


def serve(project_dir: Path, port: int = 8765, open_browser: bool = True) -> None:
    server = make_server(project_dir, port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Nolan 审片室：{url}\n按 Ctrl+C 关闭审片室。")
    try:
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                print("浏览器未自动打开，请复制上方地址访问审片室。")
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if server.context.running:
            _core().set_paused(server.context.project_dir, True)
        server.server_close()
