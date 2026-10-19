"""A private, localhost-only review room. No remote assets or dependencies."""
from __future__ import annotations

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


MAX_BODY = 32_768
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
    result["assets"] = [
        _pick(asset, ("id", "kind", "name", "prompt", "status", "version", "file", "audio_file", "feedback", "audition_profile"))
        for asset in state.get("assets", [])
    ]
    episodes = []
    for episode in state.get("episodes", []):
        item = _pick(episode, ("id", "title", "summary", "status", "version", "file", "feedback", "review", "subtitle_status", "music_status", "subtitles_burned", "render_warnings", "review_note", "subtitle_file"))
        item["has_subtitles"] = bool(episode.get("render_manifest", {}).get("subtitles"))
        item["shots"] = [
            _pick(shot, ("id", "status", "version", "file", "prompt", "duration", "review"))
            for shot in episode.get("shots", [])
        ]
        episodes.append(item)
    result["episodes"] = episodes
    return result


def _media_paths(state: dict) -> set[str]:
    paths = set()
    items = list(state.get("assets", [])) + list(state.get("episodes", []))
    for episode in state.get("episodes", []):
        items.extend(episode.get("shots", []))
    for item in items:
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

    def state(self) -> dict:
        state = public_state(_core().load_project(self.project_dir))
        with self.lock:
            state["dashboard"] = {"running": self.running, "error": self.worker_error}
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
                    self.worker_error = "执行已停止。请查看项目状态，或在终端运行 norlan run 检查原因。"
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
    server_version = "Norlan"
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
        if urlsplit(self.path).path != "/api/action":
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
        if length <= 0 or length > MAX_BODY:
            return self._error(413, "请求过大或为空。")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError
            action = data.get("action")
            with context.mutation_lock:
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
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._error(409, "状态已变化或参数无效，请刷新后重试。")
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
    print(f"Norlan 审片室：{url}\n按 Ctrl+C 关闭审片室。")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if server.context.running:
            _core().set_paused(server.context.project_dir, True)
        server.server_close()
