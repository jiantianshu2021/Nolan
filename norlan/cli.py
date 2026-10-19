"""Terminal entry point. Plain commands also make scripted installs testable."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import __version__, core


def remember(root):
    target = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "norlan"
    target.mkdir(parents=True, exist_ok=True)
    (target / "last-project").write_text(str(Path(root).resolve()), encoding="utf-8")


def last_project():
    p = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "norlan/last-project"
    try:
        root = Path(p.read_text(encoding="utf-8").strip())
        if (root / "project.json").is_file():
            return root
    except OSError:
        pass
    return None


STATUS = {"initialized": "等待策划", "awaiting_plan": "等待批准分集规划", "awaiting_assets": "等待资产审批",
    "awaiting_episode_review": "等待本集审片", "review_required": "需要人工复核镜头",
    "blocked": "等待处理", "paused": "已暂停", "complete": "全部完成", "ready": "可继续",
    "planned": "待生成", "candidate": "待批准", "approved": "已批准", "generating": "生成中",
    "awaiting_review": "待审片", "revision_requested": "待返工", "accepted": "抽样检查通过",
    "pending": "待策划", "rejected": "待修改", "reviewing": "自动检查中"}


def summary(s):
    mode = "DEMO · 合成占位素材" if s["mode"] == "demo" else "真实制作"
    rows = [f"\n{s['title']}  [{mode}]", f"状态：{STATUS.get(s['status'],s['status'])}"]
    b = s["budget"]
    rows.append(f"预算 {b['currency']}：上限 {b['limit']:.2f} / 保守已用 {b['spent']:.2f} / 在途 {b['reserved']:.2f} / 待核实 {b['unknown']:.2f}")
    p = s["plan"]
    rows.append(f"规划 v{p['version']}：{STATUS.get(p['status'],p['status'])}")
    for a in s["assets"]:
        rows.append(f"  资产 {a['id']} · {a['name']} · v{a['version']} · {STATUS.get(a['status'],a['status'])}")
    for ep in s["episodes"]:
        rows.append(f"  {ep['id']} · {ep['title']} · v{ep['version']} · {STATUS.get(ep['status'],ep['status'])}")
        for sh in ep["shots"]:
            if sh["status"] == "review_required":
                rows.append(f"    需复核 {sh['id']} v{sh.get('version',0)}：" + "；".join(sh.get("review", {}).get("issues", [])))
    if s.get("last_error"):
        rows.append("提示：" + s["last_error"])
    if s["status"] == "awaiting_plan":
        rows.append("可说：查看规划 / 规划通过。规划审批后才会生成资产。")
    elif s["status"] == "awaiting_assets":
        rows.append("可说：看一下资产 / hero 通过 / station 改成傍晚。每项单独批准。")
    elif s["status"] == "awaiting_episode_review":
        rows.append("可说：播放本集 / 本集通过 / 第一集的某个镜头需要返工。")
    return "\n".join(rows)


def preview(root, kind="", item_id=""):
    s = core.load_project(root)
    if kind == "plan":
        print(json.dumps({"plan": s["plan"], "assets": s["assets"], "episodes": s["episodes"]}, ensure_ascii=False, indent=2))
        return
    if item_id:
        items = [core._find(s, kind, item_id)]
    elif kind == "asset":
        items = [a for a in s["assets"] if a.get("file") and a["status"] == "candidate"]
    else:
        items = [e for e in s["episodes"] if e.get("file") and e["status"] == "awaiting_review"]
        if not items:
            items = [a for a in s["assets"] if a.get("file") and a["status"] == "candidate"]
    if not items:
        raise ValueError("当前没有可预览的候选素材或成片")
    for item in items:
        if not item.get("file"):
            raise ValueError(f"{item['id']} 当前尚未生成可预览文件，请先查看制作进度")
        path = core.safe_path(root, item["file"])
        if not path.is_file():
            raise ValueError(f"{item['id']} 的预览文件缺失，请先检查项目素材")
        print(f"{item['id']} v{item.get('version',0)}：{path}")
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def doctor():
    findings = {"Norlan": __version__, "Python": sys.version.split()[0]}
    for command in ("ffmpeg", "ffprobe", "codex", "jimeng"):
        findings[command] = shutil.which(command) or "未安装"
    try:
        from .brain import BrainError, resolve_codex
        selected_codex = resolve_codex()
        findings["Norlan 实际使用的 Codex"] = selected_codex
    except BrainError as exc:
        selected_codex = None
        findings["Norlan 实际使用的 Codex"] = str(exc)
    if selected_codex:
        try:
            result = subprocess.run([selected_codex, "login", "status"], text=True, capture_output=True, timeout=15)
            findings["GPT 登录"] = "已有登录" if result.returncode == 0 else "需要 codex login"
        except (OSError, subprocess.TimeoutExpired):
            findings["GPT 登录"] = "无法查询"
    for key in ("ARK_API_KEY", "OPENAI_API_KEY", "VOLC_ACCESS_KEY_ID", "VOLC_SECRET_ACCESS_KEY"):
        findings[key] = "已设置（不显示内容）" if os.environ.get(key) else "未设置"
    print(json.dumps(findings, ensure_ascii=False, indent=2))
    print("即梦 CLI 仅检测安装；其当前文字生视频能力未用于需要实际参考素材的自动生产。")
    return 0 if shutil.which("ffmpeg") and shutil.which("ffprobe") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="norlan", description="Norlan · 在终端对话，逐集监督 AI 漫剧制作")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="检查本地工具与账户配置，不显示密钥")
    for cmd in ("new", "demo"):
        p = sub.add_parser(cmd)
        p.add_argument("project", type=Path)
        if cmd == "new":
            p.add_argument("--novel", required=True, type=Path)
            p.add_argument("--mode", choices=["demo", "live"], default="live")
        p.add_argument("--brief", default="")
    for cmd in ("chat", "studio", "run", "status", "pause", "resume"):
        p = sub.add_parser(cmd)
        p.add_argument("project", type=Path, nargs="?" if cmd == "chat" else None)
        if cmd == "studio":
            p.add_argument("--port", type=int, default=8765)
            p.add_argument("--no-open", action="store_true")
        if cmd == "status":
            p.add_argument("--json", action="store_true")
    for cmd in ("approve", "reject", "preview"):
        p = sub.add_parser(cmd)
        p.add_argument("project", type=Path)
        p.add_argument("kind", choices=["plan", "asset", "episode", "shot"])
        p.add_argument("id")
        if cmd != "preview":
            p.add_argument("--version", required=True, type=int)
        if cmd == "reject":
            p.add_argument("--feedback", required=True)
            p.add_argument("--shots", nargs="*")
    p = sub.add_parser("configure")
    p.add_argument("project", type=Path)
    p.add_argument("--file", type=Path)
    p.add_argument("--budget", type=float)
    p.add_argument("--currency", choices=["CNY", "USD"])
    p = sub.add_parser("reconcile")
    p.add_argument("project", type=Path)
    p.add_argument("job")
    p.add_argument("outcome", choices=["resume", "spent", "released"])
    p.add_argument("--note", required=True)
    p.add_argument("--task-id")
    p = sub.add_parser("import-plan")
    p.add_argument("project", type=Path)
    p.add_argument("file", type=Path)
    p = sub.add_parser("captions", help="只更正本集实际字幕，不重生成视频镜头")
    p.add_argument("project", type=Path)
    p.add_argument("id")
    p.add_argument("--version", required=True, type=int)
    p.add_argument("--correction", required=True)
    args = parser.parse_args(argv)
    try:
        from . import pipeline
        if args.command == "doctor":
            return doctor()
        if args.command in (None, "chat"):
            from .chat import chat
            return chat(getattr(args, "project", None) or last_project())
        root = args.project.expanduser().resolve()
        if args.command in {"new", "demo"}:
            mode = "demo" if args.command == "demo" else args.mode
            novel = "Norlan synthetic workflow demo." if args.command == "demo" else args.novel.expanduser().read_text(encoding="utf-8")
            core.create_project(root, novel, mode)
            if args.brief:
                core.configure(root, {"creative_brief": args.brief})
            remember(root)
            print(summary(pipeline.run(root)))
            print(f"\n进入对话：norlan chat {root}")
            return 0
        if args.command == "studio":
            from .dashboard import serve
            remember(root)
            serve(root, args.port, not args.no_open)
            return 0
        if args.command == "run":
            state = pipeline.run(root)
        elif args.command == "approve":
            state = core.approve(root, args.kind, args.id, args.version)
        elif args.command == "reject":
            state = core.reject(root, args.kind, args.id, args.version, args.feedback, args.shots)
        elif args.command == "configure":
            config = json.loads(args.file.read_text(encoding="utf-8")) if args.file else None
            state = core.configure(root, config, args.budget, args.currency)
        elif args.command == "reconcile":
            state = core.reconcile(root, args.job, args.outcome, args.note, args.task_id)
        elif args.command == "import-plan":
            state = core.install_plan(root, json.loads(args.file.read_text(encoding="utf-8")))
        elif args.command == "captions":
            state = core.request_caption_fix(root, args.id, args.version, args.correction)
            core.set_paused(root, False)
            state = pipeline.run(root)
        elif args.command in {"pause", "resume"}:
            state = core.set_paused(root, args.command == "pause")
        elif args.command == "preview":
            preview(root, args.kind, args.id)
            return 0
        else:
            state = core.load_project(root)
        remember(root)
        if getattr(args, "json", False):
            print(json.dumps(state, ensure_ascii=False, indent=2))
        else:
            print(summary(state))
        return 1 if state["status"] == "blocked" else 0
    except (ValueError, OSError, KeyError) as exc:
        print("Norlan：" + str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已退出。再次运行会从已保存的进度继续；在途供应商任务可能仍在执行。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
