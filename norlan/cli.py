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
from .providers import ProviderError


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
    "awaiting_prompt_approval": "等待检查提示词并点击生成", "prompt_approved": "提示词已确认，等待提交",
    "awaiting_asset_amendment": "等待确认新增资产方案",
    "asset_generation_failed": "资产生成失败，等待处理",
    "awaiting_asset_input": "等待资产创作选择",
    "awaiting_episode_review": "等待本集审片", "review_required": "等待你审批片段",
    "generation_failed": "视频生成失败，等待处理",
    "blocked": "等待处理", "paused": "已暂停", "complete": "全部完成", "ready": "可继续",
    "planned": "待生成", "candidate": "待批准", "approved": "已批准", "generating": "生成中",
    "awaiting_review": "待审片", "revision_requested": "待返工", "accepted": "片段已接受",
    "pending": "待策划", "rejected": "待修改", "reviewing": "自动检查中"}


def summary(s):
    mode = "DEMO · 合成占位素材" if s["mode"] == "demo" else "真实制作"
    rows = [f"\n{s['title']}  [{mode}]", f"状态：{STATUS.get(s['status'],s['status'])}"]
    b = s["budget"]
    actual = f"{b['actual_cost']:.6f}".rstrip("0").rstrip(".") + " " + b["currency"] if b.get("actual_cost_jobs") else "待对账"
    usage = f"{b['usage_cost']:.6f}".rstrip("0").rstrip(".") + " " + b["currency"] if b.get("usage_cost_jobs") else "暂无可计费用量"
    rows.append(f"已核对实际费用：{actual}；按实际用量计费（未对账）：{usage}")
    rows.append(f"预算 {b['currency']}：上限 {b['limit']:.2f} / 已结束调用占用上限 {b['spent']:.2f} / 在途预留 {b['reserved']:.2f} / 结果待核实 {b['unknown']:.2f}")
    rows.append("用量计费未含折扣、赠额等账单调整；预算占用不等于实际扣款。可用 nolan costs 查看逐调用明细。")
    p = s["plan"]
    rows.append(f"规划 v{p['version']}：{STATUS.get(p['status'],p['status'])}")
    for a in s["assets"]:
        if core.is_internal_voice_source(a):
            continue
        rows.append(f"  资产 {a['id']} · {a['name']} · v{a['version']} · {STATUS.get(a['status'],a['status'])}")
    for ep in s["episodes"]:
        rows.append(f"  {ep['id']} · {ep['title']} · v{ep['version']} · {STATUS.get(ep['status'],ep['status'])}")
        for sh in ep["shots"]:
            if sh["status"] == "generation_failed":
                rows.append(f"    生成失败 {sh['id']} v{sh.get('version',0)}：" + sh.get("generation_error", "请在网页查看失败详情"))
            elif sh["status"] == "review_required":
                rows.append(f"    待你审批 {sh['id']} v{sh.get('version',0)}：" + "；".join(sh.get("review", {}).get("issues", [])))
    if s.get("last_error"):
        rows.append("提示：" + s["last_error"])
    if s["status"] == "awaiting_plan":
        rows.append("可说：查看规划 / 规划通过。批准规划后先在网页选择资产创作方式。")
    elif s["status"] == "awaiting_asset_input":
        rows.append("请先在网页选择“由 GPT 自己决定”或“我来描述”（可粘贴参考图片）；选择前不会开始生成资产。")
    elif s["status"] == "awaiting_assets":
        rows.append("可说：看一下资产 / hero 通过 / station 改成傍晚。每项单独批准。")
    elif s["status"] == "awaiting_episode_review":
        rows.append("可说：播放本集 / 本集通过 / 第一集的某个镜头需要返工。")
    return "\n".join(rows)


def preview(root, kind="", item_id=""):
    s = core.load_project(root)
    if kind == "plan":
        print(json.dumps({"plan": s["plan"], "assets": [a for a in s["assets"] if not core.is_internal_voice_source(a)], "episodes": s["episodes"]}, ensure_ascii=False, indent=2))
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
        if core.is_internal_voice_source(item):
            raise ValueError("此记录是声音的内部用途绑定；请预览对应声音资产，试听后再审批音色。")
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
    findings = {"Nolan": __version__, "Python": sys.version.split()[0]}
    for command in ("ffmpeg", "ffprobe", "codex", "jimeng"):
        findings[command] = shutil.which(command) or "未安装"
    try:
        from .brain import BrainError, resolve_codex
        selected_codex = resolve_codex()
        findings["Nolan 实际使用的 Codex"] = selected_codex
    except BrainError as exc:
        selected_codex = None
        findings["Nolan 实际使用的 Codex"] = str(exc)
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
    parser = argparse.ArgumentParser(prog="nolan", description="Nolan · 在终端对话，逐集监督 AI 漫剧制作")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--resume", action="store_true", help="继续最近一次 Nolan GPT 对话")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="检查本地工具与账户配置，不显示密钥")
    sub.add_parser("connection", help="查看 GPT 订阅/API 连接选择")
    setup_parser = sub.add_parser("setup", help="选择 ChatGPT 订阅或 OpenAI API 连接")
    setup_parser.add_argument("--provider", choices=["chatgpt", "openai"])
    setup_parser.add_argument("--model", help="明确指定模型 ID；API 模式必填")
    for cmd in ("new", "demo"):
        p = sub.add_parser(cmd)
        p.add_argument("project", type=Path)
        if cmd == "new":
            p.add_argument("--novel", required=True, type=Path)
            p.add_argument("--mode", choices=["demo", "live"], default="live")
        p.add_argument("--brief", default="")
    for cmd in ("chat", "console", "studio", "run", "status", "pause", "resume"):
        p = sub.add_parser(cmd)
        p.add_argument("project", type=Path, nargs="?" if cmd in {"chat", "console"} else None)
        if cmd == "chat":
            p.add_argument("--resume", action="store_true", dest="chat_resume", help="继续最近一次 GPT 对话")
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
    p = sub.add_parser("costs", help="查看费用、仅查询现有任务用量或导入逐调用账单")
    p.add_argument("project", type=Path)
    p.add_argument("--sync", action="store_true", help="仅 GET 已有任务的供应商用量，不生成素材")
    p.add_argument("--confirm-no-video-input", action="store_true", help="确认历史任务没有输入参考视频，允许按当前公开价折算；须与 --sync 一起使用")
    p.add_argument("--import-bill", type=Path, help="导入含 job_id/amount/currency/bill_reference/scope=per_call 的 JSON 明细列表")
    p.add_argument("--json", action="store_true", help="输出经过筛选的费用账本 JSON")
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
        if args.resume and args.command not in {None, "chat"}:
            raise ValueError("--resume 用于恢复 GPT 对话；使用 nolan --resume 或 nolan chat --resume。")
        if args.command == "setup":
            from .connection import setup
            return setup(args.provider, args.model)
        if args.command == "connection":
            from .connection import describe_connection
            print(describe_connection())
            return 0
        if args.command in (None, "chat"):
            from .interactive import launch
            return launch(getattr(args, "project", None),
                          resume=args.resume or getattr(args, "chat_resume", False))
        if args.command == "console":
            from .chat import chat
            return chat(args.project or last_project())
        from . import pipeline
        if args.command == "doctor":
            return doctor()
        root = args.project.expanduser().resolve()
        if args.command == "costs":
            from .costs import cost_report, format_report, sync_usage
            if args.confirm_no_video_input and not args.sync:
                raise ValueError("--confirm-no-video-input 必须与 --sync 一起使用。")
            imported = None
            if args.import_bill:
                try:
                    records = json.loads(args.import_bill.expanduser().read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raise ValueError("无法读取账单 JSON；请提供逐调用明细列表，不能导入账户或日期汇总金额。") from None
                core.import_bill_costs(root, records)
                imported = len(records)
            synced = sync_usage(root, args.confirm_no_video_input) if args.sync else None
            report = cost_report(root)
            if imported is not None:
                report["imported_records"] = imported
            if synced is not None:
                report["sync"] = synced
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                if imported is not None:
                    print(f"已核对导入 {imported} 条逐调用账单记录（相同记录重复导入不会重复计费）。")
                if synced is not None:
                    print(f"用量查询：已同步 {synced['synced']}，已跳过 {synced['skipped']}，无可查询用量 {synced['unavailable']}。")
                    for error in synced["errors"]:
                        print(error["job_id"] + "：" + error["error"], file=sys.stderr)
                print(format_report(report))
            return 1 if synced and synced["errors"] else 0
        if args.command in {"new", "demo"}:
            mode = "demo" if args.command == "demo" else args.mode
            novel = "Nolan synthetic workflow demo." if args.command == "demo" else args.novel.expanduser().read_text(encoding="utf-8")
            core.create_project(root, novel, mode)
            if args.brief:
                core.configure(root, {"creative_brief": args.brief})
            remember(root)
            print(summary(pipeline.run(root)))
            entry = "console" if mode == "demo" else "chat"
            print(f"\n进入{'演示控制台' if mode == 'demo' else 'GPT 对话'}：nolan {entry} {root}")
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
        if args.command in {"run", "resume"} and state["status"] in {
                "awaiting_asset_input", "awaiting_prompt_approval", "awaiting_asset_amendment", "asset_generation_failed"}:
            from .dashboard import serve
            serve(root, port=0, open_browser=True)
        return 1 if state["status"] == "blocked" else 0
    except (ValueError, OSError, KeyError, ProviderError) as exc:
        print("Nolan：" + str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已退出。再次运行会从已保存的进度继续；在途供应商任务可能仍在执行。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
