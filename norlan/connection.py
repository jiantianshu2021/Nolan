"""Operator-controlled GPT connection choice; never store or print key values."""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def connection_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "norlan/connection.json"


def validate_connection(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) - {"provider", "model"}:
        raise ValueError("GPT 连接配置只允许 provider 和 model；不要在配置中保存密钥。")
    provider = value.get("provider", "chatgpt")
    if provider not in {"chatgpt", "openai"}:
        raise ValueError("GPT 连接方式须为 chatgpt（订阅）或 openai（API）。")
    model = value.get("model", "")
    if not isinstance(model, str) or len(model) > 160 or any(c.isspace() for c in model):
        raise ValueError("请填写有效的模型 ID，不要包含空格。")
    if provider == "openai" and not model:
        raise ValueError("API 模式需要明确填写你的账户可用的模型 ID。")
    return {"provider": provider, **({"model": model} if model else {})}


def load_connection() -> dict:
    try:
        raw = connection_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"provider": "chatgpt"}
    try:
        return validate_connection(json.loads(raw))
    except (ValueError, TypeError) as exc:
        raise ValueError("Nolan GPT 连接配置无效，请运行 nolan setup 重新选择。") from exc


def save_connection(value: dict) -> None:
    value = validate_connection(value)
    target = connection_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".connection-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def describe_connection() -> str:
    value = load_connection()
    name = "ChatGPT 订阅" if value["provider"] == "chatgpt" else "OpenAI API（独立计费）"
    return f"GPT 连接：{name}\n模型：{value.get('model') or '沿用你的 Codex 模型选择'}\n切换连接：nolan setup"


def production_connection_info(config: dict) -> dict:
    """Public connection metadata only; no auth reads or configuration writes."""
    selected = load_connection()
    inherited = os.environ.get("NORLAN_GPT_BILLING")
    active = inherited if inherited in {"api", "subscription"} else (
        "api" if selected["provider"] == "openai" else "subscription")
    direct = config.get("provider") == "openai"
    model = config.get("model") or ("沿用本会话连接的模型" if inherited in {"api", "subscription"} else
                                     selected.get("model") or "沿用当前连接的模型")
    return {"transport": "responses" if direct else "codex",
            "billing": "api" if direct else active,
            "model": model,
            "connection_conflict": direct and active == "subscription",
            "note": "项目固定 API 与订阅选择冲突，制作请求将停止。" if direct and active == "subscription" else
                    "制作中的 API 调用需项目预算；自由对话费用独立于项目预算。" if direct or active == "api" else
                    "使用 ChatGPT 订阅账户额度。"}


def setup(provider: str | None = None, model: str | None = None) -> int:
    """Interactive local setup. Credentials go to official CLI stdin only."""
    from .interactive import prepare_launch
    if provider is None:
        print("选择 Nolan 的 GPT 连接方式：\n  1. ChatGPT 订阅\n  2. OpenAI API（按 API 用量计费）")
        choice = input("选择 [1/2]：").strip()
        if choice not in {"1", "2"}:
            raise ValueError("请输入 1 或 2；原连接未更改。")
        provider = "chatgpt" if choice == "1" else "openai"
    if provider == "openai" and not model:
        model = input("你的 OpenAI API 模型 ID：").strip()
    selected = validate_connection({"provider": provider, **({"model": model} if model else {})})
    command, env = prepare_launch(connection=selected)
    executable = command[0]
    # Authentication state belongs to the chosen provider's isolated home.
    forced = "chatgpt" if provider == "chatgpt" else "api"
    login_args = [executable, "-c", f'forced_login_method="{forced}"', "login"]
    try:
        result = subprocess.run(login_args + ["status"], env=env, capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            if provider == "openai":
                key = os.environ.get("OPENAI_API_KEY")
                if not key:
                    if not sys.stdin.isatty():
                        raise ValueError("请在交互终端运行 nolan setup 隐藏输入密钥，或先设置 OPENAI_API_KEY。")
                    key = getpass.getpass("OpenAI API Key（隐藏输入）：")
                if not key.strip() or "\n" in key or "\r" in key:
                    raise ValueError("未提供有效 API Key，原连接未更改。")
                result = subprocess.run(login_args + ["--with-api-key"], input=key.strip() + "\n",
                                        env=env, capture_output=True, text=True, timeout=60)
                del key
                if result.returncode != 0:
                    raise ValueError("API 登录未成功，原连接未更改。请检查密钥并重新运行 nolan setup。")
            else:
                print("将打开官方 ChatGPT 登录流程；完成登录后回到终端。")
                result = subprocess.run(login_args, env=env, timeout=600)
                if result.returncode != 0:
                    raise ValueError("ChatGPT 登录未完成，原连接未更改。")
    except subprocess.TimeoutExpired:
        raise ValueError("登录检查超时，原连接未更改；请重新运行 nolan setup。") from None
    save_connection(selected)
    print(describe_connection())
    print("运行 nolan 开始对话。" if provider == "chatgpt" else
          "运行 nolan 开始对话。自由对话的 API 费用由 OpenAI 账户计费，独立于漫剧项目预算。")
    return 0


def production_brain_config(config: dict) -> dict:
    """Use the operator's connection for Codex-backed production as well.

    An explicitly configured direct Responses provider remains explicit. API
    credentials in the native CLI cache are read by Codex, never by Nolan.
    """
    result = dict(config)
    selected = load_connection()
    # A running TUI pins its connection; changing defaults in another terminal
    # must not switch an in-flight production session's account underneath it.
    inherited = os.environ.get("NORLAN_GPT_BILLING")
    if result.get("provider", "codex") != "codex":
        if result.get("provider") == "openai" and (inherited == "subscription" or (
                inherited not in {"api", "subscription"} and selected["provider"] == "chatgpt")):
            raise ValueError("这个项目固定使用 OpenAI API，但当前选择了 ChatGPT 订阅；请将项目 brain.provider 改为 codex 以跟随连接，或用 nolan setup 明确选择 API。尚未提交请求。")
        return result
    if inherited in {"api", "subscription"} and os.environ.get("CODEX_HOME"):
        result["codex_home"] = os.environ["CODEX_HOME"]
        result["billing"] = inherited
    else:
        from .interactive import prepare_launch
        _, env = prepare_launch(connection=selected)
        result["codex_home"] = env["CODEX_HOME"]
        result["billing"] = env["NORLAN_GPT_BILLING"]
        if selected.get("model") and not result.get("model"):
            result["model"] = selected["model"]
    return result
