"""Explicit, local-only Ark credential entry and Keychain persistence."""
from __future__ import annotations

import getpass
import os
import sys
import warnings


def launch_with_ark(project=None, *, resume=False, workdir=None, no_launch=False):
    """Store a human-provided key without putting it in argv, logs or projects.

    This command is deliberately explicit. Normal launches only read saved
    credentials; they never prompt, rotate or save a key on their own.
    """
    from .credentials import CredentialError, save_ark_key

    previous = os.environ.get("ARK_API_KEY")
    value = (previous or "").strip()
    if not value:
        if not sys.stdin.isatty():
            raise ValueError(
                "请在你自己的交互终端运行 nolan ark-login --resume，隐藏输入方舟 API Key；"
                "不要发到聊天里。仅保存可加 --no-launch。"
            )
        try:
            with warnings.catch_warnings():
                # getpass may otherwise fall back to echoing input on an
                # unsuitable terminal. Never accept that fallback for secrets.
                warnings.simplefilter("error", getpass.GetPassWarning)
                value = getpass.getpass("火山方舟 API Key（隐藏输入，保存到本机钥匙串）：").strip()
        except (EOFError, getpass.GetPassWarning):
            raise ValueError("当前终端无法安全隐藏输入；请在你自己的终端运行 nolan ark-login。") from None
    if not value or any(char in value for char in "\r\n\x00"):
        raise ValueError("方舟 API Key 为空或格式无效，尚未保存或发送请求。")
    try:
        save_ark_key(value)
    except CredentialError as exc:
        raise ValueError(str(exc)) from None
    print("方舟 API Key 已安全保存到本机 macOS 钥匙串；以后启动 Nolan 会自动读取，无需每次输入。", flush=True)
    print("未调用图片或视频生成服务，未修改项目与审批进度。", flush=True)
    if no_launch:
        print("已运行的旧进程需要退出后重新启动，才能读取已保存的凭据。", flush=True)
        return 0
    try:
        os.environ["ARK_API_KEY"] = value
        from .interactive import launch
        options = {"workdir": workdir} if workdir is not None else {}
        return launch(project, resume=resume, **options)
    finally:
        if previous is None:
            os.environ.pop("ARK_API_KEY", None)
        else:
            os.environ["ARK_API_KEY"] = previous
