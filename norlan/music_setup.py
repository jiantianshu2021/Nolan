"""User-only hidden credential entry, kept in this Nolan session's memory."""
from __future__ import annotations

import getpass
import os
import sys


def launch_with_music(project=None, *, resume=False):
    names = ("VOLC_ACCESS_KEY_ID", "VOLC_SECRET_ACCESS_KEY")
    previous = {key: os.environ.get(key) for key in names}
    values = {}
    for name, label in zip(names, ("火山音乐 AccessKey ID（AK）", "火山音乐 SecretAccessKey（SK）")):
        value = os.environ.get(name, "").strip()
        if not value:
            if not sys.stdin.isatty():
                raise ValueError("请在你自己的交互终端运行 nolan music-login --resume，隐藏输入火山音乐AK/SK；不要发到聊天里。")
            value = getpass.getpass(label + "（隐藏输入）：").strip()
        if not value or any(c in value for c in "\r\n\x00"):
            raise ValueError("火山音乐凭据为空或格式无效，尚未启动或发送请求。")
        values[name] = value
    try:
        os.environ.update(values)
        print("火山音乐凭据仅用于本次Nolan进程，不写入项目或配置文件。新音乐仍须点击生成并试听确认。", flush=True)
        from .interactive import launch
        return launch(project, resume=resume)
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
