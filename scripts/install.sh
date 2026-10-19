#!/usr/bin/env bash
set -euo pipefail

NORLAN_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NORLAN_PREFIX="${NORLAN_INSTALL_DIR:-${HOME}/.local/share/norlan}"
NORLAN_BIN="${NORLAN_BIN_DIR:-${HOME}/.local/bin}"
NORLAN_PYTHON="${NORLAN_PYTHON:-python3}"

"${NORLAN_PYTHON}" -c 'import sys; assert sys.version_info >= (3, 11), "Norlan 需要 Python 3.11+"'
if [[ -L "${NORLAN_PREFIX}" || ( -e "${NORLAN_PREFIX}" && ! -d "${NORLAN_PREFIX}" ) ]]; then
  echo '安装目录不能是符号链接或普通文件；请选择新的 NORLAN_INSTALL_DIR。' >&2
  exit 1
fi
if [[ -L "${NORLAN_PREFIX}/venv" ]]; then
  echo 'venv 是符号链接，未覆盖其目标；请选择新的 NORLAN_INSTALL_DIR。' >&2
  exit 1
fi
if [[ -d "${NORLAN_PREFIX}" && -n "$(ls -A "${NORLAN_PREFIX}")" ]]; then
  if [[ ! -f "${NORLAN_PREFIX}/.norlan-install" || -L "${NORLAN_PREFIX}/.norlan-install" || "$(cat "${NORLAN_PREFIX}/.norlan-install")" != 'norlan-install-v1' ]]; then
    echo '安装目录已有其他内容，且不是 Norlan 管理的安装；未覆盖。请设置 NORLAN_INSTALL_DIR 为新目录。' >&2
    exit 1
  fi
fi
if [[ -e "${NORLAN_BIN}/norlan" && ! -L "${NORLAN_BIN}/norlan" ]]; then
  echo "${NORLAN_BIN}/norlan 已存在且不是符号链接，请选择 NORLAN_BIN_DIR。" >&2
  exit 1
fi
if [[ -L "${NORLAN_BIN}/norlan" && "$(readlink "${NORLAN_BIN}/norlan")" != "${NORLAN_PREFIX}/venv/bin/norlan" ]]; then
  echo "norlan 已指向另一个安装位置；未覆盖。" >&2
  exit 1
fi
mkdir -p "${NORLAN_PREFIX}" "${NORLAN_BIN}"
printf '%s\n' 'norlan-install-v1' > "${NORLAN_PREFIX}/.norlan-install"
"${NORLAN_PYTHON}" -m venv "${NORLAN_PREFIX}/venv"
"${NORLAN_PREFIX}/venv/bin/python" -m pip install --disable-pip-version-check "${NORLAN_SOURCE}"
ln -sfn "${NORLAN_PREFIX}/venv/bin/norlan" "${NORLAN_BIN}/norlan"
echo "Norlan 已安装到 ${NORLAN_BIN}/norlan"
echo '启动对话：norlan'
echo '检查依赖：norlan doctor'
case ":${PATH}:" in
  *":${NORLAN_BIN}:"*) ;;
  *) echo "请把 ${NORLAN_BIN} 加入 PATH，或使用上方完整路径启动。" ;;
esac
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo '媒体处理需要 FFmpeg；macOS 可用 brew install ffmpeg。'
fi
