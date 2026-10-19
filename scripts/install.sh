#!/usr/bin/env bash
set -euo pipefail

NOLAN_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep the managed installation/data location so upgrades retain projects and logins.
NOLAN_PREFIX="${NOLAN_INSTALL_DIR:-${NORLAN_INSTALL_DIR:-${HOME}/.local/share/norlan}}"
NOLAN_BIN="${NOLAN_BIN_DIR:-${NORLAN_BIN_DIR:-${HOME}/.local/bin}}"
NOLAN_PYTHON="${NOLAN_PYTHON:-${NORLAN_PYTHON:-python3}}"

"${NOLAN_PYTHON}" -c 'import sys; assert sys.version_info >= (3, 11), "Nolan 需要 Python 3.11+"'
NOLAN_VERSION="$("${NOLAN_PYTHON}" -c 'import pathlib,re,sys,tomllib; version=tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["project"]["version"]; assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9_-]{1,24})?",version), "安装版本号无效"; print(version)' "${NOLAN_SOURCE}/pyproject.toml")"
if [[ -L "${NOLAN_PREFIX}" || ( -e "${NOLAN_PREFIX}" && ! -d "${NOLAN_PREFIX}" ) ]]; then
  echo '安装目录不能是符号链接或普通文件；请选择新的 NOLAN_INSTALL_DIR。' >&2
  exit 1
fi
if [[ -L "${NOLAN_PREFIX}/venv" ]]; then
  echo 'venv 是符号链接，未覆盖其目标；请选择新的 NOLAN_INSTALL_DIR。' >&2
  exit 1
fi
if [[ -L "${NOLAN_PREFIX}/runtimes" || ( -e "${NOLAN_PREFIX}/runtimes" && ! -d "${NOLAN_PREFIX}/runtimes" ) ]]; then
  echo '运行目录不能是符号链接或普通文件；未修改安装。' >&2
  exit 1
fi
if [[ -d "${NOLAN_PREFIX}" && -n "$(ls -A "${NOLAN_PREFIX}")" ]]; then
  if [[ ! -f "${NOLAN_PREFIX}/.norlan-install" || -L "${NOLAN_PREFIX}/.norlan-install" || "$(cat "${NOLAN_PREFIX}/.norlan-install")" != 'norlan-install-v1' ]]; then
    echo '安装目录已有其他内容，且不是 Nolan 管理的安装；未覆盖。请设置 NOLAN_INSTALL_DIR 为新目录。' >&2
    exit 1
  fi
fi
check_entry() {
  local NOLAN_COMMAND="$1" NOLAN_TARGET NOLAN_RUNTIME_NAME
  if [[ -e "${NOLAN_BIN}/${NOLAN_COMMAND}" && ! -L "${NOLAN_BIN}/${NOLAN_COMMAND}" ]]; then
    echo "${NOLAN_BIN}/${NOLAN_COMMAND} 已存在且不是符号链接，请选择 NOLAN_BIN_DIR。" >&2
    exit 1
  fi
  if [[ ! -L "${NOLAN_BIN}/${NOLAN_COMMAND}" ]]; then
    return
  fi
  NOLAN_TARGET="$(readlink "${NOLAN_BIN}/${NOLAN_COMMAND}")"
  if [[ "${NOLAN_TARGET}" == "${NOLAN_PREFIX}/venv/bin/${NOLAN_COMMAND}" ]]; then
    return
  fi
  case "${NOLAN_TARGET}" in
    "${NOLAN_PREFIX}/runtimes/"*"/venv/bin/${NOLAN_COMMAND}")
      NOLAN_RUNTIME_NAME="${NOLAN_TARGET#"${NOLAN_PREFIX}/runtimes/"}"
      NOLAN_RUNTIME_NAME="${NOLAN_RUNTIME_NAME%"/venv/bin/${NOLAN_COMMAND}"}"
      if [[ "${NOLAN_RUNTIME_NAME}" =~ ^[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9_-]*-[A-Za-z0-9]{8}$ && ${#NOLAN_RUNTIME_NAME} -le 80
            && ! -L "${NOLAN_PREFIX}/runtimes/${NOLAN_RUNTIME_NAME}"
            && ! -L "${NOLAN_PREFIX}/runtimes/${NOLAN_RUNTIME_NAME}/venv"
            && ! -L "${NOLAN_PREFIX}/runtimes/${NOLAN_RUNTIME_NAME}/venv/bin" ]]; then
        return
      fi ;;
  esac
  echo "${NOLAN_COMMAND} 已指向另一个安装位置；未覆盖。" >&2
  exit 1
}
for NOLAN_COMMAND in nolan norlan; do
  check_entry "${NOLAN_COMMAND}"
done
mkdir -p "${NOLAN_PREFIX}/runtimes" "${NOLAN_BIN}"
printf '%s\n' 'norlan-install-v1' > "${NOLAN_PREFIX}/.norlan-install"
# Even the same version gets a fresh runtime: active processes may still import
# modules from an earlier installation. Never overwrite or delete their code.
NOLAN_RUNTIME="$(mktemp -d "${NOLAN_PREFIX}/runtimes/${NOLAN_VERSION}-XXXXXXXX")"
printf '%s\n' "${NOLAN_VERSION}" > "${NOLAN_RUNTIME}/.nolan-runtime"
"${NOLAN_PYTHON}" -m venv "${NOLAN_RUNTIME}/venv"
"${NOLAN_RUNTIME}/venv/bin/python" -m pip install --disable-pip-version-check "${NOLAN_SOURCE}"
for NOLAN_COMMAND in nolan norlan; do
  if [[ ! -x "${NOLAN_RUNTIME}/venv/bin/${NOLAN_COMMAND}" || "$("${NOLAN_RUNTIME}/venv/bin/${NOLAN_COMMAND}" --version)" != "${NOLAN_VERSION}" ]]; then
    echo '新运行目录未通过入口版本检查；原启动命令未切换。' >&2
    exit 1
  fi
  check_entry "${NOLAN_COMMAND}"
done
# os.replace switches each launcher atomically and cannot follow a destination
# symlink. Both launchers are validated before either is changed.
"${NOLAN_PYTHON}" -c 'import os,pathlib,sys,tempfile
directory,runtime=map(pathlib.Path,sys.argv[1:])
for command in ("nolan","norlan"):
    fd,temporary=tempfile.mkstemp(prefix="."+command+"-link-",dir=directory)
    os.close(fd)
    try:
        os.unlink(temporary)
        os.symlink(runtime/"venv"/"bin"/command,temporary)
        os.replace(temporary,directory/command)
    finally:
        if os.path.lexists(temporary): os.unlink(temporary)
' "${NOLAN_BIN}" "${NOLAN_RUNTIME}"
echo "Nolan 已安装到 ${NOLAN_BIN}/nolan"
echo '启动对话：nolan'
echo '检查依赖：nolan doctor'
echo '旧命令 norlan 仍可使用；已有项目、登录和会话保留在原目录。'
echo '本次使用独立运行目录，未覆盖运行中的旧程序；当前任务结束后用 nolan --resume 加载新版。'
case ":${PATH}:" in
  *":${NOLAN_BIN}:"*) ;;
  *) echo "请把 ${NOLAN_BIN} 加入 PATH，或使用上方完整路径启动。" ;;
esac
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo '媒体处理需要 FFmpeg；macOS 可用 brew install ffmpeg。'
fi
