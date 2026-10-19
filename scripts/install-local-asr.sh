#!/usr/bin/env bash
set -euo pipefail

# Public local inference dependencies only: no cloud transcription credentials.
NOLAN_ASR_MODEL_DIR="${NOLAN_ASR_MODEL_DIR:-${HOME}/.local/share/norlan/models}"
NOLAN_ASR_PYTHON="${NOLAN_ASR_PYTHON:-python3}"
NOLAN_ASR_SHA256='1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b'
NOLAN_ASR_REVISION='5359861c739e955e79d9a303bcbc70fb988958b1'

if ! command -v whisper-cli >/dev/null; then
  if ! command -v brew >/dev/null; then
    echo '请先安装 Homebrew，或自行安装 whisper.cpp 并把 whisper-cli 加入 PATH。' >&2
    exit 1
  fi
  HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 brew install whisper-cpp
fi
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo '还需要 FFmpeg：macOS 请运行 brew install ffmpeg，再重新运行此脚本。' >&2
  exit 1
fi
mkdir -p "${NOLAN_ASR_MODEL_DIR}"
NOLAN_ASR_TARGET="${NOLAN_ASR_MODEL_DIR}/ggml-small.bin"
verify_model() {
  "${NOLAN_ASR_PYTHON}" -c 'import hashlib,pathlib,sys
p=pathlib.Path(sys.argv[1]); h=hashlib.sha256()
with p.open("rb") as f:
    for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
assert p.stat().st_size==487601967 and h.hexdigest()==sys.argv[2], "本地模型的大小或 SHA256 校验失败"
' "$1" "${NOLAN_ASR_SHA256}"
}
if [[ -e "${NOLAN_ASR_TARGET}" ]]; then
  if ! verify_model "${NOLAN_ASR_TARGET}"; then
    echo '已有模型未通过官方校验，未覆盖；请设置 NOLAN_ASR_MODEL_DIR 为新目录或自行检查文件。' >&2
    exit 1
  fi
else
  NOLAN_ASR_TEMP="$(mktemp "${NOLAN_ASR_MODEL_DIR}/.ggml-small-XXXXXXXX")"
  trap 'rm -f "${NOLAN_ASR_TEMP:-}"' EXIT
  echo '下载 Whisper small 多语言模型（约 466 MiB，首次一次；后续识别在本机离线运行）。'
  curl --proto '=https' --tlsv1.2 --fail --location --retry 3 --connect-timeout 30 --max-time 1800 \
    --output "${NOLAN_ASR_TEMP}" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/${NOLAN_ASR_REVISION}/ggml-small.bin"
  verify_model "${NOLAN_ASR_TEMP}"
  mv -n "${NOLAN_ASR_TEMP}" "${NOLAN_ASR_TARGET}"
  verify_model "${NOLAN_ASR_TARGET}"
fi
echo "本地识别程序：$(command -v whisper-cli)"
echo "已校验多语言模型：${NOLAN_ASR_TARGET}"
echo '字幕本地识别已就绪；GPT 文字校对继续使用 Nolan 现有登录。'
