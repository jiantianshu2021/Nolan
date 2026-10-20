#!/usr/bin/env bash
set -euo pipefail
# Isolated, public local inference dependencies. No provider credentials required.
NOLAN_ALIGN_BASE="${NOLAN_ALIGN_BASE:-${HOME}/.local/share/norlan}"
NOLAN_ALIGN_PYTHON="${NOLAN_ALIGN_PYTHON:-$(command -v python3.13 || true)}"
NOLAN_ALIGN_BACKEND="${NOLAN_ALIGN_BACKEND:-sense_voice}"
case "${NOLAN_ALIGN_BACKEND}" in
  sense_voice)
    NOLAN_ALIGN_DEFAULT_DIR="${NOLAN_ALIGN_BASE}/models/sensevoice-zh-int8-2024-07-17"
    NOLAN_ALIGN_PACKAGE='sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17'
    NOLAN_ALIGN_ARCHIVE_SIZE='163002883'
    NOLAN_ALIGN_ARCHIVE_SHA='7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e'
    NOLAN_ALIGN_MODEL_SIZE='239233841'
    NOLAN_ALIGN_MODEL_SHA='c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51'
    NOLAN_ALIGN_TOKENS_SIZE='315894'
    NOLAN_ALIGN_TOKENS_SHA='f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc'
    NOLAN_ALIGN_DESCRIPTION='SenseVoice 中文独立声学检查（压缩包约 163 MB，模型约 239 MB）'
    ;;
  zipformer_ctc)
    NOLAN_ALIGN_DEFAULT_DIR="${NOLAN_ALIGN_BASE}/models/zipformer-ctc-zh-small-int8"
    NOLAN_ALIGN_PACKAGE='sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01'
    NOLAN_ALIGN_ARCHIVE_SIZE='21264113'
    NOLAN_ALIGN_ARCHIVE_SHA='b3b309f7ce4a737195fcc6963ea19b0653a7d3401580af5ae0d3e284cbb71f0b'
    NOLAN_ALIGN_MODEL_SIZE='26342340'
    NOLAN_ALIGN_MODEL_SHA='68c9c943840f7d9cf3e8a4970ba50f404feb5277f611fa82b7e72267786fa84a'
    NOLAN_ALIGN_TOKENS_SIZE='13366'
    NOLAN_ALIGN_TOKENS_SHA='6fed8c6c248516f38e7faa19404b57413e8ce259f1cbc1fa4aebc86eac32fdfd'
    NOLAN_ALIGN_DESCRIPTION='Zipformer 兼容检查器（压缩包约 21 MB）'
    ;;
  *) echo 'NOLAN_ALIGN_BACKEND 必须为 sense_voice 或 zipformer_ctc。' >&2; exit 1 ;;
esac
NOLAN_ALIGN_MODEL_DIR="${NOLAN_ALIGN_MODEL_DIR:-${NOLAN_ALIGN_DEFAULT_DIR}}"
NOLAN_ALIGN_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NOLAN_ALIGN_PACKAGE}.tar.bz2"
if [[ -z "${NOLAN_ALIGN_PYTHON}" ]]; then
  echo '字幕独立检查需要 Python 3.13。macOS 请运行 brew install python@3.13，然后重跑此脚本。' >&2
  exit 1
fi
"${NOLAN_ALIGN_PYTHON}" -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,13), "请使用 Python 3.10–3.13 的独立运行环境"'
mkdir -p "${NOLAN_ALIGN_BASE}" "${NOLAN_ALIGN_MODEL_DIR}"
NOLAN_ALIGN_TEMP="$(mktemp -d "${NOLAN_ALIGN_BASE}/.alignment-install-XXXXXXXX")"
trap 'rm -rf "${NOLAN_ALIGN_TEMP:-}"' EXIT
verify_models() {
  "${NOLAN_ALIGN_PYTHON}" - "$1" "${NOLAN_ALIGN_MODEL_SIZE}" "${NOLAN_ALIGN_MODEL_SHA}" "${NOLAN_ALIGN_TOKENS_SIZE}" "${NOLAN_ALIGN_TOKENS_SHA}" <<'PY'
import hashlib,pathlib,sys
p=pathlib.Path(sys.argv[1])
for name,size,sha in [('model.int8.onnx',int(sys.argv[2]),sys.argv[3]),('tokens.txt',int(sys.argv[4]),sys.argv[5])]:
 f=p/name
 assert f.stat().st_size==size and hashlib.sha256(f.read_bytes()).hexdigest()==sha, '模型大小或 SHA256 校验失败：'+name
PY
}
if [[ -e "${NOLAN_ALIGN_MODEL_DIR}/model.int8.onnx" || -e "${NOLAN_ALIGN_MODEL_DIR}/tokens.txt" ]]; then
  if ! verify_models "${NOLAN_ALIGN_MODEL_DIR}"; then
    echo '已有独立对齐模型未通过校验，未覆盖；请检查文件或指定新的 NOLAN_ALIGN_MODEL_DIR。' >&2
    exit 1
  fi
else
  echo "下载 ${NOLAN_ALIGN_DESCRIPTION}；仅首次下载，后续离线运行。"
  curl --proto '=https' --tlsv1.2 --fail --location --retry 3 --connect-timeout 30 --max-time 600 \
    --output "${NOLAN_ALIGN_TEMP}/model.tar.bz2" "${NOLAN_ALIGN_URL}"
  "${NOLAN_ALIGN_PYTHON}" - "${NOLAN_ALIGN_TEMP}" "${NOLAN_ALIGN_ARCHIVE_SHA}" "${NOLAN_ALIGN_ARCHIVE_SIZE}" "${NOLAN_ALIGN_PACKAGE}" <<'PY'
import hashlib,pathlib,sys,tarfile
p=pathlib.Path(sys.argv[1]);archive=p/'model.tar.bz2'
assert archive.stat().st_size==int(sys.argv[3]) and hashlib.sha256(archive.read_bytes()).hexdigest()==sys.argv[2], '官方模型压缩包校验失败'
with tarfile.open(archive) as tar:
 members={str(pathlib.PurePosixPath(m.name)):m for m in tar.getmembers()}
 for name in ('model.int8.onnx','tokens.txt'):
  member=members[sys.argv[4]+'/'+name]
  assert member.isfile() and 0<member.size<250_000_000
  with tar.extractfile(member) as stream:(p/name).write_bytes(stream.read())
 license=members.get(sys.argv[4]+'/LICENSE')
 if license and license.isfile() and 0<license.size<100_000:
  with tar.extractfile(license) as stream:(p/'LICENSE').write_bytes(stream.read())
PY
  verify_models "${NOLAN_ALIGN_TEMP}"
  mv -n "${NOLAN_ALIGN_TEMP}/model.int8.onnx" "${NOLAN_ALIGN_MODEL_DIR}/model.int8.onnx"
  mv -n "${NOLAN_ALIGN_TEMP}/tokens.txt" "${NOLAN_ALIGN_MODEL_DIR}/tokens.txt"
  if [[ -f "${NOLAN_ALIGN_TEMP}/LICENSE" ]]; then
    mv -n "${NOLAN_ALIGN_TEMP}/LICENSE" "${NOLAN_ALIGN_MODEL_DIR}/LICENSE"
  fi
  verify_models "${NOLAN_ALIGN_MODEL_DIR}"
fi
if [[ "${NOLAN_ALIGN_SKIP_RUNTIME:-0}" != '1' ]]; then
NOLAN_ALIGN_RUNTIME="$(mktemp -d "${NOLAN_ALIGN_BASE}/alignment-runtime-XXXXXXXX")"
"${NOLAN_ALIGN_PYTHON}" -m venv "${NOLAN_ALIGN_RUNTIME}"
if command -v uv >/dev/null; then
  uv pip install --python "${NOLAN_ALIGN_RUNTIME}/bin/python" 'sherpa-onnx==1.12.40' 'numpy==2.2.6' 'pypinyin==0.55.0'
else
  "${NOLAN_ALIGN_RUNTIME}/bin/python" -m pip install 'sherpa-onnx==1.12.40' 'numpy==2.2.6' 'pypinyin==0.55.0'
fi
"${NOLAN_ALIGN_RUNTIME}/bin/python" -c 'import sherpa_onnx,numpy,pypinyin; print("独立字幕检查依赖已就绪：sherpa-onnx",sherpa_onnx.__version__)'
# Existing workers finish on their own immutable runtime; atomically switch the launcher link.
NOLAN_ALIGN_LINK="${NOLAN_ALIGN_TEMP}/runtime-link"
ln -s "${NOLAN_ALIGN_RUNTIME}" "${NOLAN_ALIGN_LINK}"
"${NOLAN_ALIGN_PYTHON}" - "${NOLAN_ALIGN_LINK}" "${NOLAN_ALIGN_BASE}/alignment-runtime" <<'PY'
import os,sys
os.replace(sys.argv[1],sys.argv[2])
PY
else
  "${NOLAN_ALIGN_BASE}/alignment-runtime/bin/python" -c 'import sherpa_onnx,numpy,pypinyin'
fi
if [[ "${NOLAN_ALIGN_BACKEND}" == 'sense_voice' ]]; then
cat > "${NOLAN_ALIGN_MODEL_DIR}/MODEL_SOURCE.txt" <<'TXT'
Independent Mandarin acoustic model: SenseVoice int8 2024-07-17.
Official export/API: https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html
Pinned publisher: https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tree/2365baeacb507f821a0c8120fcee3d484dba7a07
Model SHA256: c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51
Upstream model/license: https://github.com/modelscope/FunASR?tab=readme-ov-file#license
Runtime: https://github.com/k2-fsa/sherpa-onnx (Apache-2.0)
Fixed audio windows and independent local acoustic matches estimate timestamps; final human approval remains required.
TXT
else
cat > "${NOLAN_ALIGN_MODEL_DIR}/MODEL_SOURCE.txt" <<'TXT'
Independent Chinese CTC verification model: Zipformer small CTC zh int8 2025-04-01.
Source/export: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-ctc/zipformer-ctc-models.html
Upstream checkpoint: https://huggingface.co/csukuangfj/icefall-streaming-zipformer-small-ctc-zh-2025-04-01 (Apache-2.0 model card)
Runtime: https://github.com/k2-fsa/sherpa-onnx (Apache-2.0)
This checker estimates Mandarin acoustic timestamps. Machine checks do not replace final human approval.
TXT
fi
if [[ "${NOLAN_ALIGN_BACKEND}" == 'sense_voice' ]]; then
  # This smaller, separate acoustic model can challenge a SenseVoice coverage
  # hallucination. It is never replaced with the same decoder's self-check.
  NOLAN_ALIGN_BACKEND=zipformer_ctc NOLAN_ALIGN_SKIP_RUNTIME=1 \
    NOLAN_ALIGN_MODEL_DIR="${NOLAN_ALIGN_BASE}/models/zipformer-ctc-zh-small-int8" bash "$0"
fi
echo "本地独立检查运行环境：${NOLAN_ALIGN_BASE}/alignment-runtime/bin/python"
echo "已校验独立模型：${NOLAN_ALIGN_MODEL_DIR}/model.int8.onnx"
echo "检查器类型：${NOLAN_ALIGN_BACKEND}（默认 sense_voice；兼容类型需设置 subtitles.alignment_backend）。"
echo '字幕生成后将使用另一套 CTC 声学模型核对、最多两次自动重对齐，再交由你最终验片。'
