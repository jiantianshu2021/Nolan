"""Independent, bounded local subtitle synchronization verification.

Whisper supplies observed timing candidates; a separate SenseVoice Chinese CTC
model verifies them against actual audio. The isolated worker has no GPT/API
access. Zipformer remains an explicit compatibility option.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from .providers import ProviderError
from .media import probe
from .local_transcription import DEFAULT_MODEL_PATH as DEFAULT_WHISPER_MODEL


DEFAULT_PYTHON = Path.home() / ".local/share/norlan/alignment-runtime/bin/python"
DEFAULT_MODEL = Path.home() / ".local/share/norlan/models/sensevoice-zh-int8-2024-07-17/model.int8.onnx"
MODEL_SHA256 = "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"
TOKENS_SHA256 = "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc"
ZIPFORMER_MODEL = Path.home() / ".local/share/norlan/models/zipformer-ctc-zh-small-int8/model.int8.onnx"
BACKENDS = {
    "sense_voice": {"default_model": DEFAULT_MODEL, "model_sha256": MODEL_SHA256,
                    "tokens_sha256": TOKENS_SHA256, "worker": "sensevoice_worker.py",
                    "name": "sensevoice-zh-int8-2024-07-17"},
    "zipformer_ctc": {"default_model": ZIPFORMER_MODEL,
                      "model_sha256": "68c9c943840f7d9cf3e8a4970ba50f404feb5277f611fa82b7e72267786fa84a",
                      "tokens_sha256": "6fed8c6c248516f38e7faa19404b57413e8ce259f1cbc1fa4aebc86eac32fdfd",
                      "worker": "alignment_worker.py", "name": "zipformer-small-ctc-zh-int8-2025-04-01"},
}
SCHEMA = "nolan-independent-ctc-v2"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class LocalSubtitleVerifier:
    def __init__(self, config, *, cache_dir=None, cancel_check=None):
        if not isinstance(config, dict):
            raise ProviderError("本地字幕检查配置无效。")
        self.config = dict(config)
        self.backend = config.get("alignment_backend", "sense_voice")
        if not isinstance(self.backend, str) or self.backend not in BACKENDS:
            raise ProviderError("字幕独立检查支持 sense_voice（默认）或 zipformer_ctc，请明确选择本地模型。")
        self.backend_info = BACKENDS[self.backend]
        python = config.get("alignment_python", DEFAULT_PYTHON)
        model = config.get("alignment_model_path", self.backend_info["default_model"])
        if any(not isinstance(p, (str, Path)) or not str(p) or "\x00" in str(p) for p in (python, model)):
            raise ProviderError("字幕独立检查的本地程序或模型路径无效。")
        self.python = Path(python).expanduser().absolute()
        self.model = Path(model).expanduser().resolve()
        self.tokens = self.model.parent / "tokens.txt"
        self.worker = Path(__file__).with_name(self.backend_info["worker"])
        self.shared_worker = Path(__file__).with_name("alignment_worker.py")
        coverage_model = config.get("alignment_coverage_model_path", ZIPFORMER_MODEL)
        if not isinstance(coverage_model, (str, Path)) or not str(coverage_model) or "\x00" in str(coverage_model):
            raise ProviderError("字幕独立覆盖复核模型路径无效。")
        self.coverage_model = Path(coverage_model).expanduser().resolve()
        self.coverage_tokens = self.coverage_model.parent / "tokens.txt"
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.cancel_check = cancel_check
        self.timeout = config.get("alignment_timeout", 300)
        self.threads = config.get("threads", 4)
        if (type(self.timeout) not in (int, float) or not math.isfinite(self.timeout)
                or not 1 <= self.timeout <= 3600 or type(self.threads) is not int
                or not 1 <= self.threads <= 32):
            raise ProviderError("字幕独立检查的超时或线程配置无效。")
        self._identity = {}
        self.last_metadata = {}
        self.last_observations = {}
        self._last_audio = None
        self._last_caption_hash = ""
        self.changes = []
        self._secondary_identity = {}

    @property
    def identity(self):
        return deepcopy(self._identity)

    def _cancel(self):
        if self.cancel_check and self.cancel_check():
            raise ProviderError("字幕同步检查已暂停；现有视频和字幕已保留。")

    def _run(self, args, timeout, *, capture=False):
        self._cancel()
        try:
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
        except OSError:
            raise ProviderError("无法启动本地字幕独立检查，请运行 scripts/install-subtitle-alignment.sh。") from None
        started = time.monotonic()
        try:
            while process.poll() is None:
                self._cancel()
                if time.monotonic() - started > timeout:
                    raise ProviderError("字幕同步检查超时，未标记通过；现有素材已保留。")
                time.sleep(.1)
            output = process.stdout.read(1_000_001) if capture else b""
            if process.returncode or len(output) > 1_000_000:
                raise ProviderError("本地字幕独立检查失败，未标记通过；请检查对齐运行环境和模型。")
            return output.decode("utf-8") if capture else ""
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            if process.stdout:
                process.stdout.close()

    def check_ready(self):
        if (not self.python.is_file() or not self.model.is_file() or not self.tokens.is_file()
                or (self.backend == "sense_voice" and (not self.coverage_model.is_file() or not self.coverage_tokens.is_file()))):
            raise ProviderError("尚未安装字幕独立同步检查器。请运行 scripts/install-subtitle-alignment.sh；"
                                "只下载本地模型，不需要 API Key。")
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise ProviderError("字幕独立同步检查需要 FFmpeg。")
        try:
            model_hash, tokens_hash = _sha(self.model), _sha(self.tokens)
            if model_hash != self.backend_info["model_sha256"] or tokens_hash != self.backend_info["tokens_sha256"]:
                raise ValueError("model hash mismatch")
            coverage = None
            if self.backend == "sense_voice":
                coverage = {"model_sha256": _sha(self.coverage_model), "tokens_sha256": _sha(self.coverage_tokens)}
                if (coverage["model_sha256"] != BACKENDS["zipformer_ctc"]["model_sha256"]
                        or coverage["tokens_sha256"] != BACKENDS["zipformer_ctc"]["tokens_sha256"]):
                    raise ValueError("secondary acoustic model hash mismatch")
            whisper_executable = shutil.which(self.config.get("executable", "whisper-cli"))
            whisper_model = Path(self.config.get("model_path", DEFAULT_WHISPER_MODEL)).expanduser().resolve()
            self._secondary_identity = {"available": bool(whisper_executable and whisper_model.is_file())}
            if self._secondary_identity["available"]:
                self._secondary_identity.update(executable_sha256=_sha(whisper_executable),
                                                model_sha256=_sha(whisper_model),
                                                language=self.config.get("language", "zh"))
            versions = json.loads(self._run([str(self.python), "-c",
                "import json,sherpa_onnx,numpy,pypinyin;"
                "print(json.dumps({'sherpa_onnx':sherpa_onnx.__version__,"
                "'numpy':numpy.__version__,'pypinyin':pypinyin.__version__}))"], 30, capture=True))
        except (OSError, ValueError, TypeError):
            raise ProviderError("字幕独立检查模型或运行环境未通过校验，请运行 scripts/install-subtitle-alignment.sh。") from None
        self._identity = {"schema": SCHEMA, "backend": "independent_ctc",
                          "acoustic_backend": self.backend, "model": self.backend_info["name"],
                          "model_sha256": model_hash, "tokens_sha256": tokens_hash,
                          "worker_sha256": _sha(self.worker),
                          "shared_worker_sha256": _sha(self.shared_worker),
                          "coverage_checker": coverage,
                          "secondary_whisper": deepcopy(self._secondary_identity),
                          "adapter_sha256": _sha(Path(__file__)),
                          "noise_checker_sha256": _sha(Path(__file__).with_name("subtitle_noise.py")),
                          "versions": versions,
                          "threads": self.threads, "policy": 1}
        return {"python": str(self.python), "model_path": str(self.model),
                "identity": self.identity, "method": "independent_ctc"}

    def _assert_unchanged(self, audio, audio_hash):
        try:
            if (_sha(audio) != audio_hash or _sha(self.model) != self._identity["model_sha256"]
                    or _sha(self.tokens) != self._identity["tokens_sha256"]
                    or _sha(self.worker) != self._identity["worker_sha256"]
                    or _sha(self.shared_worker) != self._identity["shared_worker_sha256"]
                    or _sha(Path(__file__)) != self._identity["adapter_sha256"]
                    or _sha(Path(__file__).with_name("subtitle_noise.py")) != self._identity["noise_checker_sha256"]):
                raise ValueError("checker input changed")
            if self.backend == "sense_voice" and (
                    _sha(self.coverage_model) != self._identity["coverage_checker"]["model_sha256"]
                    or _sha(self.coverage_tokens) != self._identity["coverage_checker"]["tokens_sha256"]):
                raise ValueError("coverage checker changed")
            if self._secondary_identity.get("available"):
                executable = shutil.which(self.config.get("executable", "whisper-cli"))
                if (not executable or _sha(executable) != self._secondary_identity["executable_sha256"]
                        or _sha(Path(self.config.get("model_path", DEFAULT_WHISPER_MODEL)).expanduser().resolve())
                        != self._secondary_identity["model_sha256"]):
                    raise ValueError("secondary recognizer changed")
        except (OSError, ValueError, KeyError):
            raise ProviderError("字幕音轨、独立模型或检查程序在处理期间发生变化，未采用本次检查。") from None

    def _crosscheck_uncovered(self, audio, observations):
        """A stable single-model hallucination is not proven missing speech."""
        from .subtitle_noise import confirm_unsupported_caption
        retained, excluded = [], []
        issues = observations.get("contradictions", [])
        if not isinstance(issues, list) or any(not isinstance(issue, dict) for issue in issues):
            return observations  # The pure checker rejects malformed evidence.
        for issue in issues:
            secondary = issue.get("secondary_ctc", {})
            if (issue.get("code") == "uncovered_speech" and secondary.get("language_supported") is True
                    and secondary.get("zero_support") is True and issue.get("observed_text")
                    and self._secondary_identity.get("available")):
                self._cancel()
                cue = {"start": issue["start"], "end": issue["end"], "text": issue["observed_text"]}
                evidence = confirm_unsupported_caption(audio, cue, self.config,
                    cache_dir=self.cache_dir or Path(audio).parent / ".sync-noise", cancel_check=self.cancel_check)
                self._cancel()
                if evidence.get("unsupported") is True:
                    record = {"kind": "unsupported_coverage_excluded", "before": deepcopy(issue),
                              "evidence": evidence, "reason": "小 CTC 全轨和两个局部上下文均零支持，两个宽上下文 Whisper 亦不支持。"}
                    excluded.append(record)
                    if not any(c.get("kind") == record["kind"] and c.get("evidence", {}).get("evidence_hash") == evidence["evidence_hash"]
                               for c in self.changes):
                        self.changes.append(deepcopy(record))
                    continue
            retained.append(issue)
        observations["contradictions"] = retained
        observations["coverage_exclusions"] = excluded
        return observations

    def verify(self, audio, captions):
        from .subtitle_sync import canonical_captions, caption_hash
        self._cancel()
        if not self._identity:
            self.check_ready()
        audio = Path(audio).expanduser().resolve()
        metadata = probe(audio)
        duration = metadata.get("duration")
        if (not audio.is_file() or not 0 < audio.stat().st_size <= 512_000_000
                or type(duration) not in (int, float) or not math.isfinite(duration)
                or not 0 < duration <= 3600 or not metadata.get("has_audio")):
            raise ProviderError("字幕同步检查需要不超过一小时的实际音轨。")
        captions = canonical_captions(captions, duration)
        audio_hash, caps_hash = _sha(audio), caption_hash(captions)
        self._assert_unchanged(audio, audio_hash)
        key = _digest({"audio": audio_hash, "captions": caps_hash, "checker": self.identity})
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="nolan-sync-") as temp:
            directory = Path(temp)
            lock_path = (self.cache_dir / (key + ".lock")) if self.cache_dir else directory / "lock"
            with lock_path.open("a+b") as lock:
                deadline = time.monotonic() + self.timeout
                while True:
                    self._cancel()
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() > deadline:
                            raise ProviderError("字幕同步检查仍在运行，未重复启动。")
                        time.sleep(.1)
                cache = self.cache_dir / (key + ".json") if self.cache_dir else None
                observations, cache_hit = None, False
                if cache and cache.is_file() and cache.stat().st_size <= 20_000_000:
                    try:
                        saved = json.loads(cache.read_text())
                        if saved.get("key") == key and saved.get("observations", {}).get("checker_identity") == self.identity:
                            observations, cache_hit = saved["observations"], True
                    except (OSError, ValueError, TypeError):
                        pass
                if observations is None:
                    normalized, request, output = directory / "audio.wav", directory / "request.json", directory / "result.json"
                    self._run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(audio),
                               "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(normalized)],
                              min(self.timeout, 120))
                    request.write_text(json.dumps({"audio": str(normalized), "model_dir": str(self.model.parent),
                        "coverage_model_dir": str(self.coverage_model.parent) if self.backend == "sense_voice" else None,
                        "captions": captions, "threads": self.threads}, ensure_ascii=False))
                    self._run([str(self.python), str(self.worker), str(request), str(output)], self.timeout)
                    if not output.is_file() or output.stat().st_size > 20_000_000:
                        raise ProviderError("本地字幕独立检查未返回有效结果。")
                    observations = json.loads(output.read_text())
                    if not isinstance(observations, dict) or not isinstance(observations.get("cues"), list):
                        raise ProviderError("本地字幕独立检查结果格式无效。")
                    if self.backend == "sense_voice":
                        observations = self._crosscheck_uncovered(audio, observations)
                    observations.update(independent=True, method="independent_ctc", audio_sha256=audio_hash,
                                        caption_hash=caps_hash, checker_identity=self.identity)
                    self._cancel()
                    self._assert_unchanged(audio, audio_hash)
                    if (_sha(audio) != audio_hash or _sha(self.model) != self._identity["model_sha256"]
                            or _sha(self.tokens) != self._identity["tokens_sha256"]
                            or _sha(self.worker) != self._identity["worker_sha256"]):
                        raise ProviderError("检查期间音轨或检查模型发生改变，旧证据已作废。")
                    if cache:
                        fd, name = tempfile.mkstemp(prefix=".sync-", dir=self.cache_dir)
                        try:
                            with os.fdopen(fd, "w") as stream:
                                json.dump({"key": key, "observations": observations}, stream, ensure_ascii=False, allow_nan=False)
                                stream.flush(); os.fsync(stream.fileno())
                            os.replace(name, cache)
                        finally:
                            if os.path.exists(name):
                                os.unlink(name)
        self._cancel()
        self._assert_unchanged(audio, audio_hash)
        self.last_observations = deepcopy(observations)
        self.last_metadata = {"cache_key": key, "cache_hit": cache_hit, "identity": self.identity,
                              "audio_sha256": audio_hash, "caption_hash": caps_hash,
                              "changes": deepcopy(self.changes)}
        self._last_audio, self._last_caption_hash = audio, caps_hash
        return deepcopy(observations)

    def repair(self, captions, report, attempt):
        from .subtitle_sync import caption_hash
        self._cancel()
        if caption_hash(captions) != self._last_caption_hash:
            raise ProviderError("字幕已改变，不能使用旧同步证据修复。")
        output = deepcopy(captions)
        for aligned in report.get("aligned_cues", []):
            index = aligned["cue_index"]
            if 0 <= index < len(output):
                old = deepcopy(output[index])
                output[index].update(start=aligned["start"], end=aligned["end"])
                if output[index] != old:
                    self.changes.append({"kind": "realigned", "cue_index": index, "attempt": attempt,
                                         "before": old, "after": deepcopy(output[index])})
        # A different decoder's failure to hear a Chinese/English cue alone
        # cannot delete it. A bounded broad-context Whisper double check is
        # requested only when full CTC and both local crops have zero support.
        excluded = set()
        from .subtitle_noise import confirm_unsupported_caption
        for row in self.last_observations.get("unsupported_cues", []):
            if (row.get("independent_text_in_interval") or len(row.get("checks", [])) != 2
                    or any(c.get("matched_anchors", 1) != 0 for c in row["checks"])):
                continue
            evidence = confirm_unsupported_caption(self._last_audio, captions[row["cue_index"]], self.config,
                                                  cache_dir=self.cache_dir or self._last_audio.parent / ".sync-noise",
                                                  cancel_check=self.cancel_check)
            if evidence.get("unsupported") is True:
                excluded.add(row["cue_index"])
                self.changes.append({"kind": "unsupported_audio_excluded", "cue_index": row["cue_index"],
                                     "attempt": attempt, "before": deepcopy(captions[row["cue_index"]]),
                                     "evidence": evidence, "independent_ctc": row})
        output = [c for index, c in enumerate(output) if index not in excluded]
        # Estimated last-character padding must never cover the next subtitle.
        for index in range(len(output) - 1):
            if output[index]["end"] > output[index + 1]["start"]:
                if output[index]["start"] >= output[index + 1]["start"]:
                    raise ProviderError("独立对齐结果顺序矛盾，未覆盖现有字幕。")
                output[index]["end"] = output[index + 1]["start"]
        additions = self.last_observations.get("supported_additions", [])
        if isinstance(additions, list) and additions:
            duration = probe(self._last_audio)["duration"]
            for addition in additions[:1000]:
                if not self._addition_supported(addition):
                    continue
                cue = {"start": addition["start"], "end": addition["end"], "text": addition["text"]}
                if (cue["start"] < 0 or cue["end"] > duration or cue["end"] <= cue["start"]
                        or any(cue["start"] < c["end"] and cue["end"] > c["start"] for c in output)):
                    continue
                output.append(cue)
                self.changes.append({"kind": "added_from_audio", "attempt": attempt,
                                     "after": deepcopy(cue), "evidence": deepcopy(addition)})
            output.sort(key=lambda row: (row["start"], row["end"]))
        self.last_metadata["changes"] = deepcopy(self.changes)
        self._cancel()
        return output

    @staticmethod
    def _addition_supported(addition):
        if (not isinstance(addition, dict) or addition.get("method") != "two_models_two_contexts_original_clock"
                or addition.get("models") != ["sense_voice", "zipformer_ctc"]
                or not isinstance(addition.get("text"), str) or not 1 < len(addition["text"].strip()) <= 4000
                or "\x00" in addition["text"] or type(addition.get("start")) not in (int, float)
                or type(addition.get("end")) not in (int, float)
                or not math.isfinite(addition["start"]) or not math.isfinite(addition["end"])):
            return False
        checks = addition.get("checks")
        if not isinstance(checks, list) or len(checks) != 4:
            return False
        for model in ("sense_voice", "zipformer_ctc"):
            pair = [c for c in checks if isinstance(c, dict) and c.get("backend") == model]
            if len(pair) != 2:
                return False
            try:
                if any(type(c.get("origin")) not in (int, float) or not math.isfinite(c["origin"]) for c in pair):
                    return False
                if abs(pair[0]["origin"] - pair[1]["origin"]) < .4:
                    return False
                for check in pair:
                    e = check["alignment"]
                    if (e.get("consistent") is not True or e.get("boundary_supported") is not True
                            or e.get("coverage") != 1. or e.get("matched_anchors", 0) < 2
                            or type(e.get("matched_anchors")) is not int
                            or e.get("target_units") != e["matched_anchors"]
                            or any(type(e.get(k)) not in (int, float) or not math.isfinite(e[k]) for k in ("start", "end"))
                            or e["end"] <= e["start"]
                            or abs(e["start"] - addition["start"]) > .28
                            or abs(e["end"] - addition["end"]) > .28):
                        return False
            except (KeyError, TypeError):
                return False
        return True
