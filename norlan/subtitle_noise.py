"""Conservative extra evidence for a caption with zero independent CTC support.

Two original-clock contexts are recognized without a script, prompt or VAD
boundaries. Failure to recognize is never sufficient by itself to delete a
caption: the caller must additionally establish zero CTC support in both
contexts and no other independently observed text in the cue's interval.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import wave

from .local_transcription import DEFAULT_MODEL_PATH, MAX_AUDIO_BYTES, _crop_pcm, _sha256
from .media import probe


METHOD = "two_origin_whisper_context"
SCHEMA = "nolan-subtitle-noise-v1"
PARAMETERS = {"contexts": [[3.0, 2.0], [4.3, 3.2]], "max_len": 0,
              "suppress_non_speech_tokens": True, "sample_rate": 16000,
              "matcher_policy": 1}
MAX_JSON_BYTES = 8_000_000
_HAN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _number(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid finite number")
    return float(value)


def _normalized(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def _supports_caption(target: str, transcript: str) -> bool:
    """Err toward retaining text whenever a meaningful fragment supports it."""
    target, observed = _normalized(target), _normalized(transcript)
    if not target or not observed:
        return False
    if target in observed or (len(observed) >= 2 and observed in target):
        return True
    matching = SequenceMatcher(None, target, observed, autojunk=False)
    longest = matching.find_longest_match(0, len(target), 0, len(observed)).size
    if _HAN.search(target):
        # Even a partial short Chinese phrase is counterevidence to deletion.
        if longest >= (1 if len(target) <= 2 else 2):
            return True
    elif longest >= min(4, len(target)):
        return True
    matched = sum(block.size for block in matching.get_matching_blocks())
    if matched >= 3 and matched / len(target) >= 0.4:
        return True
    return False


def _transcript(response) -> str:
    if not isinstance(response, dict) or not isinstance(response.get("transcription"), list):
        raise ValueError("invalid transcription")
    if len(response["transcription"]) > 10_000:
        raise ValueError("oversized transcription")
    result = []
    for row in response["transcription"]:
        text = row.get("text") if isinstance(row, dict) else None
        if not isinstance(text, str) or len(text) > 4000 or "\x00" in text:
            raise ValueError("invalid transcription text")
        result.append(text.strip())
    text = " ".join(result)
    if len(text) > 20_000 or (not result and str(response.get("text", "")).strip()):
        raise ValueError("invalid untimed response")
    return text


def _safe_result(transcripts, *, identity=None, unsupported=False, diagnostic="",
                 matcher=None, cache_hit=False) -> dict:
    return {"unsupported": unsupported is True, "method": METHOD,
            "evidence_hash": _digest({"identity": identity, "transcripts": transcripts,
                                      "unsupported": unsupported is True,
                                      "diagnostic": diagnostic, "matcher": matcher}),
            "transcripts": list(transcripts), "diagnostic": diagnostic,
            "cache_hit": cache_hit}


def confirm_unsupported_caption(audio: Path, cue: dict, config: dict, *, cache_dir: Path,
                                cancel_check=None) -> dict:
    """Return supplemental evidence; any uncertainty preserves the caption.

    Successful results cache only the two actual local transcriptions. A
    caption with Chinese text also gets a phonetic comparison through the
    existing isolated checker runtime (``alignment_python``); no new package
    or key is required. Unavailable phonetic comparison cannot reject a
    Chinese caption. All subprocesses use fixed argv and no network service.
    """
    transcripts, identity = ["", ""], None
    try:
        def cancelled():
            return bool(cancel_check and cancel_check())

        def check_cancel():
            if cancelled():
                raise InterruptedError("cancelled")

        check_cancel()
        if not isinstance(config, dict) or config.get("mode", "local") != "local":
            raise ValueError("invalid config")
        if not isinstance(cue, dict):
            raise ValueError("invalid cue")
        start, end = _number(cue.get("start")), _number(cue.get("end"))
        target = cue.get("text")
        if (not isinstance(target, str) or not target.strip() or len(target) > 4000
                or "\x00" in target or not _normalized(target) or start < 0 or end <= start or end - start > 60):
            raise ValueError("invalid cue")
        language, threads = config.get("language", "zh"), config.get("threads", 4)
        timeout = _number(config.get("timeout", 600))
        if (not isinstance(language, str) or not re.fullmatch(r"[a-z]{2}|auto", language)
                or type(threads) is not int or not 1 <= threads <= 32 or not 1 <= timeout <= 3600):
            raise ValueError("invalid recognition parameters")
        executable = config.get("executable", "whisper-cli")
        if not isinstance(executable, str) or not executable or "\x00" in executable:
            raise ValueError("invalid executable")
        executable = shutil.which(executable)
        model = Path(config.get("model_path", DEFAULT_MODEL_PATH)).expanduser().resolve()
        path = Path(audio).expanduser().resolve()
        if (not executable or not model.is_file() or not path.is_file()
                or not 0 < path.stat().st_size <= MAX_AUDIO_BYTES):
            raise ValueError("missing local input or model")
        executable = str(Path(executable).resolve())
        metadata = probe(path)
        duration = _number(metadata.get("duration"))
        if not metadata.get("has_audio") or not 0 < duration <= 3600 or end > duration + .000001:
            raise ValueError("invalid audio duration")
        crops = [[round(max(0., start - before) * 16000),
                  round(min(duration, end + after) * 16000)] for before, after in PARAMETERS["contexts"]]
        if crops[0][0] == crops[1][0] or any(first >= last for first, last in crops):
            return _safe_result(transcripts, diagnostic="distinct_origins_unavailable")
        audio_hash, model_hash, exe_hash = _sha256(path), _sha256(model), _sha256(Path(executable))
        identity = {"schema": SCHEMA, "audio_sha256": audio_hash,
                    "cue": {"start": start, "end": end, "text": target.strip()},
                    "crop_samples": crops, "model_sha256": model_hash,
                    "executable_sha256": exe_hash, "language": language, "threads": threads,
                    "parameters": PARAMETERS}
        key = _digest(identity)
        cache = Path(cache_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout

        def run(args, *, input_text=None, capture=False):
            check_cancel()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("local deadline")
            completed = subprocess.run(args, input=input_text, stdin=None if input_text is not None else subprocess.DEVNULL,
                                       stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, text=capture or input_text is not None,
                                       timeout=remaining, check=False)
            check_cancel()
            if completed.returncode:
                raise ValueError("local recognition failed")
            return completed.stdout if capture else ""

        cache_hit = False
        with (cache / (key + ".lock")).open("a+b") as lock:
            while True:
                check_cancel()
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("cache lock deadline")
                    time.sleep(.1)
            target_cache = cache / (key + ".json")
            try:
                if target_cache.is_file() and target_cache.stat().st_size <= 100_000:
                    saved = json.loads(target_cache.read_text(encoding="utf-8"))
                    values = saved.get("transcripts")
                    if (saved.get("identity") == identity and isinstance(values, list) and len(values) == 2
                            and all(isinstance(t, str) and len(t) <= 20_000 and "\x00" not in t for t in values)):
                        transcripts, cache_hit = values, True
            except (OSError, ValueError, TypeError, AttributeError):
                pass
            if not cache_hit:
                with tempfile.TemporaryDirectory(prefix=".noise-", dir=cache) as temp:
                    directory, normalized = Path(temp), Path(temp) / "normalized.wav"
                    native = False
                    try:
                        with wave.open(str(path), "rb") as incoming:
                            native = (incoming.getframerate() == 16000 and incoming.getnchannels() == 1
                                      and incoming.getsampwidth() == 2 and incoming.getcomptype() == "NONE")
                    except (OSError, wave.Error, EOFError):
                        pass
                    if native:
                        shutil.copyfile(path, normalized)
                    else:
                        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                             "-protocol_whitelist", "file,pipe", "-i", str(path), "-vn", "-map", "0:a:0",
                             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(normalized)])
                    if _sha256(path) != audio_hash:
                        raise ValueError("audio changed")
                    for index, (first, last) in enumerate(crops):
                        check_cancel()
                        cropped, output = directory / f"context-{index}.wav", directory / f"result-{index}"
                        _crop_pcm(normalized, cropped, first / 16000, last / 16000)
                        run([executable, "-m", str(model), "-f", str(cropped), "-l", language, "-t", str(threads),
                             "-ml", "0", "-oj", "-of", str(output), "-sns", "-np"])
                        response = output.with_suffix(".json")
                        if not response.is_file() or response.stat().st_size > MAX_JSON_BYTES:
                            raise ValueError("invalid local response")
                        transcripts[index] = _transcript(json.loads(response.read_text(encoding="utf-8")))
                    check_cancel()
                    if (_sha256(path) != audio_hash or _sha256(model) != model_hash
                            or _sha256(Path(executable)) != exe_hash):
                        raise ValueError("input or model changed")
                    published = directory / "cache.json"
                    published.write_text(json.dumps({"identity": identity, "transcripts": transcripts},
                                                    ensure_ascii=False, allow_nan=False), encoding="utf-8")
                    os.replace(published, target_cache)

        check_cancel()
        if any(_supports_caption(target, text) for text in transcripts):
            return _safe_result(transcripts, identity=identity, diagnostic="caption_supported",
                                matcher="unicode_fuzzy_v1", cache_hit=cache_hit)
        matcher = "unicode_fuzzy_v1"
        if _HAN.search(target):
            python = Path(config.get("alignment_python", Path.home() / ".local/share/norlan/alignment-runtime/bin/python")).expanduser().absolute()
            if not python.is_file():
                return _safe_result(transcripts, identity=identity, diagnostic="phonetic_check_unavailable", cache_hit=cache_hit)
            # Reuse the checker's existing pure-Python pinyin dependency. The
            # payload contains only candidate/observed text, never a prompt.
            code = ("import json,sys,pypinyin,unicodedata;"
                    "texts=json.load(sys.stdin);"
                    "print(json.dumps({'version':pypinyin.__version__,'units':"
                    "[pypinyin.lazy_pinyin(unicodedata.normalize('NFKC',t).lower(),"
                    "style=pypinyin.Style.NORMAL,errors=lambda s:list(s),strict=False) for t in texts]}))")
            try:
                converted = json.loads(run([str(python), "-c", code],
                                           input_text=json.dumps([target, *transcripts], ensure_ascii=False), capture=True))
                values, version = converted.get("units"), converted.get("version")
                if (not isinstance(values, list) or len(values) != 3 or not isinstance(version, str)
                        or not all(isinstance(row, list) and all(isinstance(u, str) and len(u) <= 100 for u in row)
                                   for row in values)):
                    raise ValueError("invalid phonetic response")
                phonetic = ["".join(row) for row in values]
                matcher = {"method": "unicode_fuzzy_and_phonetic_v1", "pypinyin": version}
                if any(_supports_caption(phonetic[0], text) for text in phonetic[1:]):
                    return _safe_result(transcripts, identity=identity, diagnostic="caption_phonetically_supported",
                                        matcher=matcher, cache_hit=cache_hit)
            except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired):
                return _safe_result(transcripts, identity=identity, diagnostic="phonetic_check_unavailable", cache_hit=cache_hit)
        check_cancel()
        if (_sha256(path) != audio_hash or _sha256(model) != model_hash or _sha256(Path(executable)) != exe_hash):
            raise ValueError("input or model changed")
        return _safe_result(transcripts, identity=identity, unsupported=True,
                            diagnostic="both_contexts_without_caption_support", matcher=matcher, cache_hit=cache_hit)
    except InterruptedError:
        return _safe_result(transcripts, identity=identity, diagnostic="cancelled")
    except Exception as exc:
        # Never expose exception text, CLI output, paths or credentials.
        return _safe_result(transcripts, identity=identity, diagnostic="local_check_failed_" + type(exc).__name__)
