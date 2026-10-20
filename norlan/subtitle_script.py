"""Bounded script text candidates; acoustic evidence still decides alignment.

Never feed the novel, history or visual UI text into caption selection. Match
only nearby shot dialogue, preserve real ASR timing, and do not change numbers
from a script when actual recognition disagrees.
"""
from copy import deepcopy
from difflib import SequenceMatcher
import re
import unicodedata

POLICY = 1


def _norm(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def _numbers(text):
    return re.findall(r'[0-9零〇一二三四五六七八九十百千万亿两]+', unicodedata.normalize('NFKC', text))


def dialogue_candidates(episode):
    """Extract literal dialogue only, with per-shot timing and speaker removed."""
    result, cursor = [], 0.
    for shot in episode.get('shots', []):
        duration = shot.get('duration', 0)
        if type(duration) not in (int, float) or not 0 < duration <= 300:
            continue
        rows = []
        for line in str(shot.get('dialogue') or '')[:6000].splitlines():
            line = line.strip()
            if not line:
                continue
            # Character / voice attribution is metadata, never a spoken word.
            line = re.sub(r'^[^：:\n]{1,40}[：:]\s*', '', line).strip()
            clauses = [part.strip() for part in re.findall(r'[^，、。！？；,!?;]+[，、。！？；,!?;]?', line) if part.strip()]
            for first in range(len(clauses)):
                for count in range(1, min(3, len(clauses) - first) + 1):
                    text = ''.join(clauses[first:first + count])
                    if 1 <= len(_norm(text)) <= 100:
                        rows.append(text)
        for text in dict.fromkeys(rows):
            result.append({'text': text, 'shot_id': shot.get('id'), 'start': cursor, 'end': cursor + duration})
        cursor += duration
    return result


def prefer_script(captions, episode):
    """Choose close literal script clauses, never invent timestamps or cues.

    The returned candidates are NOT verified captions. A different audio
    recognizer must corroborate them; otherwise the QA gate cannot pass. A
    missed/changed number always keeps its actual ASR form for later review.
    """
    choices = dialogue_candidates(episode)
    output, selections = deepcopy(captions), []
    for index, cue in enumerate(output):
        target = _norm(cue.get('text', ''))
        if len(target) < 2:
            continue
        center = (cue['start'] + cue['end']) / 2
        scored = []
        for row in choices:
            if not row['start'] - 2 <= center <= row['end'] + 2:
                continue
            text = _norm(row['text'])
            if not text or min(len(text), len(target)) / max(len(text), len(target)) < .65:
                continue
            # Literal digit disagreements cannot be repaired from the script.
            # Number formatting conversion requires actual phonetic evidence.
            if _numbers(target) != _numbers(text):
                continue
            score = SequenceMatcher(None, target, text, autojunk=False).ratio()
            if score >= .74:
                scored.append((score, row))
        if not scored:
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        best, row = scored[0]
        alternatives = [score for score, other in scored[1:] if _norm(other['text']) != _norm(row['text'])]
        if alternatives and best - alternatives[0] < .08:
            continue
        previous = cue['text']
        if target != _norm(row['text']):
            cue['text'] = row['text']
        selections.append({'cue_index': index, 'shot_id': row['shot_id'], 'script_text': row['text'],
                           'recognition_text': previous, 'candidate_text': cue['text'], 'similarity': round(best, 4),
                           'audio_verified': False})
    return output, {'policy': POLICY, 'method': 'script_preferred_audio_verified', 'selections': selections,
                    'note': '优先剧本中的邻近台词；候选必须经实际音轨独立核验，未说出的台词不直接上字幕。'}
