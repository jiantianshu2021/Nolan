import unittest
from norlan.subtitle_script import prefer_script, dialogue_candidates


class ScriptCaptionTests(unittest.TestCase):
    def setUp(self):
        self.ep = {'shots': [{'id': 's1', 'duration': 15, 'dialogue': '楚阳：要我喂你？\n系统：限高10米；自平衡；追踪缓冲。'},
                             {'id': 's2', 'duration': 15, 'dialogue': '角色：普通售价2999元。'}]}

    def cue(self, text, start=2):
        return {'start': start, 'end': start + 1, 'text': text}

    def test_script_is_primary_literal_text_candidate_and_timing_is_retained(self):
        raw = [self.cue('要喂你？'), self.cue('追逐缓冲。', 8)]
        chosen, proof = prefer_script(raw, self.ep)
        self.assertEqual([c['text'] for c in chosen], ['要我喂你？', '追踪缓冲。'])
        self.assertEqual([c['start'] for c in chosen], [2, 8])
        self.assertEqual(raw[0]['text'], '要喂你？')
        self.assertFalse(proof['selections'][0]['audio_verified'])
        self.assertNotIn('系统：', str(chosen))

    def test_never_adds_unrecognized_script_lines_or_wrong_episode_text(self):
        raw = [self.cue('没说过的别的话'), self.cue('MING PAO CANADA', 6)]
        chosen, _ = prefer_script(raw, self.ep)
        self.assertEqual(chosen, raw)
        self.assertEqual(prefer_script([], self.ep)[0], [])

    def test_numbers_are_not_overwritten_by_intended_script_price(self):
        raw = [self.cue('普通售价2900元。', 18)]
        self.assertEqual(prefer_script(raw, self.ep)[0], raw)

    def test_only_nearby_shot_dialogue_is_eligible(self):
        raw = [self.cue('追逐缓冲。', 25)]
        self.assertEqual(prefer_script(raw, self.ep)[0], raw)
        self.assertTrue(all('角色：' not in x['text'] and '系统：' not in x['text'] for x in dialogue_candidates(self.ep)))

    def test_ambiguous_script_candidates_keep_real_recognition(self):
        self.ep['shots'][0]['dialogue'] = '甲：追踪缓冲。\n乙：追租缓冲。'
        raw = [self.cue('追逐缓冲。')]
        self.assertEqual(prefer_script(raw, self.ep)[0], raw)

    def test_short_actual_speech_cannot_pull_in_a_whole_script_sentence(self):
        self.ep['shots'][0]['dialogue'] = '甲：好我们现在去把东西拿过来。'
        raw = [self.cue('好')]
        self.assertEqual(prefer_script(raw, self.ep)[0], raw)
