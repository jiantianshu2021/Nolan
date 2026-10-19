"""Synthetic concise-broadcast review UI checks; no model or media calls."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from norlan import dashboard


def synthetic_candidate():
    source = "民用产品发布。普通售价2999元。震惊值升级加工台；满意值兑换图纸。"
    spoken = "产品已就绪。"
    shot = {"id": "synthetic-shot", "status": "awaiting_prompt_approval", "version": 0,
            "duration": 15, "dialogue": "楚阳：知道了。\n系统播报员：" + source,
            "subshots": [{"start": 0, "end": 5, "prompt": "角色看向加工台。"},
                         {"start": 5, "end": 15, "prompt": "任务框出现，详细规则保留。"}]}
    candidate = {"id": "synthetic-candidate", "version": 1, "revision": 0,
        "request_hash": "a" * 64, "prompt": "合成测试：人物对白不变，系统短播，规则全文在画面上保留。",
        "dialogue": "楚阳：知道了。\n系统播报员：" + spoken,
        "broadcast_proposal": {"speaker": "系统播报员", "source_text": source,
            "spoken_text": spoken, "display_text": source, "summary": "只读关键结果，详细规则留在画面上。"},
        "timing_proposal": {"summary": "GPT 安排声音，保持原切点与总时长。",
            "subshots": copy.deepcopy(shot["subshots"]),
            "audio_windows": [{"start": .4, "end": 1.6, "speaker": "楚阳", "text": "知道了。"},
                              {"start": 3, "end": 5.5, "speaker": "系统播报员", "text": spoken}]}}
    return shot, candidate


class BroadcastProjectionTests(unittest.TestCase):
    def test_short_speech_and_its_timing_survive_projection_without_request_internals(self):
        shot, candidate = synthetic_candidate()
        original = copy.deepcopy(candidate)
        candidate["broadcast_proposal"]["provider_raw"] = "SECRET-BROADCAST"
        candidate["timing_proposal"]["provider_raw"] = "SECRET-TIMING"
        candidate["reference_manifest"] = {"audio": ["SECRET-PATH"], "context": {
            "audio_references": [{"index": 1, "asset_id": "voice-system", "url": "SECRET-URL"}]}}
        before = copy.deepcopy((shot, candidate))
        public = dashboard._public_prompt_candidate(candidate, shot)
        self.assertEqual(public["broadcast_proposal"], original["broadcast_proposal"])
        self.assertEqual(public["timing_proposal"], original["timing_proposal"])
        self.assertEqual(public["dialogue"], original["dialogue"])
        self.assertEqual(public["reference_manifest"]["audio_references"], [{"index": 1, "asset_id": "voice-system"}])
        self.assertNotIn("SECRET-", json.dumps(public))
        self.assertEqual((shot, candidate), before, "GET projection must not mutate consent or the plan")

    def test_invalid_broadcast_cannot_expose_a_misleading_short_dialogue_schedule(self):
        for label in ("display", "dialogue", "source", "missing-field", "wrong-type"):
            with self.subTest(label=label):
                shot, candidate = synthetic_candidate()
                if label == "display":
                    candidate["broadcast_proposal"]["display_text"] = "规则被删去了。"
                elif label == "dialogue":
                    candidate["dialogue"] = candidate["dialogue"].replace("知道了", "改写了")
                elif label == "source":
                    candidate["broadcast_proposal"]["source_text"] = "编造的原文。"
                elif label == "missing-field":
                    candidate["broadcast_proposal"].pop("spoken_text")
                else:
                    candidate["broadcast_proposal"]["summary"] = {"private": "SECRET-RAW"}
                public = dashboard._public_prompt_candidate(candidate, shot)
                self.assertIsNone(public["broadcast_proposal"])
                self.assertIsNone(public["timing_proposal"])
                self.assertNotIn("SECRET-", json.dumps(public))

    def test_legacy_candidate_remains_readable_without_a_broadcast_proposal(self):
        shot, candidate = synthetic_candidate()
        candidate.pop("broadcast_proposal")
        candidate.pop("timing_proposal")
        candidate["dialogue"] = shot["dialogue"]
        public = dashboard._public_prompt_candidate(candidate, shot)
        self.assertEqual(public["dialogue"], shot["dialogue"])
        self.assertNotIn("broadcast_proposal", public)
        self.assertNotIn("timing_proposal", public)


@unittest.skipUnless(shutil.which("node"), "Node.js required for pure DOM rendering test")
class BroadcastRenderingTests(unittest.TestCase):
    def test_editorial_details_are_folded_and_share_the_single_full_prompt_confirmation(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "promptApproval"):
            start = html.index("function " + name + "(")
            sources.append(html[start:html.index("\nfunction ", start + 1)])
        shot, candidate = synthetic_candidate()
        shot["prompt_candidate"] = dashboard._public_prompt_candidate(candidate, shot)
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const payload=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.dataset={};this.listeners={};}
 append(...nodes){this.children.push(...nodes);}
 addEventListener(key,handler){this.listeners[key]=handler;}
}
const decisions=[];
const context={document:{createElement:tag=>new Element(tag)},
 state:{paused:false,dashboard:{running:false},assets:[],asset_amendments:[]},
 perform:(...args)=>decisions.push(args),shot:payload.shot,decisions,assert};
vm.runInNewContext(payload.sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const result=promptApproval(shot),nodes=walk(result),text=nodes.map(n=>n.textContent).join(' · ');
const details=nodes.find(n=>n.tag==='details'&&n.className==='broadcast-proposal');
assert.ok(details);assert.equal(Boolean(details.open),false);
assert.equal(details.children[0].textContent,'系统播报取舍');
assert.equal(text.includes(shot.prompt_candidate.broadcast_proposal.spoken_text),true);
assert.equal(text.includes(shot.prompt_candidate.broadcast_proposal.display_text),true);
assert.equal(text.includes(shot.prompt_candidate.broadcast_proposal.summary),true);
assert.equal(text.includes('无需单独确认秒数'),true);
assert.ok(nodes.some(n=>n.tag==='details'&&n.className==='timing-proposal'));
const buttons=nodes.filter(n=>n.tag==='button');assert.equal(buttons.length,2);
assert.equal(buttons[0].textContent,'确认提示词并生成');
assert.equal(buttons[1].textContent,'修改提示词');
assert.equal(decisions.length,0,'rendering must not approve or generate anything');
buttons[0].listeners.click();assert.equal(decisions.length,1);
assert.equal(decisions[0][0].action,'approve_prompt');
assert.equal(decisions[0][0].candidate_id,shot.prompt_candidate.id);
assert.equal(decisions[0][0].request_hash,shot.prompt_candidate.request_hash);
`,context);
'''
        done = subprocess.run([shutil.which("node"), "-e", program],
            input=json.dumps({"sources": sources, "shot": shot}), text=True, capture_output=True, timeout=15)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
