"""Synthetic candidate timing review; no GPT, provider, or real project calls."""
import copy
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from norlan import core, dashboard, pipeline, prompt_gate


def original_shot():
    return {"id": "E001-S001", "status": "planned", "version": 0, "revision": 1,
            "duration": 10, "scene_id": "station", "asset_ids": ["hero", "station"],
            "prompt": "合成测试：先看人物，再切向任务框。", "dialogue": "系统：绑定成功。",
            "feedback": "交给GPT准备提示词。", "subshots": [
                {"start": 0, "end": 5, "prompt": "中景，人物放下手中的工具。"},
                {"start": 5, "end": 10, "prompt": "近景，系统任务框。"}]}


def timing_proposal():
    return {"summary": "镜头切点保持第5秒，任务框在第3秒出现，系统声音跨切镜连续说完。",
            "subshots": [
                {"start": 0, "end": 5, "prompt": "中景，人物放下工具；第3秒任务框出现，系统女声开始。"},
                {"start": 5, "end": 10, "prompt": "近景承接已有任务框，系统女声连续说完，不重新播报。"}],
            "audio_windows": [{"start": 3, "end": 7.5, "speaker": "系统", "text": "绑定成功。"}]}


def candidate_for(shot):
    return {"id": "synthetic-candidate", "version": 1, "revision": shot["revision"],
            "request_hash": "a" * 64, "prompt": "合成最终提示词：保持10秒，女声跨第5秒切镜连续播报。",
            "dialogue": shot["dialogue"], "duration": shot["duration"],
            "continuity_notes": "延续前段的空间轴线和环境底声。", "timing_proposal": timing_proposal()}


class TimingProposalProjectionTests(unittest.TestCase):
    def setUp(self):
        self.shot = original_shot()
        self.candidate = candidate_for(self.shot)

    def public(self):
        return dashboard._public_prompt_candidate(self.candidate, self.shot)

    def test_safe_draft_is_complete_and_does_not_modify_the_approved_camera_plan(self):
        proposal = self.candidate["timing_proposal"]
        proposal.update(raw_config="PRIVATE-CONFIG", signed_url="https://private.invalid")
        proposal["subshots"][0]["config"] = "PRIVATE-SHOT-CONFIG"
        proposal["audio_windows"][0]["raw"] = "PRIVATE-AUDIO-RESPONSE"
        self.candidate.update(effective_subshots=[{"raw": "PRIVATE-EFFECTIVE"}], timing_summary="PRIVATE-SUMMARY")
        before = copy.deepcopy((self.shot, self.candidate))
        public = self.public()
        self.assertEqual(public["timing_proposal"], timing_proposal())
        self.assertEqual(public["prompt"], self.candidate["prompt"])
        self.assertNotIn("PRIVATE-", json.dumps(public))
        self.assertNotIn("https://", json.dumps(public))
        self.assertEqual((self.shot, self.candidate), before)
        self.assertNotEqual(public["timing_proposal"]["subshots"], self.shot["subshots"])
        self.assertEqual([(x["start"], x["end"]) for x in public["timing_proposal"]["subshots"]],
                         [(0, 5), (5, 10)])

    def test_invalid_or_out_of_range_audio_is_not_exposed_as_a_valid_schedule(self):
        original = copy.deepcopy(self.candidate)
        for changes in ({"start": True}, {"end": 11}, {"start": float("nan")},
                        {"end": float("inf")}, {"end": 10 ** 1000}, {"start": -1},
                        {"end": 3}, {"speaker": "新角色"}, {"text": "改写后的台词。"}):
            with self.subTest(changes=changes):
                self.candidate = copy.deepcopy(original)
                self.candidate["timing_proposal"]["audio_windows"][0].update(changes)
                self.assertIsNone(self.public()["timing_proposal"])
        for windows in ([], timing_proposal()["audio_windows"] * 31, ["invalid"]):
            self.candidate = copy.deepcopy(original)
            self.candidate["timing_proposal"]["audio_windows"] = windows
            self.assertIsNone(self.public()["timing_proposal"])

    def test_changed_camera_cuts_duration_or_dialogue_never_claim_to_preserve_the_plan(self):
        for changes in ("cut", "duration", "count", "dialogue"):
            self.candidate = candidate_for(self.shot)
            with self.subTest(changes=changes):
                proposal = self.candidate["timing_proposal"]
                if changes == "cut":
                    proposal["subshots"][0]["end"] = 4
                    proposal["subshots"][1]["start"] = 4
                elif changes == "duration":
                    proposal["subshots"][-1]["end"] = 15
                elif changes == "count":
                    proposal["subshots"] = [{"start": 0, "end": 10, "prompt": "改变镜头数量。"}]
                else:
                    self.candidate["dialogue"] = "系统：新的播报。"
                self.assertIsNone(self.public()["timing_proposal"])

    def test_optional_draft_is_backward_compatible_and_silent_scenes_are_explicit(self):
        self.candidate.pop("timing_proposal")
        self.assertNotIn("timing_proposal", self.public())
        self.candidate["timing_proposal"] = None
        self.assertIsNone(self.public()["timing_proposal"])
        self.shot["dialogue"] = self.candidate["dialogue"] = ""
        self.candidate["timing_proposal"] = timing_proposal()
        self.candidate["timing_proposal"]["audio_windows"] = []
        self.assertEqual(self.public()["timing_proposal"]["audio_windows"], [])


class TimingProposalHttpTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        core.create_project(self.root, "合成网页审批测试小说，不调用任何服务。", "live")
        plan = pipeline.demo_plan()
        plan["episodes"] = plan["episodes"][:1]
        plan["episodes"][0]["shots"] = [original_shot()]
        core.install_plan(self.root, plan)
        core.approve(self.root, "plan", "plan", 1)
        with core.transaction(self.root) as state:
            for asset in state["assets"]:
                relative = "media/assets/" + asset["id"] + ".png"
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"SYNTHETIC APPROVED ASSET")
                sha = core.digest(path)
                asset.update(file=relative, sha256=sha, status="approved", version=1,
                             approved_at=core.now(), approved_sha256=sha)
            shot = state["episodes"][0]["shots"][0]
            shot.update(version=0, revision=1, feedback="系统播报需要合适的连续时间。")
            self.signature = hashlib.sha256(json.dumps([shot["revision"], shot["feedback"]],
                                                       ensure_ascii=False).encode()).hexdigest()
            shot["revision_analysis"] = {"signature": self.signature, "summary": "原定声音窗口较短。",
                                         "blockers": ["旧流程要求用户指定播报秒点。"]}
            state.update(status="blocked", last_error="旧流程要求用户指定播报秒点。")
        self.original = core.load_project(self.root)
        self.prepared = threading.Event()
        self.prepare_calls = 0

        def synthetic_prepare_only(root):
            self.prepare_calls += 1
            state = core.load_project(root)
            shot = state["episodes"][0]["shots"][0]
            refined = {"prompt": "合成最终画面提示词。", "dialogue": shot["dialogue"],
                       "continuity_notes": "原摄影切点和总时长保留。", "timing_proposal": timing_proposal()}
            manifest = {"images": [asset["file"] for asset in state["assets"]], "audio": [],
                        "context": {"image_references": [
                            {"index": index + 1, "asset_id": asset["id"], "sha256": asset["sha256"]}
                            for index, asset in enumerate(state["assets"])], "audio_references": []}}
            prompt = refined["prompt"] + "\n" + prompt_gate.candidate_timeline_prompt(shot, refined)
            prompt_gate.prepare(root, shot["id"], refined, prompt, manifest)
            self.prepared.set()
            return core.load_project(root)

        self.run = self.enterContext(patch.object(dashboard, "_run", side_effect=synthetic_prepare_only))
        self.enterContext(patch.object(pipeline, "_brain", side_effect=AssertionError("No GPT calls")))
        self.enterContext(patch.object(pipeline, "_ark", side_effect=AssertionError("No media calls")))
        self.server = dashboard.make_server(self.root, port=0)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, payload=None):
        client = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        headers = {"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token}
        try:
            client.request("POST" if payload else "GET", "/api/action" if payload else "/api/state",
                           json.dumps(payload) if payload else None, headers)
            response = client.getresponse()
            return response.status, json.loads(response.read())
        finally:
            client.close()

    def test_one_delegation_prepares_one_final_candidate_without_an_extra_seconds_approval(self):
        feedback = {"action": "planning_feedback", "id": "E001-S001", "version": 0,
                    "analysis_signature": self.signature, "feedback": "交给GPT来准备提示词，先给我确认。"}
        self.assertEqual(self.request(feedback)[0], 200)
        self.run.assert_not_called()
        self.assertEqual(self.request({"action": "run"})[0], 202)
        self.assertTrue(self.prepared.wait(3), "synthetic candidate must be prepared")
        # Synchronize after the worker has completed its no-provider preparation.
        for _ in range(100):
            with self.server.context.lock:
                running = self.server.context.running
            if not running:
                break
            threading.Event().wait(.005)
        status, public = self.request()
        self.assertEqual(status, 200)
        shot = public["episodes"][0]["shots"][0]
        candidate = shot["prompt_candidate"]
        self.assertEqual(shot["status"], "awaiting_prompt_approval")
        self.assertNotIn("planning_block", shot)
        self.assertEqual(candidate["timing_proposal"], timing_proposal())
        self.assertIn("3–7.5秒", candidate["prompt"])
        self.assertEqual(self.prepare_calls, 1)
        self.assertEqual(self.request({"action": "run"})[0], 409)
        self.assertEqual(self.request(feedback)[0], 409, "old planning form cannot overwrite this candidate")
        state = core.load_project(self.root)
        current = state["episodes"][0]["shots"][0]
        self.assertEqual(state["jobs"], {})
        self.assertNotIn("prompt_approval", current)
        self.assertNotIn("file", current)
        self.assertEqual(current["subshots"], self.original["episodes"][0]["shots"][0]["subshots"])
        approval = {"action": "approve_prompt", "id": current["id"], "candidate_id": candidate["id"],
                    "request_hash": candidate["request_hash"]}
        self.assertEqual(self.request(approval)[0], 200)
        state = core.load_project(self.root)
        self.assertEqual(state["episodes"][0]["shots"][0]["status"], "prompt_approved")
        self.assertEqual(state["jobs"], {}, "the HTTP approval handler itself never generates media")
        self.assertEqual(self.prepare_calls, 1)


@unittest.skipUnless(shutil.which("node"), "Node.js required for pure DOM rendering test")
class TimingProposalRenderingTests(unittest.TestCase):
    def test_folded_draft_shows_camera_and_audio_with_only_the_existing_confirmation(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "promptApproval"):
            start = html.index("function " + name + "(")
            sources.append(html[start:html.index("\nfunction ", start + 1)])
        shot = original_shot()
        shot.update(status="awaiting_prompt_approval", prompt_candidate=candidate_for(shot))
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {sources,shot}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.dataset={};this.listeners={};}
 append(...nodes){this.children.push(...nodes);}
 addEventListener(key,handler){this.listeners[key]=handler;}
}
const decisions=[];
const context={document:{createElement:tag=>new Element(tag)},state:{assets:[],paused:false,dashboard:{running:false}},
 perform:(...args)=>decisions.push(args),assert,shot,decisions};
vm.runInNewContext(sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const all=walk(promptApproval(shot)),text=all.map(n=>n.textContent).join(' · ');
const details=all.find(n=>n.className==='timing-proposal');
assert.equal(details.tag,'details');assert.equal(Boolean(details.open),false);
assert.equal(details.dataset.uiKey,'timing:'+shot.id+':'+shot.prompt_candidate.id);
assert.equal(text.includes('本次安排调整'),true);assert.equal(text.includes('以下安排由 GPT 拟定'),true);
assert.equal(text.includes('镜头数量、切点与总时长保持原计划'),true);
assert.equal(text.includes('点击下方“确认提示词并生成”后才提交本版视频'),true);
for(const part of shot.prompt_candidate.timing_proposal.subshots){assert.equal(text.includes(part.start+'–'+part.end+' 秒'),true);assert.equal(text.includes(part.prompt),true);}
assert.equal(text.includes('连续声音安排'),true);assert.equal(text.includes('3–7.5 秒 · 系统'),true);
assert.equal(text.includes('绑定成功。'),true);assert.equal(text.includes(shot.prompt_candidate.prompt),true);
assert.equal(all.filter(n=>n.tag==='button').length,2,'no extra timing approval');
assert.equal(all.some(n=>n.tag==='textarea'||n.tag==='input'),false,'no manual seconds prerequisite');
assert.equal(decisions.length,0,'viewing a GPT draft never authorizes generation');
const approve=all.find(n=>n.tag==='button'&&n.textContent==='确认提示词并生成');
assert.equal(approve.disabled,false);approve.listeners.click();assert.equal(decisions.length,1);
assert.equal(decisions[0][0].action,'approve_prompt');assert.equal(decisions[0][0].candidate_id,shot.prompt_candidate.id);
assert.equal(decisions[0][0].request_hash,shot.prompt_candidate.request_hash);assert.equal(decisions[0][1],true);
state.paused=true;const paused=walk(promptApproval(shot));assert.equal(paused.find(n=>n.textContent==='确认提示词并生成').disabled,true);
state.paused=false;shot.prompt_candidate.timing_proposal=null;assert.equal(walk(promptApproval(shot)).some(n=>n.className==='timing-proposal'),false);
shot.prompt_candidate.timing_proposal={summary:'无对白的合成场景。',subshots:[],audio_windows:[]};
assert.equal(walk(promptApproval(shot)).some(n=>n.textContent==='本段无原定对白，不添加新播报。'),true);
`,context);
'''
        done = subprocess.run([shutil.which("node"), "-e", program],
                              input=json.dumps({"sources": sources, "shot": shot}),
                              text=True, capture_output=True, timeout=15)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_feedback_followup_describes_prompt_preparation_and_never_approves_it(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        start = html.index("async function perform(")
        source = html[start:html.index("\nfunction ", start + 1)]
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const source=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const calls=[],messages=[];
const context={requestBusy:false,state:{assets:[],dashboard:{running:false}},
 action:async payload=>calls.push(payload),toast:(...args)=>messages.push(args),
 refresh:async()=>{},updateInputControls:()=>{},calls,messages,assert};
vm.runInNewContext(source+`
perform({action:'planning_feedback',id:'E001-S001',feedback:'交给GPT准备提示词'},true).then(()=>{
 assert.equal(calls.length,2);assert.equal(calls[0].action,'planning_feedback');assert.equal(calls[1].action,'run');
 assert.equal(calls.some(p=>p.action==='approve_prompt'),false);
 assert.equal(messages.at(-1)[0],'正在准备提示词，完成后请确认整份提示词');
});
`,context);
'''
        done = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps(source),
                              text=True, capture_output=True, timeout=15)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
