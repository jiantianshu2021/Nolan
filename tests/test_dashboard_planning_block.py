"""Synthetic web recovery for prompt-planning blockers; no paid provider calls."""
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

from norlan import dashboard


def signature(shot):
    return hashlib.sha256(json.dumps([shot.get("revision", 0), shot.get("feedback", "")], ensure_ascii=False).encode()).hexdigest()


def blocked_state():
    shot = {"id": "S7", "status": "planned", "version": 0, "revision": 1, "duration": 12,
            "feedback": "系统任务框需要女性声音播报，保留原总时长。", "prompt": "合成测试原分镜。"}
    shot["revision_analysis"] = {"signature": signature(shot), "summary": "原播报起点与台词时长可能冲突。",
        "blockers": ["保持原总时长和切点时，原定起点可能无法读完。token=PRIVATE-TOKEN https://private.invalid/task"],
        "raw_config": "PRIVATE-CONFIG"}
    return {"id": "synthetic", "mode": "live", "status": "blocked", "paused": False,
        "plan": {"status": "approved"}, "assets": [], "budget": {}, "jobs": {},
        "episodes": [{"id": "E1", "status": "generating", "shots": [shot]}]}


class PlanningBlockProjectionTests(unittest.TestCase):
    def setUp(self):
        self.state = blocked_state()
        self.shot = self.state["episodes"][0]["shots"][0]

    def test_current_block_is_bound_sanitized_and_does_not_change_the_request(self):
        before = copy.deepcopy(self.state)
        item = dashboard.public_state(self.state)["episodes"][0]["shots"][0]
        block = item["planning_block"]
        self.assertEqual(set(block), {"shot_id", "summary", "blockers", "signature", "version", "revision", "can_submit"})
        self.assertEqual(block["shot_id"], "S7")
        self.assertEqual(block["signature"], signature(self.shot))
        self.assertEqual(block["version"], 0)
        self.assertTrue(block["can_submit"])
        self.assertIn("时长", block["blockers"][0])
        self.assertNotIn("PRIVATE-", json.dumps(block))
        self.assertNotIn("https://", json.dumps(block))
        self.assertEqual(item["progress"]["stage"], "planning_blocked")
        self.assertFalse(item["progress"]["is_working"])
        self.assertTrue(item["progress"]["waiting_for_human"])
        self.assertEqual(self.state, before)
        self.assertEqual(self.state["jobs"], {})

    def test_old_revision_feedback_status_or_empty_blocker_cannot_surface_stale_controls(self):
        original = copy.deepcopy(self.shot)
        for changes in ({"revision": 2}, {"feedback": "已明确修改的要求"}, {"status": "accepted"},
                        {"revision": True}, {"version": True}, {"job_id": "already-submitted"},
                        {"file": "media/E1/S7-v1.mp4"}):
            self.shot.clear()
            self.shot.update(copy.deepcopy(original))
            self.shot.update(changes)
            public = dashboard.public_state(self.state)["episodes"][0]["shots"][0]
            self.assertNotIn("planning_block", public)
        self.shot.clear()
        self.shot.update(original)
        self.shot["revision_analysis"]["blockers"] = []
        self.assertNotIn("planning_block", dashboard.public_state(self.state)["episodes"][0]["shots"][0])

    def test_future_blocker_does_not_override_the_first_current_clip(self):
        future = copy.deepcopy(self.shot)
        future["id"] = "S8"
        earlier = {"id": "S6", "status": "review_required", "version": 1, "duration": 10,
                   "file": "media/E1/S6-v1.mp4", "sha256": "a" * 64}
        self.state["episodes"][0]["shots"] = [earlier, self.shot, future]
        shots = dashboard.public_state(self.state)["episodes"][0]["shots"]
        self.assertFalse(any("planning_block" in item for item in shots))
        earlier.update(status="accepted", approved_at="2026-01-01T00:00:00Z", approved_sha256=earlier["sha256"],
                       approved_version=1, review={"human_override": True})
        shots = dashboard.public_state(self.state)["episodes"][0]["shots"]
        self.assertIn("planning_block", shots[1])
        self.assertNotIn("planning_block", shots[2])

    def test_legacy_gpt_accepted_segment_still_precedes_a_future_planning_question(self):
        earlier = {"id": "S6", "status": "accepted", "version": 1, "duration": 10,
                   "file": "media/E1/S6-v1.mp4", "sha256": "a" * 64,
                   "review": {"verdict": "pass", "sha256": "a" * 64, "version": 1}}
        self.state["episodes"][0]["shots"].insert(0, earlier)
        self.assertIsNone(dashboard._current_planning_block(self.state))
        self.assertFalse(any("planning_block" in shot for shot in dashboard.public_state(self.state)["episodes"][0]["shots"]))
        earlier.pop("file")
        self.assertIsNone(dashboard._current_planning_block(self.state))

    def test_asset_and_plan_stages_take_priority_without_erasing_the_cached_question(self):
        analysis = copy.deepcopy(self.shot["revision_analysis"])
        self.state["plan"]["status"] = "candidate"
        self.assertIsNone(dashboard._current_planning_block(self.state))
        self.state["plan"]["status"] = "approved"
        self.state["assets"] = [{"id": "voice", "kind": "voice", "status": "planned"}]
        self.assertIsNone(dashboard._current_planning_block(self.state))
        self.assertNotIn("planning_block", dashboard.public_state(self.state)["episodes"][0]["shots"][0])
        self.state["assets"][0]["status"] = "approved"
        for status in ("candidate", "awaiting_assets"):
            self.state["asset_amendments"] = [{"id": "add-voice", "status": status, "additions": [], "shot_updates": []}]
            self.assertIsNone(dashboard._current_planning_block(self.state))
        self.assertEqual(self.shot["revision_analysis"], analysis)
        self.state["asset_amendments"] = []
        self.assertIsNotNone(dashboard._current_planning_block(self.state))

    def test_optional_suggestions_are_safe_visible_metadata_and_never_change_timing(self):
        self.shot["revision_analysis"]["resolution_options"] = [{"label": "保留总时长与切点",
            "feedback": "允许在本段第 3 秒出现任务框并开始播报，跨切镜延续。总长 10 秒、5 秒切点与整集时长保留。",
            "private": "PRIVATE-OPTION", "url": "https://private.invalid"},
            {"label": "过长选项不应截断提交", "feedback": "长" * 801}]
        before = copy.deepcopy(self.state)
        block = dashboard.public_state(self.state)["episodes"][0]["shots"][0]["planning_block"]
        self.assertEqual(len(block["resolution_options"]), 1)
        option = block["resolution_options"][0]
        self.assertEqual(set(option), {"label", "feedback"})
        self.assertEqual(option["feedback"], self.shot["revision_analysis"]["resolution_options"][0]["feedback"])
        self.assertNotIn("PRIVATE-", json.dumps(block))
        self.assertEqual(self.state, before)
        self.assertEqual(self.shot["duration"], 12)

    def test_pause_and_real_unknown_calls_disable_adjustment_submission(self):
        self.state["paused"] = True
        self.assertFalse(dashboard.public_state(self.state)["episodes"][0]["shots"][0]["planning_block"]["can_submit"])
        self.state["paused"] = False
        self.state["jobs"]["unknown-video"] = {"kind": "video", "state": "uncertain", "accounting": "unknown"}
        self.assertFalse(dashboard.public_state(self.state)["episodes"][0]["shots"][0]["planning_block"]["can_submit"])


class PlanningCore:
    def __init__(self, state):
        self.state, self.calls = state, []

    def load_project(self, _root):
        return self.state

    def submit_planning_feedback(self, _root, shot_id, feedback, expected_version, analysis_signature):
        shot = self.state["episodes"][0]["shots"][0]
        if (shot_id != shot["id"] or expected_version != shot["version"]
                or analysis_signature != signature(shot)):
            raise ValueError("PRIVATE-STALE-ERROR")
        self.calls.append((shot_id, feedback, expected_version, analysis_signature))
        shot.update(feedback=feedback, revision=shot["revision"] + 1)
        shot.pop("revision_analysis")
        self.state["status"] = "ready"
        return self.state

    def set_paused(self, _root, paused):
        self.state["paused"] = paused
        return self.state


class PlanningFeedbackHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.state = blocked_state()
        self.shot = self.state["episodes"][0]["shots"][0]
        self.core = PlanningCore(self.state)
        self.core_patch = patch.object(dashboard, "_core", return_value=self.core)
        self.core_patch.start()
        self.run_patch = patch.object(dashboard, "_run", side_effect=AssertionError("feedback HTTP must not run production"))
        self.run = self.run_patch.start()
        self.server = dashboard.make_server(self.root, port=0)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.run.assert_not_called()
        self.run_patch.stop()
        self.core_patch.stop()
        self.temp.cleanup()

    def request(self, payload, headers=None):
        client = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        options = {"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token}
        options.update(headers or {})
        try:
            client.request("POST", "/api/action", json.dumps(payload), options)
            response = client.getresponse()
            return response.status, json.loads(response.read())
        finally:
            client.close()

    def payload(self):
        return {"action": "planning_feedback", "id": "S7", "version": 0, "analysis_signature": signature(self.shot),
                "feedback": "保留原总时长与切点，重新安排播报起点，先让我检查完整提示词。"}

    def test_feedback_is_explicit_version_bound_and_only_prepares_the_next_prompt(self):
        before_jobs = copy.deepcopy(self.state["jobs"])
        payload = self.payload()
        status, result = self.request(payload)
        self.assertEqual(status, 200)
        self.assertEqual(self.core.calls, [("S7", payload["feedback"], 0, payload["analysis_signature"])])
        self.assertNotIn("planning_block", result["episodes"][0]["shots"][0])
        self.assertEqual(self.state["jobs"], before_jobs)
        self.assertEqual(self.shot["version"], 0)
        self.assertNotIn("prompt_approval", self.shot)
        self.assertNotIn("file", self.shot)

    def test_normal_continue_explains_block_without_repeating_cached_work(self):
        status, result = self.request({"action": "run"})
        self.assertEqual(status, 409)
        self.assertIn("填写调整意见", result["error"])
        self.assertIn("尚未提交视频生成", result["error"])

    def test_stale_version_or_analysis_cannot_overwrite_a_new_request(self):
        payload = self.payload()
        self.shot["version"] = 1
        status, result = self.request(payload)
        self.assertEqual(status, 409)
        self.assertNotIn("PRIVATE-", result["error"])
        self.shot["version"] = 0
        self.shot["feedback"] = "已由另一个窗口调整的要求。"
        self.shot["revision"] = 2
        self.assertEqual(self.request(payload)[0], 409)
        self.assertEqual(self.core.calls, [])

    def test_feedback_validates_empty_input_version_hash_csrf_and_origin(self):
        for change in ({"feedback": " "}, {"version": True}, {"version": -1}, {"analysis_signature": "bad"}, {"id": ""}):
            self.assertEqual(self.request({**self.payload(), **change})[0], 409)
        self.assertEqual(self.request(self.payload(), {"X-Norlan-Token": "bad"})[0], 403)
        self.assertEqual(self.request(self.payload(), {"Origin": "https://foreign.invalid"})[0], 403)
        self.assertEqual(self.core.calls, [])


@unittest.skipUnless(shutil.which("node"), "Node.js required for pure DOM rendering test")
class PlanningBlockRenderingTests(unittest.TestCase):
    def test_block_has_empty_feedback_form_and_continue_finds_it_without_generation(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "mediaURL", "appendPrompt", "failureDetails", "shotReview", "advisoryReport",
                     "subshotPlan", "promptApproval", "shotCard", "paintApprovalGate"):
            start = html.index("function " + name + "(")
            sources.append(html[start:html.index("\nfunction ", start + 1)])
        start = html.index('$("run").addEventListener("click",')
        binding = html[start:html.index('$("pause").addEventListener', start)]
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {sources,binding}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.value='';this.listeners={};this.dataset={};this.attributes={};this.classList={toggle:()=>{}};}
 append(...nodes){this.children.push(...nodes);}
 addEventListener(key,handler){this.listeners[key]=handler;}
 setAttribute(key,value){this.attributes[key]=value;}
 querySelector(tag){for(const child of this.children){if(child.tag===tag)return child;const nested=child.querySelector?.(tag);if(nested)return nested;}return null;}
 focus(){this.focused=true;}
 scrollIntoView(options){this.scrolled=options;}
}
const nodes=new Map(),decisions=[],messages=[];
const context={document:{createElement:tag=>new Element(tag),createTextNode:text=>({tag:'text',textContent:text})},
 $:id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);},
 state:{id:'synthetic',assets:[],paused:false,dashboard:{running:false}},perform:(...args)=>decisions.push(args),
 toast:(...args)=>messages.push(args),assert,nodes,decisions,messages};
vm.runInNewContext(sources.join('\n')+binding+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const shot={id:'S7',status:'planned',version:0,duration:12,planning_block:{summary:'当前起点可能读不完台词。',blockers:['先核对时长与切点。'],signature:'a'.repeat(64),version:0,revision:1,can_submit:true}};
const card=shotCard(shot,6,null),all=walk(card),box=all.find(n=>n.id==='planning-block-S7');nodes.set(box.id,box);
assert.ok(box);assert.equal(all.some(n=>n.textContent==='本段提示词尚未准备完成'),true);assert.equal(all.some(n=>n.textContent.includes('确认后才生成视频')),true);
const explanation=all.find(n=>n.className==='planning-explanation');assert.equal(explanation.tag,'details');assert.equal(Boolean(explanation.open),false);
const feedback=all.find(n=>n.tag==='textarea');assert.equal(feedback.value,'','no default timing permission');assert.equal(feedback.dataset.feedbackFor,'planning:S7:'+shot.planning_block.signature);
const submit=all.find(n=>n.tag==='button'&&n.textContent==='提交调整意见，重新准备提示词');assert.ok(submit);assert.equal(submit.disabled,false);assert.equal(decisions.length,0);
submit.listeners.click();assert.equal(decisions.length,0);assert.equal(messages.at(-1)[0],'请先填写调整意见');
feedback.value='保持原总时长和切点，重新准备完整提示词让我检查。';submit.listeners.click();
assert.equal(decisions.length,1);assert.equal(decisions[0][0].action,'planning_feedback');assert.equal(decisions[0][0].version,0);assert.equal(decisions[0][0].analysis_signature,'a'.repeat(64));assert.equal(decisions[0][0].feedback,feedback.value);assert.equal(decisions[0][1],true,'continue means prepare, never approve_prompt');
state.episodes=[{shots:[...Array.from({length:6},(_,i)=>({id:'S'+i,status:'accepted'})),shot]}];paintApprovalGate(state);assert.equal(nodes.get('run').textContent,'前往第 7 段填写意见');assert.equal(nodes.get('run').disabled,false);
nodes.get('run').listeners.click();assert.equal(decisions.length,1,'continue scrolls to the form instead of running stale analysis');assert.equal(Boolean(box.scrolled),true);assert.equal(feedback.focused,true);
state.paused=true;paintApprovalGate(state);nodes.get('run').listeners.click();assert.equal(decisions.length,2);assert.equal(decisions.at(-1)[0].action,'pause');assert.equal(decisions.at(-1)[0].paused,false,'resume only unpauses');
shot.planning_block.can_submit=false;const paused=walk(shotCard(shot,6,null));assert.equal(paused.find(n=>n.tag==='button'&&n.textContent==='提交调整意见，重新准备提示词').disabled,true);
state.paused=false;shot.planning_block.can_submit=true;const suggestion={label:'保留总时长与切点',feedback:'允许第3秒出现系统框并播报，跨切镜延续；保留原总时长与切点。'};shot.planning_block.resolution_options=[suggestion];const suggested=walk(shotCard(shot,6,null));assert.equal(decisions.length,2,'showing a suggestion is not consent');assert.equal(suggested.some(n=>n.textContent==='待确认建议'),true);assert.equal(suggested.some(n=>n.textContent===suggestion.feedback),true);assert.equal(suggested.find(n=>n.tag==='textarea').value,'','suggestion never prefills the feedback as if already chosen');
const adopt=suggested.find(n=>n.tag==='button'&&n.textContent==='采用建议，准备提示词');assert.ok(adopt);adopt.listeners.click();assert.equal(decisions.length,3);assert.equal(decisions.at(-1)[0].action,'planning_feedback');assert.equal(decisions.at(-1)[0].feedback,suggestion.feedback);assert.equal(decisions.at(-1)[0].analysis_signature,shot.planning_block.signature);assert.equal(decisions.at(-1)[1],true);assert.equal(shot.duration,12,'adoption handler does not edit canonical timing');
`,context);
'''
        done = subprocess.run([shutil.which("node"), "-e", program],
                              input=json.dumps({"sources": sources, "binding": binding}), text=True,
                              capture_output=True, timeout=15)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
