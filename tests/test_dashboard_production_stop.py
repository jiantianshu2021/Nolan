"""Synthetic blocked-preparation display regressions; no model or media calls."""
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
from unittest.mock import Mock, patch

from norlan import dashboard


def stopped_state():
    shots = []
    for index in range(1, 8):
        sha = f"{index:064x}"
        shots.append({"id": f"S{index}", "status": "accepted", "version": 1, "duration": 10,
            "file": f"media/E1/S{index}-v1.mp4", "sha256": sha, "approved_sha256": sha,
            "approved_at": "2026-01-01T00:00:00Z", "approved_version": 1, "review": {"human_override": True}})
    shots.extend([{"id": "S8", "status": "planned", "version": 0, "duration": 10},
                  {"id": "S9", "status": "planned", "version": 0, "duration": 10}])
    return {"id": "synthetic", "mode": "live", "status": "blocked", "paused": False,
        "plan": {"status": "approved"}, "assets": [], "budget": {"spent": 7, "reserved": 0, "unknown": 0},
        "jobs": {}, "last_error": "输入超过本次上下文限制；请按章节拆分小说，不能静默截断。 token=PRIVATE-TOKEN https://private.invalid/task",
        "episodes": [{"id": "E1", "status": "generating", "shots": shots,
            "progress_stage": {"stage": "shot_review", "shot_id": "S7", "started_at": "2026-01-01T00:00:00Z"}}]}


class ProductionStopProjectionTests(unittest.TestCase):
    def setUp(self):
        self.state = stopped_state()

    def test_local_preparation_error_has_correct_target_and_safe_specific_reason(self):
        before = copy.deepcopy(self.state)
        public = dashboard.public_state(self.state)
        stop = public["production_stop"]
        self.assertEqual(stop["stage"], "shot_preparation")
        self.assertEqual(stop["stage_label"], "准备提示词")
        self.assertEqual(stop["shot_id"], "S8")
        self.assertEqual(stop["shot_index"], 8)
        self.assertEqual(stop["episode_index"], 1)
        self.assertTrue(stop["can_retry"])
        self.assertIn("输入超过本次上下文限制", stop["message"])
        self.assertNotIn("PRIVATE-", json.dumps(public))
        self.assertNotIn("https://private.invalid", json.dumps(public))
        self.assertEqual(public["last_error"], stop["message"])
        self.assertEqual(self.state, before)

    def test_previous_review_hint_never_becomes_the_next_clip_phase_or_elapsed_counter(self):
        public = dashboard.public_state(self.state)
        episode = public["episodes"][0]
        old, current, future = episode["shots"][6:]
        self.assertEqual(old["progress"]["stage"], "accepted")
        self.assertNotIn("preparation_failure", old)
        self.assertEqual(current["progress"]["stage"], "preparation_failed")
        self.assertIn("准备提示词失败", current["progress"]["label"])
        self.assertFalse(current["progress"]["is_working"])
        self.assertIsNone(current["progress"]["elapsed_seconds"])
        self.assertEqual(current["preparation_failure"]["stage"], "shot_preparation")
        self.assertEqual(future["progress"]["stage"], "queued")
        self.assertNotIn("preparation_failure", future)
        self.assertEqual(episode["progress"]["stage"], "preparation_failed")
        self.assertEqual(episode["progress"]["current_shot"], "S8")
        self.assertEqual(episode["progress"]["current_shot_index"], 8)
        self.assertIsNone(episode["progress"]["elapsed_seconds"])

    def test_fresh_current_hint_reports_the_same_failure_as_an_old_hint(self):
        self.state["episodes"][0]["progress_stage"] = {"stage": "shot_preparation", "shot_id": "S8"}
        public = dashboard.public_state(self.state)
        self.assertEqual(public["production_stop"]["shot_id"], "S8")
        self.assertEqual(public["episodes"][0]["shots"][7]["progress"]["stage"], "preparation_failed")

    def test_actionable_planning_question_and_segment_plan_keep_their_existing_priority(self):
        shot = self.state["episodes"][0]["shots"][7]
        shot.update(revision=1, feedback="保留原总时长，需要明确播报起点。")
        shot["revision_analysis"] = {"signature": hashlib.sha256(json.dumps([1, shot["feedback"]], ensure_ascii=False).encode()).hexdigest(),
            "planning_policy_version": 2, "summary": "播报时间需要确认。", "blockers": ["请明确播报起点。"]}
        public = dashboard.public_state(self.state)
        self.assertNotIn("production_stop", public)
        self.assertIn("planning_block", public["episodes"][0]["shots"][7])
        self.assertNotIn("preparation_failure", public["episodes"][0]["shots"][7])
        shot.pop("revision_analysis")
        self.state["episodes"][0]["segment_proposal"] = {"id": "proposal", "status": "candidate", "shots": []}
        self.assertNotIn("production_stop", dashboard.public_state(self.state))

    def test_unknown_media_is_never_treated_as_a_retryable_local_preparation_error(self):
        self.state["jobs"]["unknown-video"] = {"kind": "video", "item_id": "S8", "state": "uncertain", "accounting": "unknown"}
        public = dashboard.public_state(self.state)
        self.assertEqual(public["production_stop"]["stage"], "unknown")
        self.assertFalse(public["production_stop"]["can_retry"])
        self.assertEqual(public["episodes"][0]["progress"]["stage"], "unknown")
        self.assertNotIn("preparation_failure", public["episodes"][0]["shots"][7])

    def test_asset_phase_and_legacy_unapproved_video_do_not_get_a_new_clip_preparation_error(self):
        self.state["assets"] = [{"id": "voice", "kind": "voice", "status": "planned"}]
        self.assertEqual(dashboard.public_state(self.state)["production_stop"]["stage"], "assets")
        self.state["assets"] = []
        previous = self.state["episodes"][0]["shots"][6]
        previous.pop("approved_at")
        public = dashboard.public_state(self.state)
        self.assertEqual(public["production_stop"]["shot_id"], "S7")
        self.assertEqual(public["production_stop"]["stage"], "shot_review")
        self.assertNotIn("preparation_failure", public["episodes"][0]["shots"][7])


class ProductionStopRetryHttpTests(unittest.TestCase):
    def test_explicit_retry_runs_once_and_does_not_authorize_a_video(self):
        state = stopped_state()
        core = Mock()
        core.load_project.side_effect = lambda _root: state
        core.set_paused.side_effect = lambda _root, paused: state.update(paused=paused)
        started = threading.Event()

        def prepare(_root):
            state["status"] = "awaiting_prompt_approval"
            started.set()
            return state

        with tempfile.TemporaryDirectory() as temporary, patch.object(dashboard, "_core", return_value=core), patch.object(dashboard, "_run", side_effect=prepare) as run:
            server = dashboard.make_server(Path(temporary), port=0)
            thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
            thread.start()
            try:
                client = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
                client.request("POST", "/api/action", json.dumps({"action": "run"}),
                    {"Content-Type": "application/json", "X-Norlan-Token": server.context.token})
                response = client.getresponse()
                response.read()
                self.assertEqual(response.status, 202)
                self.assertTrue(started.wait(timeout=1))
                client.close()
                run.assert_called_once()
                core.approve_prompt.assert_not_called()
                self.assertEqual(state["jobs"], {})
                self.assertEqual(state["episodes"][0]["shots"][7]["version"], 0)
                self.assertNotIn("file", state["episodes"][0]["shots"][7])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


@unittest.skipUnless(shutil.which("node"), "Node.js required for pure DOM rendering test")
class ProductionStopRenderingTests(unittest.TestCase):
    def test_sidebar_and_clip_display_specific_error_without_pretending_to_wait_for_approval(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "badge", "mediaURL", "appendPrompt", "failureDetails", "shotReview", "advisoryReport",
                     "reviewDetails", "episodeCard", "subshotPlan", "promptApproval", "shotCard", "paintApprovalGate", "paintProductionStop"):
            start = html.index("function " + name + "(")
            sources.append(html[start:html.index("\nfunction ", start + 1)])
        public = dashboard.public_state(stopped_state())
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {sources,state}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.listeners={};this.dataset={};this.attributes={};this.classList={toggle:(key,on)=>{this.hidden=on;}};}
 append(...nodes){this.children.push(...nodes);}
 addEventListener(key,handler){this.listeners[key]=handler;}
 setAttribute(key,value){this.attributes[key]=value;}
}
const nodes=new Map(),decisions=[];state.dashboard={running:false};
const context={document:{createElement:tag=>new Element(tag),createTextNode:text=>({tag:'text',textContent:text})},
 $:id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);},
 state,names:{generating:'制作中'},appendEpisodeProduction:()=>{},perform:(...args)=>decisions.push(args),assert,nodes,decisions};
vm.runInNewContext(sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
nodes.set('status-note',el('p'));nodes.get('status-note').textContent='在审批节点等待你的决定。';paintProductionStop(state);
assert.equal(nodes.get('production-stop').hidden,false);assert.equal(nodes.get('status').textContent,'制作已停止');assert.equal(nodes.get('status-note').textContent.includes('审批节点'),false);
assert.equal(nodes.get('stop-stage').textContent,'失败阶段：准备提示词');assert.equal(nodes.get('stop-target').textContent.includes('第 8 段（S8）'),true);assert.equal(nodes.get('stop-error').textContent.includes('输入超过本次上下文限制'),true);
assert.equal(nodes.get('run').textContent,'重新准备提示词');assert.equal(nodes.get('run').disabled,false);assert.equal(decisions.length,0,'rendering does not start a retry');
const shot=state.episodes[0].shots[7],all=walk(shotCard(shot,7,null));assert.equal(all.some(n=>n.textContent==='准备提示词失败'),true);assert.equal(all.some(n=>n.textContent.includes('输入超过本次上下文限制')),true);assert.equal(all.some(n=>n.tag==='button'&&n.textContent.includes('通过本段')),false);assert.equal(all.some(n=>n.textContent==='GPT 正在整理本版本的审查报告。'),false);
const episode=walk(episodeCard(state.episodes[0],0));assert.equal(episode.find(n=>n.className==='awaiting').textContent.includes('本集制作已停止'),true);
state.production_stop={stage:'unknown',stage_label:'核实原调用结果',message:'原视频调用结果待核实',can_retry:false};state.episodes[0].progress.stage='unknown';paintApprovalGate(state);paintProductionStop(state);assert.equal(nodes.get('run').disabled,true);assert.equal(nodes.get('stop-retry-note').textContent.includes('不能重新提交'),true);
delete state.production_stop;paintProductionStop(state);assert.equal(nodes.get('production-stop').hidden,true);
`,context);
'''
        done = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps({"sources": sources, "state": public}),
                              text=True, capture_output=True, timeout=15)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
