"""Synthetic recovery UI protocol checks; no projects or provider calls."""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

from norlan import dashboard, future_continuity


def synthetic_state():
    shot = {"id": "synthetic-s1", "status": "planned", "revision": 3, "version": 3,
            "duration": 10, "feedback": "先快速坠落，再切绝望表情，最后接住站稳。"}
    signature = hashlib.sha256(json.dumps([shot["revision"], shot["feedback"]], ensure_ascii=False).encode()).hexdigest()
    shot["revision_analysis"] = {"signature": signature,
        "summary": "需要协调下一段尚未生成的动作。", "blockers": ["下一段原定动作与本段站稳结尾冲突。"]}
    return {"id": "synthetic-recovery", "title": "合成恢复测试", "status": "blocked", "paused": False,
        "mode": "demo", "plan": {"status": "approved"}, "assets": [], "budget": {}, "jobs": {},
        "last_error": "下一段原定动作与本段站稳结尾冲突。",
        "settings": {"gpt_video_review_enabled": False},
        "episodes": [{"id": "synthetic-e1", "status": "generating", "shots": [shot]}]}


class PlanningRecoveryProjectionTests(unittest.TestCase):
    def setUp(self):
        self.state = synthetic_state()
        self.shot = self.state["episodes"][0]["shots"][0]
        # Exercise the dashboard's protocol separately from the planner's policy.
        self.protocol = future_continuity
        recovery = patch.object(future_continuity, "planning_recovery_available", side_effect=lambda state, shot: shot is self.shot)
        updates = patch.object(future_continuity, "public_updates", return_value=[])
        recovery.start()
        updates.start()
        self.addCleanup(recovery.stop)
        self.addCleanup(updates.stop)

    def test_old_creative_question_is_recoverable_without_feedback_or_state_mutation(self):
        before = copy.deepcopy(self.state)
        public = dashboard.public_state(self.state)
        shot = public["episodes"][0]["shots"][0]
        self.assertEqual(public["status"], "ready")
        self.assertNotIn("last_error", public)
        self.assertNotIn("production_stop", public)
        self.assertNotIn("planning_block", shot)
        self.assertIn("已保留修改意见", shot["planning_recovery"]["message"])
        self.assertEqual(shot["progress"]["stage"], "planning_recovery")
        self.assertFalse(shot["progress"]["waiting_for_human"])
        self.assertFalse(shot["progress"]["is_working"])
        self.assertEqual(self.state, before)
        self.assertEqual(public["video_review"], {"enabled": False})

    def test_real_hard_question_still_requires_human_input(self):
        self.protocol.planning_recovery_available.return_value = False
        self.protocol.planning_recovery_available.side_effect = None
        public = dashboard.public_state(self.state)
        shot = public["episodes"][0]["shots"][0]
        self.assertEqual(public["status"], "blocked")
        self.assertIn("planning_block", shot)
        self.assertNotIn("planning_recovery", shot)
        self.assertEqual(shot["progress"]["stage"], "planning_blocked")
        self.assertTrue(shot["progress"]["waiting_for_human"])

    def test_current_clip_and_upstream_gates_take_priority(self):
        earlier = {"id": "synthetic-earlier", "status": "review_required", "version": 1, "duration": 5,
                   "file": "media/synthetic-v1.mp4", "sha256": "a" * 64}
        self.state["episodes"][0]["shots"].insert(0, earlier)
        public = dashboard.public_state(self.state)
        self.assertFalse(any("planning_recovery" in shot for shot in public["episodes"][0]["shots"]))
        self.state["episodes"][0]["shots"].pop(0)
        self.state["episodes"][0]["segment_proposal"] = {"status": "candidate"}
        self.assertIsNone(dashboard._current_planning_recovery(self.state))
        self.state["episodes"][0].pop("segment_proposal")
        self.state["plan"]["status"] = "candidate"
        self.assertIsNone(dashboard._current_planning_recovery(self.state))

    def test_unknown_calls_cannot_be_hidden_as_ready_to_resume(self):
        self.state["jobs"]["synthetic-unknown"] = {"kind": "video", "item_id": self.shot["id"],
            "state": "uncertain", "accounting": "unknown"}
        public = dashboard.public_state(self.state)
        self.assertEqual(public["status"], "blocked")
        self.assertEqual(public["production_stop"]["stage"], "unknown")
        self.assertNotIn("planning_recovery", public["episodes"][0]["shots"][0])

    def test_continuity_summaries_are_projected_for_the_human_prompt(self):
        updates = [{"id": "synthetic-s2", "summary": "从已站稳状态接续，不重复接住动作。"}]
        self.protocol.public_updates.return_value = updates
        self.shot.update(status="awaiting_prompt_approval", prompt_candidate={"id": "synthetic-candidate",
            "version": 4, "prompt": "合成完整提示词", "request_hash": "b" * 64,
            "future_continuity_updates": copy.deepcopy(updates)})
        public = dashboard.public_state(self.state)["episodes"][0]["shots"][0]
        self.assertEqual(public["future_continuity_updates"], updates)
        self.assertEqual(public["prompt_candidate"]["future_continuity_updates"], updates)
        self.assertNotIn("prompt_approval", public)
        self.shot["prompt_candidate"]["future_continuity_updates"][0]["summary"] = "不是本候选保存的调整"
        public = dashboard.public_state(self.state)["episodes"][0]["shots"][0]
        self.assertNotIn("future_continuity_updates", public["prompt_candidate"])

    def test_latest_human_feedback_uses_event_order_without_rewriting_episode(self):
        episode = self.state["episodes"][0]
        episode["feedback"] = "较早的整集意见：接住以后站稳。"
        self.state["events"] = [
            {"action": "human_rejected", "kind": "episode", "id": episode["id"], "feedback": episode["feedback"]},
            {"action": "human_planning_feedback", "id": self.shot["id"], "feedback": "改为六秒三镜，使用下面的完整分镜表。"},
            {"action": "revision_automatically_coordinated", "id": self.shot["id"], "feedback": "机器产生的摘要不能成为用户意见"},
            {"action": "human_planning_feedback", "id": "另一集片段", "feedback": "其他集意见不属于本集"}]
        before = copy.deepcopy(self.state)
        public = dashboard.public_state(self.state)["episodes"][0]
        self.assertEqual(public["latest_feedback"], "改为六秒三镜，使用下面的完整分镜表。")
        self.assertEqual(self.state, before)
        self.state["events"].append({"action": "human_rejected", "kind": "shot", "id": self.shot["id"],
                                     "feedback": "更新：保留六秒，调整飞行板入画方向。"})
        public = dashboard.public_state(self.state)["episodes"][0]
        self.assertEqual(public["latest_feedback"], "更新：保留六秒，调整飞行板入画方向。")

    def test_launch_allows_recovery_but_still_refuses_a_hard_question(self):
        finished = threading.Event()
        fake_core = types.SimpleNamespace(load_project=lambda root: self.state,
                                         set_paused=lambda root, value: None)
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard, "_core", return_value=fake_core), \
                patch.object(dashboard, "_run", side_effect=lambda root: finished.set()) as run:
            context = dashboard._Context(Path(directory))
            self.assertTrue(context.launch())
            self.assertTrue(finished.wait(2))
            run.assert_called_once()
        self.protocol.planning_recovery_available.return_value = False
        self.protocol.planning_recovery_available.side_effect = None
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard, "_core", return_value=fake_core), \
                patch.object(dashboard, "_run") as run:
            with self.assertRaisesRegex(ValueError, "填写调整意见"):
                dashboard._Context(Path(directory)).launch()
            run.assert_not_called()


@unittest.skipUnless(shutil.which("node"), "Node.js required for synthetic DOM checks")
class PlanningRecoveryDomTests(unittest.TestCase):
    def test_recovery_runs_existing_feedback_and_prompt_lists_future_changes(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "mediaURL", "appendPrompt", "failureDetails", "shotReview", "advisoryReport",
                     "subshotPlan", "promptApproval", "shotCard", "paintApprovalGate", "episodeCard"):
            start = re.search(r"function " + name + r"\(", html).start()
            end = re.search(r"\nfunction ", html[start + 1:])
            sources.append(html[start:start + 1 + end.start()])
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
 querySelector(){return null;}
}
const nodes=new Map(),decisions=[];
const context={document:{createElement:tag=>new Element(tag),createTextNode:text=>({tag:'text',textContent:text})},
 $:id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);},
 state:{id:'synthetic',assets:[],paused:false,dashboard:{running:false}},perform:(...args)=>decisions.push(args),
 toast:()=>{},badge:()=>new Element('span'),reviewDetails:()=>null,appendEpisodeProduction:()=>{},assert,nodes,decisions};
vm.runInNewContext(sources.join('\n')+binding+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const shot={id:'synthetic-s1',status:'planned',version:3,duration:10,
 planning_recovery:{message:'已保留修改意见，继续制作将自动协调未生成片段并准备完整提示词。'}};
const all=walk(shotCard(shot,0,null));
assert.equal(all.some(node=>node.tag==='textarea'),false,'recovery never demands the same feedback again');
assert.equal(all.some(node=>node.textContent==='本段提示词尚未准备完成'),false);
const resume=all.find(node=>node.tag==='button'&&node.textContent==='继续准备提示词');assert.ok(resume);
assert.equal(decisions.length,0,'rendering does not start production');resume.listeners.click();
assert.equal(decisions.length,1);assert.equal(decisions[0][0].action,'run');assert.equal(decisions[0][0].feedback,undefined);
state.episodes=[{shots:[shot]}];paintApprovalGate(state);assert.equal(nodes.get('run').textContent,'继续准备提示词');
nodes.get('run').listeners.click();assert.equal(decisions.length,2);assert.equal(decisions[1][0].action,'run');
state.dashboard.running=true;const busy=walk(shotCard(shot,0,null));assert.equal(busy.find(node=>node.textContent==='继续准备提示词').disabled,true);
state.dashboard.running=false;shot.status='awaiting_prompt_approval';delete shot.planning_recovery;
shot.prompt_candidate={id:'synthetic-c1',version:4,prompt:'合成完整提示词',request_hash:'a'.repeat(64),
 future_continuity_updates:[{id:'synthetic-s2',summary:'从站稳状态接续，不重复接住动作。'}]};
const candidate=walk(promptApproval(shot));assert.equal(candidate.some(node=>node.textContent==='后续接续调整（尚未生成）'),true);
assert.equal(candidate.some(node=>node.textContent==='从站稳状态接续，不重复接住动作。'),true);
assert.equal(candidate.some(node=>node.textContent.includes('各自展示完整提示词')),true);
assert.ok(candidate.find(node=>node.tag==='button'&&node.textContent==='确认提示词并生成'));
assert.equal(decisions.length,2,'future summaries do not approve or generate any clip');
const latest='完整六秒分镜表。'.repeat(100),episode={id:'synthetic-e1',status:'generating',feedback:'过时的旧意见',latest_feedback:latest,shots:[]};
const episodeNodes=walk(episodeCard(episode,0)),fold=episodeNodes.find(node=>node.tag==='details'&&node.className==='latest-feedback review-details');
assert.ok(fold);assert.equal(Boolean(fold.open),false,'long human feedback is folded by default');
assert.ok(walk(fold).some(node=>node.textContent===latest),'the full human input remains available');
assert.equal(episodeNodes.some(node=>node.textContent.includes('过时的旧意见')),false);
`,context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program],
            input=json.dumps({"sources": sources, "binding": binding}), text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
