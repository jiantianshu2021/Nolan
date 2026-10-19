"""Synthetic UI contract for interrupted advisory reports; no provider activity."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from norlan import dashboard


class AdvisoryInterruptionProjectionTests(unittest.TestCase):
    def setUp(self):
        self.shot = {"id": "S6", "status": "review_required", "version": 2, "duration": 10,
            "file": "media/E1/S6-v2.mp4", "sha256": "a" * 64, "job_id": "video-synthetic",
            "review": {"verdict": "uncertain", "version": 2, "sha256": "a" * 64,
                "report_unavailable": True, "review_interrupted": True, "advisory": True,
                "issues": ["GPT 审查中断，报告不可用，视频可人工审批。"],
                "audio_checked": False, "whole_video_checked": False, "raw": "PRIVATE-REPORT"}}
        self.review_job = {"id": "gpt-synthetic", "kind": "llm", "item_id": "review", "amount": 0,
            "state": "uncertain", "accounting": "unknown", "task_id": None,
            "advisory_recovery": {"status": "interrupted", "no_retry": True, "private": "PRIVATE-RECOVERY"}}
        self.state = {"id": "synthetic", "status": "review_required", "mode": "live", "paused": False,
            "plan": {"status": "approved"}, "assets": [], "budget": {"limit": 100, "spent": 11, "unknown": 0},
            "jobs": {"gpt-synthetic": self.review_job, "video-synthetic": {"id": "video-synthetic",
                "kind": "video", "item_id": "S6", "state": "completed", "accounting": "spent", "amount": 11,
                "output": "media/E1/S6-v2.mp4", "task_id": "synthetic-original-task"}},
            "episodes": [{"id": "E1", "status": "generating", "shots": [self.shot]}]}

    def test_bound_report_interruption_keeps_video_and_human_review_without_settling_costs(self):
        before = copy.deepcopy(self.state)
        public = dashboard.public_state(self.state)
        shot = public["episodes"][0]["shots"][0]
        self.assertEqual(shot["file"], "media/E1/S6-v2.mp4")
        self.assertEqual(shot["status"], "review_required")
        self.assertTrue(shot["review"]["report_unavailable"])
        self.assertTrue(shot["review"]["review_interrupted"])
        self.assertFalse(shot["review"]["audio_checked"])
        self.assertNotIn("failure", shot)
        self.assertTrue(shot["progress"]["waiting_for_human"])
        self.assertFalse(shot["progress"]["is_working"])
        self.assertIn("GPT 报告中断", shot["progress"]["label"])
        self.assertEqual(public["episodes"][0]["progress"]["stage"], "review_required")
        self.assertIn("视频可人工审片", public["episodes"][0]["progress"]["label"])
        self.assertEqual(public["pending_advisory_count"], 1)
        self.assertIn("费用或额度待核实", public["pending_advisory_notice"])
        self.assertIn("未自动重试", public["pending_advisory_notice"])
        self.assertNotIn("PRIVATE-", json.dumps(public))
        self.assertEqual(self.state, before)
        self.assertEqual(self.review_job["accounting"], "unknown")

    def test_paid_advisory_unknown_still_displays_unsettled_budget_without_free_claim(self):
        self.review_job["amount"] = 3
        self.state["budget"]["unknown"] = 3
        public = dashboard.public_state(self.state)
        self.assertEqual(public["pending_advisory_count"], 1)
        self.assertEqual(public["budget"]["unknown"], 3)
        self.assertNotIn("免费", public["pending_advisory_notice"])
        self.assertNotIn("已结算", public["pending_advisory_notice"])

    def test_missing_or_wrong_report_binding_does_not_publish_an_interruption_notice(self):
        original = copy.deepcopy(self.shot["review"])
        for updates in ({"sha256": "b" * 64}, {"version": 1}):
            self.shot["review"] = copy.deepcopy(original)
            self.shot["review"].update(updates)
            public = dashboard.public_state(self.state)["episodes"][0]["shots"][0]
            self.assertNotIn("review", public)
            self.assertNotIn("GPT 报告中断", public["progress"]["label"])
        self.assertEqual(self.shot["file"], "media/E1/S6-v2.mp4")

    def test_real_unknown_media_and_unmarked_llm_remain_blocking(self):
        for kind in ("video", "image", "voice", "llm"):
            self.state["jobs"]["other-unknown"] = {"kind": kind, "item_id": "review" if kind == "llm" else "OTHER",
                "state": "uncertain", "accounting": "unknown", "amount": 0}
            public = dashboard.public_state(self.state)
            self.assertEqual(public["episodes"][0]["progress"]["stage"], "unknown")
            self.assertEqual(public["pending_advisory_count"], 1)
        self.state["jobs"].pop("other-unknown")
        self.review_job.pop("advisory_recovery")
        public = dashboard.public_state(self.state)
        self.assertEqual(public["pending_advisory_count"], 0)
        self.assertEqual(public["pending_advisory_notice"], "")
        self.assertEqual(public["episodes"][0]["progress"]["stage"], "unknown")

    def test_completed_video_can_never_inherit_current_unknown_video_exemption(self):
        self.state["jobs"]["video-synthetic"].update(state="uncertain", accounting="unknown")
        public = dashboard.public_state(self.state)
        shot = public["episodes"][0]["shots"][0]
        self.assertTrue(shot["failure"]["uncertain"])
        self.assertFalse(shot["can_retry"])
        self.assertEqual(shot["progress"]["stage"], "unknown")
        self.assertEqual(public["episodes"][0]["progress"]["stage"], "unknown")

    def test_explicit_failed_segment_retry_gate_ignores_only_registered_advisory_interruption(self):
        self.shot.update(status="generation_failed", generation_error="合成确定失败")
        self.shot.pop("file")
        job = self.state["jobs"]["video-synthetic"]
        job.update(state="failed", accounting="spent", failure={"message": "合成参数错误", "stage": "submission"})
        public = dashboard.public_state(self.state)["episodes"][0]["shots"][0]
        self.assertTrue(public["can_retry"])
        self.review_job.pop("advisory_recovery")
        self.assertFalse(dashboard.public_state(self.state)["episodes"][0]["shots"][0]["can_retry"])

    def test_explicit_failed_asset_retry_gate_has_same_narrow_execution_rule(self):
        self.state["assets"] = [{"id": "voice", "kind": "voice", "version": 1, "status": "generation_failed",
            "job_id": "voice-failed", "generation_error": "合成确定失败"}]
        self.state["jobs"]["voice-failed"] = {"kind": "voice", "item_id": "voice", "state": "failed",
            "accounting": "spent", "amount": 0, "failure": {"message": "合成参数错误"}}
        self.assertTrue(dashboard.public_state(self.state)["assets"][0]["can_retry"])
        self.state["jobs"]["image-unknown"] = {"kind": "image", "item_id": "OTHER", "state": "uncertain", "accounting": "unknown"}
        self.assertFalse(dashboard.public_state(self.state)["assets"][0]["can_retry"])


@unittest.skipUnless(shutil.which("node"), "Node.js required for pure DOM rendering test")
class AdvisoryInterruptionRenderingTests(unittest.TestCase):
    def test_notice_stays_outside_folded_report_and_approval_remains_explicit(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "mediaURL", "appendPrompt", "failureDetails", "shotReview", "advisoryReport",
                     "subshotPlan", "promptApproval", "shotCard", "paintAdvisoryNotice", "paintApprovalGate"):
            start = html.index("function " + name + "(")
            sources.append(html[start:html.index("\nfunction ", start + 1)])
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const sources=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.listeners={};this.dataset={};this.attributes={};this.hidden=false;this.classList={toggle:(name,on)=>{this.hidden=on;}};}
 append(...nodes){this.children.push(...nodes);}
 addEventListener(key,handler){this.listeners[key]=handler;}
 setAttribute(key,value){this.attributes[key]=value;}
}
const nodes=new Map(),decisions=[],revisions=[];
const context={document:{createElement:tag=>new Element(tag),createTextNode:text=>({tag:'text',textContent:text})},
 $:id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);},
 state:{id:'synthetic',assets:[],dashboard:{running:false}},perform:(...args)=>decisions.push(args),
 showRevision:(...args)=>revisions.push(args),assert,nodes,decisions,revisions};
vm.runInNewContext(sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const shot={id:'S6',status:'review_required',version:2,duration:10,file:'media/E1/S6-v2.mp4',review:{version:2,verdict:'uncertain',issues:['审查中断'],report_unavailable:true,review_interrupted:true}};
let card=shotCard(shot,5,null),all=walk(card),video=card.children.findIndex(n=>n.tag==='video');
assert.equal(card.children[video+1].className,'preview-pending advisory-interruption');
assert.equal(card.children[video+1].tag,'p');assert.equal(card.children[video+1].textContent.includes('GPT 审查中断，报告不可用，视频可人工审批'),true);
const report=all.find(n=>n.className==='advisory-report');assert.equal(report.tag,'details');assert.equal(report.open,false);
assert.equal(walk(report).some(n=>n.textContent.includes('GPT 报告不可用')),true);
assert.equal(all.some(n=>n.textContent==='GPT 正在整理本版本的审查报告。'),false);
assert.equal(decisions.length,0);const approve=all.find(n=>n.tag==='button'&&n.textContent==='通过本段并继续');assert.equal(approve.disabled,true);
const checkbox=all.find(n=>n.tag==='input');checkbox.checked=true;checkbox.listeners.change();approve.listeners.click();assert.equal(decisions.at(-1)[0].action,'approve');assert.equal(decisions.at(-1)[0].version,2);
all.find(n=>n.tag==='button'&&n.textContent==='填写意见并返工').listeners.click();assert.equal(revisions.at(-1)[0].id,'S6');assert.equal(decisions.length,1,'rework form must not submit work');
shot.failure={message:'原视频调用结果未知',uncertain:true};shot.can_retry=false;card=shotCard(shot,5,null);all=walk(card);
assert.equal(all.some(n=>n.tag==='video'),true);assert.equal(all.some(n=>n.tag==='button'),false);assert.equal(all.some(n=>n.className==='preview-pending advisory-interruption'),false);
paintAdvisoryNotice({pending_advisory_count:1,pending_advisory_notice:'有 1 次 GPT 审查调用的费用或额度待核实，未自动重试。'});
assert.equal(nodes.get('advisory-notice').hidden,false);assert.equal(nodes.get('advisory-notice').textContent.includes('费用或额度待核实'),true);assert.equal(nodes.get('advisory-notice').textContent.includes('已结算'),false);
paintAdvisoryNotice({pending_advisory_count:0});assert.equal(nodes.get('advisory-notice').hidden,true);
paintApprovalGate({dashboard:{running:false},episodes:[{progress:{stage:'unknown'}}]});assert.equal(nodes.get('run').disabled,true);assert.equal(nodes.get('status').textContent,'调用结果待核实');
`,context);
'''
        done = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps(sources),
                              text=True, capture_output=True, timeout=15)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
