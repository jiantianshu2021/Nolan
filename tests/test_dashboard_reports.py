"""Synthetic report/version and human-decision UI checks; no paid providers."""
import copy
import http.client
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from norlan import core, dashboard, pipeline


class ReportProjectionTests(unittest.TestCase):
    def setUp(self):
        self.shot = {"id": "S1", "version": 2, "status": "review_required", "duration": 5,
                     "file": "media/E1/S1-v2.mp4", "sha256": "b" * 64,
                     "review": {"verdict": "fail", "issues": ["合成测试：位置需要核对"],
                                "repair_prompt": "合成测试：保持原有位置", "sha256": "b" * 64,
                                "scope": "sampled_frames_only", "audio_checked": False,
                                "whole_video_checked": False, "provider_raw": "PRIVATE-REPORT"}}
        self.episode = {"id": "E1", "status": "generating", "shots": [self.shot]}
        self.state = {"status": "review_required", "plan": {"status": "approved"}, "assets": [],
                      "jobs": {}, "budget": {"currency": "CNY"}, "episodes": [self.episode]}

    def public_shot(self):
        return dashboard.public_state(self.state)["episodes"][0]["shots"][0]

    def test_all_verdicts_are_advisory_and_remain_human_gates(self):
        before = copy.deepcopy(self.state)
        for verdict in ("pass", "fail", "uncertain"):
            self.shot["review"]["verdict"] = verdict
            result = self.public_shot()
            self.assertEqual(result["review"]["verdict"], verdict)
            self.assertEqual(result["review"]["version"], 2)
            self.assertEqual(result["status"], "review_required")
            self.assertTrue(result["progress"]["waiting_for_human"])
            self.assertFalse(result["progress"]["accepted"])
            self.assertNotIn("PRIVATE-REPORT", json.dumps(result))
        self.shot["review"]["verdict"] = "fail"
        self.assertEqual(self.state, before)
        self.shot["status"] = "accepted"
        self.assertEqual(self.public_shot()["review"]["issues"], before["episodes"][0]["shots"][0]["review"]["issues"])

    def test_old_report_stays_with_history_when_new_version_is_generating(self):
        self.shot["history"] = [copy.deepcopy(self.shot)]
        self.shot.update(version=3, status="generating")
        result = self.public_shot()
        self.assertNotIn("file", result)
        self.assertNotIn("review", result)
        self.assertEqual(result["history"][0]["version"], 2)
        self.assertEqual(result["history"][0]["review"]["version"], 2)
        self.assertNotIn("PRIVATE-REPORT", json.dumps(result))
        self.shot.update(file="media/E1/S1-v3.mp4", sha256="c" * 64)
        self.assertNotIn("review", self.public_shot(), "old hash cannot label a new video")
        self.shot["review"].update(sha256="c" * 64, version=2)
        self.assertNotIn("review", self.public_shot(), "explicit report version must match too")

    def test_history_report_requires_its_own_hash(self):
        self.shot["history"] = [{"version": 1, "file": "media/E1/S1-v1.mp4", "sha256": "a" * 64,
                                 "review": copy.deepcopy(self.shot["review"])}]
        self.assertNotIn("review", self.public_shot()["history"][0])
        self.shot["history"][0]["review"]["sha256"] = "a" * 64
        self.assertEqual(self.public_shot()["history"][0]["review"]["version"], 1)

    def test_rough_report_is_attached_to_exact_cut_not_current_shot(self):
        cut = {"key": "c" * 64, "file": "exports/E1-rough-1-" + "c" * 16 + ".mp4", "sha256": "d" * 64,
               "shot_count": 1, "shots": [{"id": "S1", "version": 2, "sha256": "b" * 64}]}
        report = {**cut, "verdict": "fail", "issues": ["合成测试：粗剪接续建议"], "provider_raw": "PRIVATE-CUT"}
        self.episode.update(rough_cuts=[cut], reviews=[report])
        public = dashboard.public_state(self.state)["episodes"][0]
        self.assertEqual(public["rough_cuts"][0]["review"]["issues"], report["issues"])
        self.assertNotEqual(public["shots"][0]["review"]["issues"], report["issues"])
        self.assertNotIn("PRIVATE-CUT", json.dumps(public))
        report["sha256"] = "e" * 64
        self.assertNotIn("review", dashboard.public_state(self.state)["episodes"][0]["rough_cuts"][0])

    def test_failed_generation_has_safe_human_retry_message_without_video(self):
        for key in ("file", "sha256"):
            self.shot.pop(key)
        self.shot.update(generation_error="PRIVATE-ERROR", continuity_review={"private": "PRIVATE-REASON"})
        result = self.public_shot()
        self.assertTrue(result["progress"]["waiting_for_human"])
        self.assertFalse(result["progress"]["completed"])
        self.assertNotIn("review", result)
        self.assertIn("不会自动再次提交", result["generation_error"])
        self.assertIn("前段已返工", result["continuity_review"])
        self.assertNotIn("PRIVATE-", json.dumps(result))


class HumanReportHttpTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        core.create_project(self.root, "合成人工审批测试")
        core.install_plan(self.root, pipeline.demo_plan())
        self.run = self.enterContext(patch.object(dashboard, "_run"))
        self.server = dashboard.make_server(self.root, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, action, version=1, feedback=None):
        payload = {"action": action, "kind": "shot", "id": "E001-S001", "version": version}
        if feedback is not None:
            payload["feedback"] = feedback
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        connection.request("POST", "/api/action", json.dumps(payload),
                           {"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token})
        response = connection.getresponse()
        status, data = response.status, json.loads(response.read())
        connection.close()
        return status, data

    def prepare(self, verdict):
        path = self.root / "media/E001/E001-S001-v1.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic video artifact only")
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][0]
            shot.update(status="review_required", version=1, file="media/E001/E001-S001-v1.mp4", sha256=core.digest(path),
                        review={"verdict": verdict, "issues": ["合成审查意见"], "sha256": core.digest(path)})

    def test_human_can_approve_each_verdict_and_report_remains_available(self):
        for verdict in ("pass", "fail", "uncertain"):
            with self.subTest(verdict=verdict):
                self.prepare(verdict)
                status, data = self.request("approve")
                self.assertEqual(status, 200, data)
                shot = data["episodes"][0]["shots"][0]
                self.assertEqual(shot["status"], "accepted")
                self.assertEqual(shot["review"]["verdict"], verdict)
                self.assertEqual(self.request("approve", version=2)[0], 409)
        self.run.assert_not_called()

    def test_human_rework_requires_written_feedback_and_keeps_old_report(self):
        self.prepare("pass")
        self.assertEqual(self.request("reject", feedback=" ")[0], 409)
        status, data = self.request("reject", feedback="合成意见：请保留角色并调整站位")
        self.assertEqual(status, 200, data)
        shot = data["episodes"][0]["shots"][0]
        self.assertEqual(shot["status"], "planned")
        self.assertNotIn("file", shot)
        self.assertEqual(shot["history"][-1]["review"]["verdict"], "pass")
        self.run.assert_not_called()


@unittest.skipUnless(shutil.which("node"), "Node.js required for pure DOM rendering test")
class ReportRenderingTests(unittest.TestCase):
    def test_reports_stay_below_their_video_and_buttons_never_follow_gpt_verdict(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "mediaURL", "appendPrompt", "shotReview", "advisoryReport", "subshotPlan", "shotCard", "roughCutCard"):
            start = html.index("function " + name + "(")
            sources.append(html[start:html.index("\nfunction ", start + 1)])
        program = r'''
const assert = require("node:assert/strict"), vm = require("node:vm");
const sources = JSON.parse(require("node:fs").readFileSync(0,"utf8"));
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.attributes={};this.className="";this.textContent="";}
 append(...nodes){this.children.push(...nodes);}
 setAttribute(key,value){this.attributes[key]=value;}
 addEventListener(key,handler){this.listeners[key]=handler;}
}
const decisions=[],revisions=[];
const context={document:{createElement:tag=>new Element(tag),createTextNode:text=>({tag:"text",textContent:text})},
 perform:(...args)=>decisions.push(args),showRevision:(...args)=>revisions.push(args),assert,decisions,revisions};
vm.runInNewContext(sources.join("\n")+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
for(const verdict of ['pass','fail','uncertain']){
 const shot={id:'S1',version:2,status:'review_required',duration:5,file:'media/E1/S1-v2.mp4',review:{version:2,verdict,issues:['审查意见'],repair_prompt:'调整站位'}};
 const card=shotCard(shot,0,null),video=card.children.findIndex(n=>n.tag==='video');
 assert.equal(card.children[video+1].className,'advisory-report');
 assert.equal(card.children[video+1].children.some(n=>n.textContent.includes('GPT 建议不等于最终审批')),true);
 const nodes=walk(card),approve=nodes.find(n=>n.tag==='button'&&n.textContent==='通过本段并继续');
 assert.equal(approve.disabled,true);
 const checkbox=nodes.find(n=>n.tag==='input');checkbox.checked=true;checkbox.listeners.change();
 assert.equal(approve.disabled,false);approve.listeners.click();
 assert.equal(decisions.at(-1)[0].action,'approve');assert.equal(decisions.at(-1)[0].version,2);
 nodes.find(n=>n.tag==='button'&&n.textContent==='填写意见并返工').listeners.click();
 assert.equal(revisions.at(-1)[0].id,'S1');
 shot.status='accepted';const approved=shotCard(shot,0,null);
 assert.equal(walk(approved).filter(n=>n.className==='advisory-report').length,1);
 assert.equal(walk(approved).some(n=>n.tag==='button'&&n.textContent==='通过本段并继续'),false);
}
const failed=shotCard({id:'S1',version:3,status:'review_required',duration:5,generation_error:'生成失败'},0,null);
const nodes=walk(failed),approve=nodes.find(n=>n.tag==='button'&&n.textContent==='通过本段并继续');
assert.equal(approve.disabled,true);assert.equal(nodes.some(n=>n.tag==='input'),false);
assert.equal(nodes.some(n=>n.tag==='button'&&n.textContent==='填写意见并重试'),true);
const historical=shotCard({id:'S1',version:3,status:'generating',duration:5,history:[{version:2,file:'media/E1/S1-v2.mp4',review:{version:2,verdict:'fail',issues:['历史意见']}}]},0,null);
const history=walk(historical).find(n=>n.className==='shot-history-version');
assert.equal(history.children[1].tag,'video');assert.equal(history.children[2].className,'advisory-report');
assert.equal(historical.children.some(n=>n.className==='advisory-report'),false);
const cut=roughCutCard({file:'exports/cut.mp4',shot_count:2,review:{verdict:'uncertain',issues:['接续建议']}},true);
const cutVideo=cut.children.findIndex(n=>n.tag==='video');assert.equal(cut.children[cutVideo+1].className,'advisory-report');
`,context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps(sources), text=True,
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
