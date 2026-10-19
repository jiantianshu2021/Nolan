"""Synthetic prompt / additive-asset review HTTP checks; no paid providers."""
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

from norlan import dashboard
from tests.test_dashboard import FakeCore


class PromptCore(FakeCore):
    def __init__(self, state):
        super().__init__(state)
        self.calls = []

    def approve_prompt(self, root, item_id, candidate_id, request_hash):
        self.calls.append(("approve_prompt", item_id, candidate_id, request_hash))
        return self.state

    def reject_prompt(self, root, item_id, candidate_id, request_hash, feedback):
        self.calls.append(("reject_prompt", item_id, candidate_id, request_hash, feedback))
        return self.state

    def apply_asset_amendment(self, root, item_id, decision):
        self.calls.append(("asset_amendment", item_id, decision))
        return self.state


class DashboardPromptGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.state = {"id": "synthetic", "plan": {"status": "approved"}, "status": "awaiting_prompt_approval",
                      "budget": {}, "assets": [], "jobs": {}, "episodes": [{"id": "E1", "shots": [
                          {"id": "S1", "status": "awaiting_prompt_approval", "version": 0, "duration": 10,
                           "prompt_candidate": {"id": "candidate-one", "request_hash": "a" * 64, "version": 1,
                               "revision": 0, "prompt": "合成测试最终提示词：0–5 秒全景，5–10 秒近景。",
                               "dialogue": "系统：绑定成功。", "continuity_notes": "接前段结尾。",
                               "config": {"api_key": "SECRET-CONFIG"}, "reference_manifest": {
                                   "images": ["https://private.invalid/signed?token=SECRET-URL"], "context": {
                                       "assets": [{"id": "host", "name": "系统播报员", "voice_only": True,
                                                   "private": "SECRET-ASSET"}],
                                       "audio_references": [{"asset_id": "voice", "index": 1,
                                                             "url": "SECRET-VOICE-URL"}]}}}}]}]}
        self.core = PromptCore(self.state)
        self.core_patch = patch.object(dashboard, "_core", return_value=self.core)
        self.core_patch.start()
        self.run_patch = patch.object(dashboard, "_run", side_effect=AssertionError("HTTP approval must not run production"))
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

    def request(self, method, path="/api/action", payload=None, headers=None):
        client = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        options = {"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token}
        options.update(headers or {})
        try:
            client.request(method, path, body=json.dumps(payload) if payload is not None else None, headers=options)
            response = client.getresponse()
            return response.status, json.loads(response.read())
        finally:
            client.close()

    def payload(self, action="approve_prompt"):
        return {"action": action, "id": "S1", "candidate_id": "candidate-one", "request_hash": "a" * 64}

    def test_public_candidate_is_exact_but_excludes_request_config_and_signed_urls(self):
        before = copy.deepcopy(self.state)
        status, result = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        shot = result["episodes"][0]["shots"][0]
        candidate = shot["prompt_candidate"]
        self.assertEqual(candidate["prompt"], self.state["episodes"][0]["shots"][0]["prompt_candidate"]["prompt"])
        self.assertEqual(candidate["request_hash"], "a" * 64)
        self.assertEqual(candidate["reference_manifest"]["audio_references"], [{"index": 1, "asset_id": "voice"}])
        self.assertEqual(shot["progress"]["stage"], "awaiting_prompt_approval")
        self.assertEqual(shot["progress"]["target_version"], 1)
        self.assertTrue(shot["progress"]["waiting_for_human"])
        self.assertFalse(shot["progress"]["is_working"])
        self.assertEqual(result["episodes"][0]["progress"]["stage"], "awaiting_prompt_approval")
        self.assertNotIn("SECRET-", json.dumps(result))
        self.assertEqual(self.state, before)

    def test_exact_candidate_approval_and_rewrite_dispatch_without_starting_pipeline(self):
        self.assertEqual(self.request("POST", payload=self.payload())[0], 200)
        self.assertEqual(self.core.calls[-1], ("approve_prompt", "S1", "candidate-one", "a" * 64))
        payload = {**self.payload("reject_prompt"), "feedback": "  增加女性系统播报，先确认音色。  "}
        self.assertEqual(self.request("POST", payload=payload)[0], 200)
        self.assertEqual(self.core.calls[-1], ("reject_prompt", "S1", "candidate-one", "a" * 64,
                                              "增加女性系统播报，先确认音色。"))

    def test_hash_ids_feedback_and_csrf_validated(self):
        malformed = [{"request_hash": "not-a-hash"}, {"candidate_id": ""}, {"id": True}]
        for changes in malformed:
            self.assertEqual(self.request("POST", payload={**self.payload(), **changes})[0], 409)
        self.assertEqual(self.request("POST", payload=self.payload("reject_prompt"))[0], 409)
        self.assertEqual(self.request("POST", payload={**self.payload("reject_prompt"), "feedback": " "})[0], 409)
        self.assertEqual(self.request("POST", payload=self.payload(), headers={"X-Norlan-Token": "bad"})[0], 403)
        self.assertEqual(self.request("POST", payload=self.payload(), headers={"Origin": "https://foreign.invalid"})[0], 403)
        self.assertEqual(self.core.calls, [])

    def test_generic_continue_cannot_bypass_prompt_gate(self):
        status, result = self.request("POST", payload={"action": "run"})
        self.assertEqual(status, 409)
        self.assertIn("确认提示词", result["error"])

    def test_additive_asset_candidate_is_safe_and_decision_does_not_generate(self):
        self.state["asset_amendments"] = [{"id": "add-voice", "status": "candidate", "target_shot_id": "S1",
            "target_version": 0, "feedback": "增加女播音", "base_hash": "SECRET-BASE", "additions": [
                {"id": "host", "kind": "character", "voice_only": True, "name": "系统播报员",
                 "prompt": "不入画的系统女声。", "raw": "SECRET-RAW"},
                {"id": "voice", "kind": "voice", "character_id": "host", "prompt": "清晰中性的女声。",
                 "audition_text": "系统绑定成功。", "audition_reason": "原文中平稳完整的系统播报。",
                 "audition_source": "SECRET-SOURCE", "url": "SECRET-SIGNED-VOICE"}],
            "audio_policy": {"system_broadcast": {"enabled": True, "character_id": "host", "voice_id": "voice",
                "instruction": "每次系统任务框出现时，用同一女声播报。", "raw": "SECRET-POLICY-RAW",
                "url": "SECRET-POLICY-URL"}, "config": "SECRET-CONFIG"},
            "shot_updates": [{"id": "S1", "add_asset_ids": ["host", "voice"], "dialogue": "绑定成功。",
                              "raw": "SECRET-UPDATE"}]}]
        status, result = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(result["episodes"][0]["progress"]["stage"], "awaiting_asset_amendment")
        public = result["asset_amendments"][0]
        self.assertEqual([a["id"] for a in public["additions"]], ["voice"])
        self.assertEqual(public["additions"][0]["audition_text"], "系统绑定成功。")
        self.assertEqual(public["additions"][0]["audition_reason"], "原文中平稳完整的系统播报。")
        self.assertEqual(public["audio_policy"], {"system_broadcast": {"enabled": True,
            "character_id": "host", "voice_id": "voice", "instruction": "每次系统任务框出现时，用同一女声播报。"}})
        self.assertNotIn("SECRET-", json.dumps(result))
        self.assertEqual(self.request("POST", payload={"action": "run"})[0], 409)
        for decision in ("approve", "reject"):
            self.assertEqual(self.request("POST", payload={"action": "asset_amendment", "id": "add-voice",
                                                          "decision": decision})[0], 200)
            self.assertEqual(self.core.calls[-1], ("asset_amendment", "add-voice", decision))
        self.assertEqual(self.request("POST", payload={"action": "asset_amendment", "id": "add-voice",
                                                      "decision": True})[0], 409)

    def test_internal_voice_source_is_not_exposed_as_reviewable_asset(self):
        self.state["assets"] = [{"id": "host", "kind": "character", "voice_only": True, "name": "系统播报员",
                                 "prompt": "不入画的系统女声。", "version": 1, "status": "candidate"}]
        _, result = self.request("GET", "/api/state")
        self.assertEqual(result["assets"], [])


@unittest.skipUnless(shutil.which("node"), "Node.js required for pure DOM rendering test")
class AssetAmendmentRenderingTests(unittest.TestCase):
    def test_existing_asset_reuse_has_accurate_copy_and_never_autoruns(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "paintAssetAmendments", "paintApprovalGate"):
            start = html.index("function " + name + "(")
            sources.append(html[start:html.index("\nfunction ", start + 1)])
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const sources=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.listeners={};this.classList={toggle:()=>{}};}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];}
 addEventListener(key,handler){this.listeners[key]=handler;}
}
const nodes=new Map(),decisions=[];
const context={document:{createElement:tag=>new Element(tag)},
 $:id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);},
 kinds:{voice:'声音'},perform:(...args)=>decisions.push(args),amendmentSignature:'',assert,nodes,decisions};
vm.runInNewContext(sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
for(const adding of [true,false]){
 const data={paused:false,dashboard:{running:false},asset_amendments:[{id:adding?'add':'reuse',status:'candidate',target_shot_id:'S6',additions:adding?[{id:'voice',kind:'voice',name:'女播报员',prompt:'合成中性音色',audition_text:'系统绑定成功。',audition_reason:'稳定中性、原文完整播报。'}]:[],audio_policy:{system_broadcast:{enabled:adding,instruction:adding?'每次任务框出现均播报。':'仅当前片段复用，不启用后续自动播报。'}},shot_updates:[{id:'S6',add_asset_ids:['existing-voice'],dialogue:'绑定成功。'}]}]};
 paintAssetAmendments(data);paintApprovalGate(data);
 const content=walk(nodes.get('asset-amendments')),text=content.map(n=>n.textContent).join(' · ');
 assert.equal(nodes.get('amendment-title').textContent,adding?'新增资产方案':'已有资产复用方案');
 assert.equal(text.includes(adding?'新增资产方案':'已有资产复用方案'),true);
 assert.equal(text.includes('无需重新生成已批准资产'),!adding);
 assert.equal(nodes.get('run').disabled,true);
 assert.equal(text.includes('后续系统播报规则：'+(adding?'启用':'停用')),true);
 assert.equal(text.includes(adding?'每次任务框出现均播报。':'仅当前片段复用，不启用后续自动播报。'),true);
 if(adding){const audition=content.find(n=>n.tag==='details'&&n.children.some(c=>c.textContent==='试音原文与选句理由'));assert.ok(audition);assert.equal(Boolean(audition.open),false);assert.equal(text.includes('系统绑定成功。'),true);assert.equal(text.includes('稳定中性、原文完整播报。'),true);}
 if(!adding){assert.equal(text.includes('先回到资产选择与审批'),false);assert.equal(nodes.get('status').textContent,'等待资产复用方案审批');}
 const confirm=content.find(n=>n.tag==='button'&&n.textContent===(adding?'确认新增资产方案':'确认资产复用方案'));
 assert.ok(confirm);confirm.listeners.click();
 assert.equal(decisions.at(-1).length,1,'approval must not implicitly run production');
 assert.equal(decisions.at(-1)[0].action,'asset_amendment');assert.equal(decisions.at(-1)[0].decision,'approve');
}
`,context);
'''
        done = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps(sources),
                              text=True, capture_output=True, timeout=15)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
