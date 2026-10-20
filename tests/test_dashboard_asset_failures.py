"""Synthetic asset failure presentation; all model activity is forbidden."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import Mock, patch

from norlan import dashboard


class AssetFailureProjectionTests(unittest.TestCase):
    def setUp(self):
        self.state = {"id": "synthetic", "status": "asset_generation_failed", "plan": {"status": "approved"},
            "budget": {}, "assets": [{"id": "voice", "name": "合成女播音", "kind": "voice", "version": 2,
                "status": "generation_failed", "job_id": "voice-failed", "file": "media/assets/voice-v1.mp4",
                "audio_file": "evidence/voice-v1/audio.wav", "generation_error": "试听未成功；token=PRIVATE-GENERATION"}],
            "jobs": {"voice-failed": {"kind": "voice", "item_id": "voice", "state": "failed", "accounting": "spent",
                "failure": {"message": "合成测试：请求参数不支持", "stage": "submission", "http_status": 400,
                    "provider_code": "InvalidParameter", "request_id": "synthetic-voice-request",
                    "provider_message": "unsupported duration; token=PRIVATE-PROVIDER", "raw": "PRIVATE-RAW"}}},
            "episodes": [{"id": "E1", "shots": [{"id": "S6", "status": "planned", "duration": 10}]}]}

    def test_bound_failure_is_sanitized_and_old_media_never_appears_as_new_candidate(self):
        before = copy.deepcopy(self.state)
        result = dashboard.public_state(self.state)
        asset = result["assets"][0]
        self.assertEqual(result["status"], "asset_generation_failed")
        self.assertEqual(result["episodes"][0]["progress"]["stage"], "asset_generation_failed")
        self.assertEqual(asset["status"], "generation_failed")
        self.assertEqual(asset["version"], 2)
        self.assertEqual(asset["failure"]["http_status"], 400)
        self.assertEqual(asset["failure"]["request_id"], "synthetic-voice-request")
        self.assertTrue(asset["can_retry"])
        self.assertNotIn("file", asset)
        self.assertNotIn("audio_file", asset)
        self.assertNotIn("PRIVATE-", json.dumps(result))
        self.assertEqual(self.state, before)

    def test_wrong_kind_or_wrong_asset_cannot_publish_someone_elses_failure(self):
        job = self.state["jobs"]["voice-failed"]
        for update in ({"kind": "video"}, {"kind": "image"}, {"kind": "voice", "item_id": "another-asset"}):
            job.update(update)
            asset = dashboard.public_state(self.state)["assets"][0]
            self.assertFalse(asset["can_retry"])
            self.assertNotIn("failure", asset)
            self.assertIn("generation_error", asset)

    def test_image_failure_uses_its_own_image_job(self):
        self.state["assets"][0]["kind"] = "character"
        self.state["jobs"]["voice-failed"]["kind"] = "image"
        asset = dashboard.public_state(self.state)["assets"][0]
        self.assertTrue(asset["can_retry"])
        self.assertEqual(asset["failure"]["provider_code"], "InvalidParameter")

    def test_unknown_or_unsettled_calls_never_enable_retry(self):
        job = self.state["jobs"]["voice-failed"]
        for state, accounting in (("uncertain", "unknown"), ("submitted", "reserved")):
            job.update(state=state, accounting=accounting)
            asset = dashboard.public_state(self.state)["assets"][0]
            self.assertFalse(asset["can_retry"])
        job.update(state="failed", accounting="spent")
        self.state["jobs"]["other-unknown"] = {"kind": "image", "state": "uncertain", "accounting": "unknown"}
        self.assertFalse(dashboard.public_state(self.state)["assets"][0]["can_retry"])

    def test_new_successful_candidate_does_not_inherit_failed_diagnostics(self):
        self.state["assets"][0].update(status="candidate", version=3, file="media/assets/voice-v3.mp4")
        asset = dashboard.public_state(self.state)["assets"][0]
        self.assertNotIn("failure", asset)
        self.assertNotIn("generation_error", asset)
        self.assertEqual(asset["file"], "media/assets/voice-v3.mp4")

    def test_ordinary_continue_cannot_run_or_unpause_asset_failure(self):
        self.state["paused"] = True
        context = dashboard._Context(Path("synthetic-root"))
        core = Mock()
        core.load_project.return_value = self.state
        with patch.object(dashboard, "_core", return_value=core), patch.object(dashboard, "_run") as run:
            with self.assertRaisesRegex(ValueError, "资产失败原因"):
                context.launch()
            run.assert_not_called()
        core.set_paused.assert_not_called()
        self.assertFalse(context.running)


@unittest.skipUnless(shutil.which("node"), "Node.js required for pure DOM rendering test")
class AssetFailureRenderingTests(unittest.TestCase):
    def test_failed_asset_has_explicit_versioned_retry_but_no_media_or_approval(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "badge", "mediaURL", "appendPrompt", "failureDetails", "assetCard", "paintApprovalGate"):
            start = html.index("function " + name + "(")
            sources.append(html[start:html.index("\nfunction ", start + 1)])
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const sources=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.listeners={};this.dataset={};this.attributes={};this.classList={toggle:()=>{}};}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];}
 addEventListener(key,handler){this.listeners[key]=handler;}
 setAttribute(key,value){this.attributes[key]=value;}
}
const nodes=new Map(),revisions=[],decisions=[];
const context={document:{createElement:tag=>new Element(tag)},
 $:id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);},
 kinds:{voice:'声音',character:'人物'},names:{asset_generation_failed:'资产生成失败'},
 state:{paused:false,dashboard:{running:false}},requestBusy:false,assetDirectEditor:()=>null,openAssetPromptEditor:(...args)=>revisions.push(args),
 perform:(...args)=>decisions.push(args),assert,nodes,revisions,decisions};
vm.runInNewContext(sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
for(const kind of ['voice','character']){
 const asset={id:'synthetic-asset',kind,status:'generation_failed',version:2,file:'media/OLD-v1.mp4',can_retry:true,
   generation_error:'合成试听错误',failure:{message:'合成错误：参数不支持',provider_code:'InvalidParameter',http_status:400,request_id:'synthetic-request'}};
 const content=walk(assetCard(asset));
 assert.equal(content.some(n=>['video','audio','img'].includes(n.tag)),false,'old media is never presented as a new candidate');
 assert.equal(content.some(n=>n.tag==='button'&&n.textContent==='确认并继续'),false);
 assert.equal(content.some(n=>n.textContent.includes('synthetic-request')),true);
 assert.equal(content.some(n=>n.textContent.includes('没有可审批的新候选')),true);
 const retry=content.find(n=>n.tag==='button'&&n.textContent===(kind==='voice'?'修改提示词并重试试听':'修改提示词并重试资产'));
 assert.ok(retry);assert.equal(retry.disabled,false);assert.equal(revisions.length,kind==='voice'?0:1);
 retry.listeners.click();assert.equal(revisions.at(-1)[0].id,asset.id);assert.equal(revisions.at(-1)[0].version,2);assert.equal(revisions.at(-1).length,1);
 assert.equal(decisions.length,0,'opening the retry form cannot run production');
 state.paused=true;assert.equal(walk(assetCard(asset)).find(n=>n.tag==='button').disabled,true);state.paused=false;
 asset.can_retry=false;asset.failure.uncertain=true;assert.equal(walk(assetCard(asset)).some(n=>n.tag==='button'),false);
}
const failed={assets:[{status:'generation_failed'}],dashboard:{running:false},paused:false};paintApprovalGate(failed);
assert.equal(nodes.get('status').textContent,'资产生成失败');assert.equal(nodes.get('run').disabled,true);
failed.paused=true;paintApprovalGate(failed);assert.equal(nodes.get('run').disabled,false);assert.equal(nodes.get('run').textContent,'恢复制作');
`,context);
'''
        done = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps(sources),
                              text=True, capture_output=True, timeout=15)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
