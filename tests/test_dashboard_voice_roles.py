"""Synthetic UI regressions for internal voice roles; no generation or approvals."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from norlan import core, dashboard
from norlan.asset_inputs import empty_input
from norlan.progress import episode_progress


class VoiceRoleProjectionTests(unittest.TestCase):
    def setUp(self):
        self.source = {"id": "system-role", "kind": "character", "name": "系统播报用途",
            "prompt": "只在画外播报系统任务，不出现在画面中。token=SOURCE-SECRET https://private.invalid/signed",
            "voice_only": True, "status": "internal", "internal_role": True, "version": 1,
            "creation_input": empty_input(), "raw_config": {"key": "PRIVATE-CONFIG"}}
        self.source["sha256"] = core.voice_only_digest(self.source)
        self.voice = {"id": "system-voice", "kind": "voice", "name": "女性系统播报声音", "prompt": "合成中性稳定女声。",
            "character_id": "system-role", "status": "planned", "version": 0, "creation_input": empty_input()}
        self.policy = {"system_broadcast": {"enabled": True, "character_id": "system-role", "voice_id": "system-voice",
                                            "instruction": "系统任务出现时，用已确认女声播报。"}}
        self.state = {"id": "synthetic", "status": "awaiting_assets", "paused": False,
            "plan": {"status": "approved", "version": 1}, "assets": [self.source, self.voice], "jobs": {},
            "budget": {}, "audio_policy": self.policy, "episodes": [{"id": "E1", "shots": [
                {"id": "S1", "status": "planned", "duration": 10}]}]}

    def test_internal_source_has_no_asset_card_input_or_approval_count_entry(self):
        before = copy.deepcopy(self.state)
        public = dashboard.public_state(self.state)
        self.assertEqual([item["id"] for item in public["assets"]], ["system-voice"])
        self.assertEqual(sum(item["status"] == "approved" for item in public["assets"]), 0)
        self.assertTrue(public["assets"][0]["input_editable"])
        self.assertTrue(public["assets"][0]["input_required"])
        source = public["assets"][0]["voice_source"]
        self.assertEqual(set(source), {"name", "description", "offscreen", "purpose"})
        self.assertTrue(source["offscreen"])
        self.assertEqual(source["purpose"], "系统通知与任务播报")
        self.assertNotIn("SOURCE-SECRET", json.dumps(public))
        self.assertNotIn("PRIVATE-CONFIG", json.dumps(public))
        self.assertNotIn("https://private.invalid", json.dumps(public))
        self.assertEqual(self.state, before)

    def test_legacy_source_candidate_is_also_hidden_without_creating_a_human_decision(self):
        self.source.update(status="candidate", internal_role=False)
        public = dashboard.public_state(self.state)
        self.assertEqual([item["id"] for item in public["assets"]], ["system-voice"])
        self.assertEqual(self.source["status"], "candidate")
        self.assertNotIn("approved_at", self.source)

    def test_real_visible_character_and_its_voice_are_not_folded(self):
        hero = {"id": "hero", "kind": "character", "name": "可见演员", "status": "candidate", "version": 1,
                "file": "media/assets/hero-v1.png", "creation_input": empty_input()}
        self.state["assets"].append(hero)
        self.voice["character_id"] = "hero"
        public = dashboard.public_state(self.state)
        self.assertEqual([item["id"] for item in public["assets"]], ["system-voice", "hero"])
        self.assertNotIn("voice_source", public["assets"][0])
        self.assertEqual(public["assets"][1]["file"], hero["file"])

    def test_new_asset_proposal_merges_source_into_one_voice_and_hides_internal_reference(self):
        self.state["assets"] = []
        self.state["asset_amendments"] = [{"id": "add-system", "status": "candidate", "target_shot_id": "S1",
            "additions": [copy.deepcopy(self.source), copy.deepcopy(self.voice)],
            "shot_updates": [{"id": "S1", "add_asset_ids": ["system-role", "system-voice"], "dialogue": "绑定成功。"}],
            "audio_policy": self.policy}]
        amendment = dashboard.public_state(self.state)["asset_amendments"][0]
        self.assertEqual([item["id"] for item in amendment["additions"]], ["system-voice"])
        self.assertTrue(amendment["additions"][0]["voice_source"]["offscreen"])
        self.assertEqual(amendment["shot_updates"][0]["add_asset_ids"], ["system-voice"])
        self.assertEqual(amendment["audio_policy"]["system_broadcast"]["enabled"], True)

    def test_proposal_can_fold_an_existing_source_without_exposing_its_private_fields(self):
        self.state["asset_amendments"] = [{"id": "add-voice", "status": "candidate", "target_shot_id": "S1",
            "additions": [copy.deepcopy(self.voice)], "shot_updates": [], "audio_policy": self.policy}]
        addition = dashboard.public_state(self.state)["asset_amendments"][0]["additions"][0]
        self.assertEqual(addition["voice_source"]["name"], "系统播报用途")
        self.assertNotIn("raw_config", addition["voice_source"])

    def test_prompt_reference_summary_excludes_internal_source_as_an_image(self):
        candidate = {"id": "prompt-synthetic", "version": 1, "request_hash": "a" * 64,
            "prompt": "合成测试提示词", "reference_manifest": {"context": {
                "assets": [self.source, self.voice],
                "image_references": [{"index": 1, "asset_id": "system-role", "role": "approved_asset"}],
                "audio_references": [{"index": 1, "asset_id": "system-voice", "character_id": "system-role"}]}}}
        self.state["episodes"][0]["shots"][0].update(status="awaiting_prompt_approval", prompt_candidate=candidate)
        references = dashboard.public_state(self.state)["episodes"][0]["shots"][0]["prompt_candidate"]["reference_manifest"]
        self.assertEqual(references["image_references"], [])
        self.assertEqual([item["id"] for item in references["assets"]], ["system-voice"])
        self.assertTrue(references["assets"][0]["voice_source"]["offscreen"])
        self.assertEqual(references["audio_references"][0]["asset_id"], "system-voice")

    def test_progress_uses_internal_readiness_but_still_waits_for_real_voice_approval(self):
        episode = self.state["episodes"][0]
        self.assertEqual(episode_progress(self.state, episode)["stage"], "awaiting_assets")
        self.voice.update(status="approved", version=1, file="media/assets/system-voice-v1.mp4",
                          audio_file="evidence/system-voice-v1/audio.wav")
        self.assertEqual(episode_progress(self.state, episode)["stage"], "queued")
        self.assertEqual(self.source["status"], "internal")
        self.assertNotIn("approved_at", self.source)
        self.source.update(status="candidate", internal_role=False)
        self.assertEqual(episode_progress(self.state, episode)["stage"], "queued")


@unittest.skipUnless(shutil.which("node"), "Node.js required for pure DOM rendering test")
class VoiceRoleRenderingTests(unittest.TestCase):
    def test_voice_card_plays_audio_first_and_merges_role_without_an_extra_approval(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "badge", "mediaURL", "appendPrompt", "failureDetails", "assetCard", "paintAssetAmendments", "updateInputControls"):
            start = html.index("function " + name + "(")
            sources.append(html[start:html.index("\nfunction ", start + 1)])
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const sources=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.listeners={};this.dataset={};this.attributes={};this.classList={add:()=>{},toggle:()=>{}};}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];}
 addEventListener(key,handler){this.listeners[key]=handler;}
 setAttribute(key,value){this.attributes[key]=value;}
}
const nodes=new Map(),decisions=[];
const context={document:{createElement:tag=>new Element(tag)},
 $:id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);},
 kinds:{voice:'声音'},names:{candidate:'待确认'},perform:(...args)=>decisions.push(args),
 state:{paused:false,dashboard:{running:false}},amendmentSignature:'',inputDrafts:new Map(),hasInputDrafts:()=>false,requestBusy:false,assert,nodes,decisions};
vm.runInNewContext(sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
assert.equal(assetCard({id:'hidden-role',kind:'character',voice_only:true,status:'candidate'}),null);
const voice={id:'system-voice',kind:'voice',name:'系统女性播报声音',status:'candidate',version:1,file:'media/assets/voice-v1.mp4',audio_file:'evidence/voice-v1/audio.wav',voice_source:{name:'系统播报用途',description:'只播报系统提示。',offscreen:true,purpose:'系统通知与任务播报'}};
let content=walk(assetCard(voice));
assert.equal(content.filter(n=>n.tag==='audio').length,1);assert.equal(content.filter(n=>n.tag==='video').length,0);
const audio=content.find(n=>n.tag==='audio');assert.equal(audio.src,'/media/evidence/voice-v1/audio.wav');assert.equal(audio.controls,true);assert.equal(audio.attributes['aria-label'],'声音试听 · 系统女性播报声音');
assert.equal(content.some(n=>n.textContent==='声音试听'),true);assert.equal(content.some(n=>n.textContent.includes('系统通知与任务播报')),true);assert.equal(content.some(n=>n.textContent.includes('不出现在画面中')),true);
assert.equal(content.some(n=>n.textContent.includes('系统播报用途')),false);assert.equal(content.some(n=>n.textContent.includes('只播报系统提示。')),false,'internal role description is not product copy');
assert.equal(content.some(n=>n.tag==='button'&&n.textContent==='确认画外声源定义'),false);
content.find(n=>n.tag==='button'&&n.textContent==='确认音色并继续').listeners.click();assert.equal(decisions.at(-1)[0].id,'system-voice');assert.equal(decisions.at(-1)[0].version,1);
delete voice.audio_file;content=walk(assetCard(voice));assert.equal(content.filter(n=>n.tag==='video').length,1);assert.equal(content.find(n=>n.tag==='video').src,'/media/media/assets/voice-v1.mp4');
delete voice.file;voice.status='planned';content=walk(assetCard(voice));assert.equal(content.some(n=>['audio','video'].includes(n.tag)),false);assert.equal(content.some(n=>n.textContent==='尚未生成试听'),true);assert.equal(content.some(n=>n.tag==='button'),false);
paintAssetAmendments({dashboard:{running:false},asset_amendments:[{id:'add-voice',status:'candidate',target_shot_id:'S1',additions:[voice],shot_updates:[]}]});
const additions=walk(nodes.get('asset-amendments'));assert.equal(additions.filter(n=>n.tag==='strong'&&n.textContent==='系统女性播报声音 · 声音').length,1);assert.equal(additions.some(n=>n.textContent.includes('画外声音，不入画')),true);assert.equal(additions.some(n=>n.tag==='button'&&n.textContent==='确认画外声源定义'),false);
assert.equal(additions.some(n=>n.textContent.includes('系统播报用途')),false);assert.equal(additions.some(n=>n.textContent.includes('只播报系统提示。')),false,'amendment also hides internal source prose');
state.assets=[{...voice,input_editable:true,input_required:false}];updateInputControls();assert.equal(nodes.get('generate-assets').textContent,'生成声音试听');assert.equal(nodes.get('generate-assets').disabled,false);
state.assets.push({id:'hero',kind:'character',input_editable:true,input_required:false});updateInputControls();assert.equal(nodes.get('generate-assets').textContent,'按以上选择生成资产');
assert.equal(decisions.length,1,'labels and rendering must never submit generation or approval');
`,context);
'''
        done = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps(sources),
                              text=True, capture_output=True, timeout=15)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
