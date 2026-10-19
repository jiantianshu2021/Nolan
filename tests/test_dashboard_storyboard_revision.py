"""Synthetic draft projection and DOM behavior; no GPT or video requests."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from norlan import dashboard


def revised_shot():
    storyboard = {"summary": "先看零件，再看人物反应。", "before_count": 1, "before_duration": 5, "duration": 10,
                  "subshots": [{"start": 0, "end": 4, "prompt": "特写，手中零件放上桌面。"},
                               {"start": 4, "end": 10, "prompt": "中景，人物抬头看向同伴。"}]}
    return {"id": "E1-S16", "status": "awaiting_prompt_approval", "duration": 10, "version": 1,
            "subshots": copy.deepcopy(storyboard["subshots"]), "storyboard_revision": storyboard,
            "prompt_candidate": {"id": "draft-2", "version": 2, "request_hash": "a" * 64,
                "duration": 10, "prompt": "完整新提示词：0–4 秒特写零件，4–10 秒中景人物反应。",
                "dialogue": "", "storyboard_revision": copy.deepcopy(storyboard)}}


class StoryboardRevisionProjectionTests(unittest.TestCase):
    def state(self, shot):
        return {"id": "synthetic", "mode": "live", "status": "awaiting_prompt_approval",
                "plan": {"status": "approved"}, "budget": {}, "assets": [], "jobs": {},
                "episodes": [{"id": "E1", "shots": [shot]}]}

    def test_only_current_bounded_draft_is_projected_and_nothing_mutates(self):
        shot = revised_shot()
        private = "PRIVATE-STORYBOARD-HISTORY"
        for source in (shot["storyboard_revision"], shot["prompt_candidate"]["storyboard_revision"]):
            source.update(before={"subshots": [private] * 2000}, signature=private, provider_response=private)
            source["subshots"][0]["internal"] = private
        shot.update(storyboard_revision_history=[private] * 2000,
                    storyboard_revision_binding={"secret": private})
        state = self.state(shot)
        before = copy.deepcopy(state)
        public = dashboard.public_state(state)["episodes"][0]["shots"][0]
        expected = {"summary": "先看零件，再看人物反应。", "before_count": 1, "after_count": 2,
                    "before_duration": 5, "duration": 10,
                    "subshots": [{"start": 0, "end": 4, "prompt": "特写，手中零件放上桌面。"},
                                 {"start": 4, "end": 10, "prompt": "中景，人物抬头看向同伴。"}]}
        self.assertEqual(public["storyboard_revision"], expected)
        self.assertEqual(public["prompt_candidate"]["storyboard_revision"], expected)
        self.assertNotIn(private, json.dumps(public))
        self.assertLess(len(json.dumps(public)), 6000, "history must not enlarge the current draft")
        self.assertEqual(state, before)

    def test_bad_schedule_is_not_presented_as_a_valid_draft(self):
        mutations = [lambda value: value.update(before_count=True),
                     lambda value: value.update(before_count=4),
                     lambda value: value.update(before_duration=True),
                     lambda value: value.update(duration=3),
                     lambda value: value.update(duration=16),
                     lambda value: value.update(duration=10.0),
                     lambda value: value.update(summary="x" * 2001),
                     lambda value: value.update(subshots=value["subshots"] * 2),
                     lambda value: value["subshots"][0].update(start=True),
                     lambda value: value["subshots"][0].update(end=float("nan")),
                     lambda value: value["subshots"][0].update(end=10 ** 1000),
                     lambda value: value["subshots"][1].update(start=4.1),
                     lambda value: value["subshots"][1].update(end=11),
                     lambda value: value["subshots"][1].update(end=9),
                     lambda value: value["subshots"][0].update(prompt="x" * 2001)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                shot = revised_shot()
                mutate(shot["storyboard_revision"])
                shot["prompt_candidate"]["storyboard_revision"] = copy.deepcopy(shot["storyboard_revision"])
                public = dashboard.public_state(self.state(shot))["episodes"][0]["shots"][0]
                self.assertNotIn("storyboard_revision", public)
                self.assertIsNone(public["prompt_candidate"]["storyboard_revision"])
        for invalid in (float("nan"), float("inf"), 10 ** 1000, True, -5):
            shot = revised_shot()
            shot["duration"] = invalid
            self.assertIsNone(dashboard._public_storyboard_revision(shot["storyboard_revision"], shot))

    def test_legacy_shot_has_no_new_draft_or_gate(self):
        shot = revised_shot()
        shot.pop("storyboard_revision")
        shot["prompt_candidate"].pop("storyboard_revision")
        public = dashboard.public_state(self.state(shot))["episodes"][0]["shots"][0]
        self.assertNotIn("storyboard_revision", public)
        self.assertNotIn("storyboard_revision", public["prompt_candidate"])

    def test_editability_is_derived_from_saved_rework_and_active_gates(self):
        shot = revised_shot()
        shot.update(status="planned", history=[{"version": 1, "file": "media/original-v1.mp4", "sha256": "b" * 64}])
        shot.pop("prompt_candidate")
        state = self.state(shot)

        def editable():
            return dashboard.public_state(state)["episodes"][0]["shots"][0]["can_edit_revision"]

        self.assertTrue(editable())
        shot["pending_asset_amendment"] = "pending-asset"
        self.assertFalse(editable())
        shot.pop("pending_asset_amendment")
        state["jobs"] = {"inflight": {"kind": "video", "item_id": shot["id"],
                                       "state": "submitted", "accounting": "reserved"}}
        self.assertFalse(editable())
        state["jobs"] = {}
        shot["history"] = []
        self.assertFalse(editable(), "an untouched first-generation shot is not rework")


@unittest.skipUnless(shutil.which("node"), "Node.js required for synthetic DOM")
class StoryboardRevisionRenderingTests(unittest.TestCase):
    def test_saved_unsubmitted_revision_reuses_feedback_and_obeys_pending_asset_gate(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "shotCard"):
            start = html.index("function " + name + "(")
            sources.append(html[start:html.index("\nfunction ", start + 1)])
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const sources=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.dataset={};this.listeners={};}
 append(...nodes){this.children.push(...nodes);}
 addEventListener(key,fn){this.listeners[key]=fn;}
}
const revisions=[],sourcePrompts=[];
const context={assert,revisions,sourcePrompts,document:{createElement:tag=>new Element(tag)},
 state:{paused:false,dashboard:{running:false}},
 promptApproval:()=>null,failureDetails:()=>null,subshotPlan:()=>null,appendPrompt:(_node,prompt)=>sourcePrompts.push(prompt),
 showRevision:(...args)=>revisions.push(args)};
vm.runInNewContext(sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const shot={id:'S16',version:1,status:'planned',duration:5,can_edit_revision:true};
let content=walk(shotCard(shot,15));
const edit=content.find(n=>n.tag==='button'&&n.textContent==='修改返修意见');
assert.ok(edit);assert.equal(edit.disabled,false);assert.equal(revisions.length,0);
edit.listeners.click();assert.equal(revisions.length,1);assert.equal(revisions[0][0].id,'S16');
assert.equal(revisions[0][1],'shot','same existing revision dispatch');
shot.asset_amendment_pending=true;
assert.equal(walk(shotCard(shot,15)).some(n=>n.textContent==='修改返修意见'),false);
delete shot.asset_amendment_pending;shot.can_edit_revision=false;
assert.equal(walk(shotCard(shot,15)).some(n=>n.textContent==='修改返修意见'),false);
shot.can_edit_revision=true;state.paused=true;
assert.equal(walk(shotCard(shot,15)).find(n=>n.textContent==='修改返修意见').disabled,true);
state.paused=false;state.dashboard.running=true;
assert.equal(walk(shotCard(shot,15)).find(n=>n.textContent==='修改返修意见').disabled,true);
shot.prompt='OLD SOURCE PROMPT';shot.storyboard_revision={summary:'新分镜'};shot.prompt_candidate={id:'draft-2'};
sourcePrompts.length=0;content=walk(shotCard(shot,15));
assert.equal(sourcePrompts.length,0,'new candidate does not also present the old source as final');
assert.equal(content.some(n=>n.textContent==='原始片段说明（供对照）'),false);
delete shot.prompt_candidate;content=walk(shotCard(shot,15));
assert.equal(content.some(n=>n.textContent==='原始片段说明（供对照）'),true);
assert.equal(content.some(n=>n.textContent==='OLD SOURCE PROMPT'),true);
assert.equal(content.some(n=>n.textContent.includes('不是新版本的最终提示词')),true);
`,context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps(sources),
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_one_feedback_field_and_one_final_gate_show_the_same_new_schedule(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "showRevision", "subshotPlan", "promptApproval"):
            start = html.index("function " + name + "(")
            sources.append(html[start:html.index("\nfunction ", start + 1)])
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {sources,shot}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.dataset={};this.listeners={};}
 append(...nodes){this.children.push(...nodes);}
 addEventListener(key,fn){this.listeners[key]=fn;}
 showModal(){this.shown=true;}
 focus(){this.focused=true;}
}
const nodes=new Map(),decisions=[];
const context={assert,shot,nodes,decisions,currentAsset:null,
 document:{createElement:tag=>new Element(tag)},
 $:id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);},
 state:{assets:[],paused:false,dashboard:{running:false}},perform:(...args)=>decisions.push(args)};
vm.runInNewContext(sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const content=walk(promptApproval(shot)),text=content.map(n=>n.textContent).join(' · ');
const draft=content.find(n=>n.className==='storyboard-revision');
assert.ok(draft);assert.equal(draft.tag,'details');assert.equal(Boolean(draft.open),false);
assert.equal(draft.dataset.uiKey,'storyboard:'+shot.id+':'+shot.prompt_candidate.id);
assert.equal(text.includes('本次重新分镜 · 1 → 2 镜'),true);
assert.equal(text.includes('待生成分镜草稿'),true);assert.equal(text.includes('原 5 秒 → 新 10 秒'),true);
for(const part of shot.prompt_candidate.storyboard_revision.subshots){
 assert.equal(text.includes(part.start+'–'+part.end+' 秒'),true);assert.equal(text.includes(part.prompt),true);
}
assert.equal(text.includes(shot.prompt_candidate.prompt),true);
assert.equal(content.filter(n=>n.tag==='button').length,2,'one final approval and one edit action');
assert.equal(content.some(n=>n.tag==='textarea'||n.tag==='input'),false,'no separate structure form');
assert.equal(decisions.length,0,'rendering a draft never authorizes generation');
const approve=content.find(n=>n.textContent==='确认提示词并生成');
approve.listeners.click();assert.equal(decisions.length,1);assert.equal(decisions[0][0].action,'approve_prompt');
assert.equal(decisions[0][0].candidate_id,shot.prompt_candidate.id);
assert.equal(decisions[0][0].request_hash,shot.prompt_candidate.request_hash);
content.find(n=>n.textContent==='修改提示词').listeners.click();
assert.equal(currentAsset.reviewKind,'prompt');
assert.equal(nodes.get('dialog-note').textContent.includes('镜头数量、内容或新增资产'),true);
assert.equal(nodes.get('asset-feedback').placeholder.includes('两个镜头'),true);
const shared=nodes.get('asset-feedback');
showRevision({...shot,file:'media/original-v1.mp4'},'shot','返工片段');
assert.equal(nodes.get('asset-feedback'),shared,'video and prompt revision share the feedback field');
assert.equal(currentAsset.reviewKind,'shot');assert.equal(nodes.get('asset-dialog').shown,true);
assert.equal(nodes.get('dialog-note').textContent.includes('1–3 个内部镜头'),true);
assert.equal(decisions.length,1,'opening feedback does not submit');
const oldVideo={...shot,file:'media/original-v1.mp4'};
let plan=walk(subshotPlan(oldVideo)),copy=plan.map(n=>n.textContent).join(' · ');
assert.equal(copy.includes('待生成分镜草稿'),true);assert.equal(copy.includes('历史视频保留其原版本'),true);
oldVideo.status='review_required';delete oldVideo.prompt_candidate;
copy=walk(subshotPlan(oldVideo)).map(n=>n.textContent).join(' · ');
assert.equal(copy.includes('待生成分镜草稿'),false);
assert.equal(copy.includes('这是生成时的镜头安排'),true);
state.paused=true;
assert.equal(walk(promptApproval(shot)).find(n=>n.textContent==='确认提示词并生成').disabled,true);
`,context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program],
                                input=json.dumps({"sources": sources, "shot": revised_shot()}),
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
