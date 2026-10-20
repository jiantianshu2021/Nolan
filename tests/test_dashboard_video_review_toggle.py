"""Synthetic DOM checks for the optional advisory-review switch; no providers."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from norlan import dashboard


HTML = Path(dashboard.__file__).with_name("dashboard.html").read_text()


def function_source(name):
    start = re.search(r"(?:async )?function " + re.escape(name) + r"\(", HTML).start()
    following = re.search(r"\n(?:async )?function ", HTML[start + 1:])
    return HTML[start:start + 1 + following.start()] if following else HTML[start:]


@unittest.skipUnless(shutil.which("node"), "Node.js required for synthetic DOM checks")
class VideoReviewToggleDomTests(unittest.TestCase):
    def run_dom(self, assertions):
        sources = [function_source(name) for name in (
            "el", "mediaURL", "appendPrompt", "failureDetails", "shotReview",
            "advisoryReport", "subshotPlan", "promptApproval", "shotCard", "roughCutCard",
            "paintVideoReviewControls", "saveVideoReview", "action")]
        sources.append('$("video-review-switch").addEventListener("click",'
                       '()=>saveVideoReview(state?.video_review?.enabled!==true));')
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {sources,assertions}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.attributes={};this.textContent='';}
 append(...nodes){this.children.push(...nodes);}
 setAttribute(key,value){this.attributes[key]=value;}
 getAttribute(key){return this.attributes[key];}
 addEventListener(key,handler){this.listeners[key]=handler;}
}
const nodes=new Map(),calls=[],messages=[],painted=[];
const $=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
let serverState={id:'synthetic-review-toggle',assets:[],episodes:[],video_review:{enabled:false},dashboard:{running:true}};
let release=null,hold=false,fail=false,refreshCount=0;
const context={assert,nodes,calls,messages,painted,$,TOKEN:'synthetic-token',requestBusy:false,state:serverState,
 document:{createElement:tag=>new Element(tag),createTextNode:text=>({tag:'text',textContent:text})},
 sessionStorage:{getItem:()=>null,setItem:()=>{}},
 toast:(text,error)=>messages.push({text,error}),updateInputControls:()=>{},
 showRevision:()=>{},perform:payload=>calls.push({forbiddenProductionOrApproval:payload}),
 fetch:async(url,options)=>{
  const payload=JSON.parse(options.body);calls.push({url,options,payload});
  if(hold)await new Promise(resolve=>release=resolve);
  if(fail)return {ok:false,json:async()=>({error:'合成保存失败'})};
  serverState={...serverState,video_review:{enabled:payload.enabled}};
  return {ok:true,json:async()=>serverState};
 },
 paint:data=>{context.state=data;painted.push(data);context.paintVideoReviewControls(data);},
 refresh:async()=>{refreshCount++;context.paint(serverState);},
 walk:node=>[node,...(node.children||[]).flatMap(context.walk)],
 getServer:()=>serverState,setServer:data=>serverState=data,
 setHold:value=>hold=value,setFail:value=>fail=value,release:()=>release(),getRefreshCount:()=>refreshCount
};
vm.runInNewContext(sources.join('\n')+'\n(async()=>{'+assertions+'\n})().catch(error=>{throw error;});',context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program],
                                input=json.dumps({"sources": sources, "assertions": assertions}),
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_switch_is_in_sticky_status_panel_and_accessible_with_closed_default(self):
        sidebar = HTML[HTML.index('<aside class="sidebar">'):HTML.index('<section class="panel cost-panel">')]
        self.assertIn('id="video-review-switch"', sidebar)
        self.assertIn('role="switch" aria-checked="false"', sidebar)
        self.assertIn('aria-labelledby="video-review-label"', sidebar)
        self.assertIn('aria-describedby="video-review-help"', sidebar)
        self.assertIn('已有报告保留，正在进行的审查可能完成', sidebar)
        self.assertIn('关闭后由你看片审批', sidebar)
        self.run_dom(r'''
for(const preference of [undefined,{}, {enabled:false},{enabled:'true'},{enabled:1}]){
 paintVideoReviewControls({...state,video_review:preference});
 assert.equal($('video-review-switch').getAttribute('aria-checked'),'false');
 assert.equal($('video-review-state').textContent,'已关闭');
}
paintVideoReviewControls({...state,video_review:{enabled:true}});
assert.equal($('video-review-switch').getAttribute('aria-checked'),'true');
assert.equal($('video-review-switch').disabled,false,'running production may still be switched');
assert.equal($('video-review-state').textContent,'已开启');assert.equal(calls.length,0);
''')

    def test_click_saves_boolean_only_without_production_or_approval(self):
        self.run_dom(r'''
assert.equal(state.dashboard.running,true);paintVideoReviewControls(state);
await $('video-review-switch').listeners.click();
assert.equal(calls.length,1);assert.equal(calls[0].url,'/api/action');
assert.equal(calls[0].options.method,'POST');
assert.equal(calls[0].options.headers['X-Norlan-Token'],'synthetic-token');
assert.equal(calls[0].payload.action,'set_video_review');
assert.equal(calls[0].payload.enabled,true);assert.equal(typeof calls[0].payload.enabled,'boolean');
assert.deepEqual(Object.keys(calls[0].payload).sort(),['action','enabled']);
assert.equal($('video-review-switch').getAttribute('aria-checked'),'true');
await $('video-review-switch').listeners.click();
assert.equal(calls.length,2);assert.equal(calls[1].payload.enabled,false);
assert.equal($('video-review-switch').getAttribute('aria-checked'),'false');
assert.equal(calls.some(call=>call.forbiddenProductionOrApproval),false);
assert.equal(state.dashboard.running,true);assert.equal(painted.length,2);
''')

    def test_busy_request_disables_switch_and_duplicate_click_does_not_submit(self):
        self.run_dom(r'''
setHold(true);const pending=saveVideoReview(true);
assert.equal($('video-review-switch').disabled,true);
assert.equal($('video-review-switch').getAttribute('aria-checked'),'false','no speculative preference change');
await saveVideoReview(true);await $('video-review-switch').listeners.click();
assert.equal(calls.length,1);release();await pending;
assert.equal($('video-review-switch').disabled,false);
assert.equal($('video-review-switch').getAttribute('aria-checked'),'true');
await saveVideoReview('false');assert.equal(calls.length,1,'helper never coerces strings into preferences');
''')

    def test_failed_save_restores_server_preference(self):
        self.run_dom(r'''
state={...state,video_review:{enabled:true}};setServer(state);paintVideoReviewControls(state);setFail(true);
await saveVideoReview(false);
assert.equal(calls.length,1);assert.equal(getRefreshCount(),1);
assert.equal($('video-review-switch').getAttribute('aria-checked'),'true');
assert.equal($('video-review-state').textContent,'已开启');assert.equal($('video-review-switch').disabled,false);
assert.ok(messages.some(message=>message.error&&message.text==='合成保存失败'));
''')

    def test_disabled_version_still_requires_human_approval_and_preserves_old_reports(self):
        self.run_dom(r'''
const shot={id:'synthetic-s1',version:2,sha256:'b'.repeat(64),status:'review_required',duration:10,file:'media/synthetic-v2.mp4',
 review_skipped:{reason:'disabled',version:2,sha256:'b'.repeat(64),at:'synthetic-time'}};
const card=shotCard(shot,0,null),all=walk(card);
assert.ok(all.some(node=>node.textContent==='GPT 视频审查已关闭，本版本由你审片。'));
assert.ok(all.some(node=>node.tag==='video'&&node.src==='/media/media/synthetic-v2.mp4'));
const approve=all.find(node=>node.tag==='button'&&node.textContent==='通过本段并继续');
assert.ok(approve);assert.equal(approve.disabled,true,'human full-play confirmation still required');
assert.equal(calls.length,0,'rendering a skipped review never approves');
for(const skipped of [{...shot.review_skipped,version:1},{...shot.review_skipped,sha256:'c'.repeat(64)}]){
 const stale=walk(shotCard({...shot,review_skipped:skipped},0,null));
 assert.equal(stale.some(node=>node.className==='video-review-skipped'),false);
}
shot.review={verdict:'fail',version:2,issues:['合成历史审查意见']};
const reported=walk(shotCard(shot,0,null));
assert.ok(reported.some(node=>node.className==='advisory-report'));
assert.ok(reported.some(node=>node.textContent==='合成历史审查意见'));
assert.equal(reported.some(node=>node.className==='video-review-skipped'),false);
''')

    def test_disabled_rough_cut_keeps_media_and_existing_report_takes_priority(self):
        self.run_dom(r'''
const cut={file:'exports/synthetic-cut.mp4',sha256:'c'.repeat(64),shot_count:3,pending_shot_ids:['synthetic-s3'],
 review_skipped:{reason:'disabled',sha256:'c'.repeat(64)}};
const all=walk(roughCutCard(cut,true));
assert.ok(all.some(node=>node.tag==='video'&&node.src==='/media/exports/synthetic-cut.mp4'));
assert.ok(all.some(node=>node.textContent==='GPT 视频审查已关闭，此拼接由你审片。'));
assert.ok(all.some(node=>node.textContent.includes('预览不会代替你的审批')));
const reported=walk(roughCutCard({...cut,review:{verdict:'uncertain',issues:['合成接续建议']}},true));
assert.ok(reported.some(node=>node.className==='advisory-report'));
assert.equal(reported.some(node=>node.className==='video-review-skipped'),false);
assert.equal(calls.length,0);
''')

    def test_setting_alone_is_excluded_from_media_rebuild_signature(self):
        paint = function_source("paint")
        signature = paint[paint.index("const nextSignature="):paint.index(";if(nextSignature!==signature)")]
        self.assertNotIn("video_review", signature)
        self.assertIn("paintVideoReviewControls(data)", paint)


if __name__ == "__main__":
    unittest.main()
