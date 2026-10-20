"""Synthetic episode navigation contracts; never contact providers or live projects."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from norlan import dashboard


@unittest.skipUnless(shutil.which("node"), "Node.js required for synthetic navigation DOM")
class EpisodeNavigationTests(unittest.TestCase):
    def run_dom(self, assertions):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "episodeRoute", "navigateEpisodeRoute", "navigateToProductionEpisode",
                     "nextHumanNavigationGate", "focusNextHumanGate", "episodePlanningPanel", "episodeCard", "applyEpisodeRoute", "episodeOverviewCard", "paintEpisodeNavigation", "perform"):
            match = re.search(r"(?:async )?function " + name + r"\(", html)
            following = re.search(r"\n(?:async )?function ", html[match.start() + 1:])
            end = match.start() + 1 + following.start() if following else len(html)
            sources.append(html[match.start():end])
        start = html.index('$("run").addEventListener("click",')
        end = html.index(';$("pause").addEventListener', start)
        sources.append(html[start:end] + ";")
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {sources,assertions}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.textContent='';this.value='';this.attributes={};this.paused=true;this.currentTime=0;this.hidden=false;this.scrolls=[];this.focuses=[];
  this.classList={toggle:(name,on)=>{const set=new Set(String(this.className||'').split(' ').filter(Boolean));if(on===undefined?!set.has(name):on)set.add(name);else set.delete(name);this.className=[...set].join(' ');},contains:name=>String(this.className||'').split(' ').includes(name)};}
 append(...nodes){this.children.push(...nodes);for(const node of nodes)node.parent=this;}
 replaceChildren(...nodes){this.children=[];this.append(...nodes);}
 setAttribute(name,value){this.attributes[name]=value;this[name]=value;}
 getAttribute(name){return this.attributes[name]??this[name]??null;}
 removeAttribute(name){delete this.attributes[name];delete this[name];}
 addEventListener(name,handler){this.listeners[name]=handler;}
 querySelectorAll(selector){const matches=node=>selector==='video,audio'?['video','audio'].includes(node.tag):node.tag===selector;return this.children.flatMap(node=>[...(matches(node)?[node]:[]),...(node.querySelectorAll?.(selector)||[])]);}
 querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
 scrollIntoView(options){this.scrolls.push(options);}
 focus(options){this.focuses.push(options);}
 pause(){this.paused=true;this.pauseCount=(this.pauseCount||0)+1;}
}
const nodes=new Map(),calls=[],messages=[],scrollCalls=[];
const $=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
const first={id:'ep01',title:'合成第一集',summary:'合成概述一',status:'approved',version:5,file:'exports/ep01-v5.mp4',duration:130.6,duration_source:'measured',progress:{percent:100,completed_shots:2,total_shots:2},shots:[{id:'s1',status:'accepted',duration:10},{id:'s2',status:'accepted',duration:15}],postproduction:{final_available:true}};
const second={id:'ep02',title:'合成第二集',summary:'合成概述二',status:'planned',duration:60,duration_source:'planned',navigation:{index:2,can_produce:true,locked_reason:''},progress:{percent:0,completed_shots:0,total_shots:2,label:'等待提示词准备'},shots:[{id:'s3',duration:30,status:'planned'},{id:'s4',duration:30,status:'planned'}]};
const third={id:'ep03/特殊 名称',title:'合成第三集',status:'planned',navigation:{index:3,can_produce:false,locked_reason:'等待第 2 集最终批准'},progress:{percent:0},shots:[{id:'s5',duration:50,status:'planned'}]};
let state={title:'合成项目',plan:{status:'approved'},assets:[],asset_amendments:[],episodes:[first,second,third],paused:false,dashboard:{running:false},episode_navigation:{active_episode_id:'ep02',all_planned_complete:false}};
const window={location:{hash:''},scrollY:0,history:{pushState(_a,_b,key){window.location.hash=key;}},scrollTo(options){window.scrollY=options.top;scrollCalls.push(options);}};
for(const episode of state.episodes){const card=new Element('article');card.dataset.episodeId=episode.id;card.append(new Element('video'),new Element('textarea'));$('episodes').append(card);}
const context={assert,window,nodes,calls,messages,scrollCalls,state,$,first,second,third,
 document:{createElement:tag=>new Element(tag)},reviewDetails:()=>null,appendEpisodeProduction:()=>{},postproductionPanel:()=>null,badge:status=>Object.assign(new Element('span'),{textContent:status}),
 episodeOverviewSignature:'',episodeRouteKey:'',episodeRoutePositions:new Map(),requestBusy:false,
 toast:(...args)=>messages.push(args),updateInputControls:()=>{},refresh:async()=>{},
 action:async payload=>{calls.push(payload);if(payload.action==='approve')state.episodes.find(item=>item.id===payload.id).status='approved';},
 perform:payload=>calls.push(payload)};
vm.runInNewContext(sources.join('\n')+'\n(async()=>{'+assertions+'\n})().catch(error=>{throw error;});',context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program],
                                input=json.dumps({"sources": sources, "assertions": assertions}),
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_overview_is_default_with_per_episode_progress_and_real_duration_labels(self):
        self.run_dom(r'''
paintEpisodeNavigation(state);
assert.equal($('episode-overview-panel').classList.contains('hidden'),false);
assert.equal($('episode-detail-panel').classList.contains('hidden'),true);
assert.equal($('assets-panel').classList.contains('hidden'),true);
assert.equal($('episode-overview').children.length,3);
assert.equal($('episodes').children.every(card=>card.hidden),true);
const firstCard=$('episode-overview').children[0],secondCard=$('episode-overview').children[1];
assert.ok(firstCard.children.some(node=>node.textContent==='实测 130.6 秒 · 第 5 版'));
assert.ok(secondCard.children.some(node=>node.textContent==='计划 60.0 秒'));
assert.ok(firstCard.children.some(node=>node.textContent==='100% · 已通过 2 / 2 个片段'));
assert.equal(calls.length,0);
''')

    def test_card_link_enters_exact_episode_without_running_and_supports_encoded_ids(self):
        self.run_dom(r'''
paintEpisodeNavigation(state);
const link=$('episode-overview').children[2].querySelector('a');link.listeners.click({preventDefault(){}});
assert.equal(window.location.hash,'#episode/'+encodeURIComponent(third.id));
assert.equal($('episode-detail-panel').classList.contains('hidden'),false);
assert.equal($('episodes').children[2].hidden,false);
assert.equal($('episodes').children[0].hidden,true);
assert.equal($('episode-route-notice').textContent,'等待第 2 集最终批准');
assert.equal(calls.length,0,'viewing a future episode does not produce or approve it');
''')

    def test_polling_preserves_selected_old_episode_drafts_playback_and_scroll(self):
        self.run_dom(r'''
paintEpisodeNavigation(state);navigateEpisodeRoute('#episode/ep01');
const card=$('episodes').children[0],video=card.children[0],draft=card.children[1];
video.currentTime=32.5;video.paused=false;draft.value='尚未提交的意见';window.scrollY=670;
second.progress.percent=25;second.status='generating';state.dashboard.running=true;paintEpisodeNavigation(state);
assert.equal(window.location.hash,'#episode/ep01');assert.equal(card.hidden,false);
assert.equal(window.scrollY,670);assert.equal(video.currentTime,32.5);assert.equal(video.paused,false);
navigateEpisodeRoute('#episodes');assert.equal(video.paused,true);
navigateEpisodeRoute('#episode/ep01');assert.equal(window.scrollY,670);
assert.equal(card.children[1],draft);assert.equal(draft.value,'尚未提交的意见');assert.equal(video.currentTime,32.5);
assert.equal(calls.length,0);
''')

    def test_explicit_continue_opens_active_episode_and_runs_existing_safe_action(self):
        self.run_dom(r'''
paintEpisodeNavigation(state);navigateEpisodeRoute('#episode/ep01');
$('run').listeners.click();await Promise.resolve();await Promise.resolve();
assert.equal(window.location.hash,'#episode/ep02');
assert.equal(calls.length,1);assert.equal(calls[0].action,'run');
assert.equal(calls.some(item=>item.action==='approve'),false);
''')

    def test_approval_routes_to_next_episode_and_does_not_advance_to_unplanned_episode(self):
        self.run_dom(r'''
first.status='awaiting_review';state.episodes=[first,second];state.episode_navigation.active_episode_id='ep01';
await perform({action:'approve',kind:'episode',id:'ep01',version:5},true);
assert.equal(window.location.hash,'#episode/ep02');assert.equal(calls.length,2);assert.equal(calls[1].action,'run');
calls.length=0;state.episodes=[first];await perform({action:'approve',kind:'episode',id:'ep01',version:5},true);
assert.equal(window.location.hash,'#episodes');assert.equal(calls.length,1,'last planned approval must not run an empty pipeline');
assert.equal(state.episodes.length,1,'navigation never fabricates a sequel');
''')

    def test_all_completed_continue_returns_to_overview_without_empty_run(self):
        self.run_dom(r'''
state.episodes=[first];paintEpisodeNavigation(state);navigateEpisodeRoute('#episode/ep01');
assert.equal($('run').textContent,'查看分集总览');
assert.ok($('episode-completion-note').textContent.includes('尚未规划第 2 集'));
$('run').listeners.click();await Promise.resolve();assert.equal(window.location.hash,'#episodes');assert.equal(calls.length,0);
''')

    def test_pending_prompt_continue_is_navigation_only_and_keeps_approval_separate(self):
        self.run_dom(r'''
second.shots[0].status='awaiting_prompt_approval';paintEpisodeNavigation(state);
assert.equal($('run').textContent,'前往第 2 集确认提示词');assert.equal($('run').disabled,false);
$('run').listeners.click();await Promise.resolve();
assert.equal(window.location.hash,'#episode/ep02');assert.equal(calls.length,0);
assert.equal($('prompt-approval-s3').scrolls.length,1);
second.shots[0].status='review_required';second.shots[0].file='media/s3-v1.mp4';paintEpisodeNavigation(state);
assert.equal($('run').textContent,'前往第 2 集审批片段');$('run').listeners.click();
assert.equal(calls.length,0);assert.equal($('shot-card-s3').scrolls.length,1);
''')

    def test_completed_imported_season_does_not_claim_another_episode_is_missing(self):
        self.run_dom(r'''
state.episodes=[first];state.episode_navigation={season_complete:true,reason:'本季 30 集已全部完成并通过审批。',next_episode_number:null};paintEpisodeNavigation(state);
assert.equal($('status').textContent,'本季已全部完成');assert.equal($('episode-completion-note').textContent,'本季 30 集已全部完成并通过审批。');assert.equal($('status-note').textContent,'本季 30 集已全部完成并通过审批。');
assert.equal($('episode-completion-note').textContent.includes('尚未规划'),false);$('run').listeners.click();await Promise.resolve();assert.equal(calls.length,0);
''')

    def test_shared_asset_view_retains_pending_approvals_and_does_not_expand_episode_pages(self):
        self.run_dom(r'''
state.assets=[{id:'newVoice',kind:'voice',status:'candidate',input_editable:true,input_required:true}];
state.asset_amendments=[{status:'candidate'}];paintEpisodeNavigation(state);navigateEpisodeRoute('#assets');
assert.equal($('assets-panel').classList.contains('hidden'),false);
assert.equal($('plan-panel').classList.contains('hidden'),false);
assert.equal($('intake-panel').classList.contains('hidden'),false);
assert.equal($('amendment-panel').classList.contains('hidden'),false);
assert.equal($('episode-detail-panel').classList.contains('hidden'),true);
assert.equal($('episode-overview-panel').classList.contains('hidden'),true);
assert.equal($('assets-navigation').getAttribute('aria-current'),'page');assert.equal(calls.length,0);
''')

    def test_source_slot_without_candidate_shows_start_and_original_duration(self):
        self.run_dom(r'''
second.status='awaiting_episode_plan';second.shots=[];second.source_duration=130;second.duration=130;delete second.planning;
const card=episodeCard(second,1),walk=node=>[node,...node.children.flatMap(walk)];
assert.ok(walk(card).some(node=>node.textContent==='沿用本集原稿时长：130 秒。'));
const start=walk(card).find(node=>node.tag==='button'&&node.textContent==='规划本集分镜');assert.ok(start);assert.equal(start.disabled,false);assert.equal(calls.length,0);
paintEpisodeNavigation(state);assert.ok($('episode-overview').children[1].children.some(node=>node.textContent==='计划 130.0 秒'));
start.listeners.click();await Promise.resolve();assert.equal(calls[0].action,'run');assert.equal(calls.some(item=>item.action==='episode_plan_decide'),false);
''')

    def test_episode_candidate_shows_all_shots_and_new_assets_then_requires_bound_human_decision(self):
        self.run_dom(r'''
second.status='awaiting_episode_plan';second.shots=[];second.source_duration=130;
second.planning={status:'candidate',revision:2,signature:'a'.repeat(64),summary:'合成分镜方案说明',episode:{id:second.id,title:second.title,summary:'合成本集梗概',music_prompt:'合成低声部铺垫',shots:[{id:'ep02-s1',duration:10,prompt:'合成两个镜头总述',dialogue:'合成对白',asset_ids:['newRoom'],scene_id:'newRoom',subshots:[{start:0,end:4,prompt:'先看门口'},{start:4,end:10,prompt:'再看人物反应'}]}]},new_assets:[{id:'newRoom',kind:'location',name:'合成新车间',prompt:'合成场景描述'},{id:'newVoice',kind:'voice',name:'合成系统女声',voice_only:true,prompt:'合成音色说明'}]};
const panel=episodePlanningPanel(second,1),walk=node=>[node,...node.children.flatMap(walk)],all=walk(panel);
for(const text of ['合成分镜方案说明','片段 1 · 10 秒 · 2 个镜头','镜头 1 · 0–4 秒','镜头 2 · 4–10 秒','先看门口','再看人物反应','新增资产 · 2 项','合成系统女声 · 声音 · 仅画外声源'])assert.ok(all.some(node=>node.textContent===text),text);
assert.equal(calls.length,0,'rendering a proposal cannot approve it');
const approve=all.find(node=>node.tag==='button'&&node.textContent==='批准本集分镜与资产方案');assert.equal(approve.disabled,false);await approve.listeners.click();
assert.equal(calls[0].action,'episode_plan_decide');assert.equal(calls[0].decision,'approve');assert.equal(calls[0].id,'ep02');assert.equal(calls[0].revision,2);assert.equal(calls[0].signature,'a'.repeat(64));assert.equal(calls[1].action,'run');
''')

    def test_candidate_feedback_uses_same_bound_decision_and_navigation_never_approves(self):
        self.run_dom(r'''
second.status='awaiting_episode_plan';second.shots=[];second.planning={status:'candidate',revision:3,signature:'b'.repeat(64),episode:{shots:[]},new_assets:[]};
paintEpisodeNavigation(state);assert.equal($('run').textContent,'前往第 2 集确认分镜方案');$('run').listeners.click();assert.equal(calls.length,0);assert.equal(window.location.hash,'#episode/ep02');
const panel=episodePlanningPanel(second,1),walk=node=>[node,...node.children.flatMap(walk)],all=walk(panel),feedback=all.find(node=>node.tag==='textarea'),reject=all.find(node=>node.tag==='button'&&node.textContent==='提交意见，重新规划本集');
reject.listeners.click();assert.equal(calls.length,0,'empty feedback does not authorize replanning');feedback.value='第二段改成两个镜头，保留原稿台词。';reject.listeners.click();await Promise.resolve();
assert.equal(calls[0].action,'episode_plan_decide');assert.equal(calls[0].decision,'reject');assert.equal(calls[0].feedback,feedback.value);assert.equal(calls[0].revision,3);assert.equal(calls[0].signature,'b'.repeat(64));
''')

    def test_failed_planning_needs_feedback_but_uncertain_and_preparing_cannot_retry(self):
        self.run_dom(r'''
const walk=node=>[node,...node.children.flatMap(walk)];second.status='episode_plan_failed';second.shots=[];
second.planning={status:'failed',revision:1,signature:'c'.repeat(64),error:'合成规划失败'};
const failed=walk(episodePlanningPanel(second,1));assert.equal(failed.some(node=>node.tag==='button'&&node.textContent==='规划本集分镜'),false);
const feedback=failed.find(node=>node.tag==='textarea'),retry=failed.find(node=>node.tag==='button');feedback.value='请保留原稿时长并调整镜头分配';retry.listeners.click();await Promise.resolve();await Promise.resolve();assert.equal(calls[0].decision,'reject');assert.equal(calls[0].signature,'c'.repeat(64));calls.length=0;
for(const status of ['uncertain','preparing']){second.planning.status=status;const panel=walk(episodePlanningPanel(second,1));assert.equal(panel.some(node=>node.tag==='button'),false);assert.equal(panel.some(node=>node.tag==='textarea'),false);paintEpisodeNavigation(state);assert.equal($('run').disabled,true);}
assert.equal(calls.length,0);
''')

    def test_completed_planning_cache_can_be_restored_without_approving_or_replanning(self):
        self.run_dom(r'''
const walk=node=>[node,...node.children.flatMap(walk)];second.status='awaiting_episode_plan';second.shots=[];
second.planning={status:'preparing',revision:1,signature:'e'.repeat(64),recoverable:true};
const panel=walk(episodePlanningPanel(second,1)),resume=panel.find(node=>node.tag==='button');assert.equal(resume.textContent,'恢复已保存的分镜方案');assert.equal(resume.disabled,false);assert.equal(panel.some(node=>node.tag==='textarea'),false);
assert.equal(calls.length,0);await resume.listeners.click();assert.equal(calls.length,1);assert.equal(calls[0].action,'run');assert.equal(calls.some(item=>item.action==='episode_plan_decide'),false);
calls.length=0;paintEpisodeNavigation(state);assert.equal($('run').disabled,false);assert.equal($('run').textContent,'恢复已保存的分镜方案');$('run').listeners.click();await Promise.resolve();await Promise.resolve();assert.equal(calls.length,1);assert.equal(calls[0].action,'run');
''')

    def test_uncertain_planning_never_uses_recoverable_flag_to_enable_retry(self):
        self.run_dom(r'''
const walk=node=>[node,...node.children.flatMap(walk)];second.status='episode_plan_failed';second.shots=[];second.planning={status:'uncertain',revision:1,signature:'f'.repeat(64),recoverable:true};
assert.equal(walk(episodePlanningPanel(second,1)).some(node=>node.tag==='button'),false);paintEpisodeNavigation(state);assert.equal($('run').disabled,true);assert.equal(calls.length,0);
''')

    def test_later_episode_candidate_is_visible_but_cannot_be_adopted_before_previous_approval(self):
        self.run_dom(r'''
third.status='awaiting_episode_plan';third.planning={status:'candidate',revision:1,signature:'d'.repeat(64),episode:{shots:[]},new_assets:[]};
const panel=episodePlanningPanel(third,2),walk=node=>[node,...node.children.flatMap(walk)],all=walk(panel);
assert.ok(all.some(node=>node.textContent==='等待第 2 集最终批准'));assert.equal(all.some(node=>node.tag==='button'),false);assert.equal(all.some(node=>node.tag==='textarea'),false);assert.equal(calls.length,0);
''')

    def test_invalid_or_stale_deep_link_falls_back_to_overview_without_work(self):
        self.run_dom(r'''
for(const hash of ['#episode/%broken','#episode/not-present','#anything']){
 window.location.hash=hash;paintEpisodeNavigation(state);
 assert.equal($('episode-overview-panel').classList.contains('hidden'),false);
 assert.equal($('episode-detail-panel').classList.contains('hidden'),true);
}
assert.equal(calls.length,0);
''')


if __name__ == '__main__':
    unittest.main()
