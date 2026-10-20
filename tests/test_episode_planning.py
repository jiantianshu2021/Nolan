"""Synthetic one-episode continuation tests; no models or paid providers."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import core, episode_planning, pipeline
from norlan.brain import Brain, BrainError, EPISODE_PLAN_SCHEMA


class EpisodePlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        core.create_project(self.root, "第一集原文，后续规划不应读取。")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        self.script = "第02集 原稿：主角拿起信封，走到车站门边，决定等他回来。"
        source = self.root / "source/episodes/ep02.md"
        source.parent.mkdir()
        source.write_text(self.script, encoding="utf-8")
        previous = self.root / "exports/ep01-synthetic.mp4"
        previous.write_bytes(b"synthetic fixture, not a real video")
        with core.transaction(self.root) as state:
            state["episodes"][0].update(status="approved", version=1, file="exports/ep01-synthetic.mp4",
                                        sha256=core.digest(previous), approved_sha256=core.digest(previous))
            state["episodes"][1].update(status="awaiting_episode_plan", shots=[], duration=12,
                                        source_file="source/episodes/ep02.md", source_sha256=core.digest(source))
            state["status"] = "ready"
        self.result = {
            "episode": {"id": "E002", "title": "第 2 集：流程演示", "summary": "门边等待故人。",
                        "music_prompt": "后期柔和钢琴，门边停顿留白。", "shots": [
                            {"id": "E002-NEW-01", "duration": 12, "prompt": "主角拿信封走向车站门边，最后停步。",
                             "dialogue": "", "asset_ids": ["hero", "station"], "scene_id": "station",
                             "continuity_notes": "右手持信封，保持出门方向。",
                             "subshots": [{"start": 0, "end": 6, "prompt": "中景拿信封向右走。"},
                                          {"start": 6, "end": 12, "prompt": "近景停在门边望向远处。"}]}]},
            "new_assets": []}

    def prepare(self, value=None):
        return episode_planning.prepare_episode(self.root, "E002", planner=Mock(return_value=value or self.result))

    def with_missing_prop(self, text="本段实际使用的prop_iou，须补入参考清单。"):
        result = copy.deepcopy(self.result)
        result["new_assets"] = [{"id": "prop_iou", "kind": "prop", "name": "借据", "prompt": "折叠白纸借据",
                                 "character_id": "", "audition_text": "", "audition_reason": "", "audition_source": "",
                                 "voice_only": False, "audition_speaker": ""}]
        result["episode"]["shots"][0]["continuity_notes"] = text
        return result

    def cached_failed(self, raw):
        pending = self.prepare()["episodes"][1]["planning"]
        with core.transaction(self.root) as state:
            target = state["episodes"][1]
            target["planning"] = {key: copy.deepcopy(value) for key, value in pending.items()
                if key not in {"episode", "new_assets", "result_sha256", "summary", "local_repairs"}}
            target["planning"].update(status="failed", error="合成测试：分镜结构校验失败。")
            target["status"] = "episode_plan_failed"
        core.reserve(self.root, "gpt-synthetic-original", "llm", "plan_episode", 0)
        core.update_job(self.root, "gpt-synthetic-original", state="completed", accounting="spent",
                        post_cache_key=pending["signature"], post_result=raw,
                        post_result_hash=core._json_digest(raw))
        return core.load_project(self.root)["episodes"][1]["planning"]

    def decide(self, decision="approve", feedback=""):
        state = core.load_project(self.root)
        pending = state["episodes"][1]["planning"]
        return episode_planning.decide_episode_plan(self.root, "E002", pending["revision"],
                                                   pending["signature"], decision, feedback)

    def test_preparing_and_candidate_are_durable_without_installation(self):
        before = core.load_project(self.root)

        def planner(script, context):
            pending = core.load_project(self.root)["episodes"][1]["planning"]
            self.assertEqual(pending["status"], "preparing")
            self.assertEqual(script, self.script)
            self.assertEqual(context["episode"]["expected_duration"], 12)
            return copy.deepcopy(self.result)

        state = episode_planning.prepare_episode(self.root, "E002", planner=planner)
        self.assertEqual(state["status"], "awaiting_episode_plan")
        self.assertEqual(state["episodes"][1]["planning"]["status"], "candidate")
        self.assertEqual(state["episodes"][1]["shots"], [])
        for key in ("assets", "budget", "jobs"):
            self.assertEqual(state[key], before[key])
        self.assertEqual(state["episodes"][0], before["episodes"][0])

    def test_current_candidate_is_reused_without_model_call(self):
        first = self.prepare()
        model = Mock(side_effect=AssertionError("must reuse"))
        second = episode_planning.prepare_episode(self.root, "E002", planner=model)
        model.assert_not_called()
        self.assertEqual(first["episodes"][1]["planning"], second["episodes"][1]["planning"])

    def test_default_planner_uses_budgeted_brain_cache_boundary(self):
        with patch("norlan.pipeline._brain", return_value=self.result) as call:
            state = episode_planning.prepare_episode(self.root, "E002")
        self.assertEqual(call.call_args.args[:3], (self.root.resolve(), "plan_episode", self.script))
        self.assertEqual(call.call_args.kwargs["_cache_key"], state["episodes"][1]["planning"]["signature"])

    def test_model_context_has_no_history_media_or_other_script(self):
        with core.transaction(self.root) as state:
            state["assets"][0].update(history=[{"nested": "huge history"}], evidence={"frame": "not needed"})
            state["episodes"][0].update(postproduction={"hidden": "huge render state"})
        model = Mock(return_value=self.result)
        episode_planning.prepare_episode(self.root, "E002", planner=model)
        script, context = model.call_args.args
        self.assertEqual(script, self.script)
        self.assertEqual(set(context), {"episode", "style", "ratio", "audio_policy", "previous_episode", "existing_assets"})
        self.assertEqual(set(context["existing_assets"][0]), {"id", "kind", "name", "character_id", "voice_only"})
        self.assertNotIn("huge", repr(context))
        self.assertNotIn("第一集原文", repr(context))
        self.assertNotIn("postproduction", context["previous_episode"])

    def test_human_approval_installs_only_target_and_standard_shot_state(self):
        before = core.load_project(self.root)
        self.prepare()
        state = self.decide()
        episode = state["episodes"][1]
        self.assertEqual(episode["status"], "planned")
        self.assertEqual(episode["planning"]["status"], "approved")
        self.assertEqual(episode["duration"], 12)
        self.assertEqual(episode["shots"][0]["status"], "planned")
        self.assertEqual(episode["shots"][0]["attempts"], [])
        self.assertEqual(episode["shots"][0]["revision"], 0)
        self.assertEqual(state["episodes"][0], before["episodes"][0])
        self.assertEqual(state["assets"], before["assets"])
        with self.assertRaises(core.GateError):
            self.decide()

    def test_new_assets_require_human_plan_then_original_asset_gates(self):
        value = copy.deepcopy(self.result)
        value["new_assets"] = [{"id": "envelope", "kind": "prop", "name": "蓝色信封", "prompt": "蓝纸长信封",
                                "character_id": "", "audition_text": "", "audition_reason": "", "audition_source": "",
                                "voice_only": False, "audition_speaker": ""}]
        value["episode"]["shots"][0]["asset_ids"].append("envelope")
        before = self.prepare(value)
        self.assertEqual(len(before["assets"]), 2)
        after = self.decide()
        asset = after["assets"][-1]
        self.assertEqual(asset["id"], "envelope")
        self.assertEqual(asset["status"], "planned")
        self.assertIsNone(asset["creation_input"]["mode"])
        self.assertIs(asset["creation_input"]["confirmed"], False)
        self.assertEqual(asset["source_episode_id"], "E002")
        self.assertEqual(asset["source_file"], "source/episodes/ep02.md")
        self.assertNotIn("file", asset)
        self.assertEqual(after["jobs"], before["jobs"])

    def test_explicit_feedback_prepares_new_revision_once(self):
        first = self.prepare()["episodes"][1]["planning"]
        self.decide("reject", "保持原稿时长，把停步改为回头。")
        model = Mock(return_value=self.result)
        second = episode_planning.prepare_episode(self.root, "E002", planner=model)["episodes"][1]["planning"]
        self.assertEqual(second["revision"], first["revision"] + 1)
        self.assertNotEqual(second["signature"], first["signature"])
        self.assertIn("回头", model.call_args.args[1]["feedback"])

    def test_validation_failure_does_not_repeat_until_feedback(self):
        value = copy.deepcopy(self.result)
        value["episode"]["shots"][0]["duration"] = 11
        model = Mock(return_value=value)
        with self.assertRaises(core.GateError):
            episode_planning.prepare_episode(self.root, "E002", planner=model)
        with self.assertRaises(core.GateError):
            episode_planning.prepare_episode(self.root, "E002", planner=model)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(core.load_project(self.root)["episodes"][1]["planning"]["status"], "failed")
        self.decide("reject", "请保留原稿12秒并完整覆盖镜头。")
        self.prepare()

    def test_unknown_and_interrupted_preparing_cannot_replay_or_reject(self):
        for status in ("uncertain", "preparing"):
            with self.subTest(status=status):
                with core.transaction(self.root) as state:
                    state["episodes"][1]["planning"] = {"status": status, "revision": 1, "signature": "bound"}
                model = Mock(return_value=self.result)
                with self.assertRaises(core.GateError):
                    episode_planning.prepare_episode(self.root, "E002", planner=model)
                with self.assertRaises(core.GateError):
                    episode_planning.decide_episode_plan(self.root, "E002", 1, "bound", "reject", "再试")
                model.assert_not_called()

    def test_unapproved_predecessor_never_calls_model(self):
        with core.transaction(self.root) as state:
            state["episodes"][0]["status"] = "awaiting_review"
        model = Mock(return_value=self.result)
        with self.assertRaises(core.GateError):
            episode_planning.prepare_episode(self.root, "E002", planner=model)
        model.assert_not_called()

    def test_changed_source_or_asset_definition_blocks_adoption(self):
        self.prepare()
        original = (self.root / "source/episodes/ep02.md").read_text(encoding="utf-8")
        (self.root / "source/episodes/ep02.md").write_text("changed", encoding="utf-8")
        with self.assertRaises(core.GateError):
            self.decide()
        (self.root / "source/episodes/ep02.md").write_text(original, encoding="utf-8")
        with core.transaction(self.root) as state:
            state["assets"][0]["prompt"] = "changed definition"
        with self.assertRaises(core.GateError):
            self.decide()
        self.assertEqual(core.load_project(self.root)["episodes"][1]["shots"], [])

    def test_existing_asset_media_version_and_approval_are_bound(self):
        self.prepare()
        with core.transaction(self.root) as state:
            state["assets"][0].update(version=2, status="approved", file="media/hero-v2.png",
                                      sha256="a" * 64, approved_sha256="a" * 64)
        with self.assertRaises(core.GateError):
            self.decide()
        self.assertEqual(core.load_project(self.root)["episodes"][1]["shots"], [])

    def test_candidate_result_tampering_and_stale_revision_are_rejected(self):
        self.prepare()
        pending = core.load_project(self.root)["episodes"][1]["planning"]
        for revision in (True, 0, pending["revision"] + 1):
            with self.subTest(revision=revision), self.assertRaises(core.GateError):
                episode_planning.decide_episode_plan(self.root, "E002", revision, pending["signature"], "approve")
        with core.transaction(self.root) as state:
            state["episodes"][1]["planning"]["episode"]["shots"][0]["prompt"] = "changed"
        with self.assertRaises(core.GateError):
            self.decide()
        with self.assertRaises(core.GateError):
            self.prepare()

    def test_ids_cannot_replace_existing_assets_episodes_or_shots(self):
        for identifier in ("hero", "E001", "E001-S001"):
            value = copy.deepcopy(self.result)
            value["episode"]["shots"][0]["id"] = identifier
            with self.subTest(identifier=identifier), self.assertRaises(core.GateError):
                episode_planning.validate_proposal(core.load_project(self.root),
                    core.load_project(self.root)["episodes"][1], value, self.script)

    def test_original_title_duration_and_scene_are_enforced(self):
        for mutation in ("title", "duration", "scene"):
            value = copy.deepcopy(self.result)
            if mutation == "title":
                value["episode"]["title"] = "renamed"
            elif mutation == "duration":
                value["episode"]["shots"][0]["duration"] = 10
            else:
                value["episode"]["shots"][0]["scene_id"] = "hero"
            with self.subTest(mutation=mutation), self.assertRaises(core.GateError):
                episode_planning.validate_proposal(core.load_project(self.root),
                    core.load_project(self.root)["episodes"][1], value, self.script)

    def test_brain_episode_contract_is_separate_from_full_plan(self):
        context = episode_planning.planning_context(core.load_project(self.root), core.load_project(self.root)["episodes"][1])
        context["full_project"] = {"history": "secret old context"}
        context["existing_assets"][0]["history"] = [{"nested": "old context"}]
        with patch.object(Brain, "_call", return_value=self.result) as call:
            value = Brain().plan_episode(self.script, context)
        instruction, data, schema = call.call_args.args
        self.assertIs(schema, EPISODE_PLAN_SCHEMA)
        self.assertEqual(value, self.result)
        self.assertEqual(data["script"], self.script)
        self.assertNotIn("full_project", data["context"])
        self.assertNotIn("history", data["context"]["existing_assets"][0])
        for rule in ("原稿时长", "禁止改编其他集", "不能改写这些定义", "10–15", "试音只取本集 script"):
            self.assertIn(rule, instruction)
        self.assertIn("style 用于沿用视觉风格", instruction)

    def test_generic_failure_saved_without_raw_secret_and_no_automatic_retry(self):
        model = Mock(side_effect=RuntimeError("synthetic-secret-token"))
        with self.assertRaises(RuntimeError):
            episode_planning.prepare_episode(self.root, "E002", planner=model)
        state = core.load_project(self.root)
        self.assertNotIn("synthetic-secret-token", repr(state))
        self.assertEqual(state["episodes"][1]["planning"]["status"], "failed")
        with self.assertRaises(core.GateError):
            episode_planning.prepare_episode(self.root, "E002", planner=model)
        self.assertEqual(model.call_count, 1)

    def test_completed_budgeted_cache_recovers_interrupted_candidate_without_gpt(self):
        pending = self.prepare()["episodes"][1]["planning"]
        with core.transaction(self.root) as state:
            state["episodes"][1]["planning"] = {key: copy.deepcopy(value) for key, value in pending.items()
                                                 if key not in {"episode", "new_assets", "result_sha256", "summary"}}
            state["episodes"][1]["planning"]["status"] = "preparing"
        core.reserve(self.root, "gpt-synthetic-cache", "llm", "plan_episode", 0)
        core.update_job(self.root, "gpt-synthetic-cache", state="completed", accounting="spent",
                        post_cache_key=pending["signature"], post_result=self.result,
                        post_result_hash=core._json_digest(self.result))
        model = Mock(side_effect=AssertionError("do not resubmit"))
        state = episode_planning.prepare_episode(self.root, "E002", planner=model)
        model.assert_not_called()
        self.assertEqual(state["episodes"][1]["planning"]["status"], "candidate")
        self.assertIs(state["episodes"][1]["planning"]["recovered"], True)

    def test_pipeline_prepares_source_episode_only_and_reuses_gate(self):
        with patch("norlan.pipeline._brain", return_value=self.result) as model, \
                patch("norlan.pipeline._assets", side_effect=AssertionError("must not generate assets")) as assets, \
                patch("norlan.pipeline._episode", side_effect=AssertionError("must not generate clips")) as episode:
            first = pipeline.run(self.root)
            with core.transaction(self.root) as current:
                current.update(status="blocked", last_error="an old display state")
            second = pipeline.run(self.root)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(model.call_args.args[1], "plan_episode")
        assets.assert_not_called()
        episode.assert_not_called()
        self.assertEqual(first["status"], "awaiting_episode_plan")
        self.assertEqual(second["status"], "awaiting_episode_plan")
        self.assertIsNone(second["last_error"])
        self.assertEqual(first["episodes"][0], second["episodes"][0])
        self.assertEqual(second["episodes"][1]["shots"], [])

    def test_pipeline_failed_or_uncertain_planning_never_retries(self):
        for uncertain in (False, True):
            with self.subTest(uncertain=uncertain):
                with core.transaction(self.root) as state:
                    state["episodes"][1].pop("planning", None)
                    state["episodes"][1]["status"] = "awaiting_episode_plan"
                failure = BrainError("合成测试：规划连接未完成。", uncertain=uncertain)
                with patch("norlan.pipeline._brain", side_effect=failure) as model, \
                        patch("norlan.pipeline._assets") as assets, patch("norlan.pipeline._episode") as episode:
                    first = pipeline.run(self.root)
                    second = pipeline.run(self.root)
                self.assertEqual(model.call_count, 1)
                assets.assert_not_called()
                episode.assert_not_called()
                self.assertEqual(second["status"], "blocked")
                self.assertEqual(first["episodes"][1]["planning"]["status"], "uncertain" if uncertain else "failed")

    def test_normalization_fills_only_exact_new_id_in_same_shot(self):
        raw = self.with_missing_prop()
        original = copy.deepcopy(raw)
        normalized, repairs = episode_planning.normalize_proposal(raw)
        self.assertEqual(raw, original)
        self.assertEqual(normalized["episode"]["shots"][0]["asset_ids"], ["hero", "station", "prop_iou"])
        self.assertEqual(repairs[0]["shot_id"], "E002-NEW-01")
        self.assertEqual(repairs[0]["asset_ids"], ["prop_iou"])
        self.assertEqual(episode_planning.normalize_proposal(normalized)[1], [])

    def test_normalization_rejects_longer_id_substrings_dialogue_and_negative_mentions(self):
        for text in ("prop_iou_other", "prop_iou-v2", "Xprop_iou", "other_prop_iou", "prop_iou2",
                     "不要引用prop_iou。", "不要使用prop_iou。", "禁止prop_iou出镜。", "不引用prop_iou。",
                     "本段不使用prop_iou。", "无需prop_iou。"):
            with self.subTest(text=text):
                normalized, repairs = episode_planning.normalize_proposal(self.with_missing_prop(text))
                self.assertEqual(repairs, [])
                self.assertNotIn("prop_iou", normalized["episode"]["shots"][0]["asset_ids"])
        raw = self.with_missing_prop("右手持信封。")
        raw["episode"]["shots"][0]["dialogue"] = "prop_iou"
        self.assertEqual(episode_planning.normalize_proposal(raw)[1], [])

    def test_prompt_and_subshot_reference_normalization_are_supported(self):
        for field in ("prompt", "subshot"):
            raw = self.with_missing_prop("右手持信封。")
            if field == "prompt":
                raw["episode"]["shots"][0]["prompt"] += "，展示prop_iou。"
            else:
                raw["episode"]["shots"][0]["subshots"][0]["prompt"] += "，展示prop_iou。"
            with self.subTest(field=field):
                self.assertEqual(episode_planning.normalize_proposal(raw)[1][0]["asset_ids"], ["prop_iou"])

    def test_fresh_prepare_normalizes_before_validation_without_mutating_result(self):
        raw = self.with_missing_prop()
        original = copy.deepcopy(raw)
        state = self.prepare(raw)
        pending = state["episodes"][1]["planning"]
        self.assertEqual(raw, original)
        self.assertEqual(pending["status"], "candidate")
        self.assertEqual(pending["local_repairs"][0]["asset_ids"], ["prop_iou"])

    def test_failed_completed_cache_repairs_exact_reference_without_gpt(self):
        raw = self.with_missing_prop()
        self.cached_failed(raw)
        original = copy.deepcopy(core.load_project(self.root)["jobs"])
        model = Mock(side_effect=AssertionError("must use original cache"))
        state = episode_planning.prepare_episode(self.root, "E002", planner=model)
        model.assert_not_called()
        self.assertEqual(state["jobs"], original)
        pending = state["episodes"][1]["planning"]
        self.assertEqual(pending["status"], "candidate")
        self.assertNotIn("error", pending)
        self.assertIs(pending["recovered"], True)
        self.assertEqual(pending["local_repairs"][0]["asset_ids"], ["prop_iou"])
        self.assertNotEqual(pending["result_sha256"], state["jobs"]["gpt-synthetic-original"]["post_result_hash"])

    def test_failed_cache_without_actual_repair_or_with_remaining_errors_stays_failed(self):
        for case in ("no_repair", "still_invalid"):
            raw = copy.deepcopy(self.result) if case == "no_repair" else self.with_missing_prop()
            if case == "still_invalid":
                raw["episode"]["shots"][0]["duration"] = 11
            with self.subTest(case=case):
                # Reset the synthetic scenario without a model request.
                with core.transaction(self.root) as state:
                    state["episodes"][1].pop("planning", None)
                    state["episodes"][1]["status"] = "awaiting_episode_plan"
                    state["jobs"] = {}
                self.cached_failed(raw)
                model = Mock(side_effect=AssertionError("must stay failed"))
                with self.assertRaises(core.GateError):
                    episode_planning.prepare_episode(self.root, "E002", planner=model)
                model.assert_not_called()
                self.assertEqual(core.load_project(self.root)["episodes"][1]["planning"]["status"], "failed")

    def test_visual_reference_count_reserves_official_tail_slot(self):
        raw = copy.deepcopy(self.result)
        template = self.with_missing_prop()["new_assets"][0]
        for number in range(7):
            asset = {**template, "id": f"prop_extra_{number}", "name": f"新道具{number}"}
            raw["new_assets"].append(asset)
            raw["episode"]["shots"][0]["asset_ids"].append(asset["id"])
        state = core.load_project(self.root)
        with self.assertRaisesRegex(core.GateError, "最多8项视觉资产"):
            episode_planning.validate_proposal(state, state["episodes"][1], raw, self.script)

    def test_voice_reference_count_is_limited_to_three(self):
        raw = copy.deepcopy(self.result)
        for number in range(4):
            asset = {"id": f"voice_synthetic_{number}", "kind": "voice", "name": f"合成声音{number}",
                     "prompt": "中性中音男声", "character_id": "hero", "voice_only": False,
                     "audition_speaker": "", "audition_text": "", "audition_source": "",
                     "audition_reason": "合成测试原文无合适长句，待人工选择。"}
            raw["new_assets"].append(asset)
            raw["episode"]["shots"][0]["asset_ids"].append(asset["id"])
        state = core.load_project(self.root)
        with self.assertRaisesRegex(core.GateError, "声音为4项"):
            episode_planning.validate_proposal(state, state["episodes"][1], raw, self.script)

    def test_technical_repair_is_budgeted_once_and_still_waits_human_approval(self):
        prior = self.cached_failed(self.with_missing_prop())

        def model(*args, **kwargs):
            pending = core.load_project(self.root)["episodes"][1]["planning"]
            self.assertEqual(pending["status"], "preparing")
            self.assertEqual(pending["technical_repair_count"], 1)
            self.assertEqual(pending["revision"], prior["revision"] + 1)
            self.assertNotEqual(kwargs["_cache_key"], prior["signature"])
            self.assertIn("proposal_to_repair", args[3])
            self.assertIn("最多8项视觉资产", args[3]["technical_feedback"])
            return self.result

        before = core.load_project(self.root)
        with patch("norlan.pipeline._brain", side_effect=model) as call:
            state = episode_planning.repair_cached_structure(self.root, "E002")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(state["episodes"][1]["planning"]["status"], "candidate")
        self.assertEqual(state["episodes"][1]["shots"], [])
        self.assertEqual(state["assets"], before["assets"])
        self.assertEqual(state["episodes"][0], before["episodes"][0])
        with self.assertRaises(core.GateError):
            episode_planning.repair_cached_structure(self.root, "E002", planner=Mock())

    def test_technical_input_keeps_raw_schema_valid_when_normalization_exceeds_cap(self):
        raw = self.with_missing_prop()
        template = raw["new_assets"][0]
        for index in range(10):
            asset = {**template, "id": f"prop_background_{index}", "name": f"合成背景道具{index}"}
            raw["new_assets"].append(asset)
            raw["episode"]["shots"][0]["asset_ids"].append(asset["id"])
        self.assertEqual(len(raw["episode"]["shots"][0]["asset_ids"]), 12)
        self.assertEqual(len(episode_planning.normalize_proposal(raw)[0]["episode"]["shots"][0]["asset_ids"]), 13)
        self.cached_failed(raw)

        def planner(script, context):
            self.assertEqual(context["proposal_to_repair"], raw)
            with patch.object(Brain, "_call", return_value=self.result):
                return Brain().plan_episode(script, context)

        state = episode_planning.repair_cached_structure(self.root, "E002", planner=planner)
        self.assertEqual(state["episodes"][1]["planning"]["status"], "candidate")
        self.assertEqual(state["episodes"][1]["planning"]["technical_repair_count"], 1)

    def test_failed_technical_repair_cannot_run_again_or_auto_resume(self):
        self.cached_failed(self.with_missing_prop())
        model = Mock(side_effect=BrainError("合成测试结构校正失败。"))
        with self.assertRaises(BrainError):
            episode_planning.repair_cached_structure(self.root, "E002", planner=model)
        for function in (episode_planning.repair_cached_structure, episode_planning.prepare_episode):
            with self.assertRaises(core.GateError):
                function(self.root, "E002", planner=model)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(core.load_project(self.root)["episodes"][1]["planning"]["technical_repair_count"], 1)

    def test_technical_repair_unknown_outcome_is_never_retried(self):
        self.cached_failed(self.with_missing_prop())
        model = Mock(side_effect=BrainError("合成测试结构校正结果不明。", uncertain=True))
        with self.assertRaises(BrainError):
            episode_planning.repair_cached_structure(self.root, "E002", planner=model)
        with self.assertRaises(core.GateError):
            episode_planning.repair_cached_structure(self.root, "E002", planner=model)
        with self.assertRaises(core.GateError):
            episode_planning.prepare_episode(self.root, "E002", planner=model)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(core.load_project(self.root)["episodes"][1]["planning"]["status"], "uncertain")

    def test_technical_completed_cache_recovers_interrupted_attempt_without_gpt(self):
        self.cached_failed(self.with_missing_prop())
        state = episode_planning.repair_cached_structure(self.root, "E002", planner=Mock(return_value=self.result))
        pending = state["episodes"][1]["planning"]
        with core.transaction(self.root) as state:
            state["episodes"][1]["planning"]["status"] = "preparing"
        core.reserve(self.root, "gpt-synthetic-technical", "llm", "plan_episode", 0)
        core.update_job(self.root, "gpt-synthetic-technical", state="completed", accounting="spent",
                        post_cache_key=pending["signature"], post_result=self.result,
                        post_result_hash=core._json_digest(self.result))
        model = Mock(side_effect=AssertionError("do not resubmit technical call"))
        result = episode_planning.prepare_episode(self.root, "E002", planner=model)
        model.assert_not_called()
        self.assertEqual(result["episodes"][1]["planning"]["status"], "candidate")
        self.assertIs(result["episodes"][1]["planning"]["recovered"], True)

    def test_brain_projects_only_explicit_technical_payload_when_requested(self):
        context = episode_planning.planning_context(core.load_project(self.root), core.load_project(self.root)["episodes"][1])
        context.update(proposal_to_repair=self.result, technical_feedback="仅校正参考数量，不改变剧情。", private_history="unused")
        with patch.object(Brain, "_call", return_value=self.result) as call:
            Brain().plan_episode(self.script, context)
        data = call.call_args.args[1]
        self.assertEqual(data["context"]["proposal_to_repair"], self.result)
        self.assertEqual(data["context"]["technical_feedback"], context["technical_feedback"])
        self.assertNotIn("private_history", data["context"])


if __name__ == "__main__":
    unittest.main()
