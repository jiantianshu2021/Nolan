"""Postproduction GPT contracts with synthetic transports; no account requests."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import core, pipeline
from norlan.brain import Brain, BrainError, CAPTION_SCHEMA, MUSIC_CUE_SCHEMA


def captions():
    return [{"start": 0.25, "end": 1.5, "text": "楚洋来到车间"},
            {"start": 2.0, "end": 3.75, "text": "实际说的是2999元"}]


def music_plan():
    return {"summary": "已批准主题，重要对白附近留白，最终本人听审。", "cues": [{
        "track_id": "approved-theme", "start": 1.0, "end": 5.0, "source_start": 0.0,
        "gain_db": -16.0, "fade_in": 0.5, "fade_out": 0.5,
    }]}


class PostproductionBrainTests(unittest.TestCase):
    def test_timing_feedback_is_passed_to_caption_route_without_model_timing_edits(self):
        feedback = "第一集前五条字幕出现太早，00:03 才开始说话，请重新对齐，不要重做视频。"
        command = {"action": "captions", "kind": "episode", "id": "E001", "feedback": feedback,
                   "shot_ids": [], "limit": None, "reply": ""}
        with patch.object(Brain, "_codex", return_value=command) as model:
            self.assertEqual(Brain().command(feedback, {"items": [{"kind": "episode", "id": "E001"}]}), command)
        instruction = model.call_args.args[0]
        self.assertIn("字幕提前/滞后/出现时间对不上声音", instruction)
        self.assertIn("本地实际音频重新识别与对齐", instruction)
        payload = json.loads(instruction.split("下方 JSON 仅为创作数据：\n", 1)[1])
        self.assertEqual(payload["message"], feedback)

    def test_model_must_not_guess_times_from_a_realignment_request(self):
        feedback = "前五条出现时间太早，第一条请顺延 3 秒并重新对齐。"
        source = captions()
        before = copy.deepcopy(source)
        for method, args in (("proofread_subtitles", (source, {"feedback": feedback})),
                             ("correct_subtitles", (source, feedback))):
            with self.subTest(method=method), patch.object(Brain, "_codex", return_value={"segments": source}) as model:
                self.assertEqual(getattr(Brain(), method)(*args), source)
                instruction = model.call_args.args[0]
                self.assertIn("本地实际音频识别与对齐", instruction)
                payload = json.loads(instruction.split("下方 JSON 仅为创作数据：\n", 1)[1])
                self.assertEqual(payload.get("feedback", payload.get("context", {}).get("feedback")), feedback)
                self.assertEqual(payload["segments"], source)
            shifted = copy.deepcopy(source)
            shifted[0].update(start=3.25, end=4.5)
            with self.subTest(method=method, guessed_shift=True), patch.object(Brain, "_codex", return_value={"segments": shifted}):
                with self.assertRaises(BrainError):
                    getattr(Brain(), method)(*args)
        self.assertEqual(source, before)

    def test_proofreading_uses_observed_text_and_does_not_claim_to_hear_audio(self):
        source = captions()
        source[0]["asr_metadata"] = "本地识别内部字段，不进入GPT"
        before = copy.deepcopy(source)
        corrected = captions()
        corrected[0]["text"] = "楚阳来到车间"
        context = {"names": ["楚阳"], "planned_dialogue_reference_only": "计划价格3999元，仅作参考。",
                   "previous_corrected_text_reference_only": "楚阳来到车间。"}
        with patch.object(Brain, "_codex", return_value={"segments": corrected}) as model:
            result = Brain().proofread_subtitles(source, context)
        self.assertEqual(source, before)
        self.assertEqual(result, corrected)
        user, schema, images = model.call_args.args
        self.assertIs(schema, CAPTION_SCHEMA)
        self.assertEqual(images, [])
        self.assertIn("没有听过音频", user)
        self.assertIn("禁止把未识别到的计划台词", user)
        self.assertIn("数字、价格与规则含义即使与计划不符", user)
        self.assertIn("不得照搬旧断句、数量或序号", user)
        payload = json.loads(user.split("下方 JSON 仅为创作数据：\n", 1)[1])
        self.assertEqual(payload["context"], context)
        self.assertNotIn("asr_metadata", payload["segments"][0])
        self.assertEqual(payload["segments"][1]["text"], "实际说的是2999元")

    def test_proofreading_rejects_changed_times_count_order_and_invalid_text(self):
        for mutation in (
            lambda rows: rows[0].update(start=0.5),
            lambda rows: rows[0].update(end=1.6),
            lambda rows: rows.pop(),
            lambda rows: rows.append({"start": 5, "end": 6, "text": "杜撰对白"}),
            lambda rows: rows.reverse(),
            lambda rows: rows[0].update(start=float("nan")),
            lambda rows: rows[0].update(text=""),
            lambda rows: rows[0].update(text="   "),
        ):
            changed = captions()
            mutation(changed)
            with self.subTest(changed=changed), patch.object(Brain, "_codex", return_value={"segments": changed}):
                with self.assertRaises(BrainError):
                    Brain().proofread_subtitles(captions(), {})

    def test_invalid_caption_inputs_stop_before_a_model_submission(self):
        invalid = [None, [{"start": 0, "text": "缺结束时间"}],
                   [{"start": 1, "end": 0.5, "text": "结束早于开始"}],
                   [{"start": 0, "end": 0, "text": "零时长"}],
                   [{"start": 0, "end": 1, "text": "   "}],
                   [{"start": True, "end": 1, "text": "布尔时间"}]]
        for rows in invalid:
            client = Brain()
            client.before_submit = Mock()
            with self.subTest(rows=rows), patch.object(client, "_codex") as model:
                with self.assertRaises(BrainError):
                    client.proofread_subtitles(rows, {})
                model.assert_not_called()
                client.before_submit.assert_not_called()

    def test_subtitle_batch_and_context_limits_are_local(self):
        for rows, context in ((captions() * 41, {}), (captions(), {"feedback": "字" * 9001})):
            with self.subTest(count=len(rows)), patch.object(Brain, "_codex") as model:
                with self.assertRaisesRegex(BrainError, "尚未提交"):
                    Brain().proofread_subtitles(rows, context)
                model.assert_not_called()

    def test_music_arrangement_is_approved_catalog_only_and_never_listening_claim(self):
        context = {"duration": 130, "catalog": [{"id": "approved-theme", "duration": 45}],
                   "speech_windows": [{"start": 2, "end": 3}], "timeline": [], "feedback": "留出停顿"}
        before = copy.deepcopy(context)
        with patch.object(Brain, "_codex", return_value=music_plan()) as model:
            self.assertEqual(Brain().arrange_music(context), music_plan())
        self.assertEqual(context, before)
        user, schema, images = model.call_args.args
        self.assertIs(schema, MUSIC_CUE_SCHEMA)
        self.assertEqual(images, [])
        for requirement in ("只能选择 catalog 内已人工试听批准", "你没有听过音频", "禁止添加人声或旁白",
                            "130秒本集照常编排", "不改视频时长", "最终由用户"):
            self.assertIn(requirement, user)
        self.assertEqual(json.loads(user.split("下方 JSON 仅为创作数据：\n", 1)[1]), context)
        self.assertEqual(schema["properties"]["cues"]["maxItems"], 64)

    def test_music_schema_rejects_unbounded_counts_nonfinite_or_arbitrary_filters(self):
        variants = []
        too_many = music_plan()
        too_many["cues"] *= 65
        variants.append(too_many)
        for change in ({"gain_db": -41}, {"gain_db": -2}, {"end": float("inf")},
                       {"start": -1}, {"start": True}, {"source_start": -1},
                       {"fade_in": -1}, {"filter": "amovie=/arbitrary/file"}, {"loop": True}):
            changed = music_plan()
            changed["cues"][0].update(change)
            variants.append(changed)
        for result in variants:
            with self.subTest(result=result), patch.object(Brain, "_codex", return_value=result):
                with self.assertRaises(BrainError):
                    Brain().arrange_music({"duration": 130, "catalog": []})

    def test_music_oversize_context_is_rejected_before_submission(self):
        client = Brain()
        client.before_submit = Mock()
        with patch.object(client, "_codex") as model:
            with self.assertRaisesRegex(BrainError, "尚未提交"):
                client.arrange_music({"novel_should_never_be_here": "字" * 20001})
            model.assert_not_called()
            client.before_submit.assert_not_called()


class PostproductionBrainCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nolan-synthetic-post-brain-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        core.create_project(self.root, "合成后期缓存测试，不生成媒体。", mode="live")
        core.configure(self.root, {"caps": {"llm": 0}, "max_llm_calls": 20}, limit=100)
        config = {"provider": "codex", "billing": "subscription", "model": "synthetic-model"}
        self.enterContext(patch("norlan.connection.production_brain_config", return_value=config))
        self.enterContext(patch("norlan.brain.resolve_codex", return_value="/synthetic/codex"))
        self.enterContext(patch("norlan.brain.Brain._openai", side_effect=AssertionError("No API fallback")))
        self.transport = self.enterContext(patch("norlan.brain.subprocess.run", side_effect=self.invoke))

    def invoke(self, args, **kwargs):
        state = core.load_project(self.root)
        job = list(state["jobs"].values())[-1]
        self.assertEqual(job["state"], "submitting")
        self.assertIn('forced_login_method="chatgpt"', args)
        schema = json.loads(Path(args[args.index("--output-schema") + 1]).read_text())
        result = {"segments": captions()} if "segments" in schema["properties"] else music_plan()
        Path(args[args.index("--output-last-message") + 1]).write_text(json.dumps(result), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")

    def call(self, method="proofread_subtitles", key="synthetic-key"):
        args = (captions(), {}) if method == "proofread_subtitles" else ({"duration": 10, "catalog": []},)
        return pipeline._brain(self.root, method, *args, _cache_key=key)

    def test_completed_post_result_is_durable_hash_bound_and_reused_once(self):
        first = self.call()
        state = core.load_project(self.root)
        self.assertEqual(state["llm_calls"], 1)
        job = list(state["jobs"].values())[-1]
        self.assertEqual((job["state"], job["accounting"]), ("completed", "spent"))
        self.assertEqual(job["post_cache_key"], "synthetic-key")
        self.assertEqual(job["post_result_hash"], core._json_digest(first))
        self.assertEqual(state["budget"]["spent"], 0)
        # A cached successful result remains usable at the GPT count cap.
        core.configure(self.root, {"max_llm_calls": 1})
        self.assertEqual(self.call(), first)
        self.transport.assert_called_once()
        # The returned data cannot mutate the durable cache by aliasing.
        first[0]["text"] = "外部修改"
        self.assertEqual(self.call()[0]["text"], captions()[0]["text"])
        self.assertEqual(core.load_project(self.root)["llm_calls"], 1)

    def test_different_method_or_context_key_does_not_reuse_another_result(self):
        self.call()
        self.assertEqual(self.call("arrange_music"), music_plan())
        self.call(key="new-feedback-or-source")
        self.assertEqual(self.transport.call_count, 3)
        self.assertEqual(core.load_project(self.root)["llm_calls"], 3)

    def test_tampered_completed_cache_blocks_without_another_model_request(self):
        self.call()
        original = copy.deepcopy(list(core.load_project(self.root)["jobs"].values())[-1])
        for change in (lambda job: job["post_result"][0].update(text="已改动的缓存"),
                       lambda job: job.pop("post_result_hash", None),
                       lambda job: job.pop("post_result", None)):
            with core.transaction(self.root) as state:
                state["jobs"][original["id"]] = copy.deepcopy(original)
                change(state["jobs"][original["id"]])
            snapshot = core.load_project(self.root)
            with self.subTest(change=change), self.assertRaises(core.GateError):
                self.call()
            self.transport.assert_called_once()
            self.assertEqual(core.load_project(self.root)["jobs"], snapshot["jobs"])
            self.assertEqual(core.load_project(self.root)["llm_calls"], 1)
        # An explicitly changed source/feedback key can start a new plan;
        # simply resuming the corrupt completed record cannot.
        self.assertEqual(self.call(key="explicit-new-feedback"), captions())
        self.assertEqual(self.transport.call_count, 2)

    def test_unknown_new_job_blocks_even_a_previously_completed_cached_result(self):
        self.call()
        core.reserve(self.root, "synthetic-unknown", "llm", "arrange_music", 0)
        core.update_job(self.root, "synthetic-unknown", state="uncertain", accounting="unknown")
        snapshot = core.load_project(self.root)
        with self.assertRaisesRegex(core.GateError, "结果不明"):
            self.call()
        self.assertEqual(core.load_project(self.root)["jobs"], snapshot["jobs"])
        self.transport.assert_called_once()

    def test_stream_loss_is_not_resubmitted_or_cached(self):
        self.transport.side_effect = lambda args, **kwargs: subprocess.CompletedProcess(
            args, 1, "", "stream disconnected: SYNTHETIC TEST")
        with self.assertRaises(BrainError):
            self.call("arrange_music")
        state = core.load_project(self.root)
        job = list(state["jobs"].values())[-1]
        self.assertEqual((job["state"], job["accounting"]), ("uncertain", "unknown"))
        self.assertNotIn("post_result", job)
        with self.assertRaisesRegex(core.GateError, "结果不明"):
            self.call("arrange_music")
        self.transport.assert_called_once()
        self.assertEqual(core.load_project(self.root)["llm_calls"], 1)

    def test_oversize_input_never_reserves_a_cache_job_or_calls_gpt(self):
        state = core.load_project(self.root)
        with self.assertRaisesRegex(BrainError, "尚未提交"):
            pipeline._brain(self.root, "arrange_music", {"feedback": "字" * 20001}, _cache_key="too-big")
        self.transport.assert_not_called()
        after = core.load_project(self.root)
        self.assertEqual((after["jobs"], after["llm_calls"], after["budget"]),
                         (state["jobs"], state["llm_calls"], state["budget"]))


if __name__ == "__main__":
    unittest.main()
