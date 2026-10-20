import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from norlan import media
from norlan.media import MediaError, compose, extract_evidence, probe, synthetic_asset, synthetic_clip


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required for media integration tests")
class MediaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="norlan-media-test-")
        self.root = Path(self.directory.name).resolve()

    def tearDown(self):
        self.directory.cleanup()

    def clip(self, name, size="320x240", audio=False, duration=0.4):
        target = self.root / name
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-f", "lavfi", "-i", f"color=c=blue:s={size}:r=30:d={duration}"]
        if audio:
            args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}:sample_rate=48000"]
        args += ["-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if audio:
            args += ["-c:a", "aac"]
        subprocess.run([*args, str(target)], capture_output=True, check=True)
        return target

    def test_silent_and_native_audio_keep_order_and_timing(self):
        silent = self.clip("1: silent's [clip].mp4", audio=False)
        native = self.clip("2 有声.mp4", size="240x320", audio=True, duration=0.6)
        subtitles = [{"start": 0.1, "end": 0.35, "text": "你好，世界！"},
                     {"start": 0.5, "end": 0.9, "text": "literal <b> & %{danger} ' : [x]"}]
        result = compose([silent, native], self.root / "final: 中文's [x].mp4", subtitles)
        actual = probe(Path(result["output"]))
        self.assertEqual((actual["width"], actual["height"]), (320, 240))
        self.assertEqual((actual["channels"], actual["sample_rate"]), (2, 48000))
        self.assertAlmostEqual(actual["duration"], 1.0, delta=0.04)
        self.assertEqual([item["source"] for item in result["timeline"]], [str(silent), str(native)])
        self.assertEqual(result["timeline"][1]["start"], result["timeline"][0]["end"])
        self.assertFalse(result["music_present"])
        self.assertIn("你好", Path(result["subtitles"]).read_text())
        self.assertIn("&lt;b&gt;", Path(result["subtitles"]).read_text())
        # Audio in the second clip survives; first silent region stays quiet.
        samples = subprocess.run(["ffmpeg", "-v", "error", "-i", result["output"],
                                  "-f", "f32le", "-ac", "1", "-ar", "48000", "-"],
                                 capture_output=True, check=True).stdout
        import array
        values = array.array("f", samples)
        def power(start, end):
            region = values[round(start * 48000):round(end * 48000)]
            return sum(value * value for value in region) / len(region)
        self.assertLess(power(0.05, 0.25), 1e-7)
        self.assertGreater(power(0.55, 0.8), 1e-5)

    def test_missing_subtitle_filter_refuses_before_media_processing(self):
        source = self.root / "synthetic-input.mp4"
        source.write_bytes(b"synthetic placeholder; must not be probed")
        output = self.root / "missing-filter.mp4"
        with patch.object(media, "_has_filter", return_value=False), \
                patch.object(media, "_font_path") as font, \
                patch.object(media, "probe") as probe_mock, \
                patch.object(media, "_ffmpeg") as render:
            with self.assertRaisesRegex(MediaError, "subtitles.*libass") as raised:
                compose([source], output, [{"start": 0, "end": 1, "text": "合成字幕"}])
        self.assertTrue(raised.exception.safe_message)
        font.assert_not_called()
        probe_mock.assert_not_called()
        render.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".srt").exists())
        self.assertEqual(source.read_bytes(), b"synthetic placeholder; must not be probed")

    def test_missing_cjk_font_refuses_before_media_processing(self):
        source = self.root / "synthetic-input.mp4"
        source.write_bytes(b"synthetic placeholder; must not be probed")
        with patch.object(media, "_has_filter", return_value=True), \
                patch.object(media, "_font_path", return_value=None) as font, \
                patch.object(media, "probe") as probe_mock, \
                patch.object(media, "_ffmpeg") as render:
            with self.assertRaisesRegex(MediaError, "中文字体") as raised:
                compose([source], self.root / "missing-font.mp4",
                        [{"start": 0, "end": 1, "text": "合成字幕"}])
        self.assertTrue(raised.exception.safe_message)
        font.assert_called_once_with(cjk_only=True)
        probe_mock.assert_not_called()
        render.assert_not_called()

    def test_latin_fallback_is_not_a_chinese_subtitle_font(self):
        latin_font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        with patch.object(Path, "is_file", lambda candidate: candidate == latin_font):
            self.assertEqual(media._font_path(), latin_font)
            self.assertIsNone(media._font_path(cjk_only=True))

    def test_explicit_srt_only_export_does_not_require_subtitle_dependencies(self):
        source = self.clip("srt-only.mp4")
        with patch.object(media, "_has_filter") as filters, \
                patch.object(media, "_font_path") as font:
            result = compose([source], self.root / "srt-only-output.mp4",
                             [{"start": 0.1, "end": 0.35, "text": "仅导出合成字幕文件"}],
                             burn_subtitles=False)
        filters.assert_not_called()
        font.assert_not_called()
        self.assertFalse(result["burned_subtitles"])
        self.assertTrue(Path(result["output"]).is_file())
        self.assertIn("仅导出合成字幕文件", Path(result["subtitles"]).read_text())

    def test_real_chinese_subtitles_are_present_in_rendered_pixels(self):
        if not media._has_filter("subtitles") or not media._font_path(cjk_only=True):
            self.skipTest("requires libass and a local CJK subtitle font")
        source = self.clip("synthetic-chinese-source.mp4", size="640x360", duration=1)
        result = compose([source], self.root / "synthetic-chinese-burned.mp4",
                         [{"start": 0.2, "end": 0.9, "text": "合成中文字幕：账我看了。"}])
        self.assertTrue(result["burned_subtitles"])

        def bright_pixels(at):
            pixels = subprocess.run([
                "ffmpeg", "-v", "error", "-ss", str(at), "-i", result["output"],
                "-frames:v", "1", "-vf", "crop=iw:ih/2:0:ih/2", "-f", "rawvideo",
                "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
            return sum(value > 180 for value in pixels)

        self.assertEqual(bright_pixels(0.05), 0)
        self.assertGreater(bright_pixels(0.5), 100)

    def test_evidence_keeps_identity_and_marks_sampling_limits(self):
        source = self.clip("native.mp4", audio=True)
        evidence = extract_evidence(source, self.root / "evidence")
        self.assertEqual(len(evidence["frames"]), 3)
        self.assertTrue(all(Path(path).is_file() for path in evidence["frames"]))
        self.assertEqual(evidence["tail_frame"], evidence["frames"][-1])
        self.assertTrue(Path(evidence["audio"]).is_file())
        self.assertEqual(len(evidence["sha256"]), 64)
        self.assertIn("不能证明", evidence["sampling_note"])
        silent = self.clip("silent.mp4")
        self.assertIsNone(extract_evidence(silent, self.root / "silent-evidence")["audio"])

    def test_music_is_mixed_into_silent_clip(self):
        source = self.clip("silent.mp4", duration=0.6)
        music = self.root / "local music.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=0.3",
                        str(music)], capture_output=True, check=True)
        result = compose([source], self.root / "with-music.mp4", music=music)
        self.assertTrue(result["music_present"])
        self.assertEqual(result["music_source"], str(music))
        self.assertAlmostEqual(result["duration"], 0.6, delta=0.04)
        self.assertTrue(probe(Path(result["output"]))["has_audio"])

    def test_subshot_samples_cover_planned_cuts_and_actual_last_frame(self):
        source = self.root / "three-subshots.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                        "color=c=blue:s=160x120:r=30:d=15", "-vf",
                        "drawbox=x=0:y=0:w=iw:h=ih:color=red:t=fill:enable='eq(n,449)'",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
                       capture_output=True, check=True)
        subshots = [{"start": 0, "end": 5, "prompt": "合成子镜一"},
                    {"start": 5, "end": 10, "prompt": "合成子镜二"},
                    {"start": 10, "end": 15, "prompt": "合成子镜三"}]
        evidence = extract_evidence(source, self.root / "subshot-evidence", subshots=subshots)
        times = evidence["sample_times"]
        self.assertEqual(len(times), 9)
        self.assertEqual(times, sorted(set(times)))
        self.assertEqual(times[0], 0)
        self.assertAlmostEqual(times[-1], 449 / 30, places=5)
        for midpoint in (2.5, 7.5, 12.5):
            self.assertLess(min(abs(value - midpoint) for value in times), 1 / 30)
        for cut in (5, 10):
            self.assertTrue(any(cut - 0.2 < value < cut for value in times))
            self.assertTrue(any(cut < value < cut + 0.2 for value in times))
        self.assertEqual(evidence["frame_indices"][-1], 449)
        self.assertEqual(evidence["tail_frame"], evidence["frames"][-1])
        pixel = subprocess.run(["ffmpeg", "-v", "error", "-i", evidence["tail_frame"],
                                "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               capture_output=True, check=True).stdout
        self.assertGreater(pixel[0], 200)
        self.assertLess(pixel[2], 40)  # Only the actual final frame is red.
        self.assertIn("不代表已检测", evidence["sampling_note"])

    def test_subshot_samples_clip_to_short_actual_stream(self):
        source = self.clip("short-subshot-result.mp4", duration=0.4)
        subshots = [{"start": 0, "end": 5, "prompt": "合成子镜一"},
                    {"start": 5, "end": 10, "prompt": "合成子镜二"},
                    {"start": 10, "end": 15, "prompt": "合成子镜三"}]
        evidence = extract_evidence(source, self.root / "short-subshot-evidence", subshots=subshots)
        self.assertGreaterEqual(len(evidence["frames"]), 2)
        self.assertLessEqual(len(evidence["frames"]), 9)
        self.assertTrue(all(0 <= time < evidence["metadata"]["duration"] for time in evidence["sample_times"]))
        self.assertEqual(evidence["sample_times"], sorted(set(evidence["sample_times"])))
        self.assertAlmostEqual(evidence["sample_times"][-1], 11 / 30, places=5)
        with self.assertRaises(MediaError):
            extract_evidence(source, self.root / "invalid-subshot-evidence",
                             subshots=[{"start": 0, "end": float("nan"), "prompt": "合成"}])

    def test_demo_media_and_bad_inputs(self):
        image = synthetic_asset(self.root / "asset.png", "仅演示：%{text} $(touch unsafe)", "#234567")
        self.assertTrue(probe(image)["has_video"])
        video = synthetic_clip(self.root / "demo.mp4", "演示而非生成内容", 0.3)
        self.assertTrue(probe(video)["has_audio"])
        self.assertFalse((self.root / "unsafe").exists())
        with self.assertRaises(MediaError):
            synthetic_asset(self.root / "bad.png", "test", "red;movie=/tmp/unsafe")
        with self.assertRaises(MediaError):
            synthetic_clip(self.root / "bad.mp4", "test", -3)
        with self.assertRaises(MediaError):
            compose([], self.root / "empty.mp4")
        with self.assertRaises(MediaError):
            compose([video], video)
        with self.assertRaises(MediaError):
            compose([video], self.root / "bad-subtitle.mp4", [{"start": 0, "end": 999, "text": "bad"}])


if __name__ == "__main__":
    unittest.main()
