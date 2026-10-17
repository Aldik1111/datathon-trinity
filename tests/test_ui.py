import base64
import os
from io import BytesIO
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from src.ui.client import (
    APIClientError,
    APISettings,
    decode_evidence_image,
    format_timecode,
    get_job,
    submit_video,
)


class UIClientTests(unittest.TestCase):
    def setUp(self):
        self.settings = APISettings("http://127.0.0.1:8000", "test-secret")

    @patch("src.ui.client.requests.post")
    def test_submit_posts_mov_with_api_key_and_bounded_timeout(self, post):
        post.return_value = Mock(ok=True, json=lambda: {"job_id": "job-1", "status": "processing"})

        result = submit_video(self.settings, "clip.mov", b"video-bytes")

        self.assertEqual(result["job_id"], "job-1")
        self.assertEqual(post.call_args.kwargs["headers"], {"X-API-Key": "test-secret"})
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 120))
        file_tuple = post.call_args.kwargs["files"]["file"]
        self.assertEqual(file_tuple, ("clip.mov", b"video-bytes", "video/quicktime"))

    @patch("src.ui.client.requests.get")
    def test_poll_uses_job_endpoint_and_key(self, get):
        get.return_value = Mock(ok=True, json=lambda: {"job_id": "abc", "status": "done"})

        result = get_job(self.settings, "abc")

        self.assertEqual(result["status"], "done")
        self.assertEqual(get.call_args.args[0], "http://127.0.0.1:8000/api/v1/jobs/abc")
        self.assertEqual(get.call_args.kwargs["headers"], {"X-API-Key": "test-secret"})
        self.assertEqual(get.call_args.kwargs["timeout"], (5, 15))

    def test_rejects_missing_key_and_non_mov_upload(self):
        with self.assertRaisesRegex(APIClientError, "API_KEY"):
            submit_video(APISettings(self.settings.base_url, ""), "clip.mov", b"x")
        with self.assertRaisesRegex(APIClientError, r"\.mov"):
            submit_video(self.settings, "clip.mp4", b"x")

    def test_decodes_base64_and_skips_malformed_evidence(self):
        buffer = BytesIO()
        Image.new("RGB", (3, 2), color="red").save(buffer, format="JPEG")
        content = buffer.getvalue()
        encoded = base64.b64encode(content).decode("ascii")
        self.assertEqual(decode_evidence_image(encoded), content)
        self.assertEqual(decode_evidence_image(f"data:image/png;base64,{encoded}"), content)
        self.assertIsNone(decode_evidence_image(base64.b64encode(b"not an image").decode("ascii")))
        self.assertIsNone(decode_evidence_image("not base64!"))
        self.assertIsNone(decode_evidence_image(None))

    def test_formats_timecodes(self):
        self.assertEqual(format_timecode(62.9), "01:02.9")
        self.assertEqual(format_timecode(44.2), "00:44.2")
        self.assertEqual(format_timecode(3661), "01:01:01")
        self.assertEqual(format_timecode("bad"), "время не указано")


class StreamlitAppTests(unittest.TestCase):
    app_path = os.path.join(os.path.dirname(__file__), "..", "src", "ui", "app.py")

    @staticmethod
    def _jpeg_base64():
        buffer = BytesIO()
        Image.new("RGB", (4, 3), color="blue").save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _submit(at):
        at.file_uploader[0].set_value(("clip.mov", b"video-bytes", "video/quicktime"))
        next(button for button in at.button if button.label == "Анализировать").click()
        return at.run()

    @patch("src.ui.client.get_job")
    @patch("src.ui.client.submit_video")
    def test_renders_multiple_events_real_jpegs_missing_rule_and_rule_limitations(self, submit, get_job):
        jpeg = self._jpeg_base64()
        submit.return_value = {"job_id": "job-render", "status": "processing", "demo": False}
        get_job.return_value = {
            "job_id": "job-render",
            "status": "done",
            "verdict": "violation",
            "demo": False,
            "violations": [
                {
                    "type": "event_one",
                    "timecode_sec": 44.2,
                    "description": "Первое событие",
                    "evidence_frames": [{"timestamp_sec": 44.2, "image_base64": jpeg}],
                },
                {
                    "type": "event_two",
                    "timecode_sec": 65.5,
                    "description": "Второе событие",
                    "evidence_frames": [
                        {"timestamp_sec": 65.5, "image_base64": jpeg},
                        {"timestamp_sec": 66.0, "image_base64": jpeg},
                    ],
                    "rule": {"title": "Правило", "url": "https://example.test/rule", "section": "п. 2"},
                },
            ],
        }

        with patch.dict(os.environ, {"API_KEY": "app-test-secret"}):
            at = AppTest.from_file(self.app_path, default_timeout=10).run()
            self._submit(at)

        self.assertFalse(at.exception)
        rendered = "\n".join(str(element.value) for element in at if hasattr(element, "value"))
        self.assertIn("Обнаружены признаки возможных нарушений", rendered)
        self.assertIn("00:44.2", rendered)
        self.assertIn("Нормативное основание не подтверждено", rendered)
        self.assertIn("Связь события с указанной нормой", rendered)
        self.assertIn("event_one", rendered)
        self.assertIn("event_two", rendered)
        self.assertEqual(len(at.image), 3)
        self.assertIn("00:44.2", " ".join(caption for image in at.image for caption in image.captions))

    @patch("src.ui.client.get_job")
    @patch("src.ui.client.submit_video")
    def test_failed_demo_job_shows_persistent_test_warning(self, submit, get_job):
        submit.return_value = {"job_id": "job-failed", "status": "processing", "demo": True}
        get_job.return_value = {
            "job_id": "job-failed",
            "status": "failed",
            "demo": True,
            "error": "Видео не удалось обработать",
        }

        with patch.dict(os.environ, {"API_KEY": "app-test-secret"}):
            at = AppTest.from_file(self.app_path, default_timeout=10).run()
            self._submit(at)

        self.assertFalse(at.exception)
        rendered = "\n".join(str(element.value) for element in at if hasattr(element, "value"))
        self.assertIn("Демонстрационный режим", rendered)
        self.assertIn("Видео не удалось обработать", rendered)
        self.assertTrue(any("Демонстрационный режим" in element.value for element in at.warning))


if __name__ == "__main__":
    unittest.main()
