"""
ASR wrapper tests (mocked — no live network).
==============================================
Verifies:
  - The response parser handles all the Sarvam payload shapes the
    production wrapper accounts for (transcript / text / data.transcript /
    data.results / top-level segments).
  - transcribe_batch fans out in parallel and aggregates results by path.
  - transcribe_file gracefully returns clean=None + error message on
    Sarvam failure (matches the MVP "no Whisper fallback" decision).
  - configured / unconfigured singletons behave correctly when SARVAM_API_KEY
    is missing.

Run with:
    python -m unittest tests.test_asr
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend.asr import (
    SarvamProvider,
    _extract_transcript,
    _join_segments,
    sarvam,
    transcribe_batch,
    transcribe_file,
)


class TestResponseParser(unittest.TestCase):
    """Mirror the parser-fallback chain from production."""

    def test_empty_payload(self):
        self.assertEqual(_extract_transcript({}), "")
        self.assertEqual(_extract_transcript(None), "")

    def test_top_level_transcript(self):
        self.assertEqual(_extract_transcript({"transcript": "hello world"}), "hello world")

    def test_top_level_text_fallback(self):
        self.assertEqual(_extract_transcript({"text": "hello"}), "hello")

    def test_nested_data_transcript(self):
        self.assertEqual(_extract_transcript({"data": {"transcript": "hi"}}), "hi")

    def test_data_results_join(self):
        payload = {"data": {"results": [{"text": "foo"}, {"text": "bar"}]}}
        self.assertEqual(_extract_transcript(payload), "foo bar")

    def test_top_level_segments(self):
        payload = {"segments": [{"transcript": "alpha"}, {"transcript": "beta"}]}
        self.assertEqual(_extract_transcript(payload), "alpha beta")

    def test_segments_skip_empty_entries(self):
        segments = [{"text": "one"}, {"text": ""}, {"text": "two"}]
        self.assertEqual(_join_segments(segments), "one two")

    def test_segments_with_non_dict_entries(self):
        segments = [{"text": "good"}, "noise", 42, {"text": "fine"}]
        self.assertEqual(_join_segments(segments), "good fine")

    def test_strips_whitespace(self):
        self.assertEqual(_extract_transcript({"transcript": "  hello  "}), "hello")


class TestSarvamProviderConfig(unittest.TestCase):
    def test_unconfigured_when_no_api_key(self):
        with patch.dict(os.environ, {"SARVAM_API_KEY": ""}, clear=False):
            # Force a re-init by clearing the singleton.
            SarvamProvider._instance = None
            SarvamProvider._initialized = False
            p = SarvamProvider()
            self.assertFalse(p.configured)

    def test_configured_when_api_key_present(self):
        with patch.dict(os.environ, {"SARVAM_API_KEY": "sk_test_dummy"}, clear=False):
            SarvamProvider._instance = None
            SarvamProvider._initialized = False
            p = SarvamProvider()
            self.assertTrue(p.configured)
            self.assertEqual(p.api_key, "sk_test_dummy")

    def test_singleton_returns_same_instance(self):
        a = SarvamProvider()
        b = SarvamProvider()
        self.assertIs(a, b)


class TestTranscribeBatch(unittest.IsolatedAsyncioTestCase):
    """Async path: parallel fan-out via asyncio.gather."""

    async def test_batch_fans_out_and_aggregates(self):
        # Monkey-patch the module singleton's transcribe method.
        async def fake_transcribe(audio_bytes, filename, content_type=None,
                                  language_code=None, model=None):
            return f"transcript-for-{filename}"

        # Avoid opening actual files — fake transcribe_file at the module level.
        from backend import asr as asr_module

        async def fake_transcribe_file(path):
            return {"clean": f"transcript-for-{os.path.basename(path)}", "raw": None, "error": None}

        original = asr_module.transcribe_file
        asr_module.transcribe_file = fake_transcribe_file
        try:
            paths = ["/tmp/a.wav", "/tmp/b.wav", "/tmp/c.wav"]
            result = await transcribe_batch(paths)
            self.assertEqual(set(result.keys()), set(paths))
            self.assertEqual(result["/tmp/a.wav"]["clean"], "transcript-for-a.wav")
            self.assertIsNone(result["/tmp/a.wav"]["raw"])  # MVP: raw always None
            self.assertIsNone(result["/tmp/a.wav"]["error"])
        finally:
            asr_module.transcribe_file = original

    async def test_transcribe_file_returns_error_on_missing_file(self):
        result = await transcribe_file("/nonexistent/path/never.wav")
        self.assertIsNone(result["clean"])
        self.assertIsNone(result["raw"])
        self.assertIn("file read failed", result["error"])

    async def test_transcribe_file_returns_error_on_sarvam_failure(self):
        """When the Sarvam call raises, transcribe_file converts to a clean error result."""
        async def raising_transcribe(*args, **kwargs):
            raise RuntimeError("Sarvam transcription failed: HTTP 503")

        # Create a tiny temp file so the read step succeeds.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"RIFFfakewavheader")
            tmp_path = tmp.name

        try:
            original = sarvam.transcribe
            sarvam.transcribe = raising_transcribe
            try:
                result = await transcribe_file(tmp_path)
                self.assertIsNone(result["clean"])
                self.assertIsNone(result["raw"])
                self.assertIn("HTTP 503", result["error"])
            finally:
                sarvam.transcribe = original
        finally:
            os.unlink(tmp_path)

    async def test_transcribe_file_returns_clean_on_success(self):
        async def fake_transcribe(audio_bytes, filename, content_type=None,
                                  language_code=None, model=None):
            return "the cat sat on the mat"

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"RIFFfakewavheader")
            tmp_path = tmp.name

        try:
            original = sarvam.transcribe
            sarvam.transcribe = fake_transcribe
            try:
                result = await transcribe_file(tmp_path)
                self.assertEqual(result["clean"], "the cat sat on the mat")
                self.assertIsNone(result["raw"])  # MVP: raw always None
                self.assertIsNone(result["error"])
            finally:
                sarvam.transcribe = original
        finally:
            os.unlink(tmp_path)

    async def test_transcribe_file_returns_none_clean_on_empty_transcript(self):
        """Sarvam can return an empty string for silent audio — surface as None, not ''."""
        async def fake_transcribe(*args, **kwargs):
            return ""

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"RIFFfakewavheader")
            tmp_path = tmp.name

        try:
            original = sarvam.transcribe
            sarvam.transcribe = fake_transcribe
            try:
                result = await transcribe_file(tmp_path)
                self.assertIsNone(result["clean"])
                self.assertIsNone(result["error"])
            finally:
                sarvam.transcribe = original
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
