import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import backend.ingest as ingest
from backend.main import app


class TestSyncRateLimit(unittest.TestCase):
    def setUp(self):
        ingest.clear_rate_limited()
        self.client = TestClient(app)

    def tearDown(self):
        ingest.clear_rate_limited()

    def test_sync_status_initially_false(self):
        res = self.client.get("/api/sync/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["is_rate_limited"])
        self.assertIsNone(data["message"])

    def test_set_and_clear_rate_limit(self):
        ingest.set_rate_limited("Test 429 rate limit")
        res = self.client.get("/api/sync/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["is_rate_limited"])
        self.assertEqual(data["message"], "Test 429 rate limit")
        self.assertIsNotNone(data["timestamp"])

        # Clear via API endpoint
        clear_res = self.client.post("/api/sync/clear-rate-limit")
        self.assertEqual(clear_res.status_code, 200)

        # Verify cleared
        res2 = self.client.get("/api/sync/status")
        self.assertFalse(res2.json()["is_rate_limited"])

    def test_sync_stream_handles_rate_limited_event(self):
        async def mock_sync(*args, **kwargs):
            yield {
                "playlist": "#AskAbhijit",
                "playlist_id": "ask-abhijit",
                "current": 1,
                "total": 5,
                "video_id": "abc12345",
                "status": "rate_limited",
                "message": "YouTube rate-limited transcript requests from this IP (429). Pausing sync to protect IP.",
            }

        with patch("backend.ingest.sync_playlists", side_effect=mock_sync):
            res = self.client.post("/api/sync", json={"playlist_ids": ["ask-abhijit"]})
            self.assertEqual(res.status_code, 200)
            body = res.text
            # Must contain rate_limited event
            self.assertIn('"status": "rate_limited"', body)
            self.assertIn("YouTube rate-limited transcript requests", body)
            # Must NOT contain done event
            self.assertNotIn('"status": "done"', body)

    def test_sync_stream_handles_exception_with_429(self):
        async def mock_sync_error(*args, **kwargs):
            raise Exception("HTTP Error 429: Too Many Requests from YouTube API")
            yield {}

        with patch("backend.ingest.sync_playlists", side_effect=mock_sync_error):
            res = self.client.post("/api/sync", json={"playlist_ids": ["ask-abhijit"]})
            self.assertEqual(res.status_code, 200)
            body = res.text
            self.assertIn('"status": "rate_limited"', body)
            self.assertIn("429", body)
            self.assertNotIn('"status": "done"', body)

            # Ingest rate limit should now be set
            status_res = self.client.get("/api/sync/status")
            self.assertTrue(status_res.json()["is_rate_limited"])


if __name__ == "__main__":
    unittest.main()
