import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from backend.main import app, ChatRequest
import backend.rag as rag


class TestFreeformMode(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_chat_request_validation(self):
        req = ChatRequest(question="What is entropy?", mode="freeform")
        self.assertEqual(req.mode, "freeform")
        self.assertEqual(req.question, "What is entropy?")

    def test_build_prompt_and_sources_freeform(self):
        system_prompt, user_content, sources = rag.build_prompt_and_sources(
            question="Explain thermodynamics",
            raw_chunks=[],
            mode="freeform",
        )
        self.assertIn("helpful, versatile, and knowledgeable AI assistant", system_prompt)
        self.assertEqual(user_content, "Explain thermodynamics")
        self.assertEqual(sources, [])

    @patch("backend.rag.get_chat_client_and_model")
    def test_run_freeform_pipeline(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Thermodynamics is the study of heat and energy."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = (mock_client, "gpt-4o-mini")

        import asyncio
        result = asyncio.run(rag.run_freeform_pipeline("Explain thermodynamics", chat_key="test-key"))

        self.assertEqual(result["mode"], "freeform")
        self.assertEqual(result["answer"], "Thermodynamics is the study of heat and energy.")
        self.assertEqual(result["sources"], [])
        mock_client.chat.completions.create.assert_called_once()

    def test_prepare_rag_context_freeform_bypasses_db(self):
        import asyncio
        with patch("backend.rag.get_query_embedding") as mock_embed, \
             patch("backend.db.hybrid_search") as mock_search:
            chosen_mode, sys_p, usr_c, sources = asyncio.run(
                rag.prepare_rag_context("Hello AI", mode="freeform")
            )
            self.assertEqual(chosen_mode, "freeform")
            self.assertEqual(usr_c, "Hello AI")
            self.assertEqual(sources, [])
            mock_embed.assert_not_called()
            mock_search.assert_not_called()

    @patch("backend.rag.get_chat_client_and_model")
    def test_api_chat_endpoint_freeform_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "42 is the answer to everything."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = (mock_client, "gpt-4o-mini")

        # Freeform mode should work with just X-Api-Key, no EMBED_API_KEY needed
        res = self.client.post(
            "/api/chat",
            json={"question": "What is 42?", "mode": "freeform", "stream": False},
            headers={"X-Api-Key": "dummy-chat-key"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["mode"], "freeform")
        self.assertEqual(data["answer"], "42 is the answer to everything.")
        self.assertEqual(data["sources"], [])
