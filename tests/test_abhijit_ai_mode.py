import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from backend.main import app, ChatRequest
import backend.rag as rag


class TestAbhijitAIMode(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_chat_request_validation(self):
        req = ChatRequest(question="What would you say about the multipolar world?", mode="abhijit_ai")
        self.assertEqual(req.mode, "abhijit_ai")
        self.assertEqual(req.question, "What would you say about the multipolar world?")

    def test_classify_query_regex_abhijit_ai(self):
        import asyncio
        queries = [
            "How would Abhijit explain quantum gravity?",
            "What would Abhijit say about the Roman Empire?",
            "Ask Abhijit about ancient Indian mathematics",
            "In Abhijit voice, what is the future of AI?",
            "What is Abhijit's take on the Indus Valley Civilization?",
        ]
        for q in queries:
            mode = asyncio.run(rag.classify_query(q))
            self.assertEqual(mode, "abhijit_ai", f"Query '{q}' failed to classify as abhijit_ai")

    def test_build_prompt_and_sources_abhijit_ai_with_chunks(self):
        dummy_chunk = {
            "video_id": "test_vid_1",
            "title": "Quantum Mechanics & History",
            "upload_date": "20240115",
            "start_sec": 120,
            "end_sec": 180,
            "text": "The reality is that space exploration requires a multi-generational civilizational vision.",
        }
        sys_prompt, user_content, sources = rag.build_prompt_and_sources(
            question="What is your take on space exploration?",
            raw_chunks=[dummy_chunk],
            mode="abhijit_ai",
        )
        self.assertIn("Abhijit AI", sys_prompt)
        self.assertIn("Namaskar", sys_prompt)
        self.assertIn("Interesting question", sys_prompt)
        self.assertIn("Do NOT repeatedly start answers with 'Interesting question'", sys_prompt)
        self.assertIn("Emulate those authentic transcript opening styles", sys_prompt)
        self.assertIn("I have already answered this in one of my videos", sys_prompt)
        self.assertIn("Do NOT blindly start every answer with 'Namaskar'", sys_prompt)
        self.assertIn("Mirror or appropriately reciprocate", sys_prompt)
        self.assertIn("address them by name", sys_prompt)
        self.assertIn("Relevant excerpts from Abhijit Chavda's past video transcripts:", user_content)
        self.assertIn("Quantum Mechanics & History", user_content)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["video_id"], "test_vid_1")
        self.assertEqual(sources[0]["timestamp_str"], "02:00")

    def test_dynamic_starting_lines_across_all_modes(self):
        dummy_chunk = {
            "video_id": "vid1",
            "title": "History of India",
            "upload_date": "20230501",
            "start_sec": 60,
            "end_sec": 120,
            "text": "The Mauryan Empire established centralized administration.",
        }

        # Lookup mode
        lookup_prompt, _, _ = rag.build_prompt_and_sources("Who ruled Maurya?", [dummy_chunk], "lookup")
        self.assertIn("Avoid rigid, hard-coded boilerplate openings", lookup_prompt)
        self.assertIn("starting lines like the ones used in the transcripts", lookup_prompt)
        self.assertIn("Abhijit Chavda", lookup_prompt)
        self.assertIn("NOT Abhijit Iyer-Mitra", lookup_prompt)

        # Opinion mode
        opinion_prompt, _, _ = rag.build_prompt_and_sources("What is his opinion?", [dummy_chunk], "opinion")
        self.assertIn("Avoid repetitive, hard-coded boilerplate", opinion_prompt)
        self.assertIn("authentic starting lines like those found in the transcripts", opinion_prompt)
        self.assertIn("Abhijit Chavda", opinion_prompt)
        self.assertIn("NOT Abhijit Iyer-Mitra", opinion_prompt)

        # Evolution mode
        evolution_prompt, _, _ = rag.build_prompt_and_sources("How did views evolve?", [dummy_chunk], "evolution")
        self.assertIn("Avoid formulaic, hard-coded opening lines", evolution_prompt)
        self.assertIn("mirror the historical progression in the transcripts", evolution_prompt)
        self.assertIn("Abhijit Chavda", evolution_prompt)
        self.assertIn("NOT Abhijit Iyer-Mitra", evolution_prompt)

        # Freeform mode
        freeform_prompt, _, _ = rag.build_prompt_and_sources("Explain calculus", [], "freeform")
        self.assertIn("Avoid canned, hard-coded AI opening lines", freeform_prompt)
        self.assertIn("open naturally and dynamically", freeform_prompt)
        self.assertIn("Abhijit Chavda", freeform_prompt)
        self.assertIn("NOT Abhijit Iyer-Mitra", freeform_prompt)

        # Abhijit AI mode
        ai_prompt, _, _ = rag.build_prompt_and_sources("What is your take?", [dummy_chunk], "abhijit_ai")
        self.assertIn("Abhijit Chavda", ai_prompt)
        self.assertIn("NOT Abhijit Iyer-Mitra", ai_prompt)

    def test_build_prompt_and_sources_abhijit_ai_without_chunks(self):
        sys_prompt, user_content, sources = rag.build_prompt_and_sources(
            question="Can you imagine life on Mars?",
            raw_chunks=[],
            mode="abhijit_ai",
        )
        self.assertIn("Abhijit AI", sys_prompt)
        self.assertIn("Answer hypothetically based on Abhijit Chavda's known philosophies", user_content)
        self.assertEqual(sources, [])

    @patch("backend.rag.get_chat_client_and_model")
    def test_run_abhijit_ai_pipeline_without_embed_key(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Namaskar! Interesting question. Look, the reality is that Mars colonization requires fundamental scientific breakthroughs."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = (mock_client, "gpt-4o-mini")

        import asyncio
        result = asyncio.run(rag.run_abhijit_ai_pipeline(
            "What about Mars?",
            chat_key="test-key",
            embed_key="",
        ))

        self.assertEqual(result["mode"], "abhijit_ai")
        self.assertIn("Namaskar", result["answer"])
        self.assertEqual(result["sources"], [])
        mock_client.chat.completions.create.assert_called_once()

    @patch("backend.rag.get_chat_client_and_model")
    @patch("backend.rag.get_query_embedding")
    @patch("backend.db.hybrid_search")
    def test_run_abhijit_ai_pipeline_with_embed_key(self, mock_search, mock_embed, mock_get_client):
        mock_embed.return_value = [0.1, 0.2]
        mock_search.return_value = [
            {
                "video_id": "vid99",
                "title": "#AskAbhijit 150",
                "upload_date": "20231101",
                "start_sec": 300,
                "end_sec": 360,
                "text": "Geopolitics is driven by geography and national interest.",
            }
        ]
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Namaskar! I have already answered this in #AskAbhijit 150, but to summarize..."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = (mock_client, "gpt-4o-mini")

        import asyncio
        result = asyncio.run(rag.run_abhijit_ai_pipeline(
            "What is geopolitics?",
            chat_key="test-key",
            embed_key="test-embed-key",
        ))

        self.assertEqual(result["mode"], "abhijit_ai")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["video_id"], "vid99")
        self.assertIn("Namaskar", result["answer"])

    @patch("backend.rag.get_chat_client_and_model")
    @patch("backend.rag.get_query_embedding")
    def test_api_chat_endpoint_abhijit_ai_success(self, mock_embed, mock_get_client):
        mock_embed.return_value = [0.0] * 1536
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Namaskar! That is a very interesting question. First principles dictate that..."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = (mock_client, "gpt-4o-mini")

        # Abhijit AI should work with just X-Api-Key without throwing 401 for missing embed_key
        res = self.client.post(
            "/api/chat",
            json={"question": "What is the meaning of life?", "mode": "abhijit_ai", "stream": False},
            headers={"X-Api-Key": "dummy-chat-key"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["mode"], "abhijit_ai")
        self.assertIn("Namaskar", data["answer"])
