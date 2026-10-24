import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.rag import workflow


class WorkflowNodeTests(unittest.TestCase):
    def test_greeting_uses_direct_response_without_retrieval(self):
        state = {
            "request_id": "test-greeting",
            "question": "Hi",
            "trace": [],
        }

        with patch.object(workflow, "get_retriever") as retriever:
            result = workflow.direct_answer(state)

        retriever.assert_not_called()
        self.assertEqual(result["source_used"], "direct")
        self.assertIn("Hi!", result["answer"])

    def test_private_kb_answer_preserves_citations(self):
        document = SimpleNamespace(
            metadata={"source": "vpn-runbook.md"},
            page_content="Use the company VPN client.",
        )
        model = SimpleNamespace(
            invoke=lambda _prompt: SimpleNamespace(content="Open the VPN client.")
        )
        state = {
            "request_id": "test-kb",
            "current_query": "How do I connect to VPN?",
            "kb_docs": [document],
            "trace": [],
        }

        with patch.object(workflow, "llm", return_value=model):
            result = workflow.generate_from_kb(state)

        self.assertEqual(result["source_used"], "private_kb")
        self.assertEqual(result["answer"], "Open the VPN client.")
        self.assertEqual(result["citations"][0]["title"], "vpn-runbook.md")

    def test_retrieval_failure_returns_empty_evidence_for_fallback(self):
        state = {
            "request_id": "test-retrieval-failure",
            "current_query": "How do I connect to VPN?",
            "trace": [],
        }

        with patch.object(workflow, "get_retriever", side_effect=TimeoutError()):
            result = workflow.retrieve_kb(state)

        self.assertEqual(result["kb_docs"], [])
        self.assertEqual(workflow.after_kb({**state, **result}), "search_web")

    def test_private_model_failure_is_marked_and_does_not_retry_after_web(self):
        document = SimpleNamespace(
            metadata={"source": "vpn-runbook.md"},
            page_content="VPN setup notes.",
        )
        model = SimpleNamespace(invoke=Mock(side_effect=TimeoutError()))
        state = {
            "request_id": "test-model-failure",
            "current_query": "How do I connect to VPN?",
            "kb_docs": [document],
            "trace": [],
        }

        with patch.object(workflow, "llm", return_value=model):
            failed = workflow.generate_from_kb(state)
            state.update(failed)
            state["web_results"] = "Official instructions found."
            state["citations"] = [{"title": "Vendor guide", "url": "https://example.com"}]
            fallback = workflow.generate_from_web(state)

        self.assertTrue(state["model_failed"])
        self.assertEqual(fallback["source_used"], "insufficient_evidence")
        self.assertEqual(fallback["citations"], state["citations"])
        model.invoke.assert_called_once()

    def test_search_failure_is_safe_and_has_valid_node_result(self):
        state = {
            "request_id": "test-search-failure",
            "current_query": "VPN setup",
            "trace": [],
        }

        with patch.object(workflow, "web_search_tool", side_effect=RuntimeError("offline")):
            result = workflow.search_web(state)

        self.assertEqual(result["web_results"], "")
        self.assertEqual(result["citations"], [])
        self.assertEqual(workflow.after_web({**state, **result}), "insufficient")

    def test_tavily_uses_configured_connect_and_read_timeouts(self):
        response = SimpleNamespace(raise_for_status=Mock(), json=Mock(return_value={"results": []}))

        with patch.object(workflow.requests, "post", return_value=response) as post:
            result = workflow.BoundedTavilySearch("test-key", 12).invoke(
                {"query": "VPN setup"}
            )

        self.assertEqual(result, {"results": []})
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 12))
        self.assertNotIn("test-key", str(post.call_args.kwargs["json"]))


if __name__ == "__main__":
    unittest.main()
