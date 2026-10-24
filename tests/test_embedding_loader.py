import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.rag.embedding_loader import (
    FastEmbedAdapter,
    LazyInitializer,
    normalize_hf_timeout,
)


class EmbeddingLoaderTests(unittest.TestCase):
    def test_cached_mode_passes_local_only_and_low_memory_settings(self):
        model = SimpleNamespace(embedding_size=384)
        factory = Mock(return_value=model)

        adapter = FastEmbedAdapter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=str(Path(__file__).parent),
            local_files_only=True,
            model_factory=factory,
        )

        self.assertIsNotNone(adapter)
        self.assertEqual(factory.call_args.kwargs["local_files_only"], True)
        self.assertEqual(factory.call_args.kwargs["threads"], 1)

    def test_first_time_mode_can_download_after_local_cache_miss(self):
        model = SimpleNamespace(embedding_size=384)
        factory = Mock(return_value=model)

        FastEmbedAdapter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=str(Path(__file__).parent),
            local_files_only=False,
            model_factory=factory,
        )

        self.assertEqual(factory.call_args.kwargs["local_files_only"], False)

    def test_missing_offline_cache_fails_without_initializing_model(self):
        factory = Mock()
        missing_cache = str(Path(__file__).parent / "missing-fastembed-cache")

        with self.assertRaisesRegex(RuntimeError, "model cache is not present"):
            FastEmbedAdapter(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=missing_cache,
                local_files_only=True,
                model_factory=factory,
            )

        factory.assert_not_called()

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "expected 384"):
            FastEmbedAdapter(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=str(Path(__file__).parent),
                local_files_only=True,
                model_factory=Mock(return_value=SimpleNamespace(embedding_size=768)),
            )

    def test_adapter_uses_small_batches_and_query_encoder(self):
        vectors = [SimpleNamespace(tolist=lambda: [0.1, 0.2, 0.3])]
        model = SimpleNamespace(
            embedding_size=384,
            embed=Mock(return_value=iter(vectors)),
            query_embed=Mock(return_value=iter(vectors)),
        )
        adapter = FastEmbedAdapter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=str(Path(__file__).parent),
            local_files_only=True,
            model_factory=Mock(return_value=model),
        )

        self.assertEqual(adapter.embed_documents(["VPN setup"]), [[0.1, 0.2, 0.3]])
        self.assertEqual(adapter.embed_query("VPN setup"), [0.1, 0.2, 0.3])
        self.assertEqual(model.embed.call_args.kwargs["batch_size"], 8)
        self.assertEqual(model.query_embed.call_args.kwargs["batch_size"], 1)

    def test_hub_timeout_environment_is_normalized_to_integer_strings(self):
        with patch.dict(os.environ, {"HF_HUB_ETAG_TIMEOUT": "10.0"}, clear=False):
            self.assertEqual(normalize_hf_timeout("HF_HUB_ETAG_TIMEOUT", 8), "10")
            self.assertEqual(os.environ["HF_HUB_ETAG_TIMEOUT"], "10")

        with patch.dict(os.environ, {"HF_HUB_DOWNLOAD_TIMEOUT": "bad"}, clear=False):
            self.assertEqual(normalize_hf_timeout("HF_HUB_DOWNLOAD_TIMEOUT", 30), "30")
            self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "30")

    def test_failed_initialization_is_not_cached(self):
        initializer = LazyInitializer()
        factory = Mock(side_effect=[RuntimeError("load failed"), "ready"])

        with self.assertRaisesRegex(RuntimeError, "load failed"):
            initializer.get(factory)

        self.assertEqual(initializer.get(factory), "ready")
        self.assertEqual(factory.call_count, 2)

    def test_concurrent_first_callers_initialize_once(self):
        initializer = LazyInitializer()
        calls = 0
        calls_lock = threading.Lock()
        ready = object()

        def factory():
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.03)
            return ready

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: initializer.get(factory), range(16)))

        self.assertEqual(calls, 1)
        self.assertTrue(all(result is ready for result in results))


if __name__ == "__main__":
    unittest.main()
