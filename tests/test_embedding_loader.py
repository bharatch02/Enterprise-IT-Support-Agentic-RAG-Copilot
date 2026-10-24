import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

from app.rag.embedding_loader import LazyInitializer, resolve_embedding_model_path


class LocalCacheMiss(Exception):
    pass


class EmbeddingLoaderTests(unittest.TestCase):
    def test_cached_model_is_used_without_hub_network_check(self):
        download = Mock(return_value="/cached/model")

        path = resolve_embedding_model_path(
            repository="sentence-transformers/all-MiniLM-L6-v2",
            configured_path="",
            local_files_only=True,
            token=None,
            etag_timeout=10,
            snapshot_download=download,
            cache_miss_error=LocalCacheMiss,
        )

        self.assertEqual(path, "/cached/model")
        download.assert_called_once_with(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            local_files_only=True,
        )

    def test_configured_model_path_bypasses_hub(self):
        download = Mock()
        model_dir = str(Path(__file__).parent)
        path = resolve_embedding_model_path(
            repository="sentence-transformers/all-MiniLM-L6-v2",
            configured_path=model_dir,
            local_files_only=True,
            token=None,
            etag_timeout=10,
            snapshot_download=download,
            cache_miss_error=LocalCacheMiss,
        )

        self.assertEqual(path, str(Path(model_dir).resolve()))
        download.assert_not_called()

    def test_first_time_download_happens_only_after_cache_miss(self):
        download = Mock(side_effect=[LocalCacheMiss(), "/downloaded/model"])

        path = resolve_embedding_model_path(
            repository="sentence-transformers/all-MiniLM-L6-v2",
            configured_path="",
            local_files_only=False,
            token="secret-token",
            etag_timeout=7,
            snapshot_download=download,
            cache_miss_error=LocalCacheMiss,
        )

        self.assertEqual(path, "/downloaded/model")
        self.assertEqual(download.call_count, 2)
        self.assertTrue(download.call_args_list[0].kwargs["local_files_only"])
        self.assertFalse(download.call_args_list[1].kwargs["local_files_only"])
        self.assertEqual(download.call_args_list[1].kwargs["etag_timeout"], 7)
        self.assertEqual(download.call_args_list[1].kwargs["token"], "secret-token")

    def test_offline_cache_miss_returns_clear_failure_without_network(self):
        download = Mock(side_effect=LocalCacheMiss())

        with self.assertRaisesRegex(RuntimeError, "not present locally"):
            resolve_embedding_model_path(
                repository="sentence-transformers/all-MiniLM-L6-v2",
                configured_path="",
                local_files_only=True,
                token=None,
                etag_timeout=10,
                snapshot_download=download,
                cache_miss_error=LocalCacheMiss,
            )

        download.assert_called_once()

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
