import json

from src.data_fetch import F1Fetcher


def test_cache_only_missing_entry_does_not_hit_network(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("F1_FETCH_CACHE_ONLY", "1")
    fetcher = F1Fetcher(cache_dir=tmp_path)

    def fail_get(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("cache-only mode should not call the live API")

    monkeypatch.setattr(fetcher._session, "get", fail_get)

    assert fetcher._get("2026/7/qualifying") is None


def test_get_reads_cached_json_before_cache_only_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("F1_FETCH_CACHE_ONLY", "1")
    fetcher = F1Fetcher(cache_dir=tmp_path)
    expected = {"MRData": {"RaceTable": {"Races": [{"round": "7"}]}}}
    fetcher._cache_path("2026/7/qualifying").write_text(json.dumps(expected), encoding="utf-8")

    assert fetcher._get("2026/7/qualifying") == expected
