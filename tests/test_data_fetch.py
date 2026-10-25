import json
from datetime import datetime, timezone

from src.data_fetch import F1Fetcher, _status_is_finish, grid_from_results


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        import requests

        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))

    def json(self) -> dict:
        return self._payload


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


def test_lapped_status_counts_as_classified_finish() -> None:
    assert _status_is_finish("Finished")
    assert _status_is_finish("Lapped")
    assert _status_is_finish("+2 Laps")
    assert not _status_is_finish("Retired")
    assert not _status_is_finish("Did not start")


def test_empty_current_season_response_is_not_cached(tmp_path, monkeypatch) -> None:
    fetcher = F1Fetcher(cache_dir=tmp_path)
    year = datetime.now(timezone.utc).year
    empty = {"MRData": {"total": "0", "RaceTable": {"Races": []}}}
    monkeypatch.setattr(fetcher._session, "get", lambda *a, **k: _Response(200, empty))
    monkeypatch.setattr("src.data_fetch.time.sleep", lambda s: None)

    assert fetcher._get(f"{year}/18/qualifying") == empty
    assert not fetcher._cache_path(f"{year}/18/qualifying").exists()


def test_missing_endpoint_is_cached_as_empty(tmp_path, monkeypatch) -> None:
    fetcher = F1Fetcher(cache_dir=tmp_path)
    monkeypatch.setattr(fetcher._session, "get", lambda *a, **k: _Response(404, {}))
    monkeypatch.setattr("src.data_fetch.time.sleep", lambda s: None)

    assert fetcher._get("2026/3/practice/2") is None
    assert fetcher.fp2_classification(2026, 3) == []

    def fail_get(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("cached 404 should not be re-requested")

    monkeypatch.setattr(fetcher._session, "get", fail_get)
    assert fetcher.fp2_classification(2026, 3) == []


def test_cache_only_before_blocks_live_requests_for_past_seasons(tmp_path, monkeypatch) -> None:
    fetcher = F1Fetcher(cache_dir=tmp_path, cache_only_before=2026)
    calls = []
    monkeypatch.setattr(fetcher._session, "get", lambda url, **k: calls.append(url) or _Response(404, {}))
    monkeypatch.setattr("src.data_fetch.time.sleep", lambda s: None)

    assert fetcher._get("2025/1/results") is None
    assert calls == []
    fetcher._get("2026/1/results")
    assert len(calls) == 1


def test_grid_from_results_sends_pit_lane_starters_to_the_back() -> None:
    results = [
        {"Driver": {"driverId": "a"}, "grid": "1"},
        {"Driver": {"driverId": "b"}, "grid": "2"},
        {"Driver": {"driverId": "c"}, "grid": "0"},
    ]

    assert grid_from_results(results) == {"a": 1, "b": 2, "c": 3}


def test_qualifying_falls_back_to_openf1_when_jolpica_is_empty(tmp_path, monkeypatch) -> None:
    class FakeOpenF1:
        def qualifying_session(self, year, race_date):
            assert (year, race_date) == (2026, "2026-10-25")
            return {"session_key": 99, "date_start": "2026-10-24T21:00:00+00:00"}

        def session_results(self, key):
            return [
                {"position": 2, "driver_number": 3, "duration": [95.2, 94.4, 94.2]},
                {"position": 1, "driver_number": 12, "duration": [95.3, 94.2, 94.0]},
            ]

        def drivers(self, key):
            return [
                {"driver_number": 3, "team_name": "Red Bull Racing", "last_name": "Verstappen"},
                {"driver_number": 12, "team_name": "Mercedes", "last_name": "Antonelli"},
            ]

        def starting_grid(self, key):
            return [{"driver_number": 12, "position": 1}, {"driver_number": 3, "position": 5}]

    fetcher = F1Fetcher(cache_dir=tmp_path, openf1=FakeOpenF1())
    jolpica = {
        "2026/18/qualifying": [],
        "2026/18/results": [],
        "2026/17/results": [
            {"number": "3", "Driver": {"driverId": "max_verstappen"}, "Constructor": {"constructorId": "red_bull"}},
            {"number": "12", "Driver": {"driverId": "antonelli"}, "Constructor": {"constructorId": "mercedes"}},
        ],
    }

    def fake_mrdata(path, *keys, use_cache=True):
        if path == "2026":
            return [{"round": "17", "date": "2026-10-11"}, {"round": "18", "date": "2026-10-25"}]
        if path == "2025":
            return []
        rows = jolpica.get(path, [])
        key = "QualifyingResults" if path.endswith("qualifying") else "Results"
        return [{key: rows}] if rows else []

    monkeypatch.setattr(fetcher, "_mrdata", fake_mrdata)

    rows = fetcher.qualifying(2026, 18)
    assert [r["Driver"]["driverId"] for r in rows] == ["antonelli", "max_verstappen"]
    assert rows[0]["Constructor"]["constructorId"] == "mercedes"
    assert rows[0]["position"] == "1"
    assert rows[0]["Q3"] == "1:34.000"
    assert fetcher.starting_grid(2026, 18) == {"antonelli": 1, "max_verstappen": 5}


def test_stale_cached_empty_current_season_response_is_refetched(tmp_path, monkeypatch) -> None:
    fetcher = F1Fetcher(cache_dir=tmp_path)
    year = datetime.now(timezone.utc).year
    path = f"{year}/18/qualifying"
    fetcher._cache_path(path).write_text(
        json.dumps({"MRData": {"total": "0", "RaceTable": {"Races": []}}}), encoding="utf-8"
    )
    published = {"MRData": {"total": "22", "RaceTable": {"Races": [{"QualifyingResults": [{}]}]}}}
    monkeypatch.setattr(fetcher._session, "get", lambda *a, **k: _Response(200, published))
    monkeypatch.setattr("src.data_fetch.time.sleep", lambda s: None)

    assert fetcher._get(path) == published
