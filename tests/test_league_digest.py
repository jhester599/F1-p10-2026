from pathlib import Path

import pandas as pd

from src.league_digest import (
    ArticleSource,
    DigestCommentary,
    LeagueDigest,
    PlayerStanding,
    build_fallback_commentary,
    build_spoiler_subject,
    extract_participant_emails,
    parse_league_digest,
    render_digest_html,
    render_league_charts,
    request_gemini_commentary,
)


def sample_results_values() -> list[list[object]]:
    return [
        ["Race", "Eric", "Jeff", "Luke", "Matt", "Nick", "Tim", "Jim", "", "Race", "Eric", "Jeff", "Luke", "Matt", "Nick", "Tim", "Jim"],
        ["Total", 118, 67, 98, 93, 49, 44, 54, "", "Total", 118, 67, 98, 93, 49, 44, 54],
        ["R1", 18, 10, 25, 15, 8, 4, 6, "", "R1", 18, 10, 25, 15, 8, 4, 6],
        ["R2", 25, 18, 15, 12, 10, 8, 6, "", "R2", 43, 28, 40, 27, 18, 12, 12],
        ["R3", 10, 25, 12, 18, 15, 6, 8, "", "R3", 53, 53, 52, 45, 33, 18, 20],
        ["R4", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        ["R5", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        ["R6", 12, 8, 18, 25, 6, 15, 10, "", "R6", 65, 61, 70, 70, 39, 33, 30],
        ["R7", 18, 6, 10, 15, 8, 11, 24, "", "R7", 83, 67, 80, 85, 47, 44, 54],
    ]


def test_extract_participant_emails_deduplicates_form_responses() -> None:
    form_values = [
        ["Timestamp", "Email", "Race", "Driver"],
        ["now", "Eric@Example.com ", "R7", "COL"],
        ["now", "nick@example.com", "R7", "LAW"],
        ["now", "eric@example.com", "R8", "HAM"],
        ["now", "", "R8", "NOR"],
        ["now", "not-an-email", "R8", "PIA"],
    ]

    assert extract_participant_emails(form_values) == ["eric@example.com", "nick@example.com"]


def test_parse_league_digest_ranks_totals_and_race_points() -> None:
    digest = parse_league_digest(
        results_values=sample_results_values(),
        sheet_round=7,
        race_name="Barcelona Grand Prix",
        p10_driver="Franco Colapinto",
    )

    assert digest.race_name == "Barcelona Grand Prix"
    assert [row.name for row in digest.standings[:3]] == ["Eric", "Luke", "Matt"]
    assert digest.standings[0].total_points == 118
    assert digest.standings[0].race_points == 18
    assert digest.standings[-1].name == "Tim"
    assert digest.standings[-1].race_points == 11
    assert digest.round_labels[-2:] == ["R6", "R7"]
    assert digest.series["Eric"][-1] == 83


def test_build_spoiler_subject_and_html_include_warning() -> None:
    digest = LeagueDigest(
        league_label="Eric League",
        sheet_round=7,
        race_name="Barcelona Grand Prix",
        p10_driver="Franco Colapinto",
        standings=[PlayerStanding("Eric", 118, 18, 1, None, None, None)],
        round_labels=["R7"],
        series={"Eric": [118]},
        recipients=["eric@example.com"],
    )
    commentary = DigestCommentary(
        headline="Hamilton finally gets red-car joy",
        race_summary="Hamilton won, Colapinto landed P10, and the league table moved.",
        league_commentary="Eric stretched the lead.",
        notable_movements=["Eric led the round scoring."],
    )

    assert build_spoiler_subject(digest).startswith("[F1 P10 SPOILERS]")
    html = render_digest_html(digest, commentary, [ArticleSource("F1", "Report", "https://example.com")])
    assert "SPOILER WARNING" in html
    assert "Eric stretched the lead." in html
    assert "Franco Colapinto" in html


def test_fallback_commentary_mentions_round_winner_and_leader() -> None:
    digest = parse_league_digest(
        results_values=sample_results_values(),
        sheet_round=7,
        race_name="Barcelona Grand Prix",
        p10_driver="Franco Colapinto",
    )

    commentary = build_fallback_commentary(digest)

    assert "Eric" in commentary.league_commentary
    assert "Jim" in " ".join(commentary.notable_movements)


def test_request_gemini_commentary_uses_client_json() -> None:
    class FakeResponse:
        text = (
            '{"headline":"A lively P10 shake-up","race_summary":"Race prose",'
            '"league_commentary":"League prose","notable_movements":["Move one"]}'
        )

    class FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "gemini-test"
            assert kwargs["config"]["response_mime_type"] == "application/json"
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    digest = parse_league_digest(
        results_values=sample_results_values(),
        sheet_round=7,
        race_name="Barcelona Grand Prix",
        p10_driver="Franco Colapinto",
    )

    commentary = request_gemini_commentary(
        digest,
        [ArticleSource("F1", "Report", "https://example.com", "Hamilton wins")],
        client=FakeClient(),
        model="gemini-test",
    )

    assert commentary.headline == "A lively P10 shake-up"
    assert commentary.notable_movements == ["Move one"]


def test_request_gemini_commentary_uses_default_model_when_env_empty(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeResponse:
        text = (
            '{"headline":"Default model","race_summary":"Race prose",'
            '"league_commentary":"League prose","notable_movements":[]}'
        )

    class FakeModels:
        def generate_content(self, **kwargs):
            captured["model"] = kwargs["model"]
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setenv("RESULTS_DIGEST_MODEL", "")
    digest = parse_league_digest(
        results_values=sample_results_values(),
        sheet_round=7,
        race_name="Barcelona Grand Prix",
        p10_driver="Franco Colapinto",
    )

    request_gemini_commentary(digest, [], client=FakeClient())

    assert captured["model"] == "gemini-2.5-flash-lite"


def test_request_gemini_commentary_falls_back_on_error() -> None:
    class BrokenModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("quota")

    class BrokenClient:
        models = BrokenModels()

    digest = parse_league_digest(
        results_values=sample_results_values(),
        sheet_round=7,
        race_name="Barcelona Grand Prix",
        p10_driver="Franco Colapinto",
    )

    commentary = request_gemini_commentary(digest, [], client=BrokenClient(), model="gemini-test")

    assert commentary.headline.startswith("Barcelona Grand Prix")
    assert commentary.provider_note == "Gemini commentary unavailable; used deterministic fallback."


def test_render_league_charts_writes_png_files(tmp_path: Path) -> None:
    digest = parse_league_digest(
        results_values=sample_results_values(),
        sheet_round=7,
        race_name="Barcelona Grand Prix",
        p10_driver="Franco Colapinto",
    )

    charts = render_league_charts(digest, tmp_path)

    assert charts["standings"].exists()
    assert charts["standings"].stat().st_size > 0
    assert charts["series"].exists()
    assert charts["series"].stat().st_size > 0
