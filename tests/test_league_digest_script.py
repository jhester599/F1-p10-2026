import importlib.util
from pathlib import Path

from src.league_digest import DigestCommentary, LeagueDigest, PlayerStanding


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "67_send_league_digest.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("league_digest_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_digest_recipients_prefers_test_override() -> None:
    script = load_script_module()

    assert script.resolve_digest_recipients(
        participant_emails=["eric@example.com", "nick@example.com"],
        test_recipient="jeffrey.r.hester@gmail.com",
    ) == ["jeffrey.r.hester@gmail.com"]


def test_resolve_digest_recipients_uses_participants_without_override() -> None:
    script = load_script_module()

    assert script.resolve_digest_recipients(
        participant_emails=["eric@example.com", "nick@example.com"],
        test_recipient=None,
    ) == ["eric@example.com", "nick@example.com"]


def test_write_dry_run_artifacts_creates_html_and_charts(tmp_path: Path) -> None:
    script = load_script_module()
    digest = LeagueDigest(
        league_label="Test League",
        sheet_round=7,
        race_name="Barcelona Grand Prix",
        p10_driver="Franco Colapinto",
        standings=[PlayerStanding("Eric", 100, 18, 1)],
        round_labels=["R7"],
        series={"Eric": [100]},
        recipients=["jeffrey.r.hester@gmail.com"],
    )
    commentary = DigestCommentary(
        headline="Headline",
        race_summary="Race summary",
        league_commentary="League commentary",
        notable_movements=["Eric led the scoring."],
    )

    artifacts = script.write_dry_run_artifacts(
        digest=digest,
        commentary=commentary,
        articles=[],
        output_dir=tmp_path,
    )

    assert artifacts["html"].exists()
    assert artifacts["standings"].exists()
    assert artifacts["series"].exists()
    assert "SPOILER WARNING" in artifacts["html"].read_text(encoding="utf-8")
