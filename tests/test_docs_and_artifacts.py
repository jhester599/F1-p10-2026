from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_automation_docs_describe_current_retry_window() -> None:
    docs = "\n".join(
        [
            (ROOT / "README.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "GITHUB_ACTIONS_AUTOMATION.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "REPO_DECISIONS_2026-03-25.md").read_text(encoding="utf-8"),
        ]
    )

    assert "qualifying_time + 90 minutes" in docs
    assert "qualifying_time + 240 minutes" in docs
    assert "`+90..+240`" in docs
    assert "qualifying_time + 60 minutes` to `+90 minutes" not in docs
    assert "run window guard (`+60m` to `+90m`)" not in docs


def test_workflow_skip_message_uses_current_retry_window() -> None:
    workflow = (ROOT / ".github" / "workflows" / "qualifying-predictions-2026.yml").read_text(
        encoding="utf-8"
    )

    assert "Not in qualifying+90..+240 window." in workflow
    assert "Not in qualifying+60..+90 window." not in workflow


def test_round_7_prediction_documentation_is_backfilled() -> None:
    assert (ROOT / "results" / "prediction_2026_R07.csv").exists()
    report = ROOT / "results" / "prediction_reports" / "2026_R07_barcelona-grand-prix.md"
    assert report.exists()
    report_text = report.read_text(encoding="utf-8")
    assert "\nGenerated: 2026-06-13 21:30 UTC\n" in report_text
    assert "\n### Consensus\n" in report_text
    assert "\n        ###" not in report_text

    season_log = ROOT / "results" / "automated_predictions_2026.md"
    assert season_log.exists()
    assert "Round 07 - Barcelona Grand Prix" in season_log.read_text(encoding="utf-8")


def test_artifact_policy_is_documented_and_linked() -> None:
    policy = ROOT / "docs" / "ARTIFACT_AND_ARCHIVE_POLICY.md"
    assert policy.exists()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/ARTIFACT_AND_ARCHIVE_POLICY.md" in readme

    ignore_rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "models/" in ignore_rules
    assert "data/processed/ contains small committed feature snapshots" in ignore_rules


def test_repo_sanity_uses_pinned_dependency_snapshot() -> None:
    workflow = (ROOT / ".github" / "workflows" / "repo-sanity.yml").read_text(
        encoding="utf-8"
    )

    assert "pip install -r requirements-ci.txt pytest" in workflow
