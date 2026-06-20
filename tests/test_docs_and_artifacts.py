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


def test_candidate_replay_gate_artifacts_are_documented_and_wired() -> None:
    assert (ROOT / "results" / "scorecards" / "candidate_replay_gates.json").exists()
    assert (ROOT / "results" / "scorecards" / "candidate_replay_gates.md").exists()
    assert (ROOT / "results" / "candidate_a" / "candidate_a_rolling_cv_replay.json").exists()
    assert (ROOT / "results" / "candidate_b" / "candidate_b_rolling_cv_replay.json").exists()
    assert (ROOT / "results" / "candidate_a" / "candidate_a_live_2026_replay.json").exists()
    assert (ROOT / "results" / "candidate_b" / "candidate_b_live_2026_replay.json").exists()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "scripts/97_candidate_replay_gates.py" in readme
    assert "scripts/98_candidate_rolling_cv_replay.py" in readme
    assert "results/scorecards/candidate_replay_gates.{json,md}" in readme
    assert "candidate_a_rolling_cv_replay.json" in readme
    assert "candidate_b_rolling_cv_replay.json" in readme
    assert "candidate_a_live_2026_replay.json" in readme

    workflow = (ROOT / ".github" / "workflows" / "repo-sanity.yml").read_text(
        encoding="utf-8"
    )
    assert "scripts/97_candidate_replay_gates.py" in workflow
    assert "scripts/98_candidate_rolling_cv_replay.py" in workflow
    assert "python scripts/97_candidate_replay_gates.py" in workflow


def test_repo_sanity_uses_pinned_dependency_snapshot() -> None:
    workflow = (ROOT / ".github" / "workflows" / "repo-sanity.yml").read_text(
        encoding="utf-8"
    )

    assert "pip install -r requirements-ci.txt pytest" in workflow


def test_results_automation_is_documented_and_wired() -> None:
    workflow = (ROOT / ".github" / "workflows" / "results-automation-2026.yml").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    automation_docs = (ROOT / "docs" / "GITHUB_ACTIONS_AUTOMATION.md").read_text(
        encoding="utf-8"
    )

    assert "scripts/66_update_results_automation.py" in workflow
    assert "GOOGLE_SERVICE_ACCOUNT_JSON" in workflow
    assert "RESULTS_SPREADSHEET_IDS" in workflow
    assert "RESULTS_SPREADSHEET_ID" in workflow
    assert "RESULTS_EMAIL_TO" in workflow
    assert "scripts/66_update_results_automation.py" in readme
    assert "RESULTS_SPREADSHEET_IDS" in readme
    assert "GOOGLE_SERVICE_ACCOUNT_JSON" in automation_docs
    assert "RESULTS_SPREADSHEET_IDS" in automation_docs
    assert "RESULTS_EMAIL_TO" in automation_docs


def test_results_automation_uses_race_result_probe_window() -> None:
    workflow = (ROOT / ".github" / "workflows" / "results-automation-2026.yml").read_text(
        encoding="utf-8"
    )
    automation_docs = (ROOT / "docs" / "GITHUB_ACTIONS_AUTOMATION.md").read_text(
        encoding="utf-8"
    )

    assert "race_start + 120 minutes" in workflow
    assert "race_start + 240 minutes" in workflow
    assert "`+120..+240`" in automation_docs
    assert "window-minutes 120" in workflow
    assert "offset-minutes 120" in workflow
    assert "# R07 Barcelona +120m" in workflow
    assert "# R07 Barcelona +240m" in workflow
    assert "steps.update_results.outputs.updated_rows != ''" in workflow
    assert '30 22 * 3-12 0,1' not in workflow


def test_league_digest_workflow_is_documented_and_wired() -> None:
    workflow = (ROOT / ".github" / "workflows" / "league-digest-2026.yml").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    automation_docs = (ROOT / "docs" / "GITHUB_ACTIONS_AUTOMATION.md").read_text(
        encoding="utf-8"
    )

    assert "scripts/67_send_league_digest.py" in workflow
    assert "GEMINI_API_KEY" in workflow
    assert "RESULTS_DIGEST_TEST_RECIPIENT" in workflow
    assert "RESULTS_DIGEST_SEND_TO_PARTICIPANTS" in workflow
    assert '[ -n "$SMTP_USERNAME" ]' in workflow
    assert '[ -n "$RESULTS_EMAIL_FROM" ]' not in workflow
    assert "SPOILER" in workflow
    assert "scripts/67_send_league_digest.py" in readme
    assert "GEMINI_API_KEY" in automation_docs
    assert "RESULTS_DIGEST_TEST_RECIPIENT" in automation_docs
    assert "RESULTS_DIGEST_SEND_TO_PARTICIPANTS" in automation_docs
    assert "Form Responses 1" in automation_docs
