from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "qualifying-predictions-2026.yml"
KEEPALIVE_WORKFLOW = ROOT / ".github" / "workflows" / "cache-keepalive.yml"


def test_workflow_stages_all_prediction_outputs_before_commit() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "git add --" in workflow
    assert '"${{ steps.run_prediction.outputs.prediction_csv }}"' in workflow
    assert '"${{ steps.run_prediction.outputs.report_path }}"' in workflow
    assert '"${{ steps.run_prediction.outputs.season_log }}"' in workflow


def test_scheduled_retry_gate_runs_from_90_to_240_minutes_after_qualifying() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "offset_minutes = 90" in workflow
    assert "window_minutes = 150" in workflow
    assert "--offset-minutes 90" in workflow
    assert "--window-minutes 150" in workflow


def test_model_and_processed_caches_require_exact_keys() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    processed_restore = re.search(
        r"name: Restore processed-data cache(?P<body>.*?)- name: Restore raw-cache extraction",
        workflow,
        flags=re.S,
    )
    model_restore = re.search(
        r"name: Restore model-artifact cache(?P<body>.*?)- name: Install dependencies",
        workflow,
        flags=re.S,
    )

    assert processed_restore is not None
    assert model_restore is not None
    assert "restore-keys:" not in processed_restore.group("body")
    assert "restore-keys:" not in model_restore.group("body")


def test_prediction_workflow_does_not_commit_ignored_model_binaries() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "models/*.joblib" not in workflow
    assert "Commit rebuilt processed artifacts" in workflow


def test_cache_keepalive_uses_exact_behavior_keys() -> None:
    workflow = KEEPALIVE_WORKFLOW.read_text(encoding="utf-8")

    assert "keepalive-${{ github.run_number }}" not in workflow
    assert "restore-keys:" not in workflow
    assert "processed-2026-${{ hashFiles('config.py', 'src/feature_engineering.py', 'scripts/02_build_dataset.py', 'requirements.txt') }}" in workflow
    assert "models-2026-${{ hashFiles('config.py', 'src/models.py', 'scripts/03_train_models.py', 'requirements.txt') }}" in workflow
