from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "qualifying-predictions-2026.yml"


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
