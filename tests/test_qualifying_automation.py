from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "65_run_qualifying_automation.py"


def load_automation_module():
    spec = importlib.util.spec_from_file_location("qualifying_automation", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_round_with_missing_qualifying_data_skips_without_running_models(monkeypatch, tmp_path):
    automation = load_automation_module()

    class Fetcher:
        def schedule(self, year):
            assert year == 2026
            return [{"round": "7", "raceName": "Barcelona Grand Prix", "date": "2026-06-14"}]

        def qualifying(self, year, rnd):
            assert (year, rnd) == (2026, 7)
            return []

    def fail_if_called(*args, **kwargs):
        raise AssertionError("model pipeline should not run until qualifying rows exist")

    output_path = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setattr(automation, "F1Fetcher", Fetcher)
    monkeypatch.setattr(automation, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(automation, "ensure_processed_data", fail_if_called)
    monkeypatch.setattr(automation, "ensure_models", fail_if_called)
    monkeypatch.setattr(sys, "argv", ["65_run_qualifying_automation.py", "--year", "2026", "--round", "7"])

    automation.main()

    output = output_path.read_text(encoding="utf-8")
    assert "new_prediction=false" in output
    assert "No qualifying results found yet for 2026 R07" in output
