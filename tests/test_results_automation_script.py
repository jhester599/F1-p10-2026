import importlib.util
from pathlib import Path

from src.results_automation import RaceResult


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "66_update_results_automation.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("results_update_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_next_series_formula_updates_detects_second_league_width() -> None:
    script = load_script_module()
    results_values = [
        [
            "Race",
            "Steve Broz",
            "Craig Ewing",
            "Jeff Ewing",
            "Tricia Griffith",
            "Jeff Hester",
            "Kaitlin Marvin",
            "Ken Rolsen",
            "Rama Panguluri",
            "Mark Thomas",
            "Sean Allen",
        ],
        ["Total", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ["R1", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ]

    updates = script.next_series_formula_updates(results_values, max_round=1)

    assert updates == [
        {
            "range": "'results'!M3:W3",
            "values": [
                [
                    "=A3",
                    "=N2+B3",
                    "=O2+C3",
                    "=P2+D3",
                    "=Q2+E3",
                    "=R2+F3",
                    "=S2+G3",
                    "=T2+H3",
                    "=U2+I3",
                    "=V2+J3",
                    "=W2+K3",
                ]
            ],
        }
    ]


def test_format_summary_ignores_time_series_headers_in_totals() -> None:
    script = load_script_module()
    summary = script.format_summary(
        year=2026,
        league_label="league 2 spreadsheet ...hgjZilc",
        race_results=[RaceResult(round_number=7, driver_code="COL", position=10)],
        updates=[],
        skipped=[],
        totals=[
            [
                "Race",
                "Steve Broz",
                "Craig Ewing",
                "Jeff Ewing",
                "",
                "",
                "Steve Broz",
            ],
            ["Total", 10, 20, 30, "", "", "duplicate header"],
        ],
        series_updates=[],
        dry_run=True,
    )

    assert "- Steve Broz: 10" in summary
    assert "- Craig Ewing: 20" in summary
    assert "- Jeff Ewing: 30" in summary
    assert "duplicate header" not in summary
