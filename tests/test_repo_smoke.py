from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_layout_exists() -> None:
    assert (ROOT / "README.md").exists()
    assert (ROOT / "config.py").exists()
    assert (ROOT / "src").is_dir()
    assert (ROOT / "scripts").is_dir()
    assert (ROOT / ".github" / "workflows" / "qualifying-predictions-2026.yml").exists()


def test_archived_v6x_kept_outside_active_tests() -> None:
    # Archive mirror stays in-repo for historical reference but should not
    # be part of active pytest collection.
    assert (ROOT / "v6x").is_dir()
