from pathlib import Path

from campro.diagnostics.run_metadata import RUN_ID, log_run_metadata


def test_log_run_metadata_creates_file(tmp_path, monkeypatch):
    # Write into a temp folder
    folder = tmp_path / "runs"
    path = log_run_metadata({"status": "ok"}, folder=str(folder))
    assert Path(path).exists()
    assert RUN_ID in Path(path).name

