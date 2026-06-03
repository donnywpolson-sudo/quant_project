import polars as pl

from pipeline.session.normalize import load_session_config, normalize_session_df, session_normalize_root
from scripts.session_normalize import main as session_normalize_main


def test_session_normalization_writes_session_id_and_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "data" / "validated" / "ES" / "2024.parquet"
    p.parent.mkdir(parents=True)
    pl.DataFrame({"ts_event": [2, 1, 1], "open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0], "low": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0], "volume": [1, 1, 1]}).write_parquet(p)
    report = session_normalize_root("data/validated", "data/session_normalized")
    out = pl.read_parquet(tmp_path / "data" / "session_normalized" / "ES" / "2024.parquet")
    assert report["status"] == "PASS"
    assert "session_id" in out.columns
    assert "session_date" in out.columns
    assert out["ts_event"].n_unique() == out.height
    assert (tmp_path / "data" / "session_normalized" / "manifest.json").exists()


def test_session_config_lookup_supports_merged_market_file(tmp_path):
    p = tmp_path / "raw_data_validation.yaml"
    p.write_text(
        """
markets:
  ES:
    timezone: America/Chicago
    week_start_day: Sun
    week_start_time: "17:00"
    daily_break:
      start: "16:01"
      end: "17:00"
    session_calendar_accuracy: reviewed
    tick_policy: warn
  CL:
    timezone: America/Chicago
    week_start_day: Sun
    week_start_time: "17:00"
    daily_break:
      start: "16:00"
      end: "17:00"
    session_calendar_accuracy: reviewed
    tick_policy: warn
""",
        encoding="utf-8",
    )
    es = load_session_config(p, "ES")
    cl = load_session_config(p, "CL")
    assert es["timezone"] == "America/Chicago"
    assert es["daily_break"]["start"] == "16:01"
    assert cl["daily_break"]["start"] == "16:00"
    assert "tick_policy" not in es


def test_session_config_lookup_real_config_has_es_cl():
    es = load_session_config("configs/raw_data_validation.yaml", "ES")
    cl = load_session_config("configs/raw_data_validation.yaml", "CL")
    assert es["timezone"] == "America/Chicago"
    assert cl["timezone"] == "America/Chicago"
    assert es["week_start_day"] == "Sun"
    assert cl["week_start_day"] == "Sun"
    assert es["daily_break"]["start"] == "16:01"
    assert cl["daily_break"]["start"] == "16:00"
    assert es["session_calendar_accuracy"] == "reviewed"
    assert cl["session_calendar_accuracy"] == "reviewed"
    assert "tick_policy" not in es
    assert "tick_policy" not in cl


def test_session_normalize_root_records_merged_session_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data/validated/ES/2024.parquet"
    data.parent.mkdir(parents=True)
    pl.DataFrame({"ts_event": [1], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]}).write_parquet(data)
    cfg = tmp_path / "configs/raw_data_validation.yaml"
    cfg.parent.mkdir()
    cfg.write_text("markets:\n  ES:\n    timezone: America/Chicago\n    session_calendar_accuracy: reviewed\n", encoding="utf-8")
    report = session_normalize_root("data/validated", "data/session_normalized", cfg)
    out = pl.read_parquet(tmp_path / "data/session_normalized/ES/2024.parquet")
    assert report["files"][0]["timezone"] == "America/Chicago"
    assert out["session_timezone"].to_list() == ["America/Chicago"]
    assert out["session_calendar_accuracy"].to_list() == ["reviewed"]


def test_session_normalize_script_default_uses_merged_config(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "data/validated/ES/2024.parquet"
    p.parent.mkdir(parents=True)
    pl.DataFrame({"ts_event": [1], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]}).write_parquet(p)
    cfg = tmp_path / "configs/raw_data_validation.yaml"
    cfg.parent.mkdir()
    cfg.write_text("markets:\n  ES:\n    timezone: America/Chicago\n    session_calendar_accuracy: reviewed\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["session_normalize.py"])
    session_normalize_main()
    assert "PASS" in capsys.readouterr().out
    out = pl.read_parquet(tmp_path / "data/session_normalized/ES/2024.parquet")
    assert out["session_timezone"].to_list() == ["America/Chicago"]
