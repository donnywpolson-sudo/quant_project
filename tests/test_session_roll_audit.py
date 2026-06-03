from pathlib import Path

from pipeline.audit.session_roll import run_session_roll_audit


def test_session_roll_audit_reads_merged_market_session_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "configs/raw_data_validation.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        """
markets:
  ES:
    timezone: America/Chicago
    week_start_day: Sun
    week_start_time: "17:00"
    week_end_day: Fri
    week_end_time: "16:00"
    daily_break:
      start: "16:01"
      end: "17:00"
    tick_policy: warn
  CL:
    timezone: America/Chicago
    week_start_day: Sun
    week_start_time: "17:00"
    week_end_day: Fri
    week_end_time: "16:00"
    daily_break:
      start: "16:00"
      end: "17:00"
    tick_policy: warn
""",
        encoding="utf-8",
    )
    report = run_session_roll_audit(str(cfg), "reports/session_roll.json")
    assert report["status"] == "PASS"
    assert report["session_sections"] == ["CL", "ES"]
    assert Path("reports/session_roll.json").exists()


def test_session_roll_audit_fails_missing_session_field(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "configs/raw_data_validation.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        """
markets:
  ES:
    timezone: America/Chicago
    week_start_day: Sun
    week_start_time: "17:00"
    week_end_day: Fri
    daily_break:
      start: "16:01"
      end: "17:00"
""",
        encoding="utf-8",
    )
    report = run_session_roll_audit(str(cfg), "reports/session_roll.json")
    assert report["status"] == "FAIL"
    assert any("missing week_end_time" in f for f in report["failures"])
