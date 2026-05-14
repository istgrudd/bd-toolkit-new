from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import session_persistence as sp


@pytest.fixture(autouse=True)
def temp_session_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "SESSION_STORAGE_DIR", str(tmp_path))
    yield tmp_path


def test_session_id_generation_format():
    session_id = sp.generate_session_id()

    assert sp.SESSION_ID_PATTERN.fullmatch(session_id)
    assert session_id.startswith("BDT-")


def test_sanitize_valid_and_invalid_session_ids():
    assert sp.sanitize_session_id(" bdt-8f2a9c ") == "BDT-8F2A9C"

    with pytest.raises(ValueError):
        sp.sanitize_session_id("../BDT-8F2A9C")

    with pytest.raises(ValueError):
        sp.sanitize_session_id("BDT-TOO-LONG")


def test_metadata_expiration_logic():
    now = datetime(2026, 5, 14, 10, 0, tzinfo=timezone.utc)
    metadata = sp.build_metadata("BDT-ABC123", now=now)

    assert not sp.is_session_expired(metadata, now=now + timedelta(minutes=119))
    assert sp.is_session_expired(metadata, now=now + timedelta(minutes=120))


def test_save_and_load_workflow_state():
    state = {
        "anonymous_session_id": "BDT-ABC123",
        "df": pd.DataFrame({"a": [1, 2], "target": [0, 1]}),
        "target_column": "target",
        "task_type": "Classification",
        "temporary_widget_key": "do-not-save",
    }

    metadata = sp.save_workflow_state(state, "BDT-ABC123")
    target_state = {"anonymous_session_id": "BDT-ABC123", "df": pd.DataFrame({"old": [9]})}
    loaded_state, loaded_metadata = sp.load_workflow_state(target_state, "BDT-ABC123")

    assert metadata["session_id"] == "BDT-ABC123"
    assert loaded_metadata["session_id"] == "BDT-ABC123"
    assert loaded_state["target_column"] == "target"
    pd.testing.assert_frame_equal(target_state["df"], state["df"])
    assert "temporary_widget_key" not in loaded_state
    assert "temporary_widget_key" not in target_state


def test_cleanup_expired_sessions_removes_only_expired():
    now = datetime(2026, 5, 14, 10, 0, tzinfo=timezone.utc)
    sp.save_workflow_state({"df": pd.DataFrame({"a": [1]})}, "BDT-OLD123")
    sp.save_workflow_state({"df": pd.DataFrame({"a": [2]})}, "BDT-NEW123")
    sp._write_metadata("BDT-OLD123", sp.build_metadata("BDT-OLD123", now=now - timedelta(hours=3)))
    sp._write_metadata("BDT-NEW123", sp.build_metadata("BDT-NEW123", now=now))

    removed = sp.cleanup_expired_sessions(now=now)

    assert removed == 1
    assert not sp.get_session_dir("BDT-OLD123").exists()
    assert sp.get_session_dir("BDT-NEW123").exists()


def test_load_rejects_expired_session():
    now = datetime(2026, 5, 14, 10, 0, tzinfo=timezone.utc)
    sp.save_workflow_state({"df": pd.DataFrame({"a": [1]})}, "BDT-OLD123")
    sp._write_metadata("BDT-OLD123", sp.build_metadata("BDT-OLD123", now=now - timedelta(hours=3)))

    with pytest.raises(sp.SessionExpiredError):
        sp.load_workflow_state({}, "BDT-OLD123")

    assert not sp.get_session_dir("BDT-OLD123").exists()


def test_path_traversal_is_rejected():
    with pytest.raises(ValueError):
        sp.get_session_dir("BDT-ABC123/../../outside")

    with pytest.raises(ValueError):
        sp.load_workflow_state({}, "..\\BDT-ABC123")


def test_only_allowlisted_keys_are_saved():
    state = {
        "df": pd.DataFrame({"a": [1]}),
        "target_column": "a",
        "evaluation_plots_temp": {"plot.png": b"bytes"},
        "submission_csv": b"a\n1\n",
        "submission_upload": object(),
    }

    saved = sp.get_saveable_state(state)

    assert "df" in saved
    assert "target_column" in saved
    assert "submission_csv" in saved
    assert "evaluation_plots_temp" not in saved
    assert "submission_upload" not in saved
