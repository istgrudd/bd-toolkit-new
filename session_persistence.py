from __future__ import annotations

import json
import os
import pickle
import re
import secrets
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, MutableMapping

try:
    from config_limits import (
        MAX_SAVED_SESSION_MB,
        SESSION_ID_LENGTH,
        SESSION_ID_PREFIX,
        SESSION_STORAGE_DIR,
        SESSION_TTL_MINUTES,
    )
except ImportError:  # pragma: no cover - defensive fallback
    SESSION_STORAGE_DIR = "sessions"
    SESSION_TTL_MINUTES = 120
    SESSION_ID_PREFIX = "BDT"
    SESSION_ID_LENGTH = 6
    MAX_SAVED_SESSION_MB = 50


SESSION_ID_PATTERN = re.compile(rf"^{re.escape(SESSION_ID_PREFIX)}-[A-Z0-9]{{6,12}}$")
METADATA_FILENAME = "metadata.json"
WORKFLOW_STATE_FILENAME = "workflow_state.pkl"

SAVEABLE_WORKFLOW_KEYS = {
    "df",
    "upload_hash",
    "target_column",
    "task_type",
    "split_done",
    "pre_X_train",
    "pre_X_test",
    "pre_y_train",
    "pre_y_test",
    "cleansing_steps",
    "imputation_transformers",
    "capper_transformers",
    "dropped_columns",
    "preprocessing_steps",
    "feature_engineering_steps",
    "scalers",
    "encoders",
    "value_mappings",
    "sampled_X_train",
    "sampled_X_test",
    "sampled_y_train",
    "sampled_y_test",
    "sampling_done",
    "train_subset_indices",
    "resampled_pre_X_train",
    "resampled_pre_y_train",
    "resampling_done",
    "last_resample_source",
    "validation_summary",
    "validation_results",
    "training_evaluation_summary",
    "trained_model",
    "trained_preprocessor",
    "model_metadata",
    "submission_predictions",
    "submission_df",
    "submission_csv",
    "submission_full_df",
}


class SessionPersistenceError(Exception):
    """Raised for expected session persistence failures."""


class SessionExpiredError(SessionPersistenceError):
    """Raised when a saved anonymous session is past its TTL."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        return _utc_now()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _as_utc(value).isoformat()


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, str):
        try:
            return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _sessions_root() -> Path:
    root = Path(SESSION_STORAGE_DIR)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    return root.resolve()


def generate_session_id() -> str:
    token = secrets.token_hex(max(1, SESSION_ID_LENGTH // 2 + SESSION_ID_LENGTH % 2)).upper()
    return f"{SESSION_ID_PREFIX}-{token[:SESSION_ID_LENGTH]}"


def ensure_anonymous_session_id(session_state: MutableMapping[str, Any]) -> str:
    existing = session_state.get("anonymous_session_id")
    if existing:
        try:
            session_id = sanitize_session_id(existing)
            session_state["anonymous_session_id"] = session_id
            return session_id
        except ValueError:
            pass

    session_id = generate_session_id()
    session_state["anonymous_session_id"] = session_id
    return session_id


def sanitize_session_id(session_id: str) -> str:
    cleaned = str(session_id or "").strip().upper()
    if not SESSION_ID_PATTERN.fullmatch(cleaned):
        raise ValueError("Session ID tidak valid. Gunakan format seperti BDT-8F2A9C.")
    return cleaned


def get_session_dir(session_id: str) -> Path:
    safe_id = sanitize_session_id(session_id)
    root = _sessions_root()
    candidate = (root / safe_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Session path berada di luar folder sessions.") from exc
    return candidate


def build_metadata(session_id: str, now: datetime | None = None) -> dict[str, Any]:
    safe_id = sanitize_session_id(session_id)
    current = _as_utc(now)
    expires_at = current + timedelta(minutes=SESSION_TTL_MINUTES)
    metadata = {
        "session_id": safe_id,
        "created_at": _iso(current),
        "last_seen_at": _iso(current),
        "saved_at": _iso(current),
        "expires_at": _iso(expires_at),
        "ttl_minutes": SESSION_TTL_MINUTES,
    }
    app_version = os.environ.get("BDT_APP_VERSION")
    if app_version:
        metadata["app_version"] = app_version
    return metadata


def is_session_expired(metadata: dict[str, Any], now: datetime | None = None) -> bool:
    current = _as_utc(now)
    expires_at = _parse_datetime(metadata.get("expires_at"))
    if expires_at is None:
        last_seen = _parse_datetime(metadata.get("last_seen_at"))
        ttl_minutes = int(metadata.get("ttl_minutes") or SESSION_TTL_MINUTES)
        if last_seen is None:
            return True
        expires_at = last_seen + timedelta(minutes=ttl_minutes)
    return current >= expires_at


def _metadata_path(session_id: str) -> Path:
    return get_session_dir(session_id) / METADATA_FILENAME


def _state_path(session_id: str) -> Path:
    return get_session_dir(session_id) / WORKFLOW_STATE_FILENAME


def _read_metadata(session_id: str) -> dict[str, Any] | None:
    path = _metadata_path(session_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_session_metadata(session_id: str) -> dict[str, Any] | None:
    return _read_metadata(session_id)


def _write_metadata(session_id: str, metadata: dict[str, Any]) -> None:
    path = _metadata_path(session_id)
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _touch_metadata(metadata: dict[str, Any], session_id: str, now: datetime | None = None, mark_saved: bool = False) -> dict[str, Any]:
    current = _as_utc(now)
    ttl_minutes = int(metadata.get("ttl_minutes") or SESSION_TTL_MINUTES)
    updated = dict(metadata)
    updated["session_id"] = sanitize_session_id(session_id)
    updated.setdefault("created_at", _iso(current))
    updated["last_seen_at"] = _iso(current)
    updated["expires_at"] = _iso(current + timedelta(minutes=ttl_minutes))
    updated["ttl_minutes"] = ttl_minutes
    if mark_saved:
        updated["saved_at"] = _iso(current)
    return updated


def get_saveable_state(session_state: MutableMapping[str, Any]) -> dict[str, Any]:
    return {
        key: session_state[key]
        for key in sorted(SAVEABLE_WORKFLOW_KEYS)
        if key in session_state and session_state.get(key) is not None
    }


def restore_saveable_state(session_state: MutableMapping[str, Any], loaded_state: dict[str, Any]) -> None:
    for key in SAVEABLE_WORKFLOW_KEYS:
        session_state.pop(key, None)
    for key, value in loaded_state.items():
        if key in SAVEABLE_WORKFLOW_KEYS:
            session_state[key] = value


def save_workflow_state(session_state: MutableMapping[str, Any], session_id: str) -> dict[str, Any]:
    safe_id = sanitize_session_id(session_id)
    session_dir = get_session_dir(safe_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    state = get_saveable_state(session_state)
    state_path = _state_path(safe_id)
    tmp_state_path = state_path.with_suffix(".pkl.tmp")
    with tmp_state_path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    max_bytes = int(MAX_SAVED_SESSION_MB) * 1024 * 1024
    if max_bytes > 0 and tmp_state_path.stat().st_size > max_bytes:
        tmp_state_path.unlink(missing_ok=True)
        raise SessionPersistenceError(f"Saved session terlalu besar. Batas: {MAX_SAVED_SESSION_MB} MB.")
    tmp_state_path.replace(state_path)

    metadata = _read_metadata(safe_id) or build_metadata(safe_id)
    metadata = _touch_metadata(metadata, safe_id, mark_saved=True)
    _write_metadata(safe_id, metadata)
    return metadata


def load_workflow_state(session_state: MutableMapping[str, Any], session_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    safe_id = sanitize_session_id(session_id)
    metadata = _read_metadata(safe_id)
    state_path = _state_path(safe_id)
    if metadata is None or not state_path.exists():
        raise SessionPersistenceError("Progress dengan Session ID tersebut tidak ditemukan.")

    if is_session_expired(metadata):
        clear_saved_session(safe_id)
        raise SessionExpiredError("Progress sudah kedaluwarsa dan telah dihapus.")

    with state_path.open("rb") as handle:
        loaded_state = pickle.load(handle)
    if not isinstance(loaded_state, dict):
        raise SessionPersistenceError("File progress tidak valid.")

    restore_saveable_state(session_state, loaded_state)
    session_state["anonymous_session_id"] = safe_id

    metadata = _touch_metadata(metadata, safe_id, mark_saved=False)
    _write_metadata(safe_id, metadata)
    return loaded_state, metadata


def clear_saved_session(session_id: str) -> bool:
    session_dir = get_session_dir(session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir)
        return True
    return False


def cleanup_expired_sessions(now: datetime | None = None) -> int:
    root = _sessions_root()
    if not root.exists():
        return 0

    removed = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            session_id = sanitize_session_id(child.name)
            metadata = _read_metadata(session_id)
            if metadata is None or is_session_expired(metadata, now=now):
                clear_saved_session(session_id)
                removed += 1
        except Exception:
            continue
    return removed


def autosave_if_possible(session_state: MutableMapping[str, Any]) -> dict[str, Any] | None:
    session_id = session_state.get("anonymous_session_id")
    if not session_id:
        return None
    if not get_saveable_state(session_state):
        return None
    return save_workflow_state(session_state, session_id)
