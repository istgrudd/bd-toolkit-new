from __future__ import annotations

from collections.abc import Iterable, MutableMapping

try:
    import streamlit as st
except ImportError:  # pragma: no cover - tests use plain dicts
    st = None


BASE_STATE_KEYS = {
    "page",
    "df",
    "upload_hash",
    "target_column",
    "task_type",
}

GLOBAL_STATE_KEYS = {
    "global_seed",
}

SPLIT_STATE_KEYS = {
    "pre_X_train",
    "pre_X_test",
    "pre_y_train",
    "pre_y_test",
    "split_done",
    "split_fig_row_counts",
    "split_fig_class_prop",
}

CLEANSING_STATE_KEYS = {
    "cleansing_steps",
    "imputation_transformers",
    "capper_transformers",
    "dropped_columns",
}

PREPROCESSING_STATE_KEYS = {
    "preprocessing_steps",
    "scalers",
    "encoders",
    "value_mappings",
    "feature_engineering_steps",
    "enc_cols_cached",
}

SAMPLING_STATE_KEYS = {
    "sampled_X_train",
    "sampled_X_test",
    "sampled_y_train",
    "sampled_y_test",
    "sampling_done",
    "train_subset_indices",
}

RESAMPLING_STATE_KEYS = {
    "resampled_pre_X_train",
    "resampled_pre_y_train",
    "resampling_done",
    "resample_plot_trigger",
    "last_resample_source",
}

VALIDATION_STATE_KEYS = {
    "validation_summary",
    "validation_plots",
    "validation_results",
}

TRAINING_STATE_KEYS = {
    "trained_model",
    "trained_preprocessor",
    "training_evaluation_summary",
    "model_bundle",
    "model_metadata",
    "last_saved_bundle",
}

PLOT_ARTIFACT_STATE_KEYS = {
    "evaluation_plots_temp",
    "evaluation_plots_meta",
    "evaluation_plots_hashes",
    "eda_fig_missing_univariate",
}

EDA_STATE_KEYS = {
    "biv_corr_x",
    "biv_corr_y",
    "cust_corr_x",
    "cust_corr_y",
}

SUBMISSION_STATE_KEYS = {
    "submission_predictions",
    "submission_df",
    "submission_csv",
    "submission_full_df",
}

HEAVY_ARTIFACT_KEYS = (
    SAMPLING_STATE_KEYS
    | RESAMPLING_STATE_KEYS
    | TRAINING_STATE_KEYS
    | PLOT_ARTIFACT_STATE_KEYS
    | SUBMISSION_STATE_KEYS
)

DERIVED_STATE_KEYS = (
    SPLIT_STATE_KEYS
    | CLEANSING_STATE_KEYS
    | PREPROCESSING_STATE_KEYS
    | SAMPLING_STATE_KEYS
    | RESAMPLING_STATE_KEYS
    | VALIDATION_STATE_KEYS
    | TRAINING_STATE_KEYS
    | PLOT_ARTIFACT_STATE_KEYS
    | EDA_STATE_KEYS
    | SUBMISSION_STATE_KEYS
)

ALL_KNOWN_STATE_KEYS = BASE_STATE_KEYS | GLOBAL_STATE_KEYS | DERIVED_STATE_KEYS

DYNAMIC_DERIVED_PREFIXES = (
    "ord_order::",
    "preview_dups_",
)


def _get_state(state=None):
    if state is not None:
        return state
    if st is None:
        raise RuntimeError("Streamlit session state is unavailable.")
    return st.session_state


def _matching_dynamic_keys(state: MutableMapping, prefixes: Iterable[str]) -> set[str]:
    return {
        key
        for key in list(state.keys())
        if isinstance(key, str) and any(key.startswith(prefix) for prefix in prefixes)
    }


def pop_keys(state, keys: Iterable[str]) -> list[str]:
    removed = []
    for key in keys:
        if key in state:
            state.pop(key, None)
            removed.append(key)
    return removed


def clear_split_state(state=None) -> list[str]:
    state = _get_state(state)
    return pop_keys(state, SPLIT_STATE_KEYS)


def clear_validation_state(state=None) -> list[str]:
    state = _get_state(state)
    return pop_keys(state, VALIDATION_STATE_KEYS | PLOT_ARTIFACT_STATE_KEYS)


def clear_submission_state(state=None) -> list[str]:
    state = _get_state(state)
    return pop_keys(state, SUBMISSION_STATE_KEYS)


def clear_training_state(state=None) -> list[str]:
    state = _get_state(state)
    return pop_keys(state, TRAINING_STATE_KEYS | SUBMISSION_STATE_KEYS | PLOT_ARTIFACT_STATE_KEYS)


def clear_heavy_artifacts(state=None) -> list[str]:
    state = _get_state(state)
    return pop_keys(state, HEAVY_ARTIFACT_KEYS)


def clear_downstream_from_split(state=None) -> list[str]:
    state = _get_state(state)
    keys = (
        CLEANSING_STATE_KEYS
        | PREPROCESSING_STATE_KEYS
        | SAMPLING_STATE_KEYS
        | RESAMPLING_STATE_KEYS
        | VALIDATION_STATE_KEYS
        | TRAINING_STATE_KEYS
        | PLOT_ARTIFACT_STATE_KEYS
        | SUBMISSION_STATE_KEYS
    )
    keys |= _matching_dynamic_keys(state, DYNAMIC_DERIVED_PREFIXES)
    return pop_keys(state, keys)


def clear_downstream_from_cleansing(state=None) -> list[str]:
    state = _get_state(state)
    keys = (
        PREPROCESSING_STATE_KEYS
        | SAMPLING_STATE_KEYS
        | RESAMPLING_STATE_KEYS
        | VALIDATION_STATE_KEYS
        | TRAINING_STATE_KEYS
        | PLOT_ARTIFACT_STATE_KEYS
        | SUBMISSION_STATE_KEYS
    )
    keys |= _matching_dynamic_keys(state, ("ord_order::",))
    return pop_keys(state, keys)


def clear_downstream_from_preprocessing(state=None) -> list[str]:
    state = _get_state(state)
    keys = (
        SAMPLING_STATE_KEYS
        | RESAMPLING_STATE_KEYS
        | VALIDATION_STATE_KEYS
        | TRAINING_STATE_KEYS
        | PLOT_ARTIFACT_STATE_KEYS
        | SUBMISSION_STATE_KEYS
    )
    return pop_keys(state, keys)


def clear_downstream_from_sampling(state=None) -> list[str]:
    state = _get_state(state)
    keys = (
        RESAMPLING_STATE_KEYS
        | VALIDATION_STATE_KEYS
        | TRAINING_STATE_KEYS
        | PLOT_ARTIFACT_STATE_KEYS
        | SUBMISSION_STATE_KEYS
    )
    return pop_keys(state, keys)


def clear_downstream_from_resampling(state=None) -> list[str]:
    state = _get_state(state)
    keys = VALIDATION_STATE_KEYS | TRAINING_STATE_KEYS | PLOT_ARTIFACT_STATE_KEYS | SUBMISSION_STATE_KEYS
    return pop_keys(state, keys)


def clear_downstream_from_validation(state=None) -> list[str]:
    state = _get_state(state)
    keys = TRAINING_STATE_KEYS | PLOT_ARTIFACT_STATE_KEYS | SUBMISSION_STATE_KEYS
    return pop_keys(state, keys)


def clear_all_workflow_state(state=None, keep_base: bool = True) -> list[str]:
    state = _get_state(state)
    keys = set(DERIVED_STATE_KEYS)
    keys |= _matching_dynamic_keys(state, DYNAMIC_DERIVED_PREFIXES)
    if not keep_base:
        keys |= BASE_STATE_KEYS
    return pop_keys(state, keys)


def session_diagnostics(state=None) -> dict[str, int]:
    state = _get_state(state)
    plots = state.get("evaluation_plots_temp") or {}
    return {
        "known_keys_present": len([key for key in ALL_KNOWN_STATE_KEYS if key in state]),
        "total_keys": len(state.keys()),
        "heavy_keys_present": len([key for key in HEAVY_ARTIFACT_KEYS if key in state]),
        "evaluation_plots_count": len(plots) if hasattr(plots, "__len__") else 0,
    }
