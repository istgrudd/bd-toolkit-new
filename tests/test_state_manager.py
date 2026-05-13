from state_manager import (
    clear_all_workflow_state,
    clear_downstream_from_cleansing,
    clear_downstream_from_preprocessing,
    clear_downstream_from_split,
    clear_heavy_artifacts,
    pop_keys,
)


def test_pop_keys_removes_existing_keys():
    state = {"a": 1, "b": 2}

    removed = pop_keys(state, {"a", "missing"})

    assert removed == ["a"]
    assert state == {"b": 2}


def test_clear_downstream_from_split_removes_downstream_artifacts():
    state = {
        "pre_X_train": "split",
        "preprocessing_steps": ["scale"],
        "validation_summary": {"metric": 1},
        "trained_model": object(),
        "evaluation_plots_temp": {"plot.png": b"1"},
        "sampled_X_train": "sample",
        "resampled_pre_X_train": "resampled",
    }

    removed = clear_downstream_from_split(state)

    assert "pre_X_train" in state
    assert "preprocessing_steps" not in state
    assert "validation_summary" not in state
    assert "trained_model" not in state
    assert "evaluation_plots_temp" not in state
    assert "sampled_X_train" not in state
    assert "resampled_pre_X_train" not in state
    assert "trained_model" in removed


def test_clear_downstream_from_cleansing_keeps_cleansing_state():
    state = {
        "cleansing_steps": ["impute"],
        "imputation_transformers": {"x": object()},
        "preprocessing_steps": ["scale"],
        "validation_summary": {"metric": 1},
        "trained_model": object(),
    }

    clear_downstream_from_cleansing(state)

    assert "cleansing_steps" in state
    assert "imputation_transformers" in state
    assert "preprocessing_steps" not in state
    assert "validation_summary" not in state
    assert "trained_model" not in state


def test_clear_downstream_from_preprocessing_keeps_preprocessing_state():
    state = {
        "preprocessing_steps": ["scale"],
        "scalers": {"s": object()},
        "validation_summary": {"metric": 1},
        "trained_model": object(),
        "submission_predictions": [1, 2],
        "evaluation_plots_temp": {"plot.png": b"1"},
    }

    clear_downstream_from_preprocessing(state)

    assert "preprocessing_steps" in state
    assert "scalers" in state
    assert "validation_summary" not in state
    assert "trained_model" not in state
    assert "submission_predictions" not in state
    assert "evaluation_plots_temp" not in state


def test_clear_heavy_artifacts_removes_models_plots_and_sampled_data():
    state = {
        "df": "base",
        "trained_model": object(),
        "evaluation_plots_temp": {"plot.png": b"1"},
        "sampled_X_train": "sample",
        "resampled_pre_X_train": "resampled",
    }

    clear_heavy_artifacts(state)

    assert state == {"df": "base"}


def test_clear_all_workflow_state_keep_base_preserves_base_keys():
    state = {
        "df": "base",
        "upload_hash": "hash",
        "target_column": "target",
        "task_type": "Classification",
        "pre_X_train": "split",
        "trained_model": object(),
    }

    clear_all_workflow_state(state, keep_base=True)

    assert state == {
        "df": "base",
        "upload_hash": "hash",
        "target_column": "target",
        "task_type": "Classification",
    }


def test_clear_all_workflow_state_without_keep_base_removes_base_keys():
    state = {
        "df": "base",
        "upload_hash": "hash",
        "target_column": "target",
        "task_type": "Classification",
        "pre_X_train": "split",
    }

    clear_all_workflow_state(state, keep_base=False)

    assert state == {}
