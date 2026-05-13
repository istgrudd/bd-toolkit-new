import pandas as pd
import pytest

from ml_correctness import (
    align_features_for_inference,
    apply_fitted_scaler_to_columns,
    resolve_expected_features,
    validate_required_features,
)


def test_feature_alignment_reorders_and_drops_extra_columns():
    X = pd.DataFrame({
        "b": [1, 2],
        "a": [3, 4],
        "extra": [9, 9],
    })

    aligned = align_features_for_inference(X, ["a", "b"])

    assert list(aligned.columns) == ["a", "b"]
    assert aligned.to_dict("list") == {"a": [3, 4], "b": [1, 2]}


def test_feature_alignment_raises_for_missing_column():
    X = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="Missing required features"):
        align_features_for_inference(X, ["a", "b"])


def test_validate_required_features_reports_missing_selected_feature():
    X = pd.DataFrame({"a": [1]})

    assert validate_required_features(X, ["a", "b"]) == ["b"]


def test_resolve_expected_features_prefers_training_feature_names():
    metadata = {
        "training_feature_names": ["train_a", "train_b"],
        "selected_features": ["selected_a"],
    }

    assert resolve_expected_features(metadata) == ["train_a", "train_b"]


def test_resolve_expected_features_falls_back_to_selected_features():
    metadata = {"selected_features": ["a", "b"]}

    assert resolve_expected_features(metadata) == ["a", "b"]


def test_resolve_expected_features_falls_back_to_model_feature_names():
    class Model:
        feature_names_in_ = ["m_a", "m_b"]

    assert resolve_expected_features({}, Model()) == ["m_a", "m_b"]


def test_session_preprocessor_transform_raises_on_failed_step():
    from data_modeling import SessionPreprocessor

    preprocessor = SessionPreprocessor(
        scalers={},
        preprocessing_steps=[
            {
                "type": "scaling",
                "method": "StandardScaler",
                "columns": ["a"],
                "action": "transform",
            }
        ],
    )

    with pytest.raises(ValueError, match="Preprocessing transform failed"):
        preprocessor.transform(pd.DataFrame({"a": [1, 2]}))


def test_apply_fitted_scaler_to_columns_transforms_once():
    class AddOneScaler:
        def transform(self, values):
            return values + 1

    X = pd.DataFrame({"a": [1, 2], "b": [10, 20]})

    out = apply_fitted_scaler_to_columns(X, ["a"], AddOneScaler())

    assert out["a"].tolist() == [2, 3]
    assert out["b"].tolist() == [10, 20]
    assert X["a"].tolist() == [1, 2]
