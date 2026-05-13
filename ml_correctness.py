from __future__ import annotations

import pandas as pd


def validate_required_features(X: pd.DataFrame, required_features) -> list[str]:
    required = list(required_features or [])
    return [col for col in required if col not in X.columns]


def align_features_for_inference(X: pd.DataFrame, expected_features):
    expected = list(expected_features or [])
    if not expected:
        return X
    missing = validate_required_features(X, expected)
    if missing:
        raise ValueError("Missing required features for prediction: " + ", ".join(missing))
    return X.loc[:, expected]


def resolve_expected_features(metadata=None, model=None):
    metadata = metadata or {}
    for key in ("training_feature_names", "selected_features"):
        values = metadata.get(key)
        if values:
            return list(values)

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)

    return []


def build_training_metadata(
    *,
    model_name,
    task_type,
    target_column,
    training_feature_names,
    selected_features=None,
    extra=None,
):
    feature_names = list(training_feature_names or [])
    selected = list(selected_features or feature_names)
    metadata = {
        "model_name": model_name,
        "task_type": task_type,
        "target_column": target_column,
        "training_feature_names": feature_names,
        "selected_features": selected,
    }
    if extra:
        metadata.update(extra)
    return metadata


def apply_fitted_scaler_to_columns(df: pd.DataFrame, columns, scaler):
    missing = validate_required_features(df, columns)
    if missing:
        raise ValueError("Missing columns for scaling: " + ", ".join(missing))
    out = df.copy()
    cols = list(columns)
    out.loc[:, cols] = scaler.transform(out[cols])
    return out
