import pandas as pd

from config_limits import (
    MAX_ONE_HOT_CARDINALITY_PER_COLUMN,
    MAX_RANDOM_FOREST_ESTIMATORS,
    TRAIN_CSV_LIMIT,
    clamp_model_params,
    validate_dataframe_shape,
    validate_one_hot_request,
)


def test_dataframe_shape_limit_rejects_too_many_columns():
    df = pd.DataFrame({f"c{i}": [i] for i in range(TRAIN_CSV_LIMIT.max_columns + 1)})

    ok, message = validate_dataframe_shape(df, TRAIN_CSV_LIMIT, "Train CSV")

    assert not ok
    assert "Maximum allowed columns" in message


def test_one_hot_limit_rejects_high_cardinality_column():
    df = pd.DataFrame({"category": [f"value_{i}" for i in range(MAX_ONE_HOT_CARDINALITY_PER_COLUMN + 1)]})

    ok, message = validate_one_hot_request(df, ["category"])

    assert not ok
    assert "cardinality is too high" in message


def test_random_forest_estimators_are_capped():
    params, warnings = clamp_model_params("RandomForest", {"n_estimators": MAX_RANDOM_FOREST_ESTIMATORS + 50})

    assert params["n_estimators"] == MAX_RANDOM_FOREST_ESTIMATORS
    assert warnings
