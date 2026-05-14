from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CsvUploadLimit:
    max_upload_mb: int
    max_rows: int
    max_columns: int

    @property
    def max_upload_bytes(self):
        return self.max_upload_mb * 1024 * 1024


# Workshop-sized defaults. Keep these centralized so facilitators can loosen
# or tighten them without hunting through page files.
MAX_TRAIN_UPLOAD_MB = 10
MAX_TEST_UPLOAD_MB = 5

MAX_TRAIN_UPLOAD_BYTES = MAX_TRAIN_UPLOAD_MB * 1024 * 1024
MAX_TEST_UPLOAD_BYTES = MAX_TEST_UPLOAD_MB * 1024 * 1024

MAX_ROWS = 100_000
MAX_COLUMNS = 200

MAX_ONE_HOT_SELECTED_COLUMNS = 10
MAX_ONE_HOT_UNIQUE_VALUES_PER_COLUMN = 50
MAX_ONE_HOT_CARDINALITY_PER_COLUMN = MAX_ONE_HOT_UNIQUE_VALUES_PER_COLUMN
MAX_ONE_HOT_OUTPUT_COLUMNS = 500
MAX_FEATURE_COLUMNS_AFTER_ENCODING = MAX_COLUMNS + MAX_ONE_HOT_OUTPUT_COLUMNS

MAX_CV_FOLDS = 5
MAX_RANDOM_FOREST_ESTIMATORS = 100
MAX_PERMUTATION_IMPORTANCE_FEATURES = 30
MAX_PERMUTATION_IMPORTANCE_REPEATS = 2
MAX_PERMUTATION_IMPORTANCE_ROWS = 5_000
MAX_EVALUATION_PLOTS = 20

SESSION_STORAGE_DIR = "sessions"
SESSION_TTL_MINUTES = 120
SESSION_ID_PREFIX = "BDT"
SESSION_ID_LENGTH = 6
MAX_SAVED_SESSION_MB = 50

ALLOW_UPLOADED_MODEL_BUNDLES = False
ENABLE_PERMUTATION_IMPORTANCE_BY_DEFAULT = False
PERMUTATION_IMPORTANCE_DEFAULT_ENABLED = ENABLE_PERMUTATION_IMPORTANCE_BY_DEFAULT

TRAIN_CSV_LIMIT = CsvUploadLimit(
    max_upload_mb=MAX_TRAIN_UPLOAD_MB,
    max_rows=MAX_ROWS,
    max_columns=MAX_COLUMNS,
)
TEST_CSV_LIMIT = CsvUploadLimit(
    max_upload_mb=MAX_TEST_UPLOAD_MB,
    max_rows=MAX_ROWS,
    max_columns=MAX_COLUMNS,
)
SAMPLE_SUBMISSION_LIMIT = CsvUploadLimit(
    max_upload_mb=MAX_TEST_UPLOAD_MB,
    max_rows=MAX_ROWS,
    max_columns=MAX_COLUMNS,
)


def format_bytes(num_bytes):
    if num_bytes is None:
        return "unknown size"
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def uploaded_file_size(uploaded_file):
    size = getattr(uploaded_file, "size", None)
    if size is not None:
        return int(size)

    try:
        pos = uploaded_file.tell()
        uploaded_file.seek(0, 2)
        size = uploaded_file.tell()
        uploaded_file.seek(pos)
        return int(size)
    except Exception:
        return None


def validate_upload_size(uploaded_file, limit, label):
    size = uploaded_file_size(uploaded_file)
    if size is not None and size > limit.max_upload_bytes:
        return (
            False,
            f"{label} is too large ({format_bytes(size)}). "
            f"Maximum allowed size is {limit.max_upload_mb} MB.",
        )
    return True, None


def validate_dataframe_shape(df, limit, label):
    rows, columns = df.shape
    if rows > limit.max_rows:
        return False, f"{label} has {rows:,} rows. Maximum allowed rows: {limit.max_rows:,}."
    if columns > limit.max_columns:
        return False, f"{label} has {columns:,} columns. Maximum allowed columns: {limit.max_columns:,}."
    return True, None


def estimate_one_hot_output_columns(df, columns):
    cardinalities = {}
    total = 0
    for col in columns:
        if col not in df.columns:
            continue
        count = int(pd.Series(df[col]).nunique(dropna=False))
        cardinalities[col] = count
        total += count
    return total, cardinalities


def validate_one_hot_request(df, columns):
    if len(columns) > MAX_ONE_HOT_SELECTED_COLUMNS:
        return (
            False,
            f"OneHotEncoder can process at most {MAX_ONE_HOT_SELECTED_COLUMNS} columns at once. "
            f"You selected {len(columns)} columns.",
        )

    estimated_output, cardinalities = estimate_one_hot_output_columns(df, columns)
    too_wide = {
        col: count
        for col, count in cardinalities.items()
        if count > MAX_ONE_HOT_CARDINALITY_PER_COLUMN
    }
    if too_wide:
        details = ", ".join(f"{col}={count}" for col, count in too_wide.items())
        return (
            False,
            f"OneHotEncoder cardinality is too high ({details}). "
            f"Maximum unique values per column: {MAX_ONE_HOT_CARDINALITY_PER_COLUMN}.",
        )

    if estimated_output > MAX_ONE_HOT_OUTPUT_COLUMNS:
        return (
            False,
            f"OneHotEncoder would create about {estimated_output:,} new columns. "
            f"Maximum allowed new columns: {MAX_ONE_HOT_OUTPUT_COLUMNS:,}.",
        )

    final_columns = df.shape[1] - len(columns) + estimated_output
    if final_columns > MAX_FEATURE_COLUMNS_AFTER_ENCODING:
        return (
            False,
            f"Encoding would produce about {final_columns:,} feature columns. "
            f"Maximum allowed columns after encoding: {MAX_FEATURE_COLUMNS_AFTER_ENCODING:,}.",
        )

    return True, None


def validate_one_hot_output_width(existing_columns, removed_columns, output_columns):
    final_columns = existing_columns - removed_columns + output_columns
    if output_columns > MAX_ONE_HOT_OUTPUT_COLUMNS:
        return (
            False,
            f"OneHotEncoder would create {output_columns:,} new columns. "
            f"Maximum allowed new columns: {MAX_ONE_HOT_OUTPUT_COLUMNS:,}.",
        )
    if final_columns > MAX_FEATURE_COLUMNS_AFTER_ENCODING:
        return (
            False,
            f"Encoding would produce {final_columns:,} feature columns. "
            f"Maximum allowed columns after encoding: {MAX_FEATURE_COLUMNS_AFTER_ENCODING:,}.",
        )
    return True, None


def clamp_model_params(model_name, params):
    params = dict(params or {})
    warnings = []
    if model_name == "RandomForest" and "n_estimators" in params:
        try:
            n_estimators = int(params["n_estimators"])
        except Exception:
            n_estimators = MAX_RANDOM_FOREST_ESTIMATORS
        if n_estimators > MAX_RANDOM_FOREST_ESTIMATORS:
            warnings.append(
                f"RandomForest n_estimators capped at {MAX_RANDOM_FOREST_ESTIMATORS} "
                f"(requested {n_estimators})."
            )
            n_estimators = MAX_RANDOM_FOREST_ESTIMATORS
        params["n_estimators"] = n_estimators
    return params, warnings


def can_run_permutation_importance(X):
    rows, columns = X.shape
    if rows > MAX_PERMUTATION_IMPORTANCE_ROWS:
        return (
            False,
            f"Permutation importance skipped: {rows:,} rows exceeds "
            f"the limit of {MAX_PERMUTATION_IMPORTANCE_ROWS:,}.",
        )
    if columns > MAX_PERMUTATION_IMPORTANCE_FEATURES:
        return (
            False,
            f"Permutation importance skipped: {columns:,} features exceeds "
            f"the limit of {MAX_PERMUTATION_IMPORTANCE_FEATURES:,}.",
        )
    return True, None
