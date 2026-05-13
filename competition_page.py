import os
import io
import shutil
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from ui_components import render_info_panel, fix_arrow_compatibility
from config_limits import (
    ALLOW_UPLOADED_MODEL_BUNDLES,
    MAX_COLUMNS,
    MAX_ROWS,
    SAMPLE_SUBMISSION_LIMIT,
    TEST_CSV_LIMIT,
    validate_dataframe_shape,
    validate_upload_size,
)
from ml_correctness import align_features_for_inference, resolve_expected_features
from state_manager import clear_submission_state


def _models_dir():
    return os.path.join(os.path.dirname(__file__), "models")


def render_export_page():
    st.header("📦 Export Model Bundle (ZIP)")
    models_dir = _models_dir()
    if not os.path.exists(models_dir):
        st.info("No saved model bundles found (models/ directory missing). Save a model first.")
        return

    bundles = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    if not bundles:
        st.info("No saved model bundles found in models/.")
        return

    selection = st.selectbox("Select bundle to export", bundles, key="export_bundle_select")
    if st.button("Create ZIP"):
        bundle_name = os.path.splitext(selection)[0]
        base_tmp = os.path.join(models_dir, f"{bundle_name}_export_tmp")
        if os.path.exists(base_tmp):
            shutil.rmtree(base_tmp)
        os.makedirs(base_tmp, exist_ok=True)

        # copy pkl and metadata if present
        pkl_path = os.path.join(models_dir, selection)
        shutil.copy2(pkl_path, base_tmp)
        meta_path = os.path.join(models_dir, f"{bundle_name}_metadata.txt")
        if os.path.exists(meta_path):
            shutil.copy2(meta_path, base_tmp)

        # include evaluation_plots if present
        eval_dir = os.path.join(models_dir, "evaluation_plots", bundle_name)
        if os.path.exists(eval_dir):
            shutil.copytree(eval_dir, os.path.join(base_tmp, "evaluation_plots"))

        # include data_splits if session had saved CSVs
        data_splits_dir = os.path.join(models_dir, "data_splits")
        if os.path.exists(data_splits_dir):
            shutil.copytree(data_splits_dir, os.path.join(base_tmp, "data_splits"))

        archive_path = shutil.make_archive(base_tmp, "zip", base_tmp)
        with open(archive_path, "rb") as f:
            data = f.read()
        st.download_button(label="Download ZIP", data=data, file_name=f"{bundle_name}.zip")


def render_competition_page():
    st.header("📤 Submission / Competition")
    st.write("Upload test.csv and apply the current session model or a saved bundle to produce a submission file.")
    st.info(
        f"Test CSV maksimal {TEST_CSV_LIMIT.max_upload_mb} MB, {MAX_ROWS:,} rows, dan {MAX_COLUMNS} columns. "
        "Kolom test harus sesuai dengan fitur saat training; extra column aman, missing required feature akan ditolak."
    )

    models_dir = _models_dir()
    bundles = []
    if os.path.exists(models_dir):
        bundles = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]

    # Three-column row: Current/saved model | Test CSV | Sample submission CSV
    col_model, col_test, col_sample = st.columns(3)

    with col_model:
        model_uploaded = None
        if ALLOW_UPLOADED_MODEL_BUNDLES:
            model_uploaded = st.file_uploader(
                "Upload model bundle (.pkl/.joblib)",
                type=["pkl", "joblib"],
                key="submission_model_upload",
            )
        else:
            st.warning("Upload model bundle dinonaktifkan untuk keamanan. Gunakan model yang dilatih dari aplikasi/session ini.")
        session_bundle_available = st.session_state.get("trained_model") is not None
        use_session_model = st.checkbox(
            "Use current session trained model",
            value=session_bundle_available,
            disabled=not session_bundle_available,
            key="submission_use_session_model",
        )
        if bundles:
            selected = st.selectbox("Or choose saved bundle", bundles, key="submission_bundle_select")
        else:
            selected = None

    with col_test:
        test_uploaded = st.file_uploader("Upload test CSV (no target column)", type=["csv"], key="submission_upload")

    with col_sample:
        sample_uploaded = st.file_uploader("Upload sample submission CSV", type=["csv"], key="submission_sample_upload")

    # Read uploaded test and sample files (if provided) and show previews
    test_df = None
    sample_df = None
    if test_uploaded is not None:
        ok, message = validate_upload_size(test_uploaded, TEST_CSV_LIMIT, "Test CSV")
        if not ok:
            st.error(message)
            st.stop()
        try:
            test_uploaded.seek(0)
            test_df = pd.read_csv(test_uploaded)
            with col_test:
                st.write(f"Uploaded test: {len(test_df)} rows × {len(test_df.columns)} columns")
                try:
                    st.dataframe(fix_arrow_compatibility(test_df.head()))
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
            return
        ok, message = validate_dataframe_shape(test_df, TEST_CSV_LIMIT, "Test CSV")
        if not ok:
            st.error(message)
            st.stop()

    if sample_uploaded is not None:
        ok, message = validate_upload_size(sample_uploaded, SAMPLE_SUBMISSION_LIMIT, "Sample submission CSV")
        if not ok:
            st.error(message)
            st.stop()
        try:
            sample_uploaded.seek(0)
            sample_df = pd.read_csv(sample_uploaded)
            with col_sample:
                st.write(f"Sample submission: {len(sample_df)} rows × {len(sample_df.columns)} columns")
                try:
                    st.dataframe(fix_arrow_compatibility(sample_df.head()))
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Error reading sample submission CSV: {e}")
            return
        ok, message = validate_dataframe_shape(sample_df, SAMPLE_SUBMISSION_LIMIT, "Sample submission CSV")
        if not ok:
            st.error(message)
            st.stop()

    # Allow choosing ID column to drop from test (preserve values for output)
    if test_df is not None:
        cols = list(test_df.columns)
        drop_options = ["-- none --"] + cols
        selected_id = col_test.selectbox("Select ID column to drop (optional)", options=drop_options, index=0, key="submission_drop_id")

    # Flexible output column naming via free text inputs
    if sample_df is not None:
        default_id_name = sample_df.columns[0] if sample_df.shape[1] >= 1 else "Id"
        default_pred_name = sample_df.columns[1] if sample_df.shape[1] > 1 else "Predicted"
    elif test_df is not None:
        default_id_name = test_df.columns[0] if test_df.shape[1] >= 1 else "Id"
        default_pred_name = "Predicted"
    else:
        default_id_name = "Id"
        default_pred_name = "Predicted"

    with col_sample:
        st.text_input("Output ID column name (custom)", value=default_id_name, key="submission_output_id_name")
        st.text_input("Prediction column name (custom)", value=default_pred_name, key="submission_output_target_name")

    # Run prediction (single clean flow)
    if st.button("Run Prediction"):
        if test_df is None:
            st.error("Please upload a test CSV first.")
            return
        clear_submission_state(st.session_state)

        # Load model bundle: prefer the current in-session model, otherwise use a saved bundle.
        # User-uploaded pickle/joblib bundles are intentionally disabled.
        bundle = None
        if ALLOW_UPLOADED_MODEL_BUNDLES and model_uploaded is not None:
            try:
                model_uploaded.seek(0)
                bundle = joblib.load(io.BytesIO(model_uploaded.read()))
            except Exception as e:
                st.error(f"Error loading uploaded model bundle: {e}")
                return
        elif use_session_model and st.session_state.get("trained_model") is not None:
            bundle = {
                "model": st.session_state.get("trained_model"),
                "preprocessor": st.session_state.get("trained_preprocessor"),
                "metadata": st.session_state.get("model_metadata", {}) or {},
            }
        else:
            if selected is None:
                st.error("No model selected. Train a model in this session or choose a saved bundle.")
                return
            try:
                bundle_path = os.path.join(models_dir, selected)
                bundle = joblib.load(bundle_path)
            except Exception as e:
                st.error(f"Error loading saved bundle: {e}")
                return

        model = bundle.get("model")
        preproc = bundle.get("preprocessor")
        metadata = bundle.get("metadata", {}) or {}

        # Prepare DataFrame to transform: drop selected ID column if requested
        if st.session_state.get("submission_drop_id") and st.session_state.get("submission_drop_id") != "-- none --":
            id_col = st.session_state.get("submission_drop_id")
            if id_col in test_df.columns:
                id_vals = test_df[id_col].copy().values
                df_for_transform = test_df.drop(columns=[id_col])
            else:
                id_vals = None
                df_for_transform = test_df.copy()
        else:
            id_vals = None
            df_for_transform = test_df.copy()

        # Determine output column names (user-selected or defaults)
        final_id_col_name = st.session_state.get("submission_output_id_name") or (
            (sample_df.columns[0] if sample_df is not None and sample_df.shape[1] >= 1 else "Id")
        )
        final_pred_col_name = st.session_state.get("submission_output_target_name") or (
            (sample_df.columns[1] if sample_df is not None and sample_df.shape[1] >= 2 else "Predicted")
        )

        # Apply preprocessor if present
        try:
            if preproc is not None:
                Xp = preproc.transform(df_for_transform.copy())
            else:
                Xp = df_for_transform.copy()
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            return

        expected_features = resolve_expected_features(metadata, model)
        if expected_features:
            try:
                Xp = align_features_for_inference(Xp, expected_features)
            except ValueError as e:
                st.error(str(e))
                return
        else:
            st.warning("Feature-order validation is limited because this model bundle has no feature metadata.")

        # Predict
        try:
            preds = model.predict(Xp)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

        # Build output DataFrame
        n = len(preds)
        s_preds = pd.Series(preds)
        if sample_df is not None and isinstance(sample_df, pd.DataFrame) and sample_df.shape[1] >= 1:
            sample_cols = list(sample_df.columns)
            # determine id values preference: preserved test id -> sample id -> test index -> range
            if id_vals is not None and len(id_vals) == n:
                use_ids = list(id_vals)
            elif sample_df.shape[0] == n:
                use_ids = list(sample_df.iloc[:, 0].values)
            elif test_df.shape[0] == n:
                use_ids = list(test_df.index.values)
            else:
                use_ids = list(range(n))

            # Cast predictions to match sample target column type where possible
            preds_cast = None
            if sample_df.shape[1] > 1:
                sample_target = sample_df.iloc[:, 1]
                sample_nonnull = sample_target.dropna()
                try:
                    if not sample_nonnull.empty:
                        # numeric integer dtype
                        if pd.api.types.is_integer_dtype(sample_nonnull.dtype):
                            preds_cast = s_preds.astype(int).tolist()
                        elif pd.api.types.is_float_dtype(sample_nonnull.dtype):
                            preds_cast = s_preds.astype(float).tolist()
                        else:
                            s_sample = sample_nonnull.astype(str)
                            if s_sample.str.match(r'^-?\d+$').all():
                                preds_cast = s_preds.astype(int).tolist()
                            elif s_sample.str.match(r'^-?\d+(\.\d+)?$').all():
                                preds_cast = s_preds.astype(float).tolist()
                            else:
                                preds_cast = s_preds.astype(str).tolist()
                    else:
                        # no info in sample target, try to simplify floats that are integral
                        try:
                            s_float = s_preds.astype(float)
                            if np.all(np.isfinite(s_float)) and np.all(np.equal(np.mod(s_float, 1), 0)):
                                preds_cast = s_float.astype(int).tolist()
                            else:
                                preds_cast = s_float.tolist()
                        except Exception:
                            preds_cast = s_preds.tolist()
                except Exception:
                    preds_cast = s_preds.tolist()
            else:
                preds_cast = s_preds.tolist()

            # Cast/use ids to match sample id column type when possible
            try:
                if sample_df.shape[0] == n:
                    sample_id_series = sample_df.iloc[:, 0]
                    sample_id_nonnull = sample_id_series.dropna()
                    if not sample_id_nonnull.empty:
                        if pd.api.types.is_integer_dtype(sample_id_nonnull.dtype):
                            use_ids = [int(x) for x in use_ids]
                        elif pd.api.types.is_float_dtype(sample_id_nonnull.dtype):
                            use_ids = [float(x) for x in use_ids]
                        else:
                            use_ids = [str(x) for x in use_ids]
            except Exception:
                pass

            out_dict = {}
            out_dict[final_id_col_name] = use_ids
            out_dict[final_pred_col_name] = preds_cast
            # additional sample columns (preserve remaining sample columns order)
            other_cols = sample_cols[2:] if len(sample_cols) > 2 else []
            for col in other_cols:
                if sample_df.shape[0] == n:
                    out_dict[col] = list(sample_df[col].values)
                else:
                    out_dict[col] = [np.nan] * n

            out = pd.DataFrame(out_dict)
            cols_order = [final_id_col_name, final_pred_col_name] + other_cols
            out = out[cols_order]
        else:
            # default output format (no sample provided)
            if id_vals is not None and len(id_vals) == n:
                ids = list(id_vals)
            elif test_df.shape[0] == n:
                ids = list(test_df.index.values)
            else:
                ids = list(range(n))
            out = pd.DataFrame({final_id_col_name: ids, final_pred_col_name: list(preds)})

        st.subheader("Sample predictions")
        st.dataframe(fix_arrow_compatibility(out.sample(5) if len(out) > 5 else out))

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.session_state["submission_predictions"] = list(preds)
        st.session_state["submission_df"] = out.copy()
        st.session_state["submission_csv"] = csv_bytes
        st.download_button("Download submission.csv", data=csv_bytes, file_name="submission.csv")

        # Also show the full original test DataFrame with predictions appended
        try:
            full_df = test_df.copy()
            full_df[final_pred_col_name] = list(preds)
            st.session_state["submission_full_df"] = full_df.copy()
            st.subheader("Full test set with predictions (all columns)")
            try:
                st.dataframe(fix_arrow_compatibility(full_df.sample(5) if len(full_df) > 5 else full_df))
            except Exception:
                st.write(full_df.head())
        except Exception:
            pass

    render_info_panel("Submission")
