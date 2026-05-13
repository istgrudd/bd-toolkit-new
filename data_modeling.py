import os
import io
import joblib
import shutil
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import sys
import pickle
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance

# ensure local module imports work when pytest changes CWD
sys.path.insert(0, os.path.dirname(__file__))
from ui_components import render_info_panel, fix_arrow_compatibility, add_plot_to_session
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor

# Temporary safety flag: when True, skip writing model bundles and evaluation plots to disk
# to avoid repository bloat. Set to False to re-enable saving.
DISABLE_BUNDLE_SAVING = True


# Simple model registry + defaults for the UI
DEFAULT_HYPERPARAMETERS = {
    "RandomForest": {"n_estimators": 100, "max_depth": None},
    "LogisticRegression": {"C": 1.0, "max_iter": 200},
    "LinearRegression": {},
    "XGBoost": {"n_estimators": 100, "learning_rate": 0.1},
    "LightGBM": {"num_leaves": 31, "n_estimators": 100},
    "Dummy": {},
}


def _available_models():
    """Return the list of available model families for the UI."""
    return list(DEFAULT_HYPERPARAMETERS.keys())


def _instantiate_model(model_name, params=None):
    """Instantiate a model by family name with given params.

    Falls back to sensible sklearn defaults when optional libraries are missing.
    """
    params = params or {}
    task = st.session_state.get("task_type", "Classification")

    if model_name == "RandomForest":
        return RandomForestClassifier(**params) if task == "Classification" else RandomForestRegressor(**params)

    if model_name == "LogisticRegression":
        return LogisticRegression(**params) if task == "Classification" else LinearRegression(**params)

    if model_name == "LinearRegression":
        return LinearRegression(**params)

    if model_name == "XGBoost":
        try:
            import xgboost as xgb

            return xgb.XGBClassifier(**params) if task == "Classification" else xgb.XGBRegressor(**params)
        except Exception:
            return RandomForestClassifier(**params) if task == "Classification" else RandomForestRegressor(**params)

    if model_name == "LightGBM":
        try:
            import lightgbm as lgb

            return lgb.LGBMClassifier(**params) if task == "Classification" else lgb.LGBMRegressor(**params)
        except Exception:
            return RandomForestClassifier(**params) if task == "Classification" else RandomForestRegressor(**params)

    # default fallback
    return DummyClassifier() if task == "Classification" else DummyRegressor()


class SessionPreprocessor:
    """Minimal session preprocessor capturing fitted transformers.

    Tests only require that this can be instantiated and that `transform`
    returns a DataFrame. Implement a safe no-op transform that returns a
    copy of the input DataFrame (or converts arrays to DataFrame).
    """

    def __init__(self, imputation_transformers=None, scalers=None, encoders=None, preprocessing_steps=None):
        self.imputation_transformers = imputation_transformers or {}
        self.scalers = scalers or {}
        self.encoders = encoders or {}
        self.preprocessing_steps = preprocessing_steps or []

    def transform(self, X):
        """Apply stored preprocessing to a new DataFrame X and return transformed DataFrame.

        This method attempts to replay recorded preprocessing steps in order where
        possible using the fitted transformers saved in this object. It is
        defensive: missing transformers or missing columns are skipped rather
        than raising, to keep prediction paths robust.
        """
        if X is None:
            return X

        # Normalize input to DataFrame
        if not isinstance(X, pd.DataFrame):
            try:
                Xp = pd.DataFrame(X)
            except Exception:
                Xp = pd.DataFrame(np.asarray(X))
        else:
            Xp = X.copy()

        # Short aliases
        imps = self.imputation_transformers or {}
        scals = self.scalers or {}
        encs = self.encoders or {}

        for step in (self.preprocessing_steps or []):
            try:
                ttype = step.get("type")

                # Imputation
                if ttype == "imputation":
                    cols = step.get("columns", []) or []
                    method = step.get("method")
                    if not cols or method is None:
                        continue
                    key = f"imputer::{method}::" + ",".join(sorted(cols))
                    imputer = imps.get(key)
                    if imputer is None:
                        # try tolerant lookup (without sorting)
                        for k, v in imps.items():
                            if k.startswith("imputer::") and set(k.split("::", 2)[2].split(",")) == set(cols):
                                imputer = v
                                break
                    if imputer is not None:
                        present = [c for c in cols if c in Xp.columns]
                        if present:
                            transformed = imputer.transform(Xp[present])
                            Xp.loc[:, present] = transformed

                # Scaling
                elif ttype == "scaling":
                    cols = step.get("columns", []) or []
                    method = step.get("method")
                    if not cols or method is None:
                        continue
                    key = f"scaler::{method}::" + ",".join(sorted(cols))
                    scaler = scals.get(key)
                    if scaler is None:
                        for k, v in scals.items():
                            if k.startswith("scaler::") and set(k.split("::", 2)[2].split(",")) == set(cols):
                                scaler = v
                                break
                    if scaler is not None:
                        present = [c for c in cols if c in Xp.columns]
                        if present:
                            Xp.loc[:, present] = scaler.transform(Xp[present])

                # Encoding
                elif ttype == "encoding":
                    method = step.get("method")
                    cols = step.get("columns", []) or []
                    if not cols or method is None:
                        continue

                    # LabelEncoder (per-column)
                    if method == "LabelEncoder":
                        for col in cols:
                            if col not in Xp.columns:
                                continue
                            key_col = f"label::{col}"
                            le = encs.get(key_col) or encs.get(f"encoder::LabelEncoder::{col}")
                            if le is None:
                                # try any label-like key
                                for k, v in encs.items():
                                    if k.startswith("label::") and k.split("::", 1)[1] == col:
                                        le = v
                                        break
                            if le is not None:
                                classes = getattr(le, "classes_", [])
                                mapping = {v: i for i, v in enumerate(classes)}
                                Xp[col] = Xp[col].astype(str).map(mapping)

                    # OneHotEncoder (multi-column)
                    elif method == "OneHotEncoder":
                        cols_sorted = sorted(cols)
                        key = f"encoder::OneHotEncoder::" + ",".join(cols_sorted)
                        ohe = encs.get(key)
                        if ohe is None:
                            # try to find an encoder whose column set matches
                            for k, v in encs.items():
                                if k.startswith("encoder::OneHotEncoder::"):
                                    parts = k.split("::", 2)
                                    if len(parts) == 3:
                                        cols_k = parts[2].split(",")
                                        if set(cols_k) == set(cols_sorted):
                                            ohe = v
                                            break
                        if ohe is not None:
                            # only transform if all required columns are present
                            if all(c in Xp.columns for c in cols_sorted):
                                arr = ohe.transform(Xp[cols_sorted])
                                try:
                                    names = ohe.get_feature_names_out(cols_sorted)
                                except Exception:
                                    # fallback generic names
                                    names = [f"ohe_{i}" for i in range(arr.shape[1])]
                                df_ohe = pd.DataFrame(arr, columns=names, index=Xp.index)
                                Xp = Xp.drop(columns=cols_sorted)
                                Xp = pd.concat([Xp, df_ohe], axis=1)

                    # OrdinalEncoder (per-column expected)
                    elif method == "OrdinalEncoder":
                        for col in cols:
                            if col not in Xp.columns:
                                continue
                            key_col = f"encoder::OrdinalEncoder::{col}"
                            ord_enc = encs.get(key_col)
                            if ord_enc is None:
                                # try any ordinal encoder key that mentions the column
                                for k, v in encs.items():
                                    if k.startswith("encoder::OrdinalEncoder::") and k.endswith(f"::{col}"):
                                        ord_enc = v
                                        break
                            if ord_enc is not None:
                                cats = ord_enc.categories_[0] if hasattr(ord_enc, "categories_") and len(ord_enc.categories_) > 0 else []
                                mapping = {v: i for i, v in enumerate(cats)}
                                Xp[col] = Xp[col].astype(str).map(mapping)

                # Value mappings recorded under preprocessing (maps dict)
                elif ttype == "value_mapping":
                    mappings = step.get("mappings") or {}
                    for col_name, maps in mappings.items():
                        if col_name not in Xp.columns:
                            continue
                        for m in maps:
                            bef = m.get("before")
                            aft = m.get("after")
                            if m.get("type") == "Numerical":
                                try:
                                    bef_v = float(bef)
                                except Exception:
                                    continue
                                mask = pd.to_numeric(Xp[col_name], errors="coerce") == bef_v
                                try:
                                    aft_v = float(aft) if aft != "" else np.nan
                                except Exception:
                                    aft_v = aft
                                Xp.loc[mask, col_name] = aft_v
                            else:
                                mask = Xp[col_name].astype(str) == str(bef)
                                Xp.loc[mask, col_name] = aft

                # other step types (feature_engineering, sampling, resampling, etc.) are skipped
            except Exception:
                # swallow errors per-step to avoid breaking prediction flow
                continue

        return Xp


def render_validation():
    st.header("🔍 Validation")

    if "df" not in st.session_state or st.session_state.get("df") is None:
        st.warning("No dataset available. Upload data on the Data page first.")
        if st.button("Open Data Page"):
            st.session_state["page"] = "Data"
        render_info_panel("Validation", model_list=_available_models())
        return

    # choose dataset for validation
    datasets = ["pre_X_train"]
    if st.session_state.get("train_subset_indices") is not None:
        datasets.append("train_subset")
    if st.session_state.get("sampled_X_train") is not None:
        datasets.append("sampled_X_train")
    if st.session_state.get("resampled_pre_X_train") is not None:
        datasets.append("resampled_pre_X_train")

    dataset_choice = st.selectbox("Validation dataset", options=datasets, index=0, key="val_dataset_choice")
    # reconstruct dataset views from session_state
    if dataset_choice == "pre_X_train":
        X = st.session_state.get("pre_X_train")
        y = st.session_state.get("pre_y_train")
    elif dataset_choice == "train_subset":
        indices = st.session_state.get("train_subset_indices")
        X = st.session_state.get("pre_X_train")
        y = st.session_state.get("pre_y_train")
        if X is not None and indices is not None:
            X = X.loc[indices]
            y = y.loc[indices]
    else:
        X = st.session_state.get(dataset_choice)
        y = st.session_state.get(dataset_choice.replace("X", "y"))

    if X is None or y is None:
        st.error("Selected dataset is not available for validation.")
        return

    if st.button("Show random 5 rows", key="val_random5"):
        try:
            st.dataframe(fix_arrow_compatibility(X.sample(5)))
        except Exception:
            st.dataframe(fix_arrow_compatibility(X.head()))

    # model selection
    models = _available_models()
    model_name = st.selectbox("Model family", models, index=0, key="val_model_family")

    # hyperparameter panel
    st.subheader("Hyperparameters")
    user_params = {}
    defaults = DEFAULT_HYPERPARAMETERS.get(model_name, {})
    for k, v in defaults.items():
        if isinstance(v, bool):
            user_params[k] = st.checkbox(k, value=v)
        elif isinstance(v, int):
            user_params[k] = int(st.number_input(k, value=int(v)))
        elif isinstance(v, float):
            user_params[k] = float(st.number_input(k, value=float(v)))
        else:
            user_params[k] = st.text_input(k, value="" if v is None else str(v))

    st.subheader("Validation Method")
    method = st.selectbox("Method", ["Cross-Validation", "Train/Validation Split"], index=0, key="val_method")
    if method == "Cross-Validation":
        n_splits = st.slider("n_splits", 2, 10, 5)
        if st.session_state.get("task_type") == "Classification":
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(st.session_state.get("global_seed", 42)))
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(st.session_state.get("global_seed", 42)))
    else:
        val_size = st.slider("Validation size", 5, 50, 20, step=5)

    # Feature selection mode: All or Select Features
    features_mode = st.radio("Features to apply:", ["All", "Select Features"], horizontal=True, index=0, key="val_features_mode")
    selected_features = None
    if features_mode == "Select Features":
        opts = list(X.columns) if X is not None else []
        selected_features = st.multiselect("Select features to include", options=opts, key="val_selected_features")

    if st.button("Run Validation"):
        try:
            seed = int(st.session_state.get("global_seed", 42))
            params = {}
            # convert user_params types where possible
            for kk, vv in user_params.items():
                if isinstance(DEFAULT_HYPERPARAMETERS.get(model_name, {}).get(kk), bool):
                    params[kk] = bool(vv)
                elif isinstance(DEFAULT_HYPERPARAMETERS.get(model_name, {}).get(kk), int):
                    try:
                        params[kk] = int(vv)
                    except Exception:
                        params[kk] = DEFAULT_HYPERPARAMETERS[model_name][kk]
                elif isinstance(DEFAULT_HYPERPARAMETERS.get(model_name, {}).get(kk), float):
                    try:
                        params[kk] = float(vv)
                    except Exception:
                        params[kk] = DEFAULT_HYPERPARAMETERS[model_name][kk]
                else:
                    # string or None
                    if vv == "":
                        params[kk] = DEFAULT_HYPERPARAMETERS[model_name][kk]
                    else:
                        params[kk] = vv

            # apply feature selection if requested
            if features_mode == "Select Features":
                if not selected_features:
                    st.error("No features selected for validation. Choose 'All' or pick features.")
                    return
                X_use = X[selected_features]
            else:
                X_use = X

            model = _instantiate_model(model_name, params)

            metrics_accum = {}
            importances = []

            if method == "Cross-Validation":
                # manual CV for comprehensive metrics and feature importance
                fold = 0
                metrics_per_fold = []
                # accumulate true/pred for aggregated confusion matrix
                y_true_all = []
                y_pred_all = []
                for train_idx, val_idx in splitter.split(X_use, y):
                    fold += 1
                    X_tr, X_val = X_use.iloc[train_idx], X_use.iloc[val_idx]
                    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    m = _instantiate_model(model_name, params)
                    m.fit(X_tr, y_tr)
                    y_pred = m.predict(X_val)
                    # collect for aggregated confusion matrix
                    try:
                        y_true_all.append(np.asarray(y_val))
                        y_pred_all.append(np.asarray(y_pred))
                    except Exception:
                        pass
                    # metrics
                    if st.session_state.get("task_type") == "Classification":
                        acc = accuracy_score(y_val, y_pred)
                        prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
                        rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
                        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
                        try:
                            if hasattr(m, "predict_proba"):
                                prob = m.predict_proba(X_val)
                                if prob.shape[1] > 1:
                                    roc = roc_auc_score(y_val, prob, multi_class="ovr")
                                else:
                                    roc = roc_auc_score(y_val, prob[:, 1])
                            else:
                                roc = None
                        except Exception:
                            roc = None
                        metrics_per_fold.append({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc})
                    else:
                        mae = mean_absolute_error(y_val, y_pred)
                        mse = mean_squared_error(y_val, y_pred)
                        rmse = float(np.sqrt(mse))
                        r2 = r2_score(y_val, y_pred)
                        metrics_per_fold.append({"mae": mae, "mse": mse, "rmse": rmse, "r2": r2})

                    # feature importance
                    if hasattr(m, "feature_importances_"):
                        importances.append(m.feature_importances_)
                    elif hasattr(m, "coef_"):
                        coef = np.ravel(getattr(m, "coef_"))
                        importances.append(np.abs(coef))
                    else:
                        try:
                            r = permutation_importance(m, X_val, y_val, n_repeats=5, random_state=seed)
                            importances.append(r.importances_mean)
                        except Exception:
                            importances.append(None)

                # aggregate metrics
                # prepare validation_summary
                if st.session_state.get("task_type") == "Classification":
                    df_metrics = pd.DataFrame(metrics_per_fold)
                    summary = df_metrics.agg(["mean", "std"]).to_dict()
                else:
                    df_metrics = pd.DataFrame(metrics_per_fold)
                    summary = df_metrics.agg(["mean", "std"]).to_dict()

                # aggregate importances
                feat_names = list(X_use.columns)
                valid_imps = [imp for imp in importances if imp is not None]
                if valid_imps:
                    upload_hash = st.session_state.get("upload_hash")

                    # Convert importances to tuples for stable hashing in cache
                    imps_tuple = tuple([tuple(map(float, imp)) for imp in valid_imps])

                    @st.cache_data
                    def _aggregate_importances(_upload_hash, imps_t):
                        arr = np.vstack([np.array(t) for t in imps_t])
                        return np.mean(arr, axis=0)

                    mean_imp = _aggregate_importances(upload_hash, imps_tuple)
                    imp_series = pd.Series(mean_imp, index=feat_names).sort_values(ascending=False)
                else:
                    imp_series = None

                validation_summary = {
                    "model_name": model_name,
                    "hyperparameters": params,
                    "method": "cross_validation",
                    "metrics": summary,
                    "feature_importance": imp_series,
                }
                # aggregated confusion matrix for CV (classification only)
                if st.session_state.get("task_type") == "Classification":
                    try:
                        y_true_cat = np.concatenate(y_true_all) if y_true_all else np.array([])
                        y_pred_cat = np.concatenate(y_pred_all) if y_pred_all else np.array([])
                        if y_true_cat.size and y_pred_cat.size:
                            labels = np.unique(np.concatenate([y_true_cat, y_pred_cat]))
                            cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)
                            validation_summary["confusion_matrix"] = cm
                            validation_summary["confusion_matrix_labels"] = list(labels)
                    except Exception:
                        pass

            else:
                # Train/Validation split on chosen training data
                test_frac = int(val_size) / 100.0
                X_tr, X_val, y_tr, y_val = train_test_split(X_use, y, test_size=test_frac, random_state=seed, stratify=y if st.session_state.get("task_type") == "Classification" else None)
                m = _instantiate_model(model_name, params)
                m.fit(X_tr, y_tr)
                y_pred = m.predict(X_val)
                if st.session_state.get("task_type") == "Classification":
                    acc = accuracy_score(y_val, y_pred)
                    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
                    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
                    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
                    try:
                        if hasattr(m, "predict_proba"):
                            prob = m.predict_proba(X_val)
                            if prob.shape[1] > 1:
                                roc = roc_auc_score(y_val, prob, multi_class="ovr")
                            else:
                                roc = roc_auc_score(y_val, prob[:, 1])
                        else:
                            roc = None
                    except Exception:
                        roc = None
                    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}
                else:
                    mae = mean_absolute_error(y_val, y_pred)
                    mse = mean_squared_error(y_val, y_pred)
                    rmse = float(np.sqrt(mse))
                    r2 = r2_score(y_val, y_pred)
                    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

                # feature importance
                if hasattr(m, "feature_importances_"):
                    imp_series = pd.Series(m.feature_importances_, index=X_use.columns).sort_values(ascending=False)
                elif hasattr(m, "coef_"):
                    coef = np.ravel(getattr(m, "coef_"))
                    imp_series = pd.Series(np.abs(coef), index=X_use.columns).sort_values(ascending=False)
                else:
                    try:
                        r = permutation_importance(m, X_val, y_val, n_repeats=5, random_state=seed)
                        imp_series = pd.Series(r.importances_mean, index=X_use.columns).sort_values(ascending=False)
                    except Exception:
                        imp_series = None

                validation_summary = {
                    "model_name": model_name,
                    "hyperparameters": params,
                    "method": "train_validation_split",
                    "metrics": metrics,
                    "feature_importance": imp_series,
                }
                # confusion matrix for the holdout validation split when classification
                if st.session_state.get("task_type") == "Classification":
                    try:
                        labels = np.unique(np.concatenate([y_val, y_pred]))
                        cm = confusion_matrix(y_val, y_pred, labels=labels)
                        validation_summary["confusion_matrix"] = cm
                        validation_summary["confusion_matrix_labels"] = list(labels)
                    except Exception:
                        pass

            # store and display
            st.session_state["validation_summary"] = validation_summary
            st.success("Validation completed and summary saved to session.")

            # show metrics and importance
            st.subheader("Validation Metrics")
            if validation_summary.get("metrics"):
                st.write(validation_summary["metrics"])

            st.subheader("Feature Importance")
            if validation_summary.get("feature_importance") is not None:
                fig, ax = plt.subplots()
                validation_summary["feature_importance"].head(20).plot.bar(ax=ax)
                ax.set_title("Feature importance (mean)")
                st.pyplot(fig)
                try:
                    add_plot_to_session(fig, title="validation_feature_importance", page="Validation", kind="feature_importance")
                except Exception:
                    pass
            else:
                st.warning("Feature importance not available for this model; permutation importance may have failed.")

        except Exception as e:
            st.error(f"Validation error: {e}")

    render_info_panel("Validation", current_model_name=model_name, current_hyperparams=DEFAULT_HYPERPARAMETERS.get(model_name, {}), model_list=models)


def render_modeling():
    """Training page (keeps name `render_modeling` for app.py compatibility).

    Trains final model, evaluates on test set if present, and allows exporting bundles.
    """
    st.header("🎓 Training")

    if "df" not in st.session_state or st.session_state.get("df") is None:
        st.warning("No dataset available. Upload data on the Data page first.")
        if st.button("Open Data Page"):
            st.session_state["page"] = "Data"
        render_info_panel("Training", model_list=_available_models())
        return

    # choose model
    models = _available_models()
    model_name = st.selectbox("Model family", models, index=0, key="train_model_family")
    st.subheader("Hyperparameters")
    user_params = {}
    defaults = DEFAULT_HYPERPARAMETERS.get(model_name, {})
    for k, v in defaults.items():
        if isinstance(v, bool):
            user_params[k] = st.checkbox(k, value=v)
        elif isinstance(v, int):
            user_params[k] = int(st.number_input(k, value=int(v)))
        elif isinstance(v, float):
            user_params[k] = float(st.number_input(k, value=float(v)))
        else:
            user_params[k] = st.text_input(k, value="" if v is None else str(v))

    # choose training dataset
    train_sources = ["pre_X_train"]
    if st.session_state.get("train_subset_indices") is not None:
        train_sources.append("train_subset")
    if st.session_state.get("sampled_X_train") is not None:
        train_sources.append("sampled_X_train")
    if st.session_state.get("resampled_pre_X_train") is not None:
        train_sources.append("resampled_pre_X_train")

    train_choice = st.selectbox("Training dataset", train_sources, index=len(train_sources) - 1, key="train_choice_select")
    if train_choice == "pre_X_train":
        X_train = st.session_state.get("pre_X_train")
        y_train = st.session_state.get("pre_y_train")
    elif train_choice == "train_subset":
        indices = st.session_state.get("train_subset_indices")
        X_train = st.session_state.get("pre_X_train")
        y_train = st.session_state.get("pre_y_train")
        if X_train is not None and indices is not None:
            X_train = X_train.loc[indices]
            y_train = y_train.loc[indices]
    else:
        X_train = st.session_state.get(train_choice)
        y_train = st.session_state.get(train_choice.replace("X", "y"))

    if X_train is None or y_train is None:
        st.error("Selected training data not available.")
        return

    if st.button("Show random 5 rows", key="train_random5"):
        try:
            st.dataframe(fix_arrow_compatibility(X_train.sample(5)))
        except Exception:
            st.dataframe(fix_arrow_compatibility(X_train.head()))

    # Feature selection mode for training
    train_features_mode = st.radio("Features to apply:", ["All", "Select Features"], horizontal=True, index=0, key="train_features_mode")
    train_selected_features = None
    if train_features_mode == "Select Features":
        opts = list(X_train.columns) if X_train is not None else []
        train_selected_features = st.multiselect("Select features to include", options=opts, key="train_selected_features")

    if st.button("Train Final Model"):
        try:
            seed = int(st.session_state.get("global_seed", 42))
            params = {}
            for kk, vv in user_params.items():
                if isinstance(DEFAULT_HYPERPARAMETERS.get(model_name, {}).get(kk), bool):
                    params[kk] = bool(vv)
                elif isinstance(DEFAULT_HYPERPARAMETERS.get(model_name, {}).get(kk), int):
                    try:
                        params[kk] = int(vv)
                    except Exception:
                        params[kk] = DEFAULT_HYPERPARAMETERS[model_name][kk]
                elif isinstance(DEFAULT_HYPERPARAMETERS.get(model_name, {}).get(kk), float):
                    try:
                        params[kk] = float(vv)
                    except Exception:
                        params[kk] = DEFAULT_HYPERPARAMETERS[model_name][kk]
                else:
                    if vv == "":
                        params[kk] = DEFAULT_HYPERPARAMETERS[model_name][kk]
                    else:
                        params[kk] = vv

            # apply feature selection if requested
            if train_features_mode == "Select Features":
                if not train_selected_features:
                    st.error("No features selected for training. Choose 'All' or pick features.")
                    return
                X_train_use = X_train[train_selected_features]
            else:
                X_train_use = X_train

            model = _instantiate_model(model_name, params)
            model.fit(X_train_use, y_train)
            st.session_state["trained_model"] = model

            # evaluate on hold-out test if available
            if st.session_state.get("pre_X_test") is not None and st.session_state.get("pre_y_test") is not None:
                X_test = st.session_state.get("pre_X_test")
                y_test = st.session_state.get("pre_y_test")
                # respect selected features when evaluating on test set
                if train_features_mode == "Select Features":
                    # use intersection to avoid missing columns
                    test_cols = [c for c in train_selected_features if c in X_test.columns]
                    if not test_cols:
                        st.warning("Selected features not present in test set; skipping test evaluation.")
                        X_test_use = None
                    else:
                        X_test_use = X_test[test_cols]
                else:
                    X_test_use = X_test

                if X_test_use is not None:
                    y_pred = model.predict(X_test_use)
                    # if classification, compute confusion matrix for test evaluation
                    if st.session_state.get("task_type") == "Classification":
                        try:
                            labels = np.unique(np.concatenate([y_test, y_pred]))
                            cm = confusion_matrix(y_test, y_pred, labels=labels)
                            eval_summary["confusion_matrix"] = cm
                            eval_summary["confusion_matrix_labels"] = list(labels)
                        except Exception:
                            pass
                if st.session_state.get("task_type") == "Classification":
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
                    try:
                        if hasattr(model, "predict_proba"):
                            prob = model.predict_proba(X_test)
                            if prob.shape[1] > 1:
                                roc = roc_auc_score(y_test, prob, multi_class="ovr")
                            else:
                                roc = roc_auc_score(y_test, prob[:, 1])
                        else:
                            roc = None
                    except Exception:
                        roc = None
                    eval_summary = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}
                else:
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = float(np.sqrt(mse))
                    r2 = r2_score(y_test, y_pred)
                    eval_summary = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

                st.session_state["training_evaluation_summary"] = eval_summary
                st.write(eval_summary)
            else:
                st.info("No hold-out test set found (split). Training completed on selected training data.")

            # build a session preprocessor object capturing fitted transformers
            preprocessor = SessionPreprocessor(
                imputation_transformers=st.session_state.get("imputation_transformers", {}),
                scalers=st.session_state.get("scalers", {}),
                encoders=st.session_state.get("encoders", {}),
                preprocessing_steps=st.session_state.get("preprocessing_steps", []),
            )
            st.session_state["trained_preprocessor"] = preprocessor
            st.success("Model trained and preprocessor prepared in session.")

        except Exception as e:
            st.error(f"Training error: {e}")

    st.markdown("---")
    st.subheader("Save Model Bundle")
    bundle_name = st.text_input("Model name (file will be created under models/)", value=f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    if st.button("Save Model Bundle"):
        try:
            if DISABLE_BUNDLE_SAVING:
                st.warning(
                    "Saving model bundles is temporarily disabled to avoid repository bloat. "
                    "No files will be written to /models. To re-enable, set DISABLE_BUNDLE_SAVING=False in data_modeling.py."
                )
                st.session_state["last_saved_bundle"] = None
            else:
                models_dir = os.path.join(os.path.dirname(__file__), "models")
                os.makedirs(models_dir, exist_ok=True)
                if "trained_model" not in st.session_state or st.session_state.get("trained_model") is None:
                    st.error("No trained model in session. Train a model first.")
                else:
                    model_obj = st.session_state.get("trained_model")
                    preproc_obj = st.session_state.get("trained_preprocessor")
                    # determine selected features for metadata (respect train UI)
                    if st.session_state.get("train_features_mode") == "Select Features":
                        sel_feats_meta = st.session_state.get("train_selected_features", []) or []
                    else:
                        sel_feats_meta = list(st.session_state.get("pre_X_train").columns) if st.session_state.get("pre_X_train") is not None else []

                    metadata = {
                        "model_name": bundle_name,
                        "saved_at": datetime.utcnow().isoformat(),
                        "random_seed": int(st.session_state.get("global_seed", 42)),
                        "task_type": st.session_state.get("task_type"),
                        "target_column": st.session_state.get("target_column"),
                        "preprocessing_steps": st.session_state.get("preprocessing_steps", []),
                        "cleansing_steps": st.session_state.get("cleansing_steps", []),
                        "selected_features": sel_feats_meta,
                        "hyperparameters": user_params,
                        "validation_summary": st.session_state.get("validation_summary"),
                        "evaluation_summary": st.session_state.get("training_evaluation_summary"),
                    }

                    bundle = {"model": model_obj, "preprocessor": preproc_obj, "metadata": metadata}
                    bundle_path = os.path.join(models_dir, f"{bundle_name}.pkl")
                    joblib.dump(bundle, bundle_path)

                    # save simple metadata file
                    meta_path = os.path.join(models_dir, f"{bundle_name}_metadata.txt")
                    with open(meta_path, "w", encoding="utf-8") as f:
                        for k, v in metadata.items():
                            f.write(f"{k}: {v}\n")

                    # Save any registered evaluation plots into models/evaluation_plots/<bundle_name>/
                    try:
                        eval_plots = st.session_state.get("evaluation_plots_temp", {})
                        eval_meta = st.session_state.get("evaluation_plots_meta", {})
                        if eval_plots:
                            eval_root = os.path.join(models_dir, "evaluation_plots")
                            os.makedirs(eval_root, exist_ok=True)
                            eval_dir = os.path.join(eval_root, bundle_name)
                            os.makedirs(eval_dir, exist_ok=True)
                            for fname, data in eval_plots.items():
                                meta = eval_meta.get(fname, {})
                                created_at = meta.get("created_at") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                                page = meta.get("page", "")
                                kind = meta.get("kind", "plot")
                                title = (meta.get("title") or fname).replace(" ", "_")

                                if kind == "confusion_matrix" and page in ("Validation", "Training"):
                                    desired_name = f"{bundle_name}_{created_at}_confusion_matrix.png"
                                else:
                                    desired_name = f"{created_at}_{title}.png"

                                # sanitize filename
                                desired_name = desired_name.replace(":", "_").replace("/", "_").replace("\\\\", "_")

                                out_path = os.path.join(eval_dir, desired_name)
                                # avoid overwriting; append counter if exists
                                base, ext = os.path.splitext(out_path)
                                counter = 1
                                while os.path.exists(out_path):
                                    out_path = f"{base}_{counter}{ext}"
                                    counter += 1

                                with open(out_path, "wb") as pf:
                                    pf.write(data)
                    except Exception:
                        # non-fatal if saving plots fails
                        pass
                    st.session_state["last_saved_bundle"] = bundle_path
                    st.success(f"Saved model bundle to {bundle_path}")

        except Exception as e:
            st.error(f"Error saving bundle: {e}")

    # --- Export bundle for upload (download .pkl) ---
    st.markdown("---")
    st.subheader("Export Model Bundle (.pkl) for Submission")
    export_filename = st.text_input("Export filename", value=f"{bundle_name}.pkl", key="export_filename")
    if st.session_state.get("trained_model") is None:
        st.info("No trained model available to export. Train a model first.")
    else:
        try:
            # build export metadata similar to save flow
            if st.session_state.get("train_features_mode") == "Select Features":
                sel_feats_meta = st.session_state.get("train_selected_features", []) or []
            else:
                sel_feats_meta = list(st.session_state.get("pre_X_train").columns) if st.session_state.get("pre_X_train") is not None else []

            metadata_export = {
                "model_name": bundle_name,
                "saved_at": datetime.utcnow().isoformat(),
                "random_seed": int(st.session_state.get("global_seed", 42)),
                "task_type": st.session_state.get("task_type"),
                "target_column": st.session_state.get("target_column"),
                "preprocessing_steps": st.session_state.get("preprocessing_steps", []),
                "cleansing_steps": st.session_state.get("cleansing_steps", []),
                "selected_features": sel_feats_meta,
                "hyperparameters": user_params,
                "validation_summary": st.session_state.get("validation_summary"),
                "evaluation_summary": st.session_state.get("training_evaluation_summary"),
            }

            bundle_export = {
                "model": st.session_state.get("trained_model"),
                "preprocessor": st.session_state.get("trained_preprocessor"),
                "metadata": metadata_export,
            }

            try:
                pkl_bytes = pickle.dumps(bundle_export, protocol=pickle.HIGHEST_PROTOCOL)
                st.download_button("Download model bundle (.pkl)", data=pkl_bytes, file_name=export_filename or f"{bundle_name}.pkl", mime="application/octet-stream", key="download_export_bundle")
            except Exception as e:
                st.error(f"Error preparing export bundle: {e}")
        except Exception as e:
            st.error(f"Error preparing export UI: {e}")

    render_info_panel("Training", current_model_name=model_name, current_hyperparams=DEFAULT_HYPERPARAMETERS.get(model_name, {}), model_list=models)


def render_evaluation_summary():
    st.header("📋 Evaluation Summary")
    st.write("Aggregate and compare saved evaluation results from Validation and Training.")

    val = st.session_state.get("validation_summary")
    train = st.session_state.get("training_evaluation_summary")

    # Validation section
    st.subheader("Validation Results")
    if not val:
        st.info("No validation results found. Run validation to populate this section.")
    else:
        st.write("**Summary metrics**")
        if val.get("metrics") is not None:
            st.write(val.get("metrics"))
        else:
            st.info("No validation metrics recorded.")

        # feature importance
        if val.get("feature_importance") is not None:
            try:
                fig, ax = plt.subplots()
                val.get("feature_importance").head(20).plot.bar(ax=ax)
                ax.set_title("Validation: Feature importance (mean)")
                st.pyplot(fig)
            except Exception:
                st.write("(Feature importance not plottable)")

        # confusion matrix
        if st.session_state.get("task_type") == "Classification" and val.get("confusion_matrix") is not None:
            cm = val.get("confusion_matrix")
            labels = val.get("confusion_matrix_labels") or list(range(cm.shape[0]))
            try:
                fig, ax = plt.subplots()
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_yticklabels(labels)
                thresh = cm.max() / 2.0 if cm.size else 0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=("white" if cm[i, j] > thresh else "black"))
                ax.set_ylabel('True label')
                ax.set_xlabel('Predicted label')
                st.markdown("**Confusion Matrix (Validation)**")
                st.pyplot(fig)
                try:
                    add_plot_to_session(fig, title="validation_confusion_matrix", page="Validation", kind="confusion_matrix")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Unable to render validation confusion matrix: {e}")

    st.markdown("---")

    # Training section
    st.subheader("Training Results")
    if not train:
        st.info("No training evaluation found. Train a model to populate this section.")
    else:
        st.write("**Summary metrics**")
        if train.get("metrics") is not None:
            st.write(train.get("metrics"))
        else:
            st.info("No training metrics recorded.")

        # confusion matrix
        if st.session_state.get("task_type") == "Classification" and train.get("confusion_matrix") is not None:
            cm = train.get("confusion_matrix")
            labels = train.get("confusion_matrix_labels") or list(range(cm.shape[0]))
            try:
                fig, ax = plt.subplots()
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_yticklabels(labels)
                thresh = cm.max() / 2.0 if cm.size else 0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=("white" if cm[i, j] > thresh else "black"))
                ax.set_ylabel('True label')
                ax.set_xlabel('Predicted label')
                st.markdown("**Confusion Matrix (Training)**")
                st.pyplot(fig)
                try:
                    add_plot_to_session(fig, title="training_confusion_matrix", page="Training", kind="confusion_matrix")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Unable to render training confusion matrix: {e}")

        # feature importance if present in metadata
        # training feature importance may be stored in session metadata; try to render if present
        tr_feat = st.session_state.get("validation_summary", {}).get("feature_importance") if st.session_state.get("validation_summary") else None
        if tr_feat is not None:
            try:
                fig, ax = plt.subplots()
                tr_feat.head(20).plot.bar(ax=ax)
                ax.set_title("Training: Feature importance")
                st.pyplot(fig)
                try:
                    add_plot_to_session(fig, title="training_feature_importance", page="Training", kind="feature_importance")
                except Exception:
                    pass
            except Exception:
                pass

    render_info_panel("Evaluation Summary")
