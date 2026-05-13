import streamlit as st
import pandas as pd
import numpy as np
import random
from ui_components import render_info_panel, fix_arrow_compatibility, available_columns, available_numeric_columns
from sklearn.base import BaseEstimator, TransformerMixin
from state_manager import clear_downstream_from_cleansing


class CapperTransformer(BaseEstimator, TransformerMixin):
    """Transformer that computes clipping bounds per-column using IQR or z-score and applies clipping.

    Usage:
        capper = CapperTransformer(method='IQR', k=1.5, columns=['col1','col2'])
        capper.fit(df)
        df2 = capper.transform(df)
    """
    def __init__(self, method: str = "IQR", k: float = 1.5, columns: list = None):
        self.method = method
        self.k = float(k) if k is not None else 1.5
        self.columns = list(columns) if columns is not None else None
        self.bounds_ = {}

    def fit(self, X, y=None):
        cols = self.columns or list(X.columns)
        for c in cols:
            try:
                s = pd.to_numeric(X[c], errors="coerce").dropna()
            except Exception:
                s = pd.Series([], dtype=float)
            if s.empty:
                self.bounds_[c] = (np.nan, np.nan)
                continue
            if self.method == "IQR":
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                low = q1 - self.k * iqr
                high = q3 + self.k * iqr
            else:
                mean = s.mean()
                std = s.std()
                low = mean - self.k * std
                high = mean + self.k * std
            self.bounds_[c] = (low, high)
        return self

    def transform(self, X):
        X2 = X.copy()
        for c, (low, high) in self.bounds_.items():
            if c not in X2.columns:
                continue
            if pd.isna(low) or pd.isna(high):
                continue
            try:
                X2[c] = pd.to_numeric(X2[c], errors="coerce")
                X2[c] = X2[c].clip(lower=low, upper=high)
            except Exception:
                # if conversion/clip fails, skip column
                continue
        return X2


def _invalidate_after_cleansing_mutation():
    clear_downstream_from_cleansing(st.session_state)


def render_cleansing():
    st.header("✨ Cleansing")
    st.write("Imputation and basic cleansing operations. Use with an uploaded dataset from the Data page.")

    if "df" not in st.session_state or st.session_state.get("df") is None:
        st.warning("No dataset loaded. Please upload a dataset on the Data page first.")
        if st.button("Open Data Page"):
            st.session_state["page"] = "Data"
        render_info_panel("Cleansing")
        return

    df = st.session_state.get("df")

    # Preview (moved above tabs): show random 5 rows or head for selected subset when split active
    st.subheader("Preview Working Dataset")
    if st.session_state.get("split_done"):
        preview_choice = st.radio("Preview subset", ["Train (subset)", "Test (subset)"], index=0, key="cleansing_preview_subset")
        if preview_choice.startswith("Train"):
            df_preview = st.session_state.get("pre_X_train")
        else:
            df_preview = st.session_state.get("pre_X_test")
    else:
        df_preview = st.session_state.get("df")

    if df_preview is None:
        st.info("No data available for preview.")
    else:
        if st.button("Random 5 rows", key="random_5_rows_cleansing"):
            try:
                sample_df = df_preview.sample(5) if len(df_preview) >= 5 else df_preview.head()
                st.dataframe(fix_arrow_compatibility(sample_df))
            except Exception as e:
                st.error(f"Unable to sample dataset: {e}")
        else:
            try:
                st.dataframe(fix_arrow_compatibility(df_preview.head()))
            except Exception:
                pass

    # Top navbar (with emojis)
    tabs = st.tabs(["❗ Missing Values", "🔁 Duplicates", "⚠️ Outliers", "🗑️ Remove Columns", "🔤 Data Types"])

    # --- Missing Values ---
    with tabs[0]:
        st.subheader("Missing Values")
        sub = st.tabs(["🗑️ Deletion", "🔧 Imputation", "➖ Map → Missing"])

        # Deletion
        with sub[0]:
            st.write("Drop rows or columns based on missingness.")
            cols = st.multiselect("Columns to consider for deletion (empty = any column)", options=available_columns(), key="del_na_cols")
            how = st.selectbox("Drop rows where:", ["any", "all"], index=0, key="del_how")
            if st.session_state.get("split_done"):
                del_subsets = st.multiselect("Select subset(s) to process:", options=["Train", "Test"], default=["Train"], key="del_subsets")
            else:
                del_subsets = ["Train"]

            if st.button("Apply Deletion", key="apply_del_na"):
                if st.session_state.get("split_done"):
                    for subset in del_subsets:
                        if subset == "Train" and st.session_state.get("pre_X_train") is not None:
                            px = st.session_state.get("pre_X_train")
                            before = len(px)
                            if cols:
                                px = px.dropna(axis=0, how=how, subset=cols)
                            else:
                                px = px.dropna(axis=0, how=how)
                            st.session_state["pre_X_train"] = px
                            # align y
                            if st.session_state.get("pre_y_train") is not None:
                                st.session_state["pre_y_train"] = st.session_state["pre_y_train"].loc[px.index]
                            st.success(f"Dropped {before - len(px)} rows from Train by missingness.")
                            _invalidate_after_cleansing_mutation()
                        if subset == "Test" and st.session_state.get("pre_X_test") is not None:
                            px = st.session_state.get("pre_X_test")
                            before = len(px)
                            if cols:
                                px = px.dropna(axis=0, how=how, subset=cols)
                            else:
                                px = px.dropna(axis=0, how=how)
                            st.session_state["pre_X_test"] = px
                            if st.session_state.get("pre_y_test") is not None:
                                st.session_state["pre_y_test"] = st.session_state["pre_y_test"].loc[px.index]
                            st.success(f"Dropped {before - len(px)} rows from Test by missingness.")
                            _invalidate_after_cleansing_mutation()
                else:
                    before = len(df)
                    if cols:
                        df2 = df.dropna(axis=0, how=how, subset=cols)
                    else:
                        df2 = df.dropna(axis=0, how=how)
                    st.session_state["df"] = df2
                    st.success(f"Dropped {before - len(df2)} rows from master dataset by missingness.")
                    _invalidate_after_cleansing_mutation()

        # Imputation
        with sub[1]:
            st.write("Imputation (fit/transform semantics). Fits must include Train when split exists.")
            method = st.selectbox("Imputation Method", ["Mean", "Median", "Mode", "Custom"], index=0, key="impute_method")
            cols = st.multiselect("Select columns to impute", options=available_columns(), key="impute_columns")
            action = st.radio("Action", ["fit", "transform", "fit_transform"], index=0, key="impute_action")
            if st.session_state.get("split_done"):
                impute_subsets = st.multiselect("Select subset(s) to process:", options=["Train", "Test"], default=["Train"], key="impute_subsets")
            else:
                impute_subsets = ["Train"]

            custom_value = None
            if method == "Custom":
                custom_value = st.text_input("Custom fill value (literal). For numeric columns, provide a numeric string.", key="impute_custom_value")

            if st.button("Apply Imputation", key="apply_impute"):
                if not cols:
                    st.error("Select one or more columns to impute before applying.")
                else:
                    if "cleansing_steps" not in st.session_state:
                        st.session_state["cleansing_steps"] = []
                    if "imputation_transformers" not in st.session_state:
                        st.session_state["imputation_transformers"] = {}

                    cols_sorted = sorted(cols)
                    transformer_key = f"imputer::{method}::" + ",".join(cols_sorted)

                    from sklearn.impute import SimpleImputer

                    numeric_required = method in ("Mean", "Median")

                    # Determine which dataframe(s) to validate based on selected subsets
                    if st.session_state.get("split_done"):
                        datasets = []
                        if "Train" in impute_subsets and st.session_state.get("pre_X_train") is not None:
                            datasets.append(("Train", st.session_state.get("pre_X_train")))
                        if "Test" in impute_subsets and st.session_state.get("pre_X_test") is not None:
                            datasets.append(("Test", st.session_state.get("pre_X_test")))
                        if not datasets:
                            datasets = [("Master", df)]
                    else:
                        datasets = [("Master", df)]

                    # Validate numeric requirements. Allow coercible columns (e.g., numeric strings).
                    non_convertible = []
                    coercible = []
                    for c in cols_sorted:
                        ok_all = True
                        for name, dsrc in datasets:
                            if dsrc is None or c not in dsrc.columns:
                                continue
                            # If already numeric dtype, it's OK
                            if pd.api.types.is_numeric_dtype(dsrc[c]):
                                continue
                            # Try coercion to numeric: if after coercion there's at least one non-null value, consider coercible
                            coerced = pd.to_numeric(dsrc[c], errors="coerce")
                            if coerced.dropna().empty:
                                ok_all = False
                                break
                        if not ok_all:
                            non_convertible.append(c)
                        else:
                            # If any source wasn't numeric dtype, mark coercion needed
                            needs_coerce = any(not pd.api.types.is_numeric_dtype(dsrc[c]) for _, dsrc in datasets if dsrc is not None and c in dsrc.columns)
                            if needs_coerce:
                                coercible.append(c)

                    if numeric_required and non_convertible:
                        st.error(f"Selected columns {non_convertible} are not numeric and cannot be coerced; Mean/Median imputation requires numeric columns.")
                    else:
                        # If some columns are coercible (e.g., numeric strings), coerce them in-place for the selected datasets
                        if coercible:
                            st.info(f"Coercing columns to numeric for imputation: {coercible}")
                            for _, dsrc in datasets:
                                if dsrc is None:
                                    continue
                                for c in coercible:
                                    if c in dsrc.columns and not pd.api.types.is_numeric_dtype(dsrc[c]):
                                        try:
                                            dsrc[c] = pd.to_numeric(dsrc[c], errors="coerce")
                                        except Exception:
                                            pass
                            # write back coerced datasets to session/master
                            if st.session_state.get("split_done"):
                                if "Train" in impute_subsets and st.session_state.get("pre_X_train") is not None:
                                    st.session_state["pre_X_train"] = st.session_state.get("pre_X_train")
                                if "Test" in impute_subsets and st.session_state.get("pre_X_test") is not None:
                                    st.session_state["pre_X_test"] = st.session_state.get("pre_X_test")
                            else:
                                st.session_state["df"] = df
                    if method == "Mean":
                        strategy = "mean"
                    elif method == "Median":
                        strategy = "median"
                    elif method == "Mode":
                        strategy = "most_frequent"
                    else:
                        strategy = "constant"

                    fill_value = None
                    if method == "Custom":
                        fill_value = custom_value

                    def _record_step():
                        step = {"type": "imputation", "method": method, "columns": cols_sorted, "action": action, "subsets": impute_subsets}
                        st.session_state["cleansing_steps"].append(step)

                    # transform-only
                    if action == "transform":
                        if transformer_key not in st.session_state["imputation_transformers"]:
                            st.error("❌ You must call fit or fit_transform first before transform only.")
                        else:
                            imputer = st.session_state["imputation_transformers"][transformer_key]
                            try:
                                if st.session_state.get("split_done"):
                                    for subset in impute_subsets:
                                        if subset == "Train" and st.session_state.get("pre_X_train") is not None:
                                            px = st.session_state.get("pre_X_train")
                                            transformed = imputer.transform(px[cols_sorted])
                                            px.loc[:, cols_sorted] = transformed
                                            st.session_state["pre_X_train"] = px
                                        if subset == "Test" and st.session_state.get("pre_X_test") is not None:
                                            px = st.session_state.get("pre_X_test")
                                            transformed = imputer.transform(px[cols_sorted])
                                            px.loc[:, cols_sorted] = transformed
                                            st.session_state["pre_X_test"] = px
                                else:
                                    transformed = imputer.transform(df[cols_sorted])
                                    df.loc[:, cols_sorted] = transformed
                                    st.session_state["df"] = df
                                _record_step()
                                st.success(f"Transformed {len(cols_sorted)} columns using existing imputer ({method}).")
                                _invalidate_after_cleansing_mutation()
                            except Exception as e:
                                st.error(f"Error during transform: {e}")

                    elif action == "fit":
                        try:
                            if st.session_state.get("split_done") and "Train" not in impute_subsets:
                                st.error("Fit operations must include the Train subset when a split exists.")
                            else:
                                if strategy == "constant":
                                    fv = fill_value
                                    try:
                                        fv = pd.to_numeric(fill_value)
                                    except Exception:
                                        pass
                                    imputer = SimpleImputer(strategy="constant", fill_value=fv)
                                else:
                                    imputer = SimpleImputer(strategy=strategy)

                                if st.session_state.get("split_done"):
                                    px = st.session_state.get("pre_X_train")
                                    imputer.fit(px[cols_sorted])
                                else:
                                    imputer.fit(df[cols_sorted])

                                st.session_state["imputation_transformers"][transformer_key] = imputer
                                _record_step()
                                st.success(f"Fitted imputer ({method}) for columns: {cols_sorted}")
                                _invalidate_after_cleansing_mutation()
                        except Exception as e:
                            st.error(f"Error during fit: {e}")

                    elif action == "fit_transform":
                        try:
                            if strategy == "constant":
                                fv = fill_value
                                try:
                                    fv = pd.to_numeric(fill_value)
                                except Exception:
                                    pass
                                imputer = SimpleImputer(strategy="constant", fill_value=fv)
                            else:
                                imputer = SimpleImputer(strategy=strategy)

                            if st.session_state.get("split_done"):
                                px_train = st.session_state.get("pre_X_train")
                                imputed_train = imputer.fit_transform(px_train[cols_sorted])
                                px_train.loc[:, cols_sorted] = imputed_train
                                st.session_state["pre_X_train"] = px_train
                                st.session_state["imputation_transformers"][transformer_key] = imputer
                                if "Test" in impute_subsets and st.session_state.get("pre_X_test") is not None:
                                    px_test = st.session_state.get("pre_X_test")
                                    transformed_test = imputer.transform(px_test[cols_sorted])
                                    px_test.loc[:, cols_sorted] = transformed_test
                                    st.session_state["pre_X_test"] = px_test
                                _record_step()
                                st.success(f"Applied fit_transform ({method}) on columns: {cols_sorted} for subsets: {impute_subsets}")
                                _invalidate_after_cleansing_mutation()
                            else:
                                transformed = imputer.fit_transform(df[cols_sorted])
                                st.session_state["imputation_transformers"][transformer_key] = imputer
                                df.loc[:, cols_sorted] = transformed
                                st.session_state["df"] = df
                                _record_step()
                                st.success(f"Applied fit_transform ({method}) on columns: {cols_sorted}")
                                _invalidate_after_cleansing_mutation()
                        except Exception as e:
                            st.error(f"Error during fit_transform: {e}")

        # Mapping tab
        with sub[2]:
            st.write("Convert a specific value to missing (NaN) for selected columns.")
            map_cols = st.multiselect("Columns to map to missing", options=available_columns(), key="map_to_missing_cols")
            map_type = st.radio("Value type", ["Numerical", "Categorical"], index=0, key="map_to_missing_type")
            map_value = st.text_input("Value to treat as missing (literal)", key="map_to_missing_value")
            if st.session_state.get("split_done"):
                map_subsets = st.multiselect("Select subset(s) to process:", options=["Train", "Test"], default=["Train"], key="map_to_missing_subsets")
            else:
                map_subsets = ["Train"]

            if st.button("Apply mapping", key="apply_map_missing"):
                if not map_cols:
                    st.error("Select one or more columns to map.")
                elif map_value is None or str(map_value).strip() == "":
                    st.error("Provide a value to map to missing.")
                else:
                    if "cleansing_steps" not in st.session_state:
                        st.session_state["cleansing_steps"] = []

                    def _apply_map_obj(obj, cols_local=None):
                        # obj: DataFrame or Series
                        try:
                            if isinstance(obj, pd.Series):
                                if map_type == "Numerical":
                                    try:
                                        mv = float(map_value)
                                    except Exception:
                                        st.error("Numeric mapping requires a numeric value.")
                                        return obj
                                    mask = pd.to_numeric(obj, errors="coerce") == mv
                                    obj.loc[mask] = np.nan
                                else:
                                    mask = obj.astype(str) == str(map_value)
                                    obj.loc[mask] = np.nan
                                return obj
                            else:
                                for c in (cols_local or []):
                                    if c not in obj.columns:
                                        continue
                                    if map_type == "Numerical":
                                        try:
                                            mv = float(map_value)
                                        except Exception:
                                            st.error("Numeric mapping requires a numeric value.")
                                            return obj
                                        if pd.api.types.is_numeric_dtype(obj[c]):
                                            mask = obj[c] == mv
                                        else:
                                            mask = pd.to_numeric(obj[c], errors="coerce") == mv
                                        obj.loc[mask, c] = np.nan
                                    else:
                                        mask = obj[c].astype(str) == str(map_value)
                                        obj.loc[mask, c] = np.nan
                                return obj
                        except Exception as e:
                            st.error(f"Mapping error: {e}")
                            return obj

                    try:
                        if st.session_state.get("split_done"):
                            for subset in map_subsets:
                                if subset == "Train":
                                    # apply to features
                                    if st.session_state.get("pre_X_train") is not None:
                                        px = st.session_state.get("pre_X_train")
                                        cols_here = [c for c in map_cols if c in px.columns]
                                        if cols_here:
                                            px = _apply_map_obj(px, cols_here)
                                            st.session_state["pre_X_train"] = px
                                    # if target column selected, apply to y
                                    target = st.session_state.get("target_column")
                                    if target in map_cols and st.session_state.get("pre_y_train") is not None:
                                        py = st.session_state.get("pre_y_train")
                                        py = _apply_map_obj(py)
                                        st.session_state["pre_y_train"] = py
                                if subset == "Test":
                                    if st.session_state.get("pre_X_test") is not None:
                                        px = st.session_state.get("pre_X_test")
                                        cols_here = [c for c in map_cols if c in px.columns]
                                        if cols_here:
                                            px = _apply_map_obj(px, cols_here)
                                            st.session_state["pre_X_test"] = px
                                    target = st.session_state.get("target_column")
                                    if target in map_cols and st.session_state.get("pre_y_test") is not None:
                                        py = st.session_state.get("pre_y_test")
                                        py = _apply_map_obj(py)
                                        st.session_state["pre_y_test"] = py
                        else:
                            df2 = st.session_state.get("df")
                            df2 = _apply_map_obj(df2, map_cols)
                            st.session_state["df"] = df2

                        st.session_state["cleansing_steps"].append({"type": "map_to_missing", "columns": sorted(map_cols), "value": map_value, "value_type": map_type, "subsets": map_subsets})
                        st.success(f"Mapped value '{map_value}' to missing for columns: {map_cols} on subsets: {map_subsets}")
                        _invalidate_after_cleansing_mutation()
                    except Exception as e:
                        st.error(f"Error applying mapping: {e}")

    # --- Duplicates ---
    with tabs[1]:
        st.subheader("Duplicates")
        if st.session_state.get("split_done"):
            dup_subsets = st.multiselect("Select subset(s) to check for duplicates:", options=["Train", "Test"], default=["Train"], key="dup_subsets")
        else:
            dup_subsets = ["Train"]

        # Mode: Rows or Columns
        dup_mode = st.radio("Duplicate Type", ["Rows", "Columns"], horizontal=True, index=0, key="dup_mode")

        # Preview area: show either a random pair of duplicate rows (Rows mode)
        # or a one-line listing of duplicate column groups (Columns mode).
        # Use the first selected subset for the preview placement.
        preview_subset = dup_subsets[0] if dup_subsets else ("Train")
        if st.session_state.get("split_done"):
            if preview_subset == "Train":
                px_preview = st.session_state.get("pre_X_train")
            else:
                px_preview = st.session_state.get("pre_X_test")
        else:
            px_preview = df

        if px_preview is None:
            st.info("No data available for duplicate preview.")
        else:
            if dup_mode == "Rows":
                # Find groups of duplicate rows (by all columns). Show 2 random rows from a random duplicate group.
                try:
                    dup_mask = px_preview.duplicated(keep=False)
                    if dup_mask.any():
                        dup_rows = px_preview[dup_mask]
                        # group by all columns to find groups of identical rows
                        try:
                            grouped = dup_rows.groupby(list(px_preview.columns), sort=False)
                            groups = [g.index.tolist() for _, g in grouped]
                        except Exception:
                            # fallback: group by stringified row contents
                            temp = dup_rows.astype(str).agg("||".join, axis=1)
                            groups_map = {}
                            for idx, v in temp.items():
                                groups_map.setdefault(v, []).append(idx)
                            groups = [g for g in groups_map.values() if len(g) > 1]

                        groups = [g for g in groups if len(g) > 1]
                        if groups:
                            # persistent preview key per subset
                            preview_key = f"dup_preview_indices::{preview_subset}"
                            btn_key = f"dup_random_btn::{preview_subset}"

                            # Button above the sample table to re-sample a random duplicate pair
                            if st.button("Random 2 rows", key=btn_key):
                                # choose a random duplicate group and sample up to 2 indices
                                chosen = random.choice(groups)
                                chosen_sample = random.sample(chosen, k=min(2, len(chosen)))
                                st.session_state[preview_key] = chosen_sample

                            # Use stored sample if available, otherwise pick one once
                            chosen_sample = st.session_state.get(preview_key)
                            if not chosen_sample:
                                try:
                                    chosen = random.choice(groups)
                                    chosen_sample = random.sample(chosen, k=min(2, len(chosen)))
                                    st.session_state[preview_key] = chosen_sample
                                except Exception:
                                    chosen_sample = None

                            if chosen_sample:
                                try:
                                    sample_df = px_preview.loc[chosen_sample]
                                    st.markdown("**Example duplicate rows (random pair):**")
                                    try:
                                        st.dataframe(fix_arrow_compatibility(sample_df))
                                    except Exception:
                                        st.write(sample_df.head(2))
                                except Exception:
                                    st.info("Previously sampled rows are no longer available; try Random 2 rows again.")
                            else:
                                st.info("No duplicate row groups found for preview.")
                        else:
                            st.info("No duplicate row groups found for preview.")
                    else:
                        st.info("No duplicate rows found.")
                except Exception:
                    st.info("Duplicate row preview unavailable.")

            else:
                # Columns mode: detect identical columns and present them as groups
                try:
                    col_map = {}
                    for c in px_preview.columns:
                        try:
                            # string representation of the column values (stable for comparison)
                            sig = tuple(px_preview[c].fillna("__NA__").astype(str).tolist())
                        except Exception:
                            try:
                                sig = tuple(px_preview[c].fillna("__NA__").apply(lambda x: str(x)).tolist())
                            except Exception:
                                continue
                        col_map.setdefault(sig, []).append(c)

                    col_groups = [g for g in col_map.values() if len(g) > 1]
                    if col_groups:
                        groups_str = ", ".join("[" + ", ".join(g) + "]" for g in col_groups)
                        st.write(f"Duplicate column groups: {groups_str}")
                    else:
                        st.info("No duplicate column groups found.")
                except Exception:
                    st.info("Duplicate column group detection unavailable.")

        # report counts (explicitly say rows)
        if st.session_state.get("split_done"):
            if "Train" in dup_subsets and st.session_state.get("pre_X_train") is not None:
                train_dup = st.session_state.get("pre_X_train").duplicated().sum()
                st.write(f"Train duplicates (rows): {train_dup}")
            if "Test" in dup_subsets and st.session_state.get("pre_X_test") is not None:
                test_dup = st.session_state.get("pre_X_test").duplicated().sum()
                st.write(f"Test duplicates (rows): {test_dup}")
        else:
            dup_count = df.duplicated().sum()
            st.write(f"Master dataset duplicates (rows): {dup_count}")

        if st.button("Remove Duplicates", key="remove_duplicates"):
            if st.session_state.get("split_done"):
                for subset in dup_subsets:
                    if subset == "Train" and st.session_state.get("pre_X_train") is not None:
                        px = st.session_state.get("pre_X_train")
                        before = len(px)
                        px = px.drop_duplicates()
                        st.session_state["pre_X_train"] = px
                        if st.session_state.get("pre_y_train") is not None:
                            st.session_state["pre_y_train"] = st.session_state["pre_y_train"].loc[px.index]
                        st.success(f"Removed {before - len(px)} duplicate rows from Train.")
                        _invalidate_after_cleansing_mutation()
                    if subset == "Test" and st.session_state.get("pre_X_test") is not None:
                        px = st.session_state.get("pre_X_test")
                        before = len(px)
                        px = px.drop_duplicates()
                        st.session_state["pre_X_test"] = px
                        if st.session_state.get("pre_y_test") is not None:
                            st.session_state["pre_y_test"] = st.session_state["pre_y_test"].loc[px.index]
                        st.success(f"Removed {before - len(px)} duplicate rows from Test.")
                        _invalidate_after_cleansing_mutation()
            else:
                before = len(df)
                df2 = df.drop_duplicates()
                st.session_state["df"] = df2
                st.success(f"Removed {before - len(df2)} duplicate rows from master dataset.")
                _invalidate_after_cleansing_mutation()

    # --- Outliers ---
    with tabs[2]:
        st.subheader("Outliers")
        num_cols = available_numeric_columns()
        cols = st.multiselect("Select numeric columns for outlier detection", options=num_cols, key="outlier_cols")
        method = st.selectbox("Method", ["IQR", "Z-score"], key="outlier_method")
        if method == "IQR":
            k = st.number_input("IQR multiplier (k)", value=1.5, step=0.1, key="outlier_iqr_k")
        else:
            k = st.number_input("Z-score threshold", value=3.0, step=0.1, key="outlier_z_k")
        action = st.selectbox("Action", ["Remove", "Cap", "Convert to NaN"], index=0, key="outlier_action")
        cap_mode = None
        if action == "Cap":
            cap_mode = st.radio("Cap action", ["fit", "transform", "fit_transform"], index=2, key="outlier_cap_mode")
        if st.session_state.get("split_done"):
            out_subsets = st.multiselect("Select subset(s) to process:", options=["Train", "Test"], default=["Train"], key="out_subsets")
        else:
            out_subsets = ["Train"]

        # Preview: compute detected outliers and bounds for the currently selected columns/subsets
        if cols:
            try:
                total_outliers = 0
                overall_lows = []
                overall_highs = []

                subsets_to_check = out_subsets if st.session_state.get("split_done") else ["Master"]
                for subset in subsets_to_check:
                    if subset == "Master":
                        px = df
                    else:
                        key_x = f"pre_X_{subset.lower()}"
                        px = st.session_state.get(key_x)
                    if px is None:
                        continue

                    idx_outlier = set()
                    for c in cols:
                        try:
                            s = px[c].dropna()
                        except Exception:
                            continue
                        if s.empty:
                            continue
                        if method == "IQR":
                            q1 = s.quantile(0.25)
                            q3 = s.quantile(0.75)
                            iqr = q3 - q1
                            low = q1 - float(k) * iqr
                            high = q3 + float(k) * iqr
                        else:
                            mu = s.mean()
                            sigma = s.std()
                            low = mu - float(k) * sigma
                            high = mu + float(k) * sigma

                        overall_lows.append(low)
                        overall_highs.append(high)

                        mask = (pd.to_numeric(px[c], errors="coerce") < low) | (pd.to_numeric(px[c], errors="coerce") > high)
                        idx_outlier.update(px[mask].index.tolist())

                    total_outliers += len(idx_outlier)

                if overall_lows and overall_highs:
                    overall_low = min(overall_lows)
                    overall_high = max(overall_highs)
                    fmt = lambda v: f"{v:.6g}" if pd.notna(v) else "nan"
                    st.write(f"Detected outliers outside the range [{fmt(overall_low)}, {fmt(overall_high)}]: {total_outliers}")
                else:
                    st.write(f"Detected outliers outside the range [nan, nan]: 0")
            except Exception:
                st.write("Detected outliers: N/A")

        if st.button("Apply Outlier Handling", key="apply_outlier"):
            if not cols:
                st.error("Select one or more numeric columns to detect outliers.")
            else:
                cols_sorted = sorted(cols)

                # Ensure storage for capper transformers
                if "capper_transformers" not in st.session_state:
                    st.session_state["capper_transformers"] = {}

                transformer_key = f"capper::{method}::{k}::" + ",".join(cols_sorted)

                def _detect_iqr_bounds(series, multiplier=1.5):
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    return q1 - multiplier * iqr, q3 + multiplier * iqr

                # Remove action (unchanged)
                if action == "Remove":
                    def _handle_on_df(px):
                        idx_to_drop = set()
                        for c in cols_sorted:
                            s = px[c].dropna()
                            if method == "IQR":
                                low, high = _detect_iqr_bounds(s, multiplier=k)
                                mask = (px[c] < low) | (px[c] > high)
                            else:
                                mean = s.mean()
                                std = s.std()
                                mask = ((px[c] - mean).abs() > k * std)
                            idx_to_drop.update(px[mask].index.tolist())
                        before = len(px)
                        px = px.drop(index=list(idx_to_drop))
                        return px, before - len(px)

                    if st.session_state.get("split_done"):
                        for subset in out_subsets:
                            if subset == "Train" and st.session_state.get("pre_X_train") is not None:
                                px = st.session_state.get("pre_X_train")
                                new_px, removed = _handle_on_df(px)
                                st.session_state["pre_X_train"] = new_px
                                if removed and st.session_state.get("pre_y_train") is not None:
                                    st.session_state["pre_y_train"] = st.session_state["pre_y_train"].loc[new_px.index]
                                if removed:
                                    st.success(f"Removed {removed} outlier rows from Train.")
                                    _invalidate_after_cleansing_mutation()
                            if subset == "Test" and st.session_state.get("pre_X_test") is not None:
                                px = st.session_state.get("pre_X_test")
                                new_px, removed = _handle_on_df(px)
                                st.session_state["pre_X_test"] = new_px
                                if removed and st.session_state.get("pre_y_test") is not None:
                                    st.session_state["pre_y_test"] = st.session_state["pre_y_test"].loc[new_px.index]
                                if removed:
                                    st.success(f"Removed {removed} outlier rows from Test.")
                                    _invalidate_after_cleansing_mutation()
                    else:
                        new_df, removed = _handle_on_df(df)
                        st.session_state["df"] = new_df
                        if removed:
                            st.success(f"Removed {removed} outlier rows from master dataset.")
                            _invalidate_after_cleansing_mutation()

                # Convert outlier values to NaN
                elif action == "Convert to NaN":
                    def _convert_on_df(px):
                        total_converted = 0
                        for c in cols_sorted:
                            try:
                                s = px[c].dropna()
                            except Exception:
                                continue
                            if s.empty:
                                continue
                            if method == "IQR":
                                low, high = _detect_iqr_bounds(pd.to_numeric(s, errors="coerce").dropna(), multiplier=k)
                                mask = (pd.to_numeric(px[c], errors="coerce") < low) | (pd.to_numeric(px[c], errors="coerce") > high)
                            else:
                                vals = pd.to_numeric(s, errors="coerce").dropna()
                                mean = vals.mean()
                                std = vals.std()
                                mask = ((pd.to_numeric(px[c], errors="coerce") - mean).abs() > k * std)
                            # count and set to NaN
                            try:
                                count = int(mask.sum())
                            except Exception:
                                count = 0
                            if count > 0:
                                total_converted += count
                                try:
                                    px.loc[mask, c] = np.nan
                                except Exception:
                                    # fallback: coerce column then assign
                                    try:
                                        px[c] = pd.to_numeric(px[c], errors="coerce")
                                        px.loc[mask, c] = np.nan
                                    except Exception:
                                        continue
                        return px, total_converted

                    if st.session_state.get("split_done"):
                        total_train = 0
                        total_test = 0
                        for subset in out_subsets:
                            if subset == "Train" and st.session_state.get("pre_X_train") is not None:
                                px = st.session_state.get("pre_X_train")
                                new_px, converted = _convert_on_df(px)
                                st.session_state["pre_X_train"] = new_px
                                total_train += converted
                                if converted:
                                    st.success(f"Converted {converted} outlier values to NaN in Train.")
                                    _invalidate_after_cleansing_mutation()
                            if subset == "Test" and st.session_state.get("pre_X_test") is not None:
                                px = st.session_state.get("pre_X_test")
                                new_px, converted = _convert_on_df(px)
                                st.session_state["pre_X_test"] = new_px
                                total_test += converted
                                if converted:
                                    st.success(f"Converted {converted} outlier values to NaN in Test.")
                                    _invalidate_after_cleansing_mutation()
                        if (total_train + total_test) == 0:
                            st.info("No outlier values found to convert to NaN in selected subsets.")
                        st.session_state.setdefault("cleansing_steps", []).append({"type": "outliers_to_nan", "method": method, "k": k, "columns": cols_sorted, "subsets": out_subsets})
                    else:
                        new_df, converted = _convert_on_df(df)
                        st.session_state["df"] = new_df
                        if converted:
                            st.success(f"Converted {converted} outlier values to NaN in master dataset.")
                            _invalidate_after_cleansing_mutation()
                        else:
                            st.info("No outlier values found to convert to NaN in master dataset.")
                        st.session_state.setdefault("cleansing_steps", []).append({"type": "outliers_to_nan", "method": method, "k": k, "columns": cols_sorted, "subsets": out_subsets})

                # Cap action: use CapperTransformer with fit/transform/fit_transform semantics
                else:
                    cap_action = cap_mode or "fit_transform"

                    # Fit-only: compute bounds and save transformer
                    if cap_action == "fit":
                        try:
                            if st.session_state.get("split_done") and "Train" not in out_subsets:
                                st.error("Fit operations must include the Train subset when a split exists.")
                            else:
                                capper = CapperTransformer(method=method, k=k, columns=cols_sorted)
                                if st.session_state.get("split_done"):
                                    px = st.session_state.get("pre_X_train")
                                    capper.fit(px[cols_sorted])
                                else:
                                    capper.fit(df[cols_sorted])
                                st.session_state["capper_transformers"][transformer_key] = capper
                                if "cleansing_steps" not in st.session_state:
                                    st.session_state["cleansing_steps"] = []
                                st.session_state["cleansing_steps"].append({"type": "outlier_cap", "method": method, "k": k, "columns": cols_sorted, "action": "fit", "subsets": out_subsets})
                                st.success(f"Fitted capper for columns: {cols_sorted} (method={method}, k={k})")
                                _invalidate_after_cleansing_mutation()
                        except Exception as e:
                            st.error(f"Error during fit: {e}")

                    # Transform-only: require existing transformer
                    elif cap_action == "transform":
                        if transformer_key not in st.session_state.get("capper_transformers", {}):
                            st.error("No fitted capper found for the selected columns/method/k. Please fit first or use fit_transform.")
                        else:
                            capper = st.session_state["capper_transformers"][transformer_key]
                            try:
                                total_capped = 0
                                if st.session_state.get("split_done"):
                                    for subset in out_subsets:
                                        key_x = f"pre_X_{subset.lower()}"
                                        px = st.session_state.get(key_x)
                                        if px is None:
                                            continue
                                        before_counts = 0
                                        for c in cols_sorted:
                                            low, high = capper.bounds_.get(c, (np.nan, np.nan))
                                            if pd.isna(low) or pd.isna(high):
                                                continue
                                            mask = (pd.to_numeric(px[c], errors="coerce") < low) | (pd.to_numeric(px[c], errors="coerce") > high)
                                            before_counts += int(mask.sum())
                                        new_px = capper.transform(px[cols_sorted])
                                        # assign back transformed columns
                                        for c in cols_sorted:
                                            px[c] = new_px[c]
                                        st.session_state[key_x] = px
                                        total_capped += before_counts
                                else:
                                    before_counts = 0
                                    for c in cols_sorted:
                                        low, high = capper.bounds_.get(c, (np.nan, np.nan))
                                        if pd.isna(low) or pd.isna(high):
                                            continue
                                        mask = (pd.to_numeric(df[c], errors="coerce") < low) | (pd.to_numeric(df[c], errors="coerce") > high)
                                        before_counts += int(mask.sum())
                                    new_df = capper.transform(df[cols_sorted])
                                    for c in cols_sorted:
                                        df[c] = new_df[c]
                                    st.session_state["df"] = df
                                    total_capped += before_counts
                                st.success(f"Capped {total_capped} values across selected subsets for columns {cols_sorted}.")
                                if "cleansing_steps" not in st.session_state:
                                    st.session_state["cleansing_steps"] = []
                                st.session_state["cleansing_steps"].append({"type": "outlier_cap", "method": method, "k": k, "columns": cols_sorted, "action": "transform", "subsets": out_subsets})
                                _invalidate_after_cleansing_mutation()
                            except Exception as e:
                                st.error(f"Error during transform: {e}")

                    # Fit + Transform: fit on Train (if split) then transform selected subsets and save transformer
                    else:
                        try:
                            capper = CapperTransformer(method=method, k=k, columns=cols_sorted)
                            if st.session_state.get("split_done"):
                                if "Train" not in out_subsets:
                                    st.error("Fit operations must include the Train subset when a split exists.")
                                else:
                                    px_train = st.session_state.get("pre_X_train")
                                    capper.fit(px_train[cols_sorted])
                                    total_capped = 0
                                    for subset in out_subsets:
                                        key_x = f"pre_X_{subset.lower()}"
                                        px = st.session_state.get(key_x)
                                        if px is None:
                                            continue
                                        before_counts = 0
                                        for c in cols_sorted:
                                            low, high = capper.bounds_.get(c, (np.nan, np.nan))
                                            if pd.isna(low) or pd.isna(high):
                                                continue
                                            mask = (pd.to_numeric(px[c], errors="coerce") < low) | (pd.to_numeric(px[c], errors="coerce") > high)
                                            before_counts += int(mask.sum())
                                        new_px = capper.transform(px[cols_sorted])
                                        for c in cols_sorted:
                                            px[c] = new_px[c]
                                        st.session_state[key_x] = px
                                        total_capped += before_counts
                                    st.session_state["capper_transformers"][transformer_key] = capper
                                    st.session_state.setdefault("cleansing_steps", []).append({"type": "outlier_cap", "method": method, "k": k, "columns": cols_sorted, "action": "fit_transform", "subsets": out_subsets})
                                    st.success(f"Fitted capper and capped {total_capped} values across subsets: {out_subsets} for columns: {cols_sorted}")
                                    _invalidate_after_cleansing_mutation()
                            else:
                                capper.fit(df[cols_sorted])
                                total_capped = 0
                                before_counts = 0
                                for c in cols_sorted:
                                    low, high = capper.bounds_.get(c, (np.nan, np.nan))
                                    if pd.isna(low) or pd.isna(high):
                                        continue
                                    mask = (pd.to_numeric(df[c], errors="coerce") < low) | (pd.to_numeric(df[c], errors="coerce") > high)
                                    before_counts += int(mask.sum())
                                new_df = capper.transform(df[cols_sorted])
                                for c in cols_sorted:
                                    df[c] = new_df[c]
                                st.session_state["df"] = df
                                st.session_state["capper_transformers"][transformer_key] = capper
                                st.session_state.setdefault("cleansing_steps", []).append({"type": "outlier_cap", "method": method, "k": k, "columns": cols_sorted, "action": "fit_transform", "subsets": out_subsets})
                                st.success(f"Fitted capper and capped {before_counts} values on master dataset for columns: {cols_sorted}")
                                _invalidate_after_cleansing_mutation()
                        except Exception as e:
                            st.error(f"Error during fit_transform: {e}")

    # --- Remove Columns ---
    with tabs[3]:
        st.subheader("Remove Columns")
        drop_cols = st.multiselect("Columns to drop", options=available_columns(), key="drop_columns")
        if st.session_state.get("split_done"):
            subset_options = ["Train", "Test"]
            default_subsets = ["Train"]
        else:
            subset_options = ["Train"]
            default_subsets = ["Train"]
        selected_subsets = st.multiselect("Select subset(s) to process:", options=subset_options, default=default_subsets, key="cleansing_subsets")

        if st.button("Drop Selected", key="drop_button"):
            if not drop_cols:
                st.error("Select one or more columns to drop first.")
            else:
                if st.session_state.get("split_done"):
                    if "cleansing_steps" not in st.session_state:
                        st.session_state["cleansing_steps"] = []
                    for subset in selected_subsets:
                        try:
                            if subset == "Train" and st.session_state.get("pre_X_train") is not None:
                                px = st.session_state.get("pre_X_train")
                                px = px.drop(columns=drop_cols)
                                st.session_state["pre_X_train"] = px
                            elif subset == "Test" and st.session_state.get("pre_X_test") is not None:
                                px = st.session_state.get("pre_X_test")
                                px = px.drop(columns=drop_cols)
                                st.session_state["pre_X_test"] = px
                        except Exception as e:
                            st.error(f"Error dropping columns on {subset}: {e}")
                    st.session_state["cleansing_steps"].append({"type": "drop", "columns": sorted(drop_cols), "subsets": selected_subsets})
                    st.success(f"Dropped columns: {drop_cols} on subsets: {selected_subsets}")
                    _invalidate_after_cleansing_mutation()
                else:
                    df = df.drop(columns=drop_cols)
                    st.session_state["df"] = df
                    if "cleansing_steps" not in st.session_state:
                        st.session_state["cleansing_steps"] = []
                    st.session_state["cleansing_steps"].append({"type": "drop", "columns": sorted(drop_cols)})
                    st.success(f"Dropped columns: {drop_cols}")
                    _invalidate_after_cleansing_mutation()

    # --- Data Types ---
    with tabs[4]:
        st.subheader("Data Types")
        cols = st.multiselect("Select columns to convert", options=available_columns(), key="dtype_cols")
        convert_to = st.selectbox("Convert to", ["Numeric", "Category", "Datetime", "String"], key="dtype_convert")
        if st.session_state.get("split_done"):
            dtype_subsets = st.multiselect("Select subset(s) to process:", options=["Train", "Test"], default=["Train"], key="dtype_subsets")
        else:
            dtype_subsets = ["Train"]

        if st.button("Apply Conversion", key="apply_dtype"):
            if not cols:
                st.error("Select columns to convert first.")
            else:
                def _apply_convert(px):
                    for c in cols:
                        try:
                            if convert_to == "Numeric":
                                px[c] = pd.to_numeric(px[c], errors="coerce")
                            elif convert_to == "Category":
                                px[c] = px[c].astype("category")
                            elif convert_to == "Datetime":
                                px[c] = pd.to_datetime(px[c], errors="coerce")
                            else:
                                px[c] = px[c].astype(str)
                        except Exception:
                            continue
                    return px

                if st.session_state.get("split_done"):
                    for subset in dtype_subsets:
                        if subset == "Train" and st.session_state.get("pre_X_train") is not None:
                            px = st.session_state.get("pre_X_train")
                            px = _apply_convert(px)
                            st.session_state["pre_X_train"] = px
                        if subset == "Test" and st.session_state.get("pre_X_test") is not None:
                            px = st.session_state.get("pre_X_test")
                            px = _apply_convert(px)
                            st.session_state["pre_X_test"] = px
                    st.success(f"Applied conversion to {cols} on subsets: {dtype_subsets}")
                    _invalidate_after_cleansing_mutation()
                else:
                    df2 = _apply_convert(df.copy())
                    st.session_state["df"] = df2
                    st.success(f"Applied conversion to {cols} on master dataset.")
                    _invalidate_after_cleansing_mutation()

    # Show current cleansing steps
    st.subheader("Cleansing Steps Log")
    steps = st.session_state.get("cleansing_steps", [])
    if steps:
        st.write(steps)
    else:
        st.info("No cleansing steps recorded yet.")

    # (Old preview removed — preview moved to top of the page.)

    # Info panel at bottom
    render_info_panel("Cleansing")
