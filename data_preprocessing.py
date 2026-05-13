import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
import math

import inspect
from ui_components import render_info_panel, fix_arrow_compatibility, annotate_bar_values
from sampling import render_sampling
from resampling import render_resampling


def _ensure_pre_X_train():
    # initialize pre_X_train and pre_y_train from working copy if not present
    if "pre_X_train" not in st.session_state or st.session_state.get("pre_X_train") is None:
        df_master = st.session_state.get("df")
        target = st.session_state.get("target_column")
        if df_master is None:
            st.session_state["pre_X_train"] = None
            st.session_state["pre_y_train"] = None
            return
        if target and target in df_master.columns:
            st.session_state["pre_X_train"] = df_master.drop(columns=[target]).copy()
            st.session_state["pre_y_train"] = df_master[target].copy()
        else:
            st.session_state["pre_X_train"] = df_master.copy()
            st.session_state["pre_y_train"] = None


def _record_preprocessing_step(step: dict):
    if "preprocessing_steps" not in st.session_state:
        st.session_state["preprocessing_steps"] = []
    st.session_state["preprocessing_steps"].append(step)


def render_preprocessing():
    st.header("⚙️ Preprocessing")

    if "df" not in st.session_state or st.session_state.get("df") is None:
        st.warning("No dataset uploaded yet. Please upload data on the Data page.")
        if st.button("Open Data Page"):
            st.session_state["page"] = "Data"
        render_info_panel("Preprocessing")
        return

    # prepare working X
    _ensure_pre_X_train()
    X = st.session_state.get("pre_X_train")
    task_type = st.session_state.get("task_type")
    # ensure encoders dict exists in session state
    st.session_state.setdefault("encoders", {})
    # Tabs will be created after the preview so the preview sits immediately under the header

    # Preview (above tabs): show random 5 rows or head for selected subset when split active
    st.subheader("Preview Working Dataset")
    if st.session_state.get("split_done"):
        preview_choice = st.radio("Preview subset", ["Train (subset)", "Test (subset)"], index=0, key="preproc_preview_subset")
        if preview_choice.startswith("Train"):
            df_preview = st.session_state.get("pre_X_train")
            y_preview = st.session_state.get("pre_y_train")
        else:
            df_preview = st.session_state.get("pre_X_test")
            y_preview = st.session_state.get("pre_y_test")
    else:
        # when no split, preview the working master or the working X
        pre_x = st.session_state.get("pre_X_train")
        df_master = st.session_state.get("df")
        df_preview = pre_x if pre_x is not None else df_master

    if df_preview is None:
        st.info("No data available for preview.")
    else:
        if st.button("Random 5 rows", key="random_5_rows_preproc"):
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

        # If resampling was recently applied, show comparison plots under the preview
        if st.session_state.get("resample_plot_trigger"):
            # prefer the explicit resampled view if available
            train_key = "resampled_pre_X_train" if st.session_state.get("resampled_pre_X_train") is not None else "pre_X_train"
            y_train_key = train_key.replace("X", "y", 1)
            y_test_key = "pre_y_test"
            y_train = st.session_state.get(y_train_key)
            y_test = st.session_state.get(y_test_key)
            if y_train is not None and y_test is not None:
                st.markdown(f"**Comparison using `{train_key}`**")
                try:
                    train_counts = y_train.value_counts(normalize=True)
                    test_counts = y_test.value_counts(normalize=True)
                    df_prop = pd.DataFrame({"Train": train_counts, "Test": test_counts}).fillna(0)

                    train_abs = y_train.value_counts()
                    test_abs = y_test.value_counts()
                    df_count = pd.DataFrame({"Train": train_abs, "Test": test_abs}).fillna(0)

                    categories = list(dict.fromkeys(list(df_prop.index) + list(df_count.index)))
                    df_prop = df_prop.reindex(categories, fill_value=0)
                    df_count = df_count.reindex(categories, fill_value=0)

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    df_prop.plot(kind="bar", ax=axes[0])
                    annotate_bar_values(axes[0], fmt="{:.2f}", integer=False)
                    axes[0].set_title("Class Proportions: Train vs Test")
                    axes[0].set_ylabel("Proportion")

                    df_count.plot(kind="bar", ax=axes[1])
                    annotate_bar_values(axes[1], integer=True)
                    axes[1].set_title("Class Counts: Train vs Test")
                    axes[1].set_ylabel("Count")

                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error rendering resampling comparison plots: {e}")

    # Create horizontal navigation tabs below the preview (so preview is directly under header)
    tab_scaling, tab_encoding, tab_fe, tab_mapping, tab_sampling, tab_resampling = st.tabs([
        "🔢 Scaling",
        "🏷️ Encoding",
        "⚙️ Feature Engineering",
        "🗺️ Value Mapping",
        "🧪 Sampling",
        "⚖️ Resampling",
    ])

    # NOTE: per-tab subset selectors are used below (to match Cleansing layout)

    # --- Tabs ---
    with tab_scaling:
        st.subheader("Scaling")
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist() if X is not None else []
        if not num_cols:
            st.info("No numeric features available for scaling.")
        else:
            scaler_name = st.selectbox("Scaler", ["MinMaxScaler", "StandardScaler", "RobustScaler"], key="scaler_name")
            sel_cols_scaling = st.multiselect("Numeric columns to scale", options=num_cols, default=num_cols[:3], key="scaler_cols")
            action_scale = st.radio("Action", ["fit", "transform", "fit_transform"], index=0, key="scaler_action")
            if st.session_state.get("split_done"):
                scaling_subsets = st.multiselect("Select subset(s) to process:", options=["Train", "Test"], default=["Train"], key="scaling_subsets")
            else:
                scaling_subsets = ["Train"]

            if st.button("Apply Scaling", key="apply_scaling"):
                if not sel_cols_scaling:
                    st.error("Select numeric columns to scale.")
                else:
                    if scaler_name == "MinMaxScaler":
                        scaler = MinMaxScaler()
                    elif scaler_name == "StandardScaler":
                        scaler = StandardScaler()
                    else:
                        scaler = RobustScaler()

                    transformer_key = f"scaler::{scaler_name}::" + ",".join(sorted(sel_cols_scaling))
                    if "scalers" not in st.session_state:
                        st.session_state["scalers"] = {}
                    try:
                        if action_scale == "transform":
                            if transformer_key not in st.session_state["scalers"]:
                                st.error("You must fit or fit_transform before transform-only.")
                            else:
                                fitted = st.session_state["scalers"][transformer_key]
                                if st.session_state.get("split_done"):
                                    for subset in scaling_subsets:
                                        if subset == "Train" and st.session_state.get("pre_X_train") is not None:
                                            px = st.session_state.get("pre_X_train")
                                            px.loc[:, sel_cols_scaling] = fitted.transform(px[sel_cols_scaling])
                                            st.session_state["pre_X_train"] = px
                                        elif subset == "Test" and st.session_state.get("pre_X_test") is not None:
                                            px = st.session_state.get("pre_X_test")
                                            px.loc[:, sel_cols_scaling] = fitted.transform(px[sel_cols_scaling])
                                            st.session_state["pre_X_test"] = px
                                    else:
                                        pre_x_local = st.session_state.get("pre_X_train")
                                        df_master_local = st.session_state.get("df")
                                        X_local = pre_x_local if pre_x_local is not None else df_master_local
                                    X_local.loc[:, sel_cols_scaling] = fitted.transform(X_local[sel_cols_scaling])
                                    st.session_state["pre_X_train"] = X_local
                                _record_preprocessing_step({"type": "scaling", "method": scaler_name, "columns": sorted(sel_cols_scaling), "action": action_scale, "target_data": scaling_subsets})
                                st.success(f"Applied {action_scale} with {scaler_name} to {sel_cols_scaling} on subsets {scaling_subsets}")

                        elif action_scale == "fit":
                            if st.session_state.get("split_done") and "Train" not in scaling_subsets:
                                st.error("Fit operations must include Train when a split exists.")
                            else:
                                if st.session_state.get("split_done"):
                                    px = st.session_state.get("pre_X_train")
                                    scaler.fit(px[sel_cols_scaling])
                                else:
                                    df_master = st.session_state.get("df")
                                    scaler.fit(df_master[sel_cols_scaling])
                                st.session_state["scalers"][transformer_key] = scaler
                                _record_preprocessing_step({"type": "scaling", "method": scaler_name, "columns": sorted(sel_cols_scaling), "action": action_scale, "target_data": scaling_subsets})
                                st.success(f"Fitted {scaler_name} on {sel_cols_scaling}")

                        else:  # fit_transform
                            if st.session_state.get("split_done"):
                                if "Train" not in scaling_subsets:
                                    st.error("Fit operations must include Train when a split exists.")
                                else:
                                    px = st.session_state.get("pre_X_train")
                                    transformed = scaler.fit_transform(px[sel_cols_scaling])
                                    px.loc[:, sel_cols_scaling] = transformed
                                    st.session_state["pre_X_train"] = px
                                    st.session_state["scalers"][transformer_key] = scaler
                                    if "Test" in scaling_subsets and st.session_state.get("pre_X_test") is not None:
                                        px_test = st.session_state.get("pre_X_test")
                                        px_test.loc[:, sel_cols_scaling] = scaler.transform(px_test[sel_cols_scaling])
                                        st.session_state["pre_X_test"] = px_test
                            else:
                                df_master = st.session_state.get("df")
                                transformed = scaler.fit_transform(df_master[sel_cols_scaling])
                                df_master.loc[:, sel_cols_scaling] = transformed
                                st.session_state["df"] = df_master
                                st.session_state["scalers"][transformer_key] = scaler
                            _record_preprocessing_step({"type": "scaling", "method": scaler_name, "columns": sorted(sel_cols_scaling), "action": action_scale, "target_data": scaling_subsets})
                            st.success(f"Applied fit_transform with {scaler_name} to {sel_cols_scaling} on subsets: {scaling_subsets}")
                    except Exception as e:
                        st.error(f"Scaling error: {e}")

    with tab_encoding:
        st.subheader("Encoding")
        cat_cols = X.select_dtypes(include=["category", "object", "bool"]).columns.tolist()
        # preserve previous selection in case widget options change after a rerun
        cached_sel = st.session_state.get("enc_cols_cached", None)
        # normalize cached selection to a list if it's a single scalar
        if isinstance(cached_sel, (str, int)):
            cached_sel = [cached_sel]
        default_sel = cached_sel if cached_sel is not None else cat_cols[:2]
        # sanitize default values so Streamlit doesn't raise when defaults are not in options
        if default_sel:
            sanitized_default = [c for c in default_sel if c in cat_cols]
        else:
            sanitized_default = []
        sel_enc_cols = st.multiselect("Categorical columns to encode", options=cat_cols, default=sanitized_default, key="enc_cols")
        # update cached selection when user explicitly picks columns
        if sel_enc_cols:
            st.session_state["enc_cols_cached"] = sel_enc_cols
        else:
            # clear cached selection if it no longer matches available options
            if cached_sel and list(cached_sel) != sanitized_default:
                st.session_state["enc_cols_cached"] = sanitized_default

        encoder_name = st.selectbox("Encoder", ["LabelEncoder", "OneHotEncoder", "OrdinalEncoder"], key="encoder_name")

        # User-facing note: behavior when a split exists
        st.caption("Note: when a split exists, the encoder will be fitted on the Train subset and applied to Test (fit_transform on Train, transform on Test).")

        # Allow using previously-fitted encoders when current categorical options are missing
        chosen_transformer_key = None
        available = st.session_state.get("encoders", {})
        if not sel_enc_cols and available:
            # LabelEncoder choices
            label_keys = [k for k in available.keys() if k.startswith("label::")]
            label_cols = [k.split("::", 1)[1] for k in label_keys]
            if label_cols:
                chosen_labels = st.multiselect("Choose fitted LabelEncoder column(s) to apply", options=label_cols, key="enc_label_choice")
                if chosen_labels:
                    sel_enc_cols = chosen_labels
                    st.session_state["enc_cols_cached"] = sel_enc_cols

            # Encoders like OneHot/Ordinal saved with prefix encoder::Name::col1,col2
            prefix = f"encoder::{encoder_name}::"
            avail = [k for k in available.keys() if k.startswith(prefix)]
            if avail:
                display = [k.replace(prefix, "") for k in avail]
                choice = st.selectbox("Use one of the fitted encoders (choose columns)", options=["-- choose --"] + display, key="enc_pick_transformer")
                if choice and choice != "-- choose --":
                    idx = display.index(choice)
                    chosen_transformer_key = avail[idx]
                    cols_from_key = chosen_transformer_key.split("::", 2)[2].split(",")
                    sel_enc_cols = cols_from_key
                    st.session_state["enc_cols_cached"] = sel_enc_cols

            # Fallback: choose any fitted encoder
            if not sel_enc_cols:
                display_all = []
                keys_all = []
                for k in available.keys():
                    if k.startswith("label::"):
                        display_all.append(k.replace("label::", "LabelEncoder::"))
                        keys_all.append(k)
                    elif k.startswith("encoder::"):
                        display_all.append(k.replace("encoder::", ""))
                        keys_all.append(k)
                    else:
                        display_all.append(k)
                        keys_all.append(k)

                choice_any = st.selectbox("Or choose any fitted encoder to apply", options=["-- choose --"] + display_all, key="enc_choose_any")
                if choice_any and choice_any != "-- choose --":
                    idx = display_all.index(choice_any)
                    chosen_transformer_key = keys_all[idx]
                    if chosen_transformer_key.startswith("label::"):
                        cols_from_key = [chosen_transformer_key.split("::", 1)[1]]
                    else:
                        parts = chosen_transformer_key.split("::", 2)
                        cols_from_key = parts[2].split(",")
                    sel_enc_cols = cols_from_key
                    st.session_state["enc_cols_cached"] = sel_enc_cols

        # When a split exists we will fit encoders on Train and apply to Test automatically
        encoding_subsets = ["Train", "Test"] if st.session_state.get("split_done") else ["Train"]

        # Special UI: for OrdinalEncoder allow ordering of unique values per-column
        ord_orders = {}
        if encoder_name == "OrdinalEncoder" and sel_enc_cols:
            for col in sel_enc_cols:
                try:
                    if st.session_state.get("split_done") and st.session_state.get("pre_X_train") is not None:
                        uniq = list(pd.Series(st.session_state.get("pre_X_train")[col].astype(str).unique()))
                    else:
                        uniq = list(pd.Series(X[col].astype(str).unique())) if X is not None and col in X.columns else []
                except Exception:
                    uniq = []
                cached_order = st.session_state.get(f"ord_order::{col}", None)
                if isinstance(cached_order, (str, int)):
                    cached_order = [cached_order]
                default_ord = cached_order if cached_order is not None else uniq
                # sanitize default
                sanitized_ord = [v for v in default_ord if v in uniq]
                sel_order = st.multiselect(f"Order unique values for `{col}` (first = lowest ordinal)", options=uniq, default=sanitized_ord, key=f"ord_order::{col}")
                # Streamlit manages the widget-backed session_state entry for the key
                # Avoid assigning to the same `st.session_state` key after widget creation
                if sel_order:
                    ord_orders[col] = sel_order
                else:
                    ord_orders[col] = sanitized_ord

        if st.button("Apply Encoding"):
            if not sel_enc_cols:
                st.error("Select categorical columns to encode or choose a fitted encoder.")
            else:
                cols_sorted = sorted(sel_enc_cols)
                transformer_key = f"encoder::{encoder_name}::" + ",".join(cols_sorted)

                if encoder_name == "LabelEncoder":
                    # apply per-column and support subsets; use existing encoder if chosen
                    for col in cols_sorted:
                        key_col = f"label::{col}"
                        try:
                            if chosen_transformer_key and chosen_transformer_key.startswith("label::"):
                                le = st.session_state.get("encoders", {}).get(chosen_transformer_key)
                                if le is None:
                                    st.error(f"Chosen LabelEncoder '{chosen_transformer_key}' not found in session encoders.")
                                else:
                                    # apply using existing encoder via safe mapping to handle unknowns
                                    mapping = {v: i for i, v in enumerate(getattr(le, 'classes_', []))}
                                    if st.session_state.get("split_done"):
                                        for subset in encoding_subsets:
                                            if subset == "Train":
                                                px = st.session_state.get("pre_X_train")
                                                if px is not None and col in px.columns:
                                                    px[col] = px[col].astype(str).map(mapping)
                                                    st.session_state["pre_X_train"] = px
                                            elif subset == "Test":
                                                px = st.session_state.get("pre_X_test")
                                                if px is not None and col in px.columns:
                                                    px[col] = px[col].astype(str).map(mapping)
                                                    st.session_state["pre_X_test"] = px
                            else:
                                le = LabelEncoder()
                                if st.session_state.get("split_done"):
                                    px = st.session_state.get("pre_X_train")
                                    le.fit(px[col].astype(str))
                                    # apply mapping to train
                                    mapping = {v: i for i, v in enumerate(le.classes_)}
                                    px[col] = px[col].astype(str).map(mapping)
                                    st.session_state["pre_X_train"] = px
                                    st.session_state.setdefault("encoders", {})[key_col] = le
                                    # safely apply mapping to test (unknowns become NaN)
                                    if "Test" in encoding_subsets and st.session_state.get("pre_X_test") is not None:
                                        px_test = st.session_state.get("pre_X_test")
                                        if col in px_test.columns:
                                            px_test[col] = px_test[col].astype(str).map(mapping)
                                            st.session_state["pre_X_test"] = px_test
                                else:
                                    X[col] = le.fit_transform(X[col].astype(str))
                                    st.session_state.setdefault("encoders", {})[key_col] = le
                                    st.session_state["pre_X_train"] = X

                            _record_preprocessing_step({
                                "type": "encoding",
                                "method": "LabelEncoder",
                                "columns": [col],
                                "parameters": {},
                                "action_performed": "fit_transform",
                                "target_data": encoding_subsets,
                            })
                            st.success(f"Applied LabelEncoder to {col}")
                        except Exception as e:
                            st.error(f"Error processing LabelEncoder for {col}: {e}")

                elif encoder_name == "OneHotEncoder":
                    try:
                        # If a fitted encoder key was chosen, apply it; otherwise instantiate and fit
                        if chosen_transformer_key and chosen_transformer_key in st.session_state.get("encoders", {}):
                            ohe = st.session_state.get("encoders", {}).get(chosen_transformer_key)
                            if ohe is None:
                                st.error(f"Chosen OneHotEncoder '{chosen_transformer_key}' not found in session encoders.")
                            else:
                                # determine feature names safely
                                try:
                                    names = ohe.get_feature_names_out(cols_sorted)
                                except Exception:
                                    try:
                                        names = ohe.get_feature_names(cols_sorted)
                                    except Exception:
                                        names = [f"ohe_{i}" for i in range(len(cols_sorted))]

                                if st.session_state.get("split_done"):
                                    px = st.session_state.get("pre_X_train")
                                    if px is not None:
                                        arr = ohe.transform(px[cols_sorted])
                                        if hasattr(arr, "toarray"):
                                            arr = arr.toarray()
                                        df_ohe = pd.DataFrame(arr, columns=names, index=px.index)
                                        px = px.drop(columns=cols_sorted)
                                        px = pd.concat([px, df_ohe], axis=1)
                                        st.session_state["pre_X_train"] = px
                                    px_test = st.session_state.get("pre_X_test")
                                    if px_test is not None:
                                        arr_t = ohe.transform(px_test[cols_sorted])
                                        if hasattr(arr_t, "toarray"):
                                            arr_t = arr_t.toarray()
                                        df_ohe_t = pd.DataFrame(arr_t, columns=names, index=px_test.index)
                                        px_test = px_test.drop(columns=cols_sorted)
                                        px_test = pd.concat([px_test, df_ohe_t], axis=1)
                                        st.session_state["pre_X_test"] = px_test
                                else:
                                    arr = ohe.transform(X[cols_sorted])
                                    if hasattr(arr, "toarray"):
                                        arr = arr.toarray()
                                    df_ohe = pd.DataFrame(arr, columns=names, index=X.index)
                                    X = X.drop(columns=cols_sorted)
                                    X = pd.concat([X, df_ohe], axis=1)

                                _record_preprocessing_step({
                                    "type": "encoding",
                                    "method": "OneHotEncoder",
                                    "columns": cols_sorted,
                                    "parameters": {"handle_unknown": "ignore"},
                                    "action_performed": "transform",
                                    "target_data": encoding_subsets,
                                })
                                st.success(f"Transformed columns {cols_sorted} with existing OneHotEncoder for subsets: {encoding_subsets}")
                        else:
                            # instantiate OneHotEncoder compatibly across sklearn versions
                            def _make_onehot_encoder(handle_unknown="ignore", dense=True):
                                params = inspect.signature(OneHotEncoder).parameters
                                kwargs = {"handle_unknown": handle_unknown}
                                if "sparse_output" in params:
                                    kwargs["sparse_output"] = False if dense else True
                                elif "sparse" in params:
                                    kwargs["sparse"] = False if dense else True
                                return OneHotEncoder(**kwargs)

                            ohe = _make_onehot_encoder(handle_unknown="ignore", dense=True)

                            if st.session_state.get("split_done"):
                                px = st.session_state.get("pre_X_train")
                                if px is not None:
                                    arr = ohe.fit_transform(px[cols_sorted])
                                    if hasattr(arr, "toarray"):
                                        arr = arr.toarray()
                                    try:
                                        names = ohe.get_feature_names_out(cols_sorted)
                                    except Exception:
                                        try:
                                            names = ohe.get_feature_names(cols_sorted)
                                        except Exception:
                                            names = [f"ohe_{i}" for i in range(arr.shape[1])]
                                    df_ohe = pd.DataFrame(arr, columns=names, index=px.index)
                                    px = px.drop(columns=cols_sorted)
                                    px = pd.concat([px, df_ohe], axis=1)
                                    st.session_state["pre_X_train"] = px
                                    st.session_state.setdefault("encoders", {})[transformer_key] = ohe
                                if "Test" in encoding_subsets and st.session_state.get("pre_X_test") is not None:
                                    px_test = st.session_state.get("pre_X_test")
                                    arr_t = ohe.transform(px_test[cols_sorted])
                                    if hasattr(arr_t, "toarray"):
                                        arr_t = arr_t.toarray()
                                    try:
                                        names = ohe.get_feature_names_out(cols_sorted)
                                    except Exception:
                                        try:
                                            names = ohe.get_feature_names(cols_sorted)
                                        except Exception:
                                            names = [f"ohe_{i}" for i in range(arr_t.shape[1])]
                                    df_ohe_t = pd.DataFrame(arr_t, columns=names, index=px_test.index)
                                    px_test = px_test.drop(columns=cols_sorted)
                                    px_test = pd.concat([px_test, df_ohe_t], axis=1)
                                    st.session_state["pre_X_test"] = px_test
                            else:
                                arr = ohe.fit_transform(X[cols_sorted])
                                if hasattr(arr, "toarray"):
                                    arr = arr.toarray()
                                try:
                                    names = ohe.get_feature_names_out(cols_sorted)
                                except Exception:
                                    try:
                                        names = ohe.get_feature_names(cols_sorted)
                                    except Exception:
                                        names = [f"ohe_{i}" for i in range(arr.shape[1])]
                                df_ohe = pd.DataFrame(arr, columns=names, index=X.index)
                                X = X.drop(columns=cols_sorted)
                                X = pd.concat([X, df_ohe], axis=1)
                                st.session_state.setdefault("encoders", {})[transformer_key] = ohe

                            _record_preprocessing_step({
                                "type": "encoding",
                                "method": "OneHotEncoder",
                                "columns": cols_sorted,
                                "parameters": {"handle_unknown": "ignore"},
                                "action_performed": "fit_transform",
                                "target_data": encoding_subsets,
                            })
                            st.success(f"Applied OneHotEncoder to {cols_sorted} for subsets: {encoding_subsets}")
                    except Exception as e:
                        st.error(f"OneHotEncoder error: {e}")

                elif encoder_name == "OrdinalEncoder":
                    try:
                        # process each column with its own ordered categories
                        for col in cols_sorted:
                            key_col = f"encoder::OrdinalEncoder::{col}"
                            cat_order = ord_orders.get(col, None)
                            if chosen_transformer_key and chosen_transformer_key in st.session_state.get("encoders", {}):
                                ord_enc = st.session_state.get("encoders", {}).get(chosen_transformer_key)
                                if ord_enc is None:
                                    st.error(f"Chosen OrdinalEncoder '{chosen_transformer_key}' not found in session encoders.")
                                else:
                                    # safe mapping from fitted encoder categories
                                    cats = ord_enc.categories_[0] if hasattr(ord_enc, 'categories_') and len(ord_enc.categories_) > 0 else []
                                    mapping = {v: i for i, v in enumerate(cats)}
                                    if st.session_state.get("split_done"):
                                        for subset in encoding_subsets:
                                            if subset == "Train":
                                                px = st.session_state.get("pre_X_train")
                                                if px is not None and col in px.columns:
                                                    px[col] = px[col].astype(str).map(mapping)
                                                    st.session_state["pre_X_train"] = px
                                            elif subset == "Test":
                                                px = st.session_state.get("pre_X_test")
                                                if px is not None and col in px.columns:
                                                    px[col] = px[col].astype(str).map(mapping)
                                                    st.session_state["pre_X_test"] = px
                            else:
                                # build categories list for this column
                                if not cat_order or len(cat_order) == 0:
                                    st.error(f"No category order provided for {col}. Provide an ordered list of unique values.")
                                    continue
                                ord_enc = OrdinalEncoder(categories=[cat_order])
                                if st.session_state.get("split_done"):
                                    px = st.session_state.get("pre_X_train")
                                    ord_enc.fit(px[[col]].astype(str))
                                    mapping = {v: i for i, v in enumerate(cat_order)}
                                    px[col] = px[col].astype(str).map(mapping)
                                    st.session_state["pre_X_train"] = px
                                    st.session_state.setdefault("encoders", {})[key_col] = ord_enc
                                    if "Test" in encoding_subsets and st.session_state.get("pre_X_test") is not None:
                                        px_test = st.session_state.get("pre_X_test")
                                        px_test[col] = px_test[col].astype(str).map(mapping)
                                        st.session_state["pre_X_test"] = px_test
                                else:
                                    res = ord_enc.fit_transform(X[[col]].astype(str))
                                    X[col] = res[:, 0]
                                    st.session_state.setdefault("encoders", {})[key_col] = ord_enc

                            _record_preprocessing_step({
                                "type": "encoding",
                                "method": "OrdinalEncoder",
                                "columns": [col],
                                "parameters": {"order": ord_orders.get(col)},
                                "action_performed": "fit_transform",
                                "target_data": encoding_subsets,
                            })
                        st.success(f"Applied OrdinalEncoder to {cols_sorted}")
                    except Exception as e:
                        st.error(f"OrdinalEncoder error: {e}")

    with tab_fe:
        st.subheader("Feature Engineering (Creation)")
        st.write("Enter a formula using column names and basic operators. Allowed functions: log, exp, sqrt, np.")
        formula = st.text_area("Formula (e.g., col1 * col2, log(col3))", key="fe_formula")
        new_col = st.text_input("New column name", value=f"feat_{len(st.session_state['preprocessing_steps'])+1}")
        # Per-tab subset selector for feature engineering
        if st.session_state.get("split_done"):
            fe_subsets = st.multiselect("Select subset(s) to process:", options=["Train", "Test"], default=["Train"], key="fe_subsets")
        else:
            fe_subsets = ["Train"]

        if st.button("Create Feature"):
            if not formula.strip():
                st.error("Provide a formula for the new feature.")
            else:
                import numpy as _np
                # apply formula to selected subsets when split exists
                applied_subsets = []
                if st.session_state.get("split_done"):
                    for subset in fe_subsets:
                        try:
                            if subset == "Train":
                                px = st.session_state.get("pre_X_train")
                                if px is None:
                                    st.warning("Train subset not available to create feature.")
                                    continue
                                local_vars = {c: px[c] for c in px.columns}
                                local_vars.update({"log": _np.log, "exp": _np.exp, "sqrt": _np.sqrt, "np": _np})
                                result = eval(formula, {"__builtins__": {}}, local_vars)
                                if isinstance(result, (pd.Series, np.ndarray, list)):
                                    px[new_col] = result
                                    st.session_state["pre_X_train"] = px
                                    applied_subsets.append("Train")
                            elif subset == "Test":
                                px = st.session_state.get("pre_X_test")
                                if px is None:
                                    st.warning("Test subset not available to create feature.")
                                    continue
                                local_vars = {c: px[c] for c in px.columns}
                                local_vars.update({"log": _np.log, "exp": _np.exp, "sqrt": _np.sqrt, "np": _np})
                                result = eval(formula, {"__builtins__": {}}, local_vars)
                                if isinstance(result, (pd.Series, np.ndarray, list)):
                                    px[new_col] = result
                                    st.session_state["pre_X_test"] = px
                                    applied_subsets.append("Test")
                        except Exception as e:
                            st.error(f"Error applying formula to {subset}: {e}")
                else:
                    try:
                        px = st.session_state.get("pre_X_train")
                        local_vars = {c: px[c] for c in px.columns}
                        local_vars.update({"log": _np.log, "exp": _np.exp, "sqrt": _np.sqrt, "np": _np})
                        result = eval(formula, {"__builtins__": {}}, local_vars)
                        if isinstance(result, (pd.Series, np.ndarray, list)):
                            st.session_state["pre_X_train"][new_col] = result
                            applied_subsets.append("Train")
                        else:
                            st.error("Formula did not return a sequence-like result (Series/array).")
                    except Exception as e:
                        st.error(f"Error evaluating formula: {e}")

                if applied_subsets:
                    cols_used = [c for c in (st.session_state.get("pre_X_train") or []).columns if c in formula]
                    _record_preprocessing_step({
                        "type": "feature_engineering",
                        "method": "expression",
                        "columns": cols_used,
                        "parameters": {"formula": formula},
                        "action_performed": "create",
                        "target_data": fe_subsets,
                    })
                    st.success(f"Created feature {new_col} on subsets: {applied_subsets}")

    with tab_mapping:
        st.subheader("Value Mapping")
        map_cols = X.columns.tolist() if X is not None else []
        if not map_cols:
            st.info("No features available to map. Ensure data is loaded.")
        else:
            sel_col = st.selectbox("Feature to map", options=map_cols, key="value_map_feature")
            map_type = st.radio("Value type", ["Numerical", "Categorical"], index=0, key="value_map_type")
            before_val = st.text_input("Value BEFORE (to replace)", key="value_map_before")
            after_val = st.text_input("Value AFTER (replacement)", key="value_map_after")

            # Per-tab subset selector for value mapping
            if st.session_state.get("split_done"):
                mapping_subsets = st.multiselect("Select subset(s) to process:", options=["Train", "Test"], default=["Train"], key="mapping_subsets")
            else:
                mapping_subsets = ["Train"]

            if "value_mappings" not in st.session_state:
                st.session_state["value_mappings"] = {}

            if st.button("Add mapping", key="add_value_mapping"):
                if not sel_col or before_val is None or str(before_val).strip() == "":
                    st.error("Select a feature and provide the value to replace.")
                else:
                    st.session_state.setdefault("value_mappings", {})
                    st.session_state["value_mappings"].setdefault(sel_col, [])
                    st.session_state["value_mappings"][sel_col].append({"before": before_val, "after": after_val, "type": map_type})
                    st.success(f"Added mapping for {sel_col}: '{before_val}' → '{after_val}'")

            # show current mappings for selected column
            current = st.session_state.get("value_mappings", {}).get(sel_col, [])
            if current:
                st.write("Current mappings for selected feature:")
                for i, m in enumerate(current):
                    st.write(f"{i+1}. '{m['before']}' → '{m['after']}' ({m['type']})")
                if st.button("Clear mappings for this feature", key="clear_mappings_feature"):
                    st.session_state["value_mappings"][sel_col] = []
                    st.success(f"Cleared mappings for {sel_col}.")

            if st.button("Apply mappings to selected subsets", key="apply_value_mappings"):
                vm = st.session_state.get("value_mappings", {})
                if not vm:
                    st.error("No mappings defined. Use 'Add mapping' to add at least one mapping.")
                else:
                    def _apply_map_df(df_local, col_name, mappings):
                        for m in mappings:
                            bef = m.get("before")
                            aft = m.get("after")
                            if m.get("type") == "Numerical":
                                try:
                                    bef_v = float(bef)
                                except Exception:
                                    st.error(f"Numeric mapping value '{bef}' is not a valid number for column {col_name}.")
                                    continue
                                mask = pd.to_numeric(df_local[col_name], errors="coerce") == bef_v
                                try:
                                    aft_v = float(aft) if aft != "" else np.nan
                                except Exception:
                                    aft_v = aft
                                df_local.loc[mask, col_name] = aft_v
                            else:
                                mask = df_local[col_name].astype(str) == str(bef)
                                df_local.loc[mask, col_name] = aft
                        return df_local

                    try:
                        for col_name, mappings in vm.items():
                            if st.session_state.get("split_done"):
                                for subset in mapping_subsets:
                                    key_x = f"pre_X_{subset.lower()}"
                                    px = st.session_state.get(key_x)
                                    if px is not None and col_name in px.columns:
                                        px = _apply_map_df(px, col_name, mappings)
                                        st.session_state[key_x] = px
                            else:
                                df_master = st.session_state.get("df")
                                if df_master is not None and col_name in df_master.columns:
                                    df_master = _apply_map_df(df_master, col_name, mappings)
                                    st.session_state["df"] = df_master

                        _record_preprocessing_step({
                            "type": "value_mapping",
                            "mappings": st.session_state.get("value_mappings", {}),
                            "target_data": mapping_subsets,
                        })
                        st.success("Applied value mappings to selected subsets.")
                    except Exception as e:
                        st.error(f"Error applying mappings: {e}")

    with tab_sampling:
        # render sampling UI inside preprocessing as requested
        render_sampling(show_info_panel=False)

    with tab_resampling:
        # render resampling UI inside preprocessing as requested
        render_resampling(show_info_panel=False)

    # Show preprocessing steps and preview (move below tabs)
    st.subheader("Preprocessing Steps Log")
    steps = st.session_state.get("preprocessing_steps", [])
    if steps:
        st.write(steps)
    else:
        st.info("No preprocessing steps recorded yet.")

    st.markdown("---")
    render_info_panel("Preprocessing")
