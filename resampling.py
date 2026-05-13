import streamlit as st
import pandas as pd
from ui_components import fix_arrow_compatibility, render_info_panel
from state_manager import clear_downstream_from_resampling


def _check_imblearn():
    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        return True
    except Exception:
        return False


def render_resampling(show_info_panel: bool = True):
    st.header("⚖️ Resampling - Handle Imbalanced Classes")

    if st.session_state.get("task_type") != "Classification":
        st.error("Resampling is available only for Classification tasks.")
        if show_info_panel:
            render_info_panel("Resampling")
        return

    if not st.session_state.get("split_done"):
        st.warning("Please create a train/test split first on the 'Split Dataset' page.")
        if show_info_panel:
            render_info_panel("Resampling")
        return

    # Offer any available training views as source (pre, sampled, resampled)
    base_key_options = []
    if st.session_state.get("pre_X_train") is not None:
        base_key_options.append("pre_X_train")
    if st.session_state.get("sampled_X_train") is not None:
        base_key_options.append("sampled_X_train")
    if st.session_state.get("resampled_pre_X_train") is not None:
        base_key_options.append("resampled_pre_X_train")

    src = st.selectbox("Source training data", base_key_options, key="resample_source")
    X_src = st.session_state.get(src)
    # derive matching y key by replacing the first occurrence of 'X' with 'y'
    try:
        y_key = src.replace("X", "y", 1)
    except Exception:
        y_key = src.replace("X", "y")
    y_src = st.session_state.get(y_key)

    if X_src is None or y_src is None:
        st.error("Selected source does not contain both X and y. Ensure the training split and target are available.")
        if show_info_panel:
            render_info_panel("Resampling")
        return

    if not _check_imblearn():
        st.error("Package 'imbalanced-learn' not installed. Install with: pip install imbalanced-learn")
        if show_info_panel:
            render_info_panel("Resampling")
        return

    methods = ["RandomOverSampler", "RandomUnderSampler", "SMOTE"]
    method = st.selectbox("Resampling method", methods, key="resample_method")
    seed = int(st.session_state.get("global_seed", 42))

    if st.button("Apply Resampling"):
        try:
            from imblearn.over_sampling import RandomOverSampler, SMOTE
            from imblearn.under_sampling import RandomUnderSampler

            if method == "RandomOverSampler":
                sampler = RandomOverSampler(random_state=seed)
            elif method == "RandomUnderSampler":
                sampler = RandomUnderSampler(random_state=seed)
            else:
                sampler = SMOTE(random_state=seed)

            X_res, y_res = sampler.fit_resample(X_src, y_src)
            st.session_state["resampled_pre_X_train"] = pd.DataFrame(X_res, columns=X_src.columns)
            st.session_state["resampled_pre_y_train"] = pd.Series(y_res)
            # trigger plot rendering in the Preprocessing preview
            st.session_state["resample_plot_trigger"] = True
            st.session_state["last_resample_source"] = src
            st.session_state["resampling_done"] = True
            clear_downstream_from_resampling(st.session_state)
            st.success(f"Resampling applied: {method}. Result rows: {len(X_res)}")
            try:
                st.write(st.session_state["resampled_pre_y_train"].value_counts())
            except Exception:
                pass
        except Exception as e:
            st.error(f"Error during resampling: {e}")

    if show_info_panel:
        st.markdown("---")
        render_info_panel("Resampling")
