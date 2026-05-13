import streamlit as st
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from ui_components import render_info_panel, fix_arrow_compatibility, annotate_bar_values, add_plot_to_session
from state_manager import clear_downstream_from_split, clear_split_state


def render_split_dataset():
    st.header("✂️ Split Dataset")
    st.write("Create a train/test split and store split views in session state.")

    if "df" not in st.session_state or st.session_state.get("df") is None:
        st.warning("No dataset available. Upload a dataset on the Data page first.")
        if st.button("Open Data Page"):
            st.session_state["page"] = "Data"
            try:
                st.rerun()
            except AttributeError:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass
        render_info_panel("Split Dataset")
        return

    df_master = st.session_state.get("df")
    target = st.session_state.get("target_column")
    task_type = st.session_state.get("task_type")

    test_size = st.slider("Test set size (%)", min_value=5, max_value=50, value=20, step=1, key="split_test_size")
    stratify_option = False
    if task_type == "Classification":
        stratify_option = st.checkbox("Stratify by target (recommended)", value=True, key="split_stratify")
    else:
        st.caption("Stratified sampling is available only for Classification tasks.")

    seed = int(st.session_state.get("global_seed", 42))

    split_disabled = bool(st.session_state.get("split_done", False))
    if st.button("Run Split", key="run_split", disabled=split_disabled):
        if not target or target not in df_master.columns:
            st.error("Target column not set or not present in dataset. Set the target on the Data page first.")
        else:
            try:
                X = df_master.drop(columns=[target])
                y = df_master[target]
                test_frac = float(test_size) / 100.0
                stratify_col = y if (task_type == "Classification" and stratify_option) else None
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_frac, random_state=seed, stratify=stratify_col)

                clear_split_state(st.session_state)
                clear_downstream_from_split(st.session_state)
                st.session_state["pre_X_train"] = X_tr.copy()
                st.session_state["pre_X_test"] = X_te.copy()
                st.session_state["pre_y_train"] = y_tr.copy()
                st.session_state["pre_y_test"] = y_te.copy()
                st.session_state["split_done"] = True
                st.success(f"Split completed. Train rows: {len(X_tr)}, Test rows: {len(X_te)}")
            except Exception as e:
                st.error(f"Error during split: {e}")

    if st.session_state.get("split_done"):
        st.info("A split is currently active.")
        x_train = st.session_state.get("pre_X_train")
        x_test = st.session_state.get("pre_X_test")
        st.write({
            "pre_X_train_rows": len(x_train) if x_train is not None else 0,
            "pre_X_test_rows": len(x_test) if x_test is not None else 0,
        })
        # Visual comparison for Train vs Test
        try:
            y_train = st.session_state.get("pre_y_train")
            y_test = st.session_state.get("pre_y_test")
            if y_train is not None and y_test is not None:
                if task_type == "Classification":
                    # proportions
                    train_counts = y_train.value_counts(normalize=True)
                    test_counts = y_test.value_counts(normalize=True)
                    df_prop = pd.DataFrame({"Train": train_counts, "Test": test_counts}).fillna(0)

                    # absolute counts
                    train_abs = y_train.value_counts()
                    test_abs = y_test.value_counts()
                    df_count = pd.DataFrame({"Train": train_abs, "Test": test_abs}).fillna(0)

                    # ensure same index/order for both tables
                    categories = list(dict.fromkeys(list(df_prop.index) + list(df_count.index)))
                    df_prop = df_prop.reindex(categories, fill_value=0)
                    df_count = df_count.reindex(categories, fill_value=0)

                    # side-by-side plots: proportions (left) and counts (right)
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
                    # save split comparison plot into session for bundle export
                    try:
                        add_plot_to_session(fig, title="split_class_proportions_and_counts", page="Split Dataset", kind="split_comparison")
                    except Exception:
                        pass
                else:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(y_train, bins=30, alpha=0.5, label='Train', density=True)
                    ax.hist(y_test, bins=30, alpha=0.5, label='Test', density=True)
                    ax.set_title('Target Distribution: Train vs Test')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    try:
                        add_plot_to_session(fig, title="split_target_histogram", page="Split Dataset", kind="split_comparison")
                    except Exception:
                        pass
        except Exception:
            # non-fatal: continue without plots
            pass
        if st.button("Reset Split", key="reset_split"):
            clear_split_state(st.session_state)
            clear_downstream_from_split(st.session_state)
            st.success("Split cleared. You can run a new split now.")

    render_info_panel("Split Dataset")
