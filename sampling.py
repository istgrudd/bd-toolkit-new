import streamlit as st
import pandas as pd
import numpy as np
from ui_components import fix_arrow_compatibility, render_info_panel
from state_manager import clear_downstream_from_sampling


def render_sampling(show_info_panel: bool = True):
    st.header("🧪 Sampling - Create Subset")

    if not st.session_state.get("split_done"):
        st.warning("Please create a train/test split first on the 'Split Dataset' page.")
        if show_info_panel:
            render_info_panel("Sampling")
        return

    subset = st.radio("Select subset to sample from", ["Train", "Test"], index=0, key="sampling_subset")
    X = st.session_state.get(f"pre_X_{subset.lower()}")
    y = st.session_state.get(f"pre_y_{subset.lower()}")
    if X is None:
        st.error(f"No {subset} data available to sample from.")
        if show_info_panel:
            render_info_panel("Sampling")
        return

    max_rows = len(X)
    n = st.number_input("Number of samples", min_value=1, max_value=max_rows, value=min(100, max_rows), step=1)
    sampling_types = ["Random"]
    if st.session_state.get("task_type") == "Classification" and y is not None:
        sampling_types.append("Stratified")
    sampling_type = st.selectbox("Sampling type", sampling_types, key="sampling_type")

    if st.button("Create Sample"):
        try:
            seed = int(st.session_state.get("global_seed", 42))
            if sampling_type == "Random":
                sampled_X = X.sample(n=int(n), random_state=seed)
                sampled_y = y.loc[sampled_X.index] if y is not None else None
            else:
                # Stratified sampling: compute counts per class preserving proportions
                proportions = y.value_counts(normalize=True)
                raw = proportions * int(n)
                base = raw.astype(int)
                rem = raw - base
                needed = int(n) - base.sum()
                add_idx = np.argsort(-rem.values)[:needed] if needed > 0 else []
                counts = base.copy()
                for i in add_idx:
                    counts.iloc[i] += 1

                indices = []
                for cls, cnt in counts.items():
                    cls_idx = y[y == cls].index
                    if cnt >= len(cls_idx):
                        sel = list(cls_idx)
                    else:
                        sel = list(pd.Series(cls_idx).sample(n=int(cnt), random_state=seed).values)
                    indices.extend(sel)

                sampled_X = X.loc[indices]
                sampled_y = y.loc[indices]

            # Save sampled data to session
            key_x = f"sampled_X_{subset.lower()}"
            key_y = f"sampled_y_{subset.lower()}"
            st.session_state[key_x] = sampled_X.copy()
            st.session_state[key_y] = sampled_y.copy() if sampled_y is not None else None
            st.session_state["sampling_done"] = True
            clear_downstream_from_sampling(st.session_state)

            st.success(f"Created sample of {len(sampled_X)} rows from {subset} set (method={sampling_type}).")
            try:
                st.dataframe(fix_arrow_compatibility(sampled_X.head()))
            except Exception:
                st.dataframe(sampled_X.head())
        except Exception as e:
            st.error(f"Sampling error: {e}")

    if show_info_panel:
        st.markdown("---")
        render_info_panel("Sampling")
