import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import base64
import io

from input_data import render_input_data
from eda import render_eda
from data_cleansing import render_cleansing
from data_preprocessing import render_preprocessing
from data_modeling import render_modeling, render_validation, render_evaluation_summary
from split_dataset import render_split_dataset
from data_visualization import render_data_visualization
from competition_page import render_competition_page, render_export_page
from ui_components import render_info_panel

SESSION_KEYS = [
    "df",
    "task_type",
    "target_column",
    "global_seed",
    "page",
        
    "cleansing_steps",
    "preprocessing_steps",
    "pre_X_train",
    "pre_X_test",
    "pre_y_train",
    "pre_y_test",
    "resampled_pre_X_train",
    "resampled_pre_y_train",
    "trained_model",
    "trained_preprocessor",
    "last_saved_bundle",
    "eda_fig_missing_univariate",
    "split_fig_row_counts",
    "split_fig_class_prop",
]

SIDEBAR_PAGES = [
    "Landing",
    "Data",
    "EDA",
    "Split Dataset",
    "Cleansing",
    "Preprocessing",
    "Validation",
    "Training",
    "Evaluation Summary",
    "Export",
    "Submission",
]


def init_session_state():
    for key in SESSION_KEYS:
        if key not in st.session_state:
            if key == "global_seed":
                st.session_state[key] = 42
            elif key == "page":
                # Start with a brief splash screen on new sessions
                st.session_state[key] = "Splash"
            elif key in ("cleansing_steps", "preprocessing_steps"):
                st.session_state[key] = []
            else:
                st.session_state[key] = None


def reset_data():
    preserve = {"global_seed", "page"}
    for key in SESSION_KEYS:
        if key in preserve:
            continue
        if isinstance(st.session_state.get(key), list):
            st.session_state[key] = []
        else:
            st.session_state[key] = None
    # Streamlit automatically re-runs when session state changes; explicit rerun removed.





def main():
    st.set_page_config(page_title="Big Data Toolkit", layout="wide")
    init_session_state()
    # Respect URL query param `?page=...` for direct navigation (used by splash link)
    try:
        params = st.query_params
        if "page" in params and params["page"]:
            requested = params["page"][0]
            allowed = SIDEBAR_PAGES + ["Splash"]
            if requested in allowed:
                st.session_state["page"] = requested
    except Exception:
        pass
    # Sidebar title intentionally omitted to keep navigation compact.
    try:
        base_dir = Path(__file__).parent
    except Exception:
        base_dir = Path(os.getcwd())

    # Sidebar: hide on Splash, render otherwise
    logo_candidates = [base_dir.parent / 'bd-toolkit' / 'big_data.png', base_dir.parent / 'bd-toolkit-2' / 'big_data.png', base_dir / 'big_data.png']
    logo_path = None
    for p in logo_candidates:
        try:
            if p.exists():
                logo_path = p
                break
        except Exception:
            continue

    # Determine current page early
    page = st.session_state.get("page", "Splash")

    if page == "Splash":
        # Hide the sidebar entirely during splash
        hide_css = """
        <style>
        [data-testid="stSidebar"]{display: none;}
        .css-1v3fvcr {padding-left: 1rem;}
        </style>
        """
        st.markdown(hide_css, unsafe_allow_html=True)
    else:
        # render logo in sidebar (visible on non-splash pages)
        if logo_path is not None:
            try:
                sidebar_img = Image.open(logo_path)
                buf = io.BytesIO()
                sidebar_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                img_html = f"<div style='padding:8px 10px; text-align:center;'><img src='data:image/png;base64,{b64}' style='max-width:100%; height:auto; display:block; margin:0 auto; border-radius:6px;'/></div>"
                st.sidebar.markdown(img_html, unsafe_allow_html=True)
            except Exception:
                st.sidebar.markdown("**Big Data Toolkit**")
        else:
            st.sidebar.markdown("**Big Data Toolkit**")

        st.sidebar.markdown("---")
        seed = st.sidebar.number_input("Global Random Seed", value=int(st.session_state.get("global_seed", 42)), min_value=0, step=1)
        st.session_state.global_seed = int(seed)
        st.sidebar.markdown("---")

        st.sidebar.markdown("### Navigation")
        # Emoji-enhanced navigation
        page_emojis = {
            'Landing': '🏠',
            'Data': '📊',
            'EDA': '📈',
            'Split Dataset': '✂️',
            'Cleansing': '✨',
            'Preprocessing': '⚙️',
            'Validation': '🔍',
            'Training': '🎓',
            'Evaluation Summary': '📋',
            'Export': '📦',
            'Submission': '📤'
        }
        page_names = list(page_emojis.keys())
        # Determine radio index only when current page is one of the sidebar pages
        if st.session_state.get("page") in page_names:
            current_index = page_names.index(st.session_state.get("page", "Landing"))
        else:
            current_index = 0
        menu_options = [f"{page_emojis[p]} {p}" for p in page_names]
        selected_label = st.sidebar.radio("Navigate", menu_options, index=current_index)
        # extract page name after the emoji and space
        selected = selected_label.split(' ', 1)[1]
        # Only change session page from the sidebar when the current session page
        # is one of the sidebar pages (prevents overwriting Splash start page).
        if st.session_state.get("page") in page_names and selected != st.session_state.get("page"):
            st.session_state.page = selected
            try:
                st.rerun()
            except AttributeError:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass

        st.sidebar.markdown("---")
        if st.sidebar.button("Reset Data (clear)"):
            reset_data()

    page = st.session_state.get("page", "Landing")

    # Splash screen (initial opener) - centered minimal layout with large logo
    if page == "Splash":
        # prepare base64 embedded logo if available
        b64 = None
        if logo_path is not None:
            try:
                splash_img = Image.open(logo_path)
                buf = io.BytesIO()
                splash_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                b64 = None

        # splash HTML template (defined outside try/except so it's always available)
        splash_html = """
        <div style="min-height:60vh; display:flex; align-items:center; justify-content:center;">
          <div style="text-align:center; width:100%;">
            {img}
            <h1 style="margin:8px 0 6px 0; font-size:42px;">Big Data Toolkit</h1>
            <p style="margin:0 0 4px 0; font-size:18px;">Your end-to-end machine learning companion</p>
          </div>
        </div>
        """

        img_tag = f"<div style='display:block; margin:0 auto 18px auto;'><img src='data:image/png;base64,{b64}' style='width:320px; height:auto; display:block; margin:0 auto;'/></div>" if b64 else ""
        st.markdown(splash_html.format(img=img_tag), unsafe_allow_html=True)

        # Splash-only CSS to style and center the Streamlit button rendered below
        splash_btn_css = """
        <style>
        /* Memposisikan wrapper tombol ke tengah */
        /* Memposisikan wrapper tombol ke tengah dan menariknya ke atas */
        .stButton, div.stButton {
            display: flex !important;
            justify-content: center !important;
            margin-top: -55px !important; /* MENGGUNAKAN MINUS UNTUK MENARIK KE ATAS */
            margin-bottom: 20px !important;
            margin-left: 110px !important;
        }
        
        /* Memaksa warna background dan border tombol */
        div[data-testid="stButton"] button {
            background-color: #009E42 !important; 
            border: 1px solid #009E42 !important;
            border-radius: 8px !important;
            padding: 10px 22px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }

        /* INI KUNCINYA: Memaksa warna TULISAN di dalam tombol menjadi putih */
        div[data-testid="stButton"] button p, 
        div[data-testid="stButton"] button div, 
        div[data-testid="stButton"] button * {
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            margin: 0 !important;
        }
        
        /* Efek Hover (Saat disorot mouse) */
        div[data-testid="stButton"] button:hover {
            background-color: #008738 !important;
            border: 1px solid #008738 !important;
            box-shadow: 0 6px 14px rgba(0,0,0,0.2) !important;
        }
        div[data-testid="stButton"] button:hover * {
            color: #ffffff !important;
        }
        </style>
        """
        st.markdown(splash_btn_css, unsafe_allow_html=True)

        # Centered Streamlit button (reliable same-tab navigation)
        cols_cta = st.columns([1, 2, 1])
        with cols_cta[1]:
            if st.button("🚀 Start your journey", key="splash_start"):
                st.session_state["page"] = "Landing"
                try:
                    st.rerun()
                except AttributeError:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass

        return

    # Landing page (now simplified — major CTA moved to Splash)
    if page == "Landing":
        st.title("🏠 Landing")
        st.markdown("---")
        st.markdown("**Quick Start**")
        st.markdown("1. Navigate to `Data` to upload or point to a dataset.  ")
        st.markdown("2. Explore data in `EDA`.  ")
        st.markdown("3. Apply cleansing and preprocessing.  ")
        st.markdown("4. Train and validate models.  ")
        render_info_panel("Landing")

    elif page == "Data":
        render_input_data()

    elif page == "EDA":
        render_eda()

    elif page == "Split Dataset":
        render_split_dataset()

    # Sampling and Resampling moved into Preprocessing tabs

    elif page == "Cleansing":
        render_cleansing()

    elif page == "Preprocessing":
        render_preprocessing()

    elif page == "Validation":
        render_validation()

    elif page == "Training":
        render_modeling()

    elif page == "Evaluation Summary":
        render_evaluation_summary()

    elif page == "Export":
           render_export_page()

    elif page == "Submission":
        render_competition_page()

    else:
        st.write("Unknown page — showing landing.")
        st.session_state.page = "Landing"

    # Each page's render function calls the info panel individually.

    # Sticky footer (non-overlapping): add bottom padding to main container
    footer_html = """
    <style>
    .stApp .main .block-container{ padding-bottom:60px; }
    .bd-footer{
        position: fixed;
        left: 0;
        bottom: 0;
        right: 0;
        background: transparent;
        text-align: center;
        padding: 5px;
        font-size: 12px;
        border-top: 1px solid rgba(0,0,0,0.06);
    }
    </style>
    <div class="bd-footer"><small>© 2024 MBC Big Data</small></div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
