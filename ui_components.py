import streamlit as st
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import io
import hashlib
from datetime import datetime
from config_limits import MAX_EVALUATION_PLOTS


def annotate_bar_values(ax, fmt: Optional[str] = "{:.2f}", integer: bool = False, fontsize: int = 9):
    """Annotate bars on a matplotlib Axes with their heights.

    - `fmt`: format string for non-integer values (default two decimals).
    - `integer`: if True, values are formatted as integers.
    """
    try:
        for p in ax.patches:
            try:
                h = p.get_height()
            except Exception:
                continue
            if h is None or (isinstance(h, float) and np.isnan(h)):
                continue
            x = p.get_x() + p.get_width() / 2
            if integer:
                label = f"{int(h):d}"
            else:
                try:
                    label = fmt.format(h)
                except Exception:
                    label = str(h)
            ax.text(x, h, label, ha="center", va="bottom", fontsize=fontsize)
    except Exception:
        # non-fatal: fail silently to avoid breaking plotting
        return


def fix_arrow_compatibility(df):
    """Convert object columns to pandas 'string' dtype to avoid Arrow serialization errors.

    Returns a shallow copy with object columns cast to 'string' where possible.
    """
    if df is None:
        return df
    try:
        df2 = df.copy()
    except Exception:
        return df
    try:
        obj_cols = df2.select_dtypes(include=["object"]).columns.tolist()
        for col in obj_cols:
            try:
                df2[col] = df2[col].astype("string")
            except Exception:
                # leave as-is if conversion fails
                continue
    except Exception:
        return df
    return df2


def render_info_panel(page_name: str, current_model_name: Optional[str] = None, current_hyperparams: Optional[Any] = None, model_list: Optional[list] = None):
    """Reusable info panel displayed at the bottom of each page.

    - Renders a `Show:` radio for model pages (Validation/Training) that toggles
      between `Model` (model-specific docs/hyperparams) and `Other` (concepts).
    - Renders a set of top-level expanders (no nested expanders).
    - When `Model` is active a `⚙️ Hyperparameter Info` expander exposes a
      selectbox of hyperparameters and shows descriptions + recommended values.
    """

    # Example metadata / docs / terms
    page_descriptions = {
        "Landing": "Overview and quick start for Big Data Toolkit 2.",
        "Data": "Upload or connect to datasets, preview and inspect types.",
        "EDA": "Explore distributions, correlations, missingness, and summaries.",
        "Cleansing": "Impute missing values, remove duplicates, and handle outliers.",
        "Preprocessing": "Scale, encode, and engineer features for modeling.",
        "Validation": "Set up cross-validation and compute evaluation metrics.",
        "Training": "Train models, track hyperparameters and artifacts.",
        "Evaluation Summary": "Summarize model performance across chosen metrics.",
        "Export": "Serialize models and export bundles for deployment.",
        "Submission": "Prepare submission files for competitions or evaluation.",
    }

    terms_by_page = {
        "Landing": ["Overview", "Quick Start"],
        "Data": ["CSV", "Preview", "Missing Values", "Dtypes"],
        "EDA": ["Histogram", "Box Plot", "Correlation", "Missingness", "Summary Stats"],
        "Cleansing": ["Imputation", "Outlier Detection", "Duplicate Removal", "Scaling"],
        "Preprocessing": ["Scaling", "One-Hot Encoding", "Label Encoding", "Feature Selection"],
        "Validation": ["K-Fold", "StratifiedKFold", "Holdout"],
        "Training": ["Grid Search", "Random Search", "Early Stopping", "Ensembling"],
        "Evaluation Summary": ["Confusion Matrix", "ROC Curve", "Precision", "Recall", "F1-Score"],
        "Export": ["Pickle", "Joblib", "ONNX"],
        "Submission": ["Submission CSV", "ID Column", "Target Column"],
    }

    docs_by_page = {
        "default": "https://scikit-learn.org/stable/",
        "Landing": "https://streamlit.io/",
        "Data": "https://pandas.pydata.org/",
        "EDA": "https://seaborn.pydata.org/",
        "Cleansing": "https://scikit-learn.org/stable/modules/impute.html",
        "Preprocessing": "https://scikit-learn.org/stable/modules/preprocessing.html",
        "Validation": "https://scikit-learn.org/stable/modules/cross_validation.html",
        "Training": "https://scikit-learn.org/stable/supervised_learning.html",
        "Evaluation Summary": "https://scikit-learn.org/stable/modules/model_evaluation.html",
        "Export": "https://scikit-learn.org/",
        "Submission": "https://pandas.pydata.org/",
    }

    packages_by_page = {
        "Data": ["pandas", "pyarrow"],
        "EDA": ["seaborn", "matplotlib"],
        "Cleansing": ["scikit-learn"],
        "Preprocessing": ["scikit-learn"],
        "Validation": ["scikit-learn"],
        "Training": ["scikit-learn", "xgboost", "lightgbm"],
        "Evaluation Summary": ["scikit-learn"],
        "Export": ["joblib", "pickle"],
        "Submission": ["pandas"],
    }

    # Top-level separator/title (not an expander to avoid nested expanders)
    st.markdown("---")
    st.markdown("**ℹ️ Info Panel**")

    non_model_pages = {"Landing", "Data", "EDA", "Cleansing", "Preprocessing", "Evaluation Summary", "Export", "Submission"}
    model_pages = {"Validation", "Training"}

    # Helper for term explanations
    term_explanations = {
        "Histogram": "Shows distribution of a single variable.",
        "Box Plot": "Visualizes distribution and potential outliers.",
        "Correlation": "Shows pairwise linear relationships between variables.",
        "Correlation Matrix": "A matrix of Pearson correlation coefficients between features.",
        "Missingness": "Patterns and counts of missing values.",
        "Missing Value Analysis": "Techniques to identify and quantify missing values in the dataset.",
        "Imputation": "Replacing missing values with estimates.",
        "One-Hot Encoding": "Convert categorical to binary indicator columns.",
        "Grid Search": "Systematic search over hyperparameter grid.",
        "Random Search": "Randomized sampling of hyperparameter space.",
        "Confusion Matrix": "Counts of true/false positives/negatives.",
        "Describe Method": "`DataFrame.describe()` computes summary statistics for numeric and object columns.",
        "Info Method": "`DataFrame.info()` shows dtypes and non-null counts for each column.",
    }

    # Landing special-case: show a short guided 'start your journey' selector
    if page_name == "Landing":
        terms = [
            '📊 Data Page',
            '📈 EDA Page',
            '✂️ Split Dataset Page',
            '✨ Cleansing Page',
            '⚙️ Preprocessing Page',
            '🔍 Validation Page',
            '🎓 Training Page',
            '📋 Evaluation Summary Page',
            '📦 Export Page',
            '📤 Submission Page',
        ]
        term_explanations = {
            '📊 Data Page': 'Upload your training dataset (CSV), select the target column, and choose the task type (Classification or Regression). This page initializes the entire workflow.',
            '📈 EDA Page': 'Explore your data with univariate (histograms, boxplots, bar charts) and bivariate (scatter plots, heatmaps) visualizations. Understand distributions, correlations, and missing values.',
            '✂️ Split Dataset Page': 'Split your data into training and testing sets. For classification, see class proportions; for regression, compare target distributions. Essential for honest evaluation.',
            '✨ Cleansing Page': 'Handle missing values (imputation with fit/transform), remove duplicates, detect and cap outliers, drop unnecessary columns, and convert data types. Each operation can be applied to selected subsets.',
            '⚙️ Preprocessing Page': 'Scale numerical features, encode categorical variables, and engineer new features using custom formulas. All steps support fit/transform/fit_transform on train/test subsets.',
            '🔍 Validation Page': 'Select a machine learning model, adjust hyperparameters, and validate using cross-validation or a train-validation split. View feature importance (with permutation fallback) and performance metrics.',
            '🎓 Training Page': 'Train the final model on the full training set (optionally with resampling). Save the model together with a fitted preprocessor (pipeline) and metadata. Ready for export.',
            '📋 Evaluation Summary Page': 'Compare validation and final training results side by side. View confusion matrix, ROC/PR curves, residuals, and more. Upload multiple bundles to compare on the same test set.',
            '📦 Export Page': 'Download a ZIP archive of any saved model bundle, including the model file, metadata, evaluation plots, and data splits. Supports batch export of all models.',
            '📤 Submission Page': 'Upload a test CSV (without target column) and a saved model bundle. Apply preprocessing and generate predictions. Download submission CSV in the required format.'
        }
        selected_term = st.selectbox("Select a page to learn about", terms, key="landing_info_term")
        st.markdown(f"### {selected_term}")
        st.markdown(term_explanations[selected_term])
        st.markdown("📖 [Documentation](https://github.com/your-repo/big-data-toolkit)")
        return

    # For model pages show a Mode toggle above the Topic selectbox
    if page_name in model_pages:
        mode = st.radio("Show:", ["Model", "Other"], horizontal=True, key=f"info_mode_{page_name}")
        if mode == "Other":
            terms = ["Cross-Validation", "Overfitting", "Bias-Variance", "Hyperparameter Tuning", "Feature Importance"]
        else:
            # model mode: use provided model list if available, fall back to page terms
            terms = model_list if (model_list and isinstance(model_list, list) and len(model_list) > 0) else terms_by_page.get(page_name, ["General"])
    else:
        terms = terms_by_page.get(page_name, ["General"])

    # Avoid fixed widget keys here to prevent duplicate-key errors when panel is rendered multiple times
    page_key = page_name.replace(" ", "_")
    selected_term = st.selectbox("Topic", terms, key=f"info_topic_{page_key}")

    if page_name not in model_pages:
        # show five expanders for non-model pages / other concepts
        for exp in ["📖 Name", "📝 Short Explanation", "💡 When to Use", "🔗 Documentation Link", "📦 Packages"]:
            with st.expander(exp, expanded=False):
                if exp == "📖 Name":
                    st.write(f"**Page:** {page_name}")
                    st.write(f"**Topic:** {selected_term}")
                    if current_model_name:
                        st.write(f"**Model:** {current_model_name}")

                elif exp == "📝 Short Explanation":
                    st.write(term_explanations.get(selected_term, page_descriptions.get(page_name, "General information.")))

                elif exp == "💡 When to Use":
                    st.write(f"Guidance: When to use **{selected_term}** in the context of {page_name}.")

                elif exp == "🔗 Documentation Link":
                    term_doc = docs_by_page.get(selected_term)
                    url = term_doc or docs_by_page.get(page_name, docs_by_page["default"])
                    st.markdown(f"Documentation: [{url}]({url})")

                elif exp == "📦 Packages":
                    pkgs = packages_by_page.get(page_name, ["scikit-learn"])
                    st.write(", ".join(pkgs))

    else:
        # Model pages: show model-focused expanders including Hyperparameter Info
        for exp in ["📌 Name", "🏷️ Category", "📖 Short Explanation", "⚙️ Hyperparameter Info", "💡 When to Use", "🔗 Documentation Link"]:
            with st.expander(exp, expanded=False):
                if exp == "📌 Name":
                    st.write(f"**Page:** {page_name}")
                    st.write(f"**Topic:** {selected_term}")
                    if current_model_name:
                        st.write(f"**Model:** {current_model_name}")

                elif exp == "🏷️ Category":
                    st.write("Supervised / Unsupervised / Ensemble etc. (placeholder)")

                elif exp == "📖 Short Explanation":
                    st.write(term_explanations.get(selected_term, page_descriptions.get(page_name, "Modeling related page.")))

                elif exp == "⚙️ Hyperparameter Info":
                    # Provide hyperparameter details for the selected model or fallbacks
                    hp_map = {}
                    if current_hyperparams and isinstance(current_hyperparams, dict):
                        # support both nested dicts {param: {desc,recommended}} or simple defaults
                        for k, v in current_hyperparams.items():
                            if isinstance(v, dict) and ("desc" in v or "recommended" in v):
                                hp_map[k] = {"desc": v.get("desc"), "recommended": v.get("recommended")}
                            else:
                                # simple default value -> recommend that value
                                hp_map[k] = {"desc": None, "recommended": [v]}
                    else:
                        # light fallback
                        hp_map = {
                            "n_estimators": {"desc": "Number of trees in ensemble.", "recommended": [100, 200]},
                            "max_depth": {"desc": "Maximum depth of tree.", "recommended": [None, 10, 20]},
                            "C": {"desc": "Inverse of regularization strength.", "recommended": [0.01, 0.1, 1, 10]},
                        }

                    names = list(hp_map.keys())
                    if names:
                        sel = st.selectbox("Hyperparameter", names, key=f"info_hp_{page_key}")
                        info = hp_map.get(sel, {})
                        desc = info.get("desc")
                        rec = info.get("recommended")
                        if desc:
                            st.write(desc)
                        if rec is not None:
                            st.write("**Recommended:**", rec)
                        if not desc and rec is None:
                            st.write("No detailed info available for this hyperparameter.")
                    else:
                        st.write("No hyperparameters available.")

                elif exp == "💡 When to Use":
                    st.write(f"Guidance: When to use **{selected_term}** for {page_name}.")

                elif exp == "🔗 Documentation Link":
                    term_doc = docs_by_page.get(selected_term)
                    url = term_doc or docs_by_page.get(page_name, docs_by_page["default"])
                    st.markdown(f"Documentation: [{url}]({url})")


def render_learning_panel(page_name: str, terms: Optional[list] = None):
    """Render a small learning panel expander with a term selectbox and docs.

    This is placed at the very bottom of a page (outside other expanders).
    """
    default_terms = {
        "Data": ["CSV", "Preview", "Missing Values", "Dtypes"],
        "EDA": ["Histogram", "Box Plot", "Correlation Matrix", "Missing Value Analysis", "Describe Method", "Info Method"],
        "Cleansing": ["Imputation", "Outlier Detection"],
        "Preprocessing": ["Scaling", "Encoding"],
        "Training": ["Hyperparameters", "Grid Search"],
    }

    explanations = {
        "Histogram": "A histogram shows the distribution of numeric values for a feature.",
        "Box Plot": "A box plot summarizes distribution and highlights outliers per group.",
        "Correlation Matrix": "A matrix of Pearson correlation coefficients between features.",
        "Missing Value Analysis": "Techniques to identify and quantify missing values in the dataset.",
        "Describe Method": "`DataFrame.describe()` computes summary statistics for numeric and object columns.",
        "Info Method": "`DataFrame.info()` shows dtypes and non-null counts for each column.",
        "CSV": "Comma-separated values file format commonly used for tabular data.",
        "Preview": "A small sample of the dataset (head) to validate uploaded data.",
        "Dtypes": "Column data types (int, float, object, category) used for selecting analyses.",
    }

    docs = {
        "Histogram": "https://seaborn.pydata.org/",
        "Box Plot": "https://seaborn.pydata.org/",
        "Correlation Matrix": "https://scikit-learn.org/stable/",
        "Missing Value Analysis": "https://pandas.pydata.org/",
        "Describe Method": "https://pandas.pydata.org/",
        "Info Method": "https://pandas.pydata.org/",
        "CSV": "https://en.wikipedia.org/wiki/Comma-separated_values",
    }

    terms = terms or default_terms.get(page_name, ["General"])

    with st.expander("📚 Learning Panel", expanded=False):
        choice = st.selectbox("Select a term to learn:", terms, key=f"learn_{page_name}")
        text = explanations.get(choice, "See documentation link for details.")
        doc = docs.get(choice, "https://scikit-learn.org/stable/")
        st.markdown(f"**{choice}**\n\n{text}")
        st.markdown(f"Documentation: [{doc}]({doc})")


def available_columns() -> list:
    """Return a sorted list of available feature columns present in session state.

    Scans `pre_X_train`, `pre_X_test`, `df`, and common training variants (resampled/sample)
    and returns the union of column names so UIs reflect newly created features.
    """
    cols = set()
    keys = ["pre_X_train", "pre_X_test", "df", "resampled_pre_X_train", "sampled_X_train"]
    for k in keys:
        df_local = st.session_state.get(k)
        if df_local is None:
            continue
        try:
            cols.update(list(df_local.columns))
        except Exception:
            continue
    return sorted(cols)


def available_numeric_columns() -> list:
    """Return sorted numeric-like columns present in any session-state dataframe.

    A column is considered numeric-like if any available dataframe reports a numeric dtype
    or can be coerced to numeric with at least one non-null value.
    """
    all_cols = available_columns()
    num_cols = set()
    keys = ["pre_X_train", "pre_X_test", "df", "resampled_pre_X_train", "sampled_X_train"]
    for c in all_cols:
        for k in keys:
            df_local = st.session_state.get(k)
            if df_local is None or c not in df_local.columns:
                continue
            try:
                s = df_local[c]
                if pd.api.types.is_numeric_dtype(s):
                    num_cols.add(c)
                    break
                else:
                    coerced = pd.to_numeric(s, errors="coerce")
                    if coerced.dropna().size > 0:
                        num_cols.add(c)
                        break
            except Exception:
                continue
    return sorted(num_cols)


def add_plot_to_session(fig, title: Optional[str] = None, page: Optional[str] = None, kind: Optional[str] = None):
    """Save a matplotlib figure into session state for inclusion in bundles.

    - fig: matplotlib.figure.Figure
    - title: human readable title used for default filename
    - page: originating page name (e.g., 'EDA', 'Validation')
    - kind: semantic kind ('confusion_matrix', 'pr_curve', 'split_plot', etc.)

    Returns the chosen filename (str) if added, or None if duplicate detected.
    """
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        data = buf.getvalue()
    except Exception:
        return None

    # compute content hash to avoid duplicates
    h = hashlib.sha1(data).hexdigest()
    if "evaluation_plots_hashes" not in st.session_state:
        st.session_state["evaluation_plots_hashes"] = set()
    if h in st.session_state["evaluation_plots_hashes"]:
        # already saved identical plot
        return None

    # ensure container exists
    if "evaluation_plots_temp" not in st.session_state:
        st.session_state["evaluation_plots_temp"] = {}
    if "evaluation_plots_meta" not in st.session_state:
        st.session_state["evaluation_plots_meta"] = {}

    # build default filename using timestamp and provided title
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_title = (title or "plot").strip().replace(" ", "_")[:60]
    fname_base = f"{ts}_{safe_title}.png"

    # ensure unique filename in session
    fname = fname_base
    idx = 1
    while fname in st.session_state["evaluation_plots_temp"]:
        fname = f"{ts}_{safe_title}_{idx}.png"
        idx += 1

    # store bytes and metadata
    st.session_state["evaluation_plots_temp"][fname] = data
    st.session_state["evaluation_plots_meta"][fname] = {"title": title or safe_title, "page": page or "unknown", "kind": kind or "plot", "created_at": ts, "hash": h}
    st.session_state["evaluation_plots_hashes"].add(h)
    while len(st.session_state["evaluation_plots_temp"]) > MAX_EVALUATION_PLOTS:
        oldest_name = next(iter(st.session_state["evaluation_plots_temp"]))
        oldest_meta = st.session_state["evaluation_plots_meta"].pop(oldest_name, {})
        st.session_state["evaluation_plots_temp"].pop(oldest_name, None)
        oldest_hash = oldest_meta.get("hash")
        if oldest_hash is not None:
            st.session_state["evaluation_plots_hashes"].discard(oldest_hash)
    return fname
