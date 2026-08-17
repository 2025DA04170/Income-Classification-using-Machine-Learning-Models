"""
Income Classification using Machine Learning Models
--------------------------------------------------
Streamlit app for the Adult Census Income classification project.

App sections:
- Model Metrics: Shows stored training/evaluation metrics for each model.
- Predictions: Supports uploaded CSV data. If no CSV is uploaded, the app uses bundled test_data.csv.
- Model Comparison: Compares model performance and shows Random Forest feature importance when available.
"""

import json
import os
from typing import Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
icon = Image.open("assets/favicon.png")

st.set_page_config(
    page_title="Income Classification · ML Models",
    page_icon=icon,
    layout="wide",
)

# ----------------------------------------------------------------------------
# PATHS AND CONSTANTS
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

TEST_DATA_CANDIDATES = [
    os.path.join(HERE, "data", "processed", "test_data.csv"),
    os.path.join(HERE, "test_data.csv"),
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "kNN": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest (Ensemble)": "random_forest_model.pkl",
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": "A linear classification model that works well when the relationship between input features and the target is approximately linear.",
    "Decision Tree": "A rule-based model that splits the data into decision paths. It is easy to interpret but can overfit if not controlled.",
    "kNN": "An instance-based model that classifies a record based on nearby examples in the feature space.",
    "Naive Bayes": "A probabilistic classifier based on Bayes theorem. It assumes features are conditionally independent.",
    "Random Forest (Ensemble)": "An ensemble model that combines multiple decision trees to improve stability and generalisation.",
}

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "income"
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# ----------------------------------------------------------------------------
# CUSTOM CSS AND SVG ASSETS
# ----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
    }

    .svg-header {
        margin-bottom: 1rem;
    }

    .info-card {
        border: 1px solid #d8dee9;
        border-radius: 8px;
        padding: 14px;
        background-color: #f8fafc;
        margin-bottom: 12px;
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .section-title h2 {
        margin: 0;
        color: #1f2937;
    }

    .source-uploaded {
        background-color: #dcfce7;
        border: 1px solid #86efac;
        color: #14532d;
        padding: 10px;
        border-radius: 7px;
        margin-bottom: 12px;
        font-weight: 600;
    }

    .source-preloaded {
        background-color: #e0f2fe;
        border: 1px solid #7dd3fc;
        color: #075985;
        padding: 10px;
        border-radius: 7px;
        margin-bottom: 12px;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        font-size: 24px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f3f4f6;
        border-radius: 8px;
        padding: 10px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

HEADER_SVG = """
<div class="svg-header" style="margin-top: 2rem;">
<svg width="100%" height="95" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="headerGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:rgba(17,24,39,0.85); backdrop-filter: blur(20px);" />
      <stop offset="100%" style="stop-color:rgba(17,24,39,0.85);stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="100%" height="95" rx="10" fill="url(#headerGradient)"/>
  <circle cx="42" cy="48" r="23" fill="#ffffff" opacity="0.16"/>
    <!-- Input Nodes -->
    <circle cx="15" cy="20" r="6" fill="#60A5FA"/>
    <circle cx="15" cy="45" r="6" fill="#34D399"/>
    <circle cx="15" cy="70" r="6" fill="#FBBF24"/>
    <!-- Hidden Model -->
    <circle cx="45" cy="45" r="12" fill="#2563EB"/>
    <!-- Output Classes -->
    <circle cx="75" cy="30" r="8" fill="#10B981"/>
    <circle cx="75" cy="60" r="8" fill="#F59E0B"/>
    <!-- Connections -->
    <line x1="21" y1="20" x2="35" y2="40"stroke="#94A3B8" stroke-width="2"/>
    <line x1="21" y1="45" x2="33" y2="45" stroke="#94A3B8" stroke-width="2"/>
    <line x1="21" y1="70" x2="35" y2="50" stroke="#94A3B8" stroke-width="2"/>
    <line x1="57" y1="40" x2="67" y2="30" stroke="#94A3B8" stroke-width="2"/>
    <line x1="57" y1="50" x2="67" y2="60" stroke="#94A3B8" stroke-width="2"/>
  <text x="50%" y="40" text-anchor="middle" font-size="28" font-family="Arial" fill="white" font-weight="600">
    Income Classification · ML Models Metrics · Dashboard
  </text>
  <text x="50%" y="68" text-anchor="middle" font-size="14" font-family="Arial" fill="#dddddd">
    Adult Census Income Dataset · 5 Machine Learning Models
  </text>
</svg>
</div>
"""

METRICS_ICON = """
<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <rect x="3" y="12" width="4" height="8" fill="#2563eb"/>
  <rect x="10" y="7" width="4" height="13" fill="#2563eb"/>
  <rect x="17" y="4" width="4" height="16" fill="#2563eb"/>
</svg>
"""

PREDICTION_ICON = """
<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="12" r="9" fill="#2563eb"/>
  <path d="M8 12 L11 15 L16 9" stroke="white" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

COMPARISON_ICON = """
<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <rect x="3" y="5" width="18" height="3" fill="#2563eb"/>
  <rect x="3" y="11" width="14" height="3" fill="#2563eb"/>
  <rect x="3" y="17" width="10" height="3" fill="#2563eb"/>
</svg>
"""

# ----------------------------------------------------------------------------
# CACHED LOADERS
# ----------------------------------------------------------------------------

@st.cache_resource
def load_preprocessor():
    return joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))


@st.cache_resource
def load_model(model_file: str):
    return joblib.load(os.path.join(MODEL_DIR, model_file))


@st.cache_data
def load_metrics_json():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics_csv():
    path = os.path.join(MODEL_DIR, "metrics.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_feature_names():
    path = os.path.join(MODEL_DIR, "feature_names.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def load_csv_from_path(path: str):
    return pd.read_csv(path)


# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------

def get_test_data_path() -> Optional[str]:
    for path in TEST_DATA_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(r"^\s*\?\s*$", np.nan, regex=True)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    return df


def encode_true_labels(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.replace(".", "", regex=False)
    return cleaned.map({"<=50K": 0, ">50K": 1})


def build_report_dataframe(report):
    if not isinstance(report, dict):
        return pd.DataFrame({"Classification Report": [str(report)]})

    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"Label": label}
            row.update(values)
            rows.append(row)
        else:
            rows.append({"Label": label, "Value": values})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("Label")


def get_comparison_dataframe(metrics_json: Optional[Dict], metrics_csv: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if metrics_csv is not None and "Model" in metrics_csv.columns:
        return metrics_csv.set_index("Model")

    if metrics_json is not None:
        return pd.DataFrame(metrics_json).T

    return None


def render_section_title(icon_svg: str, title: str):
    # st.markdown(
    #     f"""
    #         {icon_svg} 
    #     """,
    #     unsafe_allow_html=True,
    # )
    st.subheader(title)


def render_data_source_badge(source_label: str):
    if source_label == "Live Uploaded Dataset":
        css_class = "source-uploaded"
    else:
        css_class = "source-preloaded"

    st.markdown(
        f'<div class="{css_class}">Data Source: {source_label}</div>',
        unsafe_allow_html=True,
    )


def safe_metric_value(metrics: Dict, key: str) -> str:
    value = metrics.get(key, None)
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def plot_confusion_matrix(cm, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"],
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------

st.markdown(HEADER_SVG, unsafe_allow_html=True)

st.markdown(
    """
This app compares five classification models on the Adult Income dataset and predicts whether an individual's annual income exceeds 50K. It supports both a bundled test dataset and user-uploaded CSV files.
    """
)

metrics_json = load_metrics_json()
metrics_csv = load_metrics_csv()
comparison_df_global = get_comparison_dataframe(metrics_json, metrics_csv)

# Default values used by sidebar after tab execution
active_df = None
active_source = "Preloaded Repository Dataset"
selected_model_for_sidebar = list(MODEL_FILES.keys())[0]

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Model Metrics", "Predictions", "Model Comparison"])

# ----------------------------------------------------------------------------
# TAB 1: MODEL METRICS
# ----------------------------------------------------------------------------

with tab1:
    render_section_title(METRICS_ICON, "Model Evaluation Metrics")

    if metrics_json is None:
        st.warning("model/metrics.json was not found. Run python model/train_models.py first to generate the stored metrics.")
    else:
        with st.container(border=True):
            selected_model_metrics = st.selectbox(
                "Selected Model",
                list(MODEL_FILES.keys()),
                key="metrics_model_selector",
            )
        st.markdown("</div>", unsafe_allow_html=True)
        selected_model_for_sidebar = selected_model_metrics

        comparison_df = pd.DataFrame(metrics_json).T
        available_metric_cols = [c for c in METRIC_COLUMNS if c in comparison_df.columns]

        st.caption("These metrics are computed from the held-out test split during model training.")
        st.dataframe(
            comparison_df[available_metric_cols].astype(float).style.format("{:.4f}"),
            use_container_width=True,
        )

        m = metrics_json[selected_model_metrics]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", safe_metric_value(m, "Accuracy"))
            st.metric("Precision", safe_metric_value(m, "Precision"))
        with col2:
            st.metric("Recall", safe_metric_value(m, "Recall"))
            st.metric("F1 Score", safe_metric_value(m, "F1"))
        with col3:
            st.metric("AUC Score", safe_metric_value(m, "AUC"))
            st.metric("MCC Score", safe_metric_value(m, "MCC"))

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.subheader("Confusion Matrix")
            if "ConfusionMatrix" in m:
                plot_confusion_matrix(m["ConfusionMatrix"])
            else:
                st.info("Confusion matrix is not available in metrics.json for this model.")

        with detail_col2:
            st.subheader("ROC Curve")
            if "FPR" in m and "TPR" in m:
                roc_fig, roc_ax = plt.subplots(figsize=(5, 4))
                roc_ax.plot(
                    m["FPR"], 
                    m["TPR"], 
                    linewidth=3, 
                    label=f"AUC = {float(m.get('AUC', 0)):.3f}")
                roc_ax.fill_between(
                m["FPR"],
                    m["TPR"],
                    alpha=0.2,
                    color="#234cb2"
                )
                roc_ax.plot([0, 1], [0, 1], linestyle="--", color="red")
                roc_ax.set_xlabel("False Positive Rate")
                roc_ax.set_ylabel("True Positive Rate")
                roc_ax.legend()
                st.pyplot(roc_fig)
            else:
                st.info("ROC data is not available in metrics.json for this model.")

        st.subheader("Classification Report")
        if "ClassificationReport" in m:
            report_df = build_report_dataframe(m["ClassificationReport"])
            numeric_cols = report_df.select_dtypes(include=["number"]).columns
            report_df[numeric_cols] = report_df[numeric_cols].round(3)
            st.dataframe(report_df, use_container_width=True)
        else:
            st.info("Classification report is not available in metrics.json for this model.")

# ----------------------------------------------------------------------------
# TAB 2: PREDICTIONS
# ----------------------------------------------------------------------------

with tab2:
    render_section_title(PREDICTION_ICON, "Prediction using custom dataset")

    with st.container(border=True):
        selected_model_prediction = st.selectbox(
            "Selected Model",
            list(MODEL_FILES.keys()),
            key="prediction_model_selector",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    selected_model_for_sidebar = selected_model_prediction

    test_data_path = get_test_data_path()

    uploaded_file = st.file_uploader(
        "Upload Test CSV",
        type=["csv"],
        help="The CSV should contain the Adult dataset feature columns. If it also contains income, live evaluation metrics will be displayed.",
    )

    if test_data_path is not None:
        with open(test_data_path, "rb") as f:
            st.download_button(
                "Download test_data.csv",
                data=f,
                file_name="test_data.csv",
                mime="text/csv",
            )
    else:
        st.warning("Bundled test_data.csv was not found. Expected location: data/processed/test_data.csv or project root test_data.csv.")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        active_source = "Live Uploaded Dataset"
    elif test_data_path is not None:
        raw_df = load_csv_from_path(test_data_path)
        active_source = "Preloaded Repository Dataset"
    else:
        raw_df = None
        active_source = "No Dataset Available"

    active_df = raw_df

    if raw_df is None:
        st.error("No input data is available for prediction. Upload a CSV file or add test_data.csv to data/processed.")
    else:
        render_data_source_badge(active_source)

        st.subheader("Dataset Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.caption(f"{raw_df.shape[0]} rows x {raw_df.shape[1]} columns")

        with st.expander("Dataset Statistics"):
            numeric_summary = raw_df.select_dtypes(include=[np.number]).describe().T
            if not numeric_summary.empty:
                st.dataframe(numeric_summary.round(3), use_container_width=True)
            else:
                st.info("No numeric columns were found for statistical summary.")

        try:
            features_df = prepare_features(raw_df)
            preprocessor = load_preprocessor()
            model = load_model(MODEL_FILES[selected_model_prediction])

            X = features_df[FEATURE_COLS]
            X_t = preprocessor.transform(X)

            y_pred = model.predict(X_t)
            pred_labels = pd.Series(y_pred).map({0: "<=50K", 1: ">50K"})

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_t)[:, 1]
            else:
                y_proba = np.zeros(len(y_pred))

            results_df = raw_df.copy()
            results_df["Prediction"] = pred_labels.values
            results_df["Probability_>50K"] = np.round(y_proba, 4)

            if TARGET in raw_df.columns:
                y_true = encode_true_labels(raw_df[TARGET])
                valid_mask = y_true.notna()

                if valid_mask.sum() > 0:
                    y_true_v = y_true[valid_mask].astype(int)
                    y_pred_v = y_pred[valid_mask.values]
                    y_proba_v = y_proba[valid_mask.values]

                    auc_value = np.nan
                    if y_true_v.nunique() == 2:
                        auc_value = roc_auc_score(y_true_v, y_proba_v)

                    live_metrics = {
                        "Accuracy": accuracy_score(y_true_v, y_pred_v),
                        "AUC": auc_value,
                        "Precision": precision_score(y_true_v, y_pred_v, zero_division=0),
                        "Recall": recall_score(y_true_v, y_pred_v, zero_division=0),
                        "F1": f1_score(y_true_v, y_pred_v, zero_division=0),
                        "MCC": matthews_corrcoef(y_true_v, y_pred_v),
                    }

                    st.subheader(f"Live Evaluation Metrics: {selected_model_prediction}")
                    st.dataframe(pd.DataFrame([live_metrics]).round(4), use_container_width=True)

                    live_col1, live_col2 = st.columns(2)
                    with live_col1:
                        st.markdown("**Confusion Matrix**")
                        cm = confusion_matrix(y_true_v, y_pred_v)
                        plot_confusion_matrix(cm, title="Live Confusion Matrix")

                    with live_col2:
                        st.markdown("**Classification Report**")
                        rep = classification_report(
                            y_true_v,
                            y_pred_v,
                            target_names=["<=50K", ">50K"],
                            output_dict=True,
                            zero_division=0,
                        )
                        live_report_df = build_report_dataframe(rep)
                        numeric_cols = live_report_df.select_dtypes(include=["number"]).columns
                        live_report_df[numeric_cols] = live_report_df[numeric_cols].round(3)
                        st.dataframe(live_report_df, use_container_width=True)
                else:
                    st.info("The income column is present, but no valid labels were found for live evaluation.")
            else:
                st.info("The dataset does not include the income column, so only predictions are shown.")

            st.subheader("Prediction Results")
            st.dataframe(results_df.head(200), use_container_width=True)
            st.caption(f"Showing up to 200 of {len(results_df)} predicted rows.")

            st.download_button(
                "Download predictions.csv",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ----------------------------------------------------------------------------
# TAB 3: MODEL COMPARISON
# ----------------------------------------------------------------------------

with tab3:
    render_section_title(COMPARISON_ICON, "Model Comparison")

    comparison_df = get_comparison_dataframe(metrics_json, metrics_csv)

    if comparison_df is None:
        st.warning("Model comparison data was not found. Run python model/train_models.py first to generate metrics files.")
    else:
        available_metric_cols = [c for c in METRIC_COLUMNS if c in comparison_df.columns]
        st.dataframe(comparison_df[available_metric_cols].round(4), use_container_width=True)

        winner_model = comparison_df["Accuracy"].idxmax()
        winner_accuracy = comparison_df["Accuracy"].max()

        st.subheader("Best Performing Model")
        st.info(f"{winner_model} achieved the highest accuracy score of {winner_accuracy:.4f} on the stored evaluation results.")

        st.subheader("Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            x=comparison_df.index,
            y=comparison_df["Accuracy"],
            palette="viridis",
            hue=comparison_df.index,
            legend=False,
            ax=ax,
        )
        plt.xticks(rotation=20, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("")
        st.pyplot(fig)

        st.subheader("Random Forest Feature Importance")
        try:
            rf_model = load_model(MODEL_FILES["Random Forest (Ensemble)"])
            feature_names = load_feature_names()

            if feature_names is None:
                raise FileNotFoundError("feature_names.pkl was not found")

            if not hasattr(rf_model, "feature_importances_"):
                raise AttributeError("Random Forest model does not expose feature_importances_")

            importances_array = rf_model.feature_importances_

            if len(feature_names) != len(importances_array):
                st.warning(
                    "Feature importance values were found, but their count does not match feature_names.pkl. "
                    "This usually happens when feature_names.pkl contains original columns instead of transformed one-hot encoded columns."
                )
            else:
                importances = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance": importances_array,
                    }
                ).sort_values("Importance", ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(
                    data=importances,
                    x="Importance",
                    y="Feature",
                    palette="rocket",
                    hue="Feature",
                    legend=False,
                    ax=ax,
                    # color="#2563eb",
                )
                st.pyplot(fig)

        except Exception as e:
            st.warning(f"Feature importance could not be displayed: {e}")

        st.subheader("Observations")
        st.write(
            """
Random Forest is expected to perform strongly because ensemble models can capture non-linear relationships and reduce the instability of a single decision tree.

Logistic Regression provides a useful baseline and performs well when the transformed features show a mostly linear relationship with the target.

Naive Bayes can be faster and simpler, but its independence assumption may limit performance on a dataset where demographic and employment attributes are related.

The final model choice should consider accuracy, AUC, F1 score and MCC together rather than accuracy alone.
            """
        )

# ----------------------------------------------------------------------------
# SIDEBAR: INFORMATION PANEL
# ----------------------------------------------------------------------------

with st.sidebar:
    st.title("Information Panel")

    st.subheader("Project Overview")
    st.markdown(
        """
**Problem Type**  
Binary classification

**Dataset**  
Adult Census Income

**Target Variable**  
income

**Classes**  
<=50K and >50K
        """
    )

    st.markdown("---")
    st.subheader("Dataset Status")

    if active_df is not None:
        if active_source == "Live Uploaded Dataset":
            st.success(active_source)
        elif active_source == "Preloaded Repository Dataset":
            st.info(active_source)
        else:
            st.warning(active_source)

        st.metric("Rows", f"{len(active_df):,}")
        st.metric("Columns", len(active_df.columns))
        st.metric("Missing Values", int(active_df.isna().sum().sum()))

        if TARGET in active_df.columns:
            st.markdown("**Target Distribution**")
            target_counts = active_df[TARGET].astype(str).str.strip().value_counts()
            for label, count in target_counts.items():
                label = label.replace(">", "&gt;").replace("<", "&lt;")
                st.write(f"{label}: {count}")
    else:
        st.warning("No active dataset loaded")

    st.markdown("---")
    st.subheader("Feature Selection")
    st.markdown("**Numeric Features**")
    for feature in NUMERIC_FEATURES:
        st.write(f"- {feature}")

    st.markdown("**Categorical Features**")
    for feature in CATEGORICAL_FEATURES:
        st.write(f"- {feature}")

    st.markdown("---")
    st.subheader("Preprocessing Summary")
    st.markdown(
        """
- Missing values represented by ? are handled
- Numerical features are scaled
- Categorical features are one-hot encoded
- Target labels are encoded as 0 and 1
- Models are evaluated using a held-out test split
        """
    )

    st.markdown("---")
    st.subheader("Selected Model Summary")
    st.write(selected_model_for_sidebar)
    st.info(MODEL_DESCRIPTIONS.get(selected_model_for_sidebar, "Model description is not available."))

    if metrics_json is not None and selected_model_for_sidebar in metrics_json:
        selected_sidebar_metrics = metrics_json[selected_model_for_sidebar]
        st.metric("Accuracy", safe_metric_value(selected_sidebar_metrics, "Accuracy"))
        st.metric("AUC", safe_metric_value(selected_sidebar_metrics, "AUC"))
        st.metric("F1", safe_metric_value(selected_sidebar_metrics, "F1"))
        st.metric("MCC", safe_metric_value(selected_sidebar_metrics, "MCC"))

    if comparison_df_global is not None and "Accuracy" in comparison_df_global.columns:
        st.markdown("---")
        st.subheader("Overall Best Model")
        best_model = comparison_df_global["Accuracy"].idxmax()
        best_score = comparison_df_global["Accuracy"].max()
        st.success(f"{best_model}: {best_score:.4f}")

    st.markdown("---")
    st.subheader("Technology Stack")
    st.markdown(
        """
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib
        """
    )

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------

st.markdown("---")
st.caption("Adult Census Income dataset | Income Classification using Machine Learning Models · Logistic Regression · Decision Tree · kNN · Naive Bayes · Random Forest")
