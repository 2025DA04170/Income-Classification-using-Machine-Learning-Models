import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(
    page_title="Income Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------------------------------
# TITLE
# ----------------------------------------------------------

st.title("📊 Income Classification using Machine Learning Models")

st.markdown("""
This application demonstrates and compares multiple machine learning
classification models trained on the Adult Income Dataset.
""")

# ----------------------------------------------------------
# LOAD METRICS
# ----------------------------------------------------------

with open("metrics.json", "r") as f:
    metrics = json.load(f)

# ----------------------------------------------------------
# MODEL PATHS
# ----------------------------------------------------------

MODELS = {
    "Logistic Regression": "model/logistic_regression_model.pkl",
    "Decision Tree": "model/decision_tree_model.pkl",
    "KNN": "model/knn_model.pkl",
    "Naive Bayes": "model/naive_bayes_model.pkl",
    "Random Forest": "model/random_forest_model.pkl"
}

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.title("⚙️ Configuration")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODELS.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

# ----------------------------------------------------------
# TABS
# ----------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "📈 Metrics",
        "🚀 Predictions",
        "🏆 Model Comparison"
    ]
)

# ==========================================================
# TAB 1 - METRICS
# ==========================================================

with tab1:

    st.header("Model Evaluation Metrics")

    comparison_df = pd.DataFrame(metrics).T

    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]

    st.dataframe(
        comparison_df[metric_columns].style.format("{:.4f}"),
        use_container_width=True
    )

    st.subheader(f"Selected Model : {selected_model}")

    selected_metrics = metrics[selected_model]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            f"{selected_metrics['Accuracy']:.4f}"
        )

        st.metric(
            "Precision",
            f"{selected_metrics['Precision']:.4f}"
        )

    with col2:
        st.metric(
            "Recall",
            f"{selected_metrics['Recall']:.4f}"
        )

        st.metric(
            "F1 Score",
            f"{selected_metrics['F1']:.4f}"
        )

    with col3:
        st.metric(
            "AUC Score",
            f"{selected_metrics['AUC']:.4f}"
        )

        st.metric(
            "MCC Score",
            f"{selected_metrics['MCC']:.4f}"
        )

    # ------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = np.array(
        selected_metrics["ConfusionMatrix"]
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # ------------------------------------------------------
    # ROC CURVE
    # ------------------------------------------------------

    st.subheader("ROC Curve")

    if (
        "FPR" in selected_metrics
        and "TPR" in selected_metrics
    ):

        roc_fig, roc_ax = plt.subplots(
            figsize=(7, 5)
        )

        roc_ax.plot(
            selected_metrics["FPR"],
            selected_metrics["TPR"],
            linewidth=2,
            label=f"AUC = {selected_metrics['AUC']:.3f}"
        )

        roc_ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="red"
        )

        roc_ax.set_xlabel(
            "False Positive Rate"
        )

        roc_ax.set_ylabel(
            "True Positive Rate"
        )

        roc_ax.legend()

        st.pyplot(roc_fig)

    # ------------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------------

    st.subheader("Classification Report")

    if "ClassificationReport" in selected_metrics:

        report_df = pd.DataFrame(
            selected_metrics["ClassificationReport"]
        ).T

        st.dataframe(
            report_df,
            use_container_width=True
        )

# ==========================================================
# TAB 2 - PREDICTIONS
# ==========================================================

with tab2:

    st.header("Prediction on Uploaded Dataset")

    if uploaded_file is not None:

        test_df = pd.read_csv(
            uploaded_file
        )

        st.subheader("Uploaded Dataset")

        st.dataframe(
            test_df.head(10)
        )

        try:

            model = joblib.load(
                MODELS[selected_model]
            )

            predictions = model.predict(
                test_df
            )

            result_df = test_df.copy()

            result_df["Prediction"] = predictions

            result_df["Prediction"] = (
                result_df["Prediction"]
                .replace(
                    {
                        0: "<=50K",
                        1: ">50K"
                    }
                )
            )

            st.subheader(
                "Prediction Results"
            )

            st.dataframe(
                result_df
            )

            csv = result_df.to_csv(
                index=False
            )

            st.download_button(
                label="⬇ Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

    else:

        st.info(
            "Upload test_data.csv to generate predictions."
        )

# ==========================================================
# TAB 3 - MODEL COMPARISON
# ==========================================================

with tab3:

    st.header(
        "Overall Model Performance Comparison"
    )

    comparison_df = pd.DataFrame(metrics).T

    st.dataframe(
        comparison_df[metric_columns],
        use_container_width=True
    )

    # ------------------------------------------------------
    # WINNER MODEL
    # ------------------------------------------------------

    winner_model = (
        comparison_df["Accuracy"]
        .idxmax()
    )

    winner_accuracy = (
        comparison_df["Accuracy"]
        .max()
    )

    st.success(
        f"🏆 Best Model : {winner_model}"
    )

    st.metric(
        "Highest Accuracy",
        f"{winner_accuracy:.4f}"
    )

    # ------------------------------------------------------
    # ACCURACY CHART
    # ------------------------------------------------------

    st.subheader(
        "Accuracy Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.barplot(
        x=comparison_df.index,
        y=comparison_df["Accuracy"],
        palette="viridis",
        ax=ax
    )

    plt.xticks(rotation=20)

    st.pyplot(fig)

    # ------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------

    st.subheader(
        "Random Forest Feature Importance"
    )

    try:

        rf_model = joblib.load(
            "model/random_forest_model.pkl"
        )

        feature_names = joblib.load(
            "model/feature_names.pkl"
        )

        importances = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": rf_model.feature_importances_
            }
        )

        importances = (
            importances
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(10)
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.barplot(
            data=importances,
            x="Importance",
            y="Feature",
            palette="rocket",
            ax=ax
        )

        st.pyplot(fig)

    except Exception:

        st.warning(
            "feature_names.pkl or Random Forest model not found."
        )

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------

st.markdown("---")

st.caption(
    "BITS Pilani WILP | Machine Learning Assignment 2"
)