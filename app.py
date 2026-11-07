
"""Streamlit app for comparing five breast cancer classifiers."""

from pathlib import Path
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from train_models import MODEL_FILES



# 1. Page configuration


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Breast Cancer Classifier Comparison",
    layout="wide"
)

st.title("Breast Cancer Wisconsin — Five Classifiers")

st.caption(
    "0 = Malignant · 1 = Benign | "
    "Same dataset and stratified 80/20 split for all models"
)

st.info(
    "Educational demonstration only. "
    "These predictions are not medical advice "
    "or a clinical diagnostic tool."
)


# 2. Load trained models and test data

@st.cache_resource
def load_models():
    return {
        name: joblib.load(ROOT / "model" / filename)
        for name, filename in MODEL_FILES.items()
    }


@st.cache_data
def load_test_data():
    return pd.read_csv(ROOT / "test_data.csv")


try:
    models = load_models()
    reference = load_test_data()

except FileNotFoundError:
    st.error(
        "Missing model files or test_data.csv. "
        "Run `python train_models.py` first."
    )
    st.stop()


features = [
    col for col in reference.columns
    if col != "target"
]


# 3. Sidebar controls with GO button

st.sidebar.header("Controls")

with st.sidebar.form("classifier_form"):

    selected = st.selectbox(
        "Select classifier",
        list(models.keys())
    )

    upload = st.file_uploader(
        "Optional: upload CSV for predictions",
        type=["csv"]
    )

    go_button = st.form_submit_button(
        "GO",
        type="primary",
        use_container_width=True
    )



if go_button:

    st.session_state["show_results"] = True

    st.session_state["selected_model"] = selected

    if upload is not None:
        st.session_state["uploaded_data"] = upload.getvalue()
    else:
        st.session_state["uploaded_data"] = None


if not st.session_state.get("show_results", False):

    st.info(
        "Please select a classifier and click GO "
        "to view the evaluation results and predictions."
    )

    st.stop()



selected = st.session_state["selected_model"]

if st.session_state.get("uploaded_data") is not None:

    upload = BytesIO(
        st.session_state["uploaded_data"]
    )

else:
    upload = None


st.success(
    f"Showing results for: {selected}"
)


# 4. Evaluation metrics for all five models

st.subheader("Held-out test-set performance")

X_test = reference[features]
y_test = reference["target"]

rows = []

for name, model in models.items():

    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)[
        :, list(model.classes_).index(1)
    ]

    rows.append({
        "ML Model Name": name,

        "Accuracy": accuracy_score(
            y_test, pred
        ),

        "AUC": roc_auc_score(
            y_test, prob
        ),

        "Precision": precision_score(
            y_test, pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test, pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_test, pred,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test, pred
        )
    })


results = pd.DataFrame(rows).set_index(
    "ML Model Name"
)

st.dataframe(
    results.style
    .format("{:.4f}")
    .highlight_max(axis=0),
    use_container_width=True
)

st.caption(
    "Metrics use positive class 1 (benign); "
    "AUC uses the probability of class 1."
)


# 5. Selected classifier

model = models[selected]

pred = model.predict(X_test)


# 6. Confusion matrix and ROC curves

left, right = st.columns(2)


# Confusion matrix
with left:

    st.subheader(
        f"Confusion matrix — {selected}"
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        confusion_matrix(
            y_test,
            pred,
            labels=[0, 1]
        ),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Malignant (0)",
            "Benign (1)"
        ],
        yticklabels=[
            "Malignant (0)",
            "Benign (1)"
        ],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    plt.close(fig)


# ROC curves for all five models
with right:

    st.subheader(
        "ROC curves — all models"
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    for name, fitted in models.items():

        p = fitted.predict_proba(X_test)[
            :, list(fitted.classes_).index(1)
        ]

        fpr, tpr, _ = roc_curve(
            y_test, p
        )

        auc = roc_auc_score(
            y_test, p
        )

        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.3f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        alpha=0.5
    )

    ax.set_xlabel(
        "False positive rate"
    )

    ax.set_ylabel(
        "True positive rate"
    )

    ax.legend(fontsize=8)

    st.pyplot(fig)

    plt.close(fig)


# 7. Predictions

st.subheader("Predictions")


if upload is None:

    st.write(
        "Previewing predictions on the included test set. "
        "Upload a CSV with the same 30 feature columns "
        "to predict new samples."
    )

    incoming = reference.copy()

else:

    try:

        incoming = pd.read_csv(upload)

    except Exception as exc:

        st.error(
            f"Could not read CSV: {exc}"
        )

        st.stop()



# 8. Validate uploaded data

missing = [
    col for col in features
    if col not in incoming.columns
]

if missing:

    st.error(
        "Missing required feature columns: "
        + ", ".join(missing)
    )

else:

    try:

        input_features = incoming[features].apply(
            pd.to_numeric,
            errors="raise"
        )

        if (
            input_features.isna().any().any()
            or not np.isfinite(
                input_features.to_numpy()
            ).all()
        ):

            raise ValueError(
                "Feature values must be finite "
                "and non-missing."
            )


        # 9. Generate predictions

        predictions = model.predict(
            input_features
        )

        probabilities = model.predict_proba(
            input_features
        )[
            :, list(model.classes_).index(1)
        ]


        output = incoming.copy()

        output["predicted_target"] = predictions

        output["predicted_diagnosis"] = np.where(
            predictions == 1,
            "Benign",
            "Malignant"
        )

        output["probability_benign"] = probabilities


        st.dataframe(
            output.head(100),
            use_container_width=True
        )


        # 10. Download prediction results

        st.download_button(
            label="Download predictions CSV",
            data=output.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )


        if "target" in incoming.columns:

            st.caption(
                "The uploaded target column is shown "
                "for reference, but predictions never use it."
            )


    except (ValueError, TypeError) as exc:

        st.error(
            f"Invalid input data: {exc}"
        )
