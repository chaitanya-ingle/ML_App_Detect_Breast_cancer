# Breast Cancer Wisconsin: Five-Model Classification App

A Streamlit application derived from `ML_Assignment_2.ipynb`, comparing **Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine** on the **same Breast Cancer Wisconsin (Diagnostic)** dataset (569 samples, 30 numeric features).

**Target:** `0 = malignant`, `1 = benign`. This is an educational ML project, **not** a medical diagnostic application.

## Project layout

```text
ML_Classification_Streamlit/
├── app.py                      # Streamlit UI, predictions, plots and metrics
├── train_models.py             # Reproducible training/evaluation source code
├── ML_Assignment_2.ipynb       # Original notebook (retained for submission)
├── requirements.txt
├── README.md
├── test_data.csv               # 114 held-out examples; 30 features + target
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── evaluation_metrics.csv
```

## Setup and run

From the project directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The **five trained models and test_data.csv are already included**. Retrain at any time:

```bash
python train_models.py
```

## Model configurations

- Logistic Regression: `StandardScaler` + `LogisticRegression(max_iter=1000, random_state=42)`
- Decision Tree: `DecisionTreeClassifier(max_depth=5, random_state=42)`
- KNN: `StandardScaler` + `KNeighborsClassifier(n_neighbors=7)`
- Gaussian Naive Bayes: `GaussianNB()`
- SVM: `StandardScaler` + `SVC(kernel='rbf', probability=True, random_state=42)`

**Preprocessing artifacts:** The `StandardScaler` is fitted inside each relevant saved sklearn pipeline, so no separate scaler file is needed. Tree and Naive Bayes use raw numeric features.

**Dataset:** The original notebook loads UCI Breast Cancer Wisconsin (Diagnostic), dataset ID 17, via `ucimlrepo`. The reproducible training script uses scikit-learn's bundled copy of **the same underlying dataset** to work offline. The feature headers in `test_data.csv` follow scikit-learn's feature naming convention, rather than the original UCI header names. The notebook is retained unchanged.

## Evaluation

An 80/20 stratified split with `random_state=42` is used for all models. All six test metrics are shown in the app and saved to `model/evaluation_metrics.csv`:

1. Accuracy
2. ROC AUC (probability of class `1`)
3. Precision (positive class `1`)
4. Recall (positive class `1`)
5. F1 (positive class `1`)
6. Matthews Correlation Coefficient (MCC)

The app also shows a confusion matrix, ROC curves, and a model comparison table.

## Uploading your own test data

Upload a CSV containing all 30 columns from `test_data.csv` (excluding the optional `target` column). The app reorders columns to match training order, validates numeric inputs, and downloads predictions including `predicted_target`, `predicted_diagnosis`, and `probability_benign`.

**Security note:** Only load `.joblib` model files from trusted sources; joblib deserialization can execute arbitrary code. Use a compatible scikit-learn version for inference.


## Updated official UCI data source

`train_models.py` now fetches Breast Cancer Wisconsin (Diagnostic) directly from the UCI Machine Learning Repository using `ucimlrepo.fetch_ucirepo(id=17)`. UCI diagnosis labels are mapped as `M → 0` and `B → 1`. The script regenerates `test_data.csv`, all five saved model pipelines, and `model/evaluation_metrics.csv`. **Run `python train_models.py` after installing requirements with an internet connection.** The previously packaged models/test CSV are from the equivalent scikit-learn dataset and have **not** been retrained from the UCI API in this offline environment. The app does not require network access after retraining.

UCI source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
