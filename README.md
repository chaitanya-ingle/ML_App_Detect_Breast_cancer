# Breast Cancer Wisconsin — ML Classification & Model Comparison

### An Interactive Machine Learning Application Built with Python and Streamlit

This project implements and compares **five supervised machine learning classification algorithms** using the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the UCI Machine Learning Repository.

The application provides an interactive interface built with Streamlit, allowing users to select classification models, evaluate their performance, visualize results, and generate predictions from uploaded CSV files.

The objective is to evaluate multiple classification techniques on the same dataset using consistent preprocessing, training, and testing procedures.

**GitHub Repository:** [ML_App_Detect_Breast_cancer](https://github.com/chaitanya-ingle/ML_App_Detect_Breast_cancer)

> **Disclaimer:** This application is developed for educational and academic purposes only. It is not intended for medical diagnosis or clinical decision-making.

---

## 1. Project Overview

The project evaluates five classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five models are trained and evaluated on the same dataset using an **80:20 stratified train-test split** with `random_state=42`.

Each model is evaluated using six performance metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The application also provides a comparative performance table, confusion matrices, ROC curves, and CSV-based prediction functionality.

## 2. Dataset Information

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository

**Dataset ID:** 17

**Dataset URL:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

| Attribute | Description |
|---|---|
| Total observations | 569 |
| Input features | 30 numerical features |
| Target variable | Diagnosis |
| Classification type | Binary classification |
| Training samples | 455 (80%) |
| Testing samples | 114 (20%) |
| Missing values | None |

### Target Encoding

| Encoded Value | Diagnosis |
|---|---|
| 0 | Malignant (M) |
| 1 | Benign (B) |

The dataset contains measurements derived from digitized images of breast cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, and symmetry.

### Dataset Retrieval

The training script retrieves the dataset directly from the UCI Machine Learning Repository using the `ucimlrepo` package.

```python
from ucimlrepo import fetch_ucirepo

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

y = y.iloc[:, 0].map({
    "M": 0,
    "B": 1
})
```

The dataset is downloaded during model training. Once trained, the application uses locally saved model files and test data without requiring a new dataset download for every prediction.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | Interactive web application |
| Scikit-learn | Model training and evaluation |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |
| Joblib | Model serialization |
| UCI ML Repository | Dataset source |

## 4. Project Structure

```text
ML_App_Detect_Breast_cancer/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── evaluation_metrics.csv
```

### File Descriptions

| File | Description |
|---|---|
| `app.py` | Streamlit user interface, visualizations and predictions |
| `train_models.py` | Dataset retrieval, preprocessing, training and evaluation |
| `requirements.txt` | Required Python dependencies |
| `README.md` | Project documentation |
| `test_data.csv` | Held-out test dataset |
| `model/*.joblib` | Five trained classification models |
| `evaluation_metrics.csv` | Performance metrics for all five classifiers |

The original `ML_Assignment_2.ipynb` notebook may also be included for academic reference.

## 5. Machine Learning Models

### Logistic Regression

A linear classification algorithm that estimates class probabilities using a logistic function.

**Configuration:**
- StandardScaler for feature normalization
- `LogisticRegression(max_iter=1000, random_state=42)`

### Decision Tree Classifier

A tree-based classification algorithm that recursively divides observations based on feature thresholds.

**Configuration:**
- `DecisionTreeClassifier(max_depth=5, random_state=42)`
- No feature scaling required

### K-Nearest Neighbors

An instance-based learning algorithm that classifies observations using nearby training examples.

**Configuration:**
- StandardScaler for feature normalization
- `KNeighborsClassifier(n_neighbors=7)`

### Gaussian Naive Bayes

A probabilistic classification algorithm based on Bayes' theorem and the assumption of conditional feature independence.

**Configuration:**
- `GaussianNB()`
- Default model parameters

### Support Vector Machine

A classification algorithm that identifies a decision boundary between classes, with an RBF kernel supporting nonlinear decision boundaries.

**Configuration:**
- StandardScaler for feature normalization
- `SVC(kernel="rbf", probability=True, random_state=42)`

### Preprocessing

Feature scaling is applied to Logistic Regression, KNN, and SVM through scikit-learn pipelines.

The preprocessing steps are saved within the corresponding `.joblib` model files, eliminating the need for separate scaler files.

## 6. Model Evaluation Metrics

The following six metrics are calculated on the held-out test dataset.

| Metric | Purpose |
|---|---|
| Accuracy | Proportion of correctly classified observations |
| AUC | Ability to distinguish between classes across classification thresholds |
| Precision | Proportion of predicted positive cases that are correct |
| Recall | Proportion of actual positive cases correctly identified |
| F1 Score | Harmonic mean of Precision and Recall |
| MCC | Balanced measure of classification quality considering all confusion matrix outcomes |

**Evaluation convention:** Benign (`1`) is treated as the positive class for Precision, Recall, F1, and AUC calculations.

## 7. Model Performance Comparison

The following results were obtained from the previously generated held-out test set containing 114 observations.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.9825** | **0.9954** | **0.9861** | 0.9861 | **0.9861** | **0.9623** |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9737 | 0.9884 | 0.9600 | **1.0000** | 0.9796 | 0.9442 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | **0.9825** | 0.9950 | **0.9861** | 0.9861 | **0.9861** | **0.9623** |

**Note:** These results correspond to the previously generated model artifacts. Metrics should be regenerated and verified after retraining using the UCI dataset retrieval script.

## 8. Model Performance Analysis

### Logistic Regression

Logistic Regression achieved one of the highest classification accuracies at **98.25%**, with an AUC of **0.9954**.

It demonstrated strong classification performance while maintaining a relatively simple and interpretable model structure.

### Decision Tree

The Decision Tree achieved **92.11% accuracy**, the lowest among the five models.

Although decision trees are easy to interpret, the model produced more classification errors and showed weaker generalization performance on this test set.

### K-Nearest Neighbors

KNN achieved **97.37% accuracy** and a recall of **100%** for the benign class.

Its performance demonstrates the effectiveness of distance-based classification when numerical features are appropriately standardized.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved **93.86% accuracy** and an AUC of **0.9878**.

Despite its simplifying feature independence assumption, the algorithm demonstrated competitive probability-ranking performance.

### Support Vector Machine

SVM achieved **98.25% accuracy**, matching Logistic Regression on Accuracy, Precision, Recall, F1, and MCC.

The RBF kernel performed effectively on the standardized numerical feature space.

### Overall Findings

**Logistic Regression and Support Vector Machine were the strongest-performing classifiers on this test set.**

Both achieved an accuracy of 98.25%, while Logistic Regression recorded a marginally higher AUC.

Logistic Regression is a reasonable preferred model for this experiment due to its strong performance, simplicity, and interpretability.

However, the small performance difference between the leading models should not be interpreted as conclusive evidence of superiority across other datasets.

## 9. Streamlit Application Features

The application provides an interactive interface with the following capabilities.

### Classifier Selection

Users can select any of the five trained classifiers from the sidebar dropdown.

### GO Button

After selecting a classifier, users click the **GO** button to display results.

This allows users to configure their selection before running predictions and viewing model evaluation outputs.

### Model Comparison

The application displays a comparison table containing all six evaluation metrics for the five classifiers.

The highest values in each metric are highlighted for easier comparison.

### Confusion Matrix

A confusion matrix is generated for the selected classifier to visualize correct and incorrect predictions.

### ROC Curves

ROC curves for all five models are displayed together, allowing comparison of their classification performance across thresholds.

### CSV Upload and Predictions

Users can upload CSV files containing the required 30 input features and generate predictions using the selected model.

### Download Predictions

Predictions can be exported as a CSV file containing:

- `predicted_target`
- `predicted_diagnosis`
- `probability_benign`

The application uses previously trained models for inference and does not retrain on uploaded data.

## 10. Installation and Local Execution

### Prerequisites

- Python
- pip
- Internet connection for the initial UCI dataset retrieval
- Required packages listed in `requirements.txt`

### Clone the Repository

```bash
git clone https://github.com/chaitanya-ingle/ML_App_Detect_Breast_cancer.git
cd ML_App_Detect_Breast_cancer
```

### Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train or Regenerate Models

```bash
python train_models.py
```

This step downloads the dataset from UCI and regenerates the trained model files, evaluation metrics, and test data.

It is necessary when reproducing the training pipeline or replacing previously saved model artifacts.

### Launch Streamlit

```bash
streamlit run app.py
```

Streamlit will start a local server and open the application in your browser.

If compatible trained model artifacts are already present, the training step can be skipped for normal application usage.

## 11. Using the Application

1. Launch the Streamlit application.
2. Select a classification algorithm from the sidebar.
3. Optionally upload a CSV containing the required 30 features.
4. Click **GO** to display results.
5. Review the model comparison table, confusion matrix, and ROC curves.
6. Inspect the generated predictions.
7. Download prediction results if required.

### CSV Input Requirements

Uploaded CSV files must contain the same 30 feature columns expected by the trained model.

The optional `target` column may be included for reference but is not used for generating predictions.

All feature values must be numeric, finite, and non-missing.

## 12. Deployment

The application is designed for deployment using **Streamlit Community Cloud**.

Deployment procedure:

1. Upload the project files to a GitHub repository.
2. Ensure `app.py` and `requirements.txt` are in the repository root.
3. Keep the five trained `.joblib` files inside the `model/` directory.
4. Open [Streamlit Community Cloud](https://share.streamlit.io/).
5. Connect the GitHub repository.
6. Select `app.py` as the application entry point.
7. Deploy the application.

Once deployed, users can access the application through a public Streamlit URL.

## 13. Reproducibility and Limitations

- All five classifiers use the same training and testing split.
- A stratified 80:20 split is used with `random_state=42`.
- Feature scaling is applied within the relevant model pipelines.
- Models are evaluated on held-out test data.
- The reported metrics are specific to the chosen dataset, split, and hyperparameters.
- The UCI source requires an internet connection during data retrieval.
- Model artifacts should be regenerated after changes to dataset loading or preprocessing.
- Saved `.joblib` files should only be loaded from trusted sources.
- Compatible scikit-learn versions should be used for training and inference.
- This application is not validated for clinical use.

## 14. Conclusion

This project demonstrates the implementation, evaluation, and deployment of five supervised machine learning classification algorithms on a common dataset.

The results highlight differences in classification performance across linear, tree-based, distance-based, probabilistic, and kernel-based learning approaches.

Logistic Regression and SVM achieved the strongest overall performance in the recorded experiment, while KNN demonstrated perfect recall for the benign class.

By integrating trained models with Streamlit, the project provides an accessible platform for model comparison, visualization, and prediction.

---

**Project Type:** Supervised Machine Learning — Binary Classification

**Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI Repository

**Framework:** Streamlit

**Source Code:** [GitHub Repository](https://github.com/chaitanya-ingle/ML_App_Detect_Breast_cancer)

**Purpose:** Academic learning, model evaluation, and interactive machine learning demonstration.
