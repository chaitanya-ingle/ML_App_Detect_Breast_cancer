# Breast Cancer Wisconsin — Classification Model Comparison

## a. Problem Statement

The objective of this project is to implement and compare five supervised machine learning classification algorithms on the same dataset to evaluate their effectiveness in binary classification.

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** to classify breast tumour samples as malignant or benign based on numerical features extracted from digitized images of cell nuclei.

The five classification algorithms implemented are Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM).

Each model is evaluated using six performance metrics: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit application is developed to compare model performance, visualize confusion matrices and ROC curves, and generate predictions using uploaded test data.

*This project is intended solely for educational purposes and is not a clinical diagnostic tool.*

## b. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The dataset contains **569 observations and 30 numerical input features** derived from digitized images of breast cell nuclei. These features describe characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, and symmetry.

| Attribute | Description |
|---|---|
| Dataset type | Binary classification |
| Total observations | 569 |
| Number of input features | 30 |
| Target variable | Diagnosis |
| Malignant samples | 212 |
| Benign samples | 357 |
| Training samples (80%) | 455 |
| Testing samples (20%) | 114 |
| Missing values | None |

**Target Encoding:**
- `0` — Malignant (M)
- `1` — Benign (B)

The dataset is retrieved using the Python `ucimlrepo` library with `fetch_ucirepo(id=17)`.

An **80:20 stratified train-test split** with `random_state=42` is used to ensure consistent evaluation across all five classifiers.

## c. GitHub Repository Link

**GitHub Repository:** https://github.com/chaitanya-ingle/ML_App_Detect_Breast_cancer

The repository contains the Streamlit application (`app.py`), model training source code (`train_models.py`), `requirements.txt`, `README.md`, test dataset (`test_data.csv`), and five saved classification models in `.joblib` format, along with the evaluation metrics CSV file.

## d. Models Used and Evaluation Metrics

Five classification algorithms were implemented and evaluated on the same held-out test dataset.

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.9825** | **0.9954** | **0.9861** | 0.9861 | **0.9861** | **0.9623** |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9737 | 0.9884 | 0.9600 | **1.0000** | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | **0.9825** | 0.9950 | **0.9861** | 0.9861 | **0.9861** | **0.9623** |

**Evaluation Details:** Precision, Recall, F1 Score, and AUC are calculated considering benign (`1`) as the positive class. The results shown are from the previously generated 114-sample test set and should be verified after retraining from the UCI source.

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Achieved the joint-highest accuracy of 98.25%, with the highest AUC (0.9954) and an MCC of 0.9623. It demonstrated excellent classification performance with a relatively simple and interpretable model. |
| **Decision Tree** | Recorded the lowest accuracy (92.11%) and MCC (0.8341). Although the model is easy to interpret, its performance was comparatively weaker, suggesting that the chosen tree configuration did not generalize as effectively on the test data. |
| **KNN** | Achieved 97.37% accuracy and perfect recall (1.0000) for the benign class. Its strong performance demonstrates the effectiveness of distance-based classification with standardized features, although its precision was lower than Logistic Regression and SVM. |
| **Naive Bayes** | Achieved 93.86% accuracy with a high AUC of 0.9878. Despite the assumption of conditional feature independence, the model performed reasonably well, although its accuracy and MCC were below the leading classifiers. |
| **Support Vector Machine** | Achieved 98.25% accuracy, matching Logistic Regression in Accuracy, Precision, Recall, F1, and MCC. Its RBF kernel performed effectively on the standardized feature space, with an AUC of 0.9950. |
| **Overall Winner for the Dataset** | **Logistic Regression**, narrowly preferred over SVM. Both models achieved identical Accuracy (98.25%), F1 (0.9861), and MCC (0.9623). Logistic Regression recorded a marginally higher AUC (0.9954 versus 0.9950) while offering a simpler and more interpretable classification approach. |

### Conclusion

Among the five classification algorithms, **Logistic Regression and Support Vector Machine demonstrated the strongest overall performance** on the Breast Cancer Wisconsin dataset.

Logistic Regression was selected as the preferred model based on its combination of high accuracy, highest AUC, strong MCC, and model simplicity.

The experiment demonstrates that different machine learning algorithms can achieve varying classification performance on the same dataset, emphasizing the importance of evaluating multiple models using consistent performance metrics.
