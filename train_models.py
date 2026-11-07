"""Train five classifiers using UCI Breast Cancer Wisconsin (Diagnostic), dataset ID 17.
Requires internet access on training runs to fetch the dataset via ucimlrepo.
"""
from pathlib import Path
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Support Vector Machine': 'svm.joblib',
}

def evaluate(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return {
        'Accuracy': accuracy_score(y, predictions),
        'AUC': roc_auc_score(y, probabilities),
        'Precision': precision_score(y, predictions, zero_division=0),
        'Recall': recall_score(y, predictions, zero_division=0),
        'F1': f1_score(y, predictions, zero_division=0),
        'MCC': matthews_corrcoef(y, predictions),
    }

def main():
    # Official UCI Machine Learning Repository dataset (ID 17).
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    X = breast_cancer_wisconsin_diagnostic.data.features.copy()
    raw_y = breast_cancer_wisconsin_diagnostic.data.targets.iloc[:, 0]
    # UCI diagnosis is M (malignant) or B (benign); match app convention.
    y = raw_y.map({'M': 0, 'B': 1}).rename('target')
    if y.isna().any():
        raise ValueError(f'Unexpected UCI diagnosis labels: {raw_y.unique().tolist()}')
    if X.isna().any().any():
        raise ValueError('UCI features contain missing values; inspect the source dataset.')
    print('Dataset source: UCI ML Repository, ID 17')
    print(f'Samples: {len(X)} | Features: {X.shape[1]}')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    models = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)),
    }
    (ROOT / 'model').mkdir(exist_ok=True)
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, ROOT / 'model' / MODEL_FILES[name])
        rows.append({'ML Model Name': name, **evaluate(model, X_test, y_test)})
    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / 'test_data.csv', index=False)
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / 'model' / 'evaluation_metrics.csv', index=False)
    print(results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'Exported {len(models)} models and {len(X_test)} test samples.')

if __name__ == '__main__':
    main()
