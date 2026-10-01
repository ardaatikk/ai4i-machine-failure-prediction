from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data.analysis import load_data
from src.data.preprocessing import prepare_features, split_data
from src.models.train import MODEL_PATH
from src.models.config import BASELINE_THRESHOLD

def load_model(model_path: Path) -> Pipeline:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run 'python -m src.models.train' first."
        )

    return joblib.load(model_path)


def evaluate_model(
    model: Pipeline,
    X_test,
    y_test,
) -> None:
    
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= BASELINE_THRESHOLD
    ).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"Threshold: {BASELINE_THRESHOLD:.2f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, predictions))

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, predictions))


def main() -> None:
    df = load_data()

    X, y = prepare_features(df)
    (
        _,
        _,
        X_test,
        _,
        _,
        y_test,
    ) = split_data(X, y)

    model = load_model(MODEL_PATH)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()