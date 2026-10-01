import json
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
from src.features.engineering import add_engineered_features
from src.models.config import (
    FINAL_MODEL_NAME,
    FINAL_MODEL_THRESHOLD,
)
from src.utils.config import (
    FINAL_EVALUATION_REPORT_PATH,
    FINAL_MODEL_PATH,
)


def load_model(
    model_path: Path,
) -> Pipeline:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run 'python -m src.models.train' first."
        )

    return joblib.load(
        model_path
    )


def evaluate_model(
    model: Pipeline,
    X_test,
    y_test,
    threshold: float,
) -> dict:
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print(
        "\n=== FINAL TEST PERFORMANCE ==="
    )

    print(
        f"Model: {FINAL_MODEL_NAME}"
    )

    print(
        f"Threshold: {threshold:.2f}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    print(
        "\n=== CONFUSION MATRIX ==="
    )

    print(
        matrix
    )

    print(
        "\n=== CLASSIFICATION REPORT ==="
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return {
        "model": FINAL_MODEL_NAME,
        "threshold": threshold,
        "test_samples": int(
            len(y_test)
        ),
        "positive_samples": int(
            y_test.sum()
        ),
        "metrics": {
            "accuracy": float(
                accuracy
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1_score": float(
                f1
            ),
            "roc_auc": float(
                roc_auc
            ),
            "pr_auc": float(
                pr_auc
            ),
        },
        "confusion_matrix": {
            "true_negative": int(
                matrix[0, 0]
            ),
            "false_positive": int(
                matrix[0, 1]
            ),
            "false_negative": int(
                matrix[1, 0]
            ),
            "true_positive": int(
                matrix[1, 1]
            ),
        },
    }


def save_evaluation_report(
    report: dict,
) -> None:
    FINAL_EVALUATION_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with FINAL_EVALUATION_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        "\nEvaluation report saved: "
        f"{FINAL_EVALUATION_REPORT_PATH}"
    )


def main() -> None:
    df = load_data()

    X, y = prepare_features(
        df
    )

    X = add_engineered_features(
        X
    )

    (
        _,
        _,
        X_test,
        _,
        _,
        y_test,
    ) = split_data(
        X,
        y,
    )

    model = load_model(
        FINAL_MODEL_PATH
    )

    report = evaluate_model(
        model,
        X_test,
        y_test,
        FINAL_MODEL_THRESHOLD,
    )

    save_evaluation_report(
        report
    )


if __name__ == "__main__":
    main()