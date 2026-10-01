import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.data.analysis import load_data
from src.data.preprocessing import prepare_features, split_data
from src.models.evaluate import load_model
from src.models.train import MODEL_PATH
from src.utils.config import FIGURES_DIR


THRESHOLDS = np.arange(0.75, 0.901, 0.01)

FIGURE_FILENAME = "threshold_analysis.png"
FIGURE_PATH = FIGURES_DIR / FIGURE_FILENAME


def analyze_thresholds(
    model: Pipeline,
    X_validation,
    y_validation,
) -> list[dict]:
    probabilities = model.predict_proba(X_validation)[:, 1]

    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )
        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )
        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return results

def print_results(results: list[dict]) -> None:
    print("\n=== THRESHOLD ANALYSIS ===")
    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    for result in results:
        print(
            f"{result['threshold']:<12.2f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
        )
        
def find_best_threshold(results: list[dict]) -> dict:
    return max(
        results,
        key=lambda result: result["f1"],
    )
    
def plot_thresholds(results: list[dict]) -> None:
    thresholds = [
        result["threshold"]
        for result in results
    ]

    precision_values = [
        result["precision"]
        for result in results
    ]

    recall_values = [
        result["recall"]
        for result in results
    ]

    f1_values = [
        result["f1"]
        for result in results
    ]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(
        thresholds,
        precision_values,
        marker="o",
        label="Precision",
    )

    plt.plot(
        thresholds,
        recall_values,
        marker="o",
        label="Recall",
    )

    plt.plot(
        thresholds,
        f1_values,
        marker="o",
        label="F1",
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Analysis on Validation Set")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close()

    print(f"\nFigure saved: {FIGURE_PATH}")
    
def main() -> None:
    df = load_data()

    X, y = prepare_features(df)

    (
        _,
        X_validation,
        _,
        _,
        y_validation,
        _,
    ) = split_data(X, y)

    model = load_model(MODEL_PATH)

    results = analyze_thresholds(
        model,
        X_validation,
        y_validation,
    )

    print_results(results)

    best_result = find_best_threshold(results)

    print("\n=== BEST VALIDATION THRESHOLD ===")
    print(f"Threshold: {best_result['threshold']:.2f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall:    {best_result['recall']:.4f}")
    print(f"F1:        {best_result['f1']:.4f}")

    plot_thresholds(results)


if __name__ == "__main__":
    main()