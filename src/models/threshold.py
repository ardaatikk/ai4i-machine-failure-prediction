import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.data.analysis import load_data
from src.data.preprocessing import prepare_features, split_data
from src.features.engineering import add_engineered_features
from src.models.evaluate import load_model
from src.utils.config import (
    BASELINE_MODEL_PATH,
    ENGINEERED_MODEL_PATH,
    FIGURES_DIR,
    GRADIENT_BOOSTING_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
)


THRESHOLDS = np.arange(0.01, 0.991, 0.01)

THRESHOLD_COMPARISON_FIGURE_PATH = (
    FIGURES_DIR / "threshold_comparison.png"
)

MODEL_REGISTRY = {
    "Baseline Logistic": {
        "path": BASELINE_MODEL_PATH,
        "engineered": False,
    },
    "Engineered Logistic": {
        "path": ENGINEERED_MODEL_PATH,
        "engineered": True,
    },
    "Random Forest": {
        "path": RANDOM_FOREST_MODEL_PATH,
        "engineered": True,
    },
    "Gradient Boosting": {
        "path": GRADIENT_BOOSTING_MODEL_PATH,
        "engineered": True,
    },
}


def analyze_thresholds(
    model: Pipeline,
    X_validation,
    y_validation,
) -> list[dict]:
    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    results = []

    for threshold in THRESHOLDS:
        predictions = (
            probabilities >= threshold
        ).astype(int)

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


def print_results(
    results: list[dict],
) -> None:
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


def find_best_threshold(
    results: list[dict],
) -> dict:
    return max(
        results,
        key=lambda result: result["f1"],
    )


def plot_model_comparison(
    model_results: dict[str, list[dict]],
) -> None:
    plt.figure(
        figsize=(10, 6)
    )

    for model_name, results in model_results.items():
        thresholds = [
            result["threshold"]
            for result in results
        ]

        f1_scores = [
            result["f1"]
            for result in results
        ]

        plt.plot(
            thresholds,
            f1_scores,
            label=model_name,
        )

    plt.xlabel(
        "Decision Threshold"
    )

    plt.ylabel(
        "F1 Score"
    )

    plt.title(
        "Threshold Optimization by Model"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        THRESHOLD_COMPARISON_FIGURE_PATH,
        dpi=300,
    )

    plt.close()

    print(
        "\nFigure saved: "
        f"{THRESHOLD_COMPARISON_FIGURE_PATH}"
    )


def run_threshold_analysis(
    model_path,
    X_validation,
    y_validation,
    model_name: str,
) -> tuple[dict, list[dict]]:
    model = load_model(
        model_path
    )

    results = analyze_thresholds(
        model,
        X_validation,
        y_validation,
    )

    print(
        f"\n=== {model_name.upper()} ==="
    )

    print_results(
        results
    )

    best_result = find_best_threshold(
        results
    )

    print(
        "\n=== BEST VALIDATION THRESHOLD ==="
    )

    print(
        f"Threshold: "
        f"{best_result['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{best_result['f1']:.4f}"
    )

    return best_result, results


def main() -> None:
    df = load_data()

    X, y = prepare_features(
        df
    )

    (
        _,
        X_validation,
        _,
        _,
        y_validation,
        _,
    ) = split_data(
        X,
        y,
    )

    X_engineered = add_engineered_features(
        X
    )

    (
        _,
        X_engineered_validation,
        _,
        _,
        y_engineered_validation,
        _,
    ) = split_data(
        X_engineered,
        y,
    )

    best_results = {}
    threshold_results = {}

    for model_name, config in MODEL_REGISTRY.items():
        print(
            "\n"
            + "#" * 40
        )

        print(
            model_name.upper()
        )

        print(
            "#" * 40
        )

        if config["engineered"]:
            X_model_validation = (
                X_engineered_validation
            )

            y_model_validation = (
                y_engineered_validation
            )

        else:
            X_model_validation = (
                X_validation
            )

            y_model_validation = (
                y_validation
            )

        best_result, results = (
            run_threshold_analysis(
                config["path"],
                X_model_validation,
                y_model_validation,
                model_name,
            )
        )

        best_results[
            model_name
        ] = best_result

        threshold_results[
            model_name
        ] = results

    print(
        "\n=== VALIDATION COMPARISON ==="
    )

    for model_name, result in best_results.items():
        print(
            f"{model_name:<22} "
            f"F1={result['f1']:.4f} "
            f"| Precision={result['precision']:.4f} "
            f"| Recall={result['recall']:.4f} "
            f"| Threshold={result['threshold']:.2f}"
        )

    plot_model_comparison(
        threshold_results
    )


if __name__ == "__main__":
    main()