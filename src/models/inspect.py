import pandas as pd

from src.models.evaluate import load_model
from src.utils.config import RANDOM_FOREST_MODEL_PATH


def inspect_random_forest() -> None:
    model = load_model(
        RANDOM_FOREST_MODEL_PATH
    )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    importances = (
        classifier.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    print(
        "\n=== RANDOM FOREST FEATURE IMPORTANCE ==="
    )

    print(
        importance_df.to_string(
            index=False,
        )
    )

    print(
        "\n=== IMPORTANCE SUM ==="
    )

    print(
        importance_df["importance"].sum()
    )


def inspect_model_structure() -> None:
    model = load_model(
        RANDOM_FOREST_MODEL_PATH
    )

    classifier = model.named_steps[
        "classifier"
    ]

    print(
        "\n=== RANDOM FOREST STRUCTURE ==="
    )

    print(
        f"Number of trees: "
        f"{len(classifier.estimators_)}"
    )

    print(
        f"Input features: "
        f"{classifier.n_features_in_}"
    )

    print(
        f"Classes: "
        f"{classifier.classes_}"
    )


def main() -> None:
    inspect_random_forest()
    inspect_model_structure()


if __name__ == "__main__":
    main()