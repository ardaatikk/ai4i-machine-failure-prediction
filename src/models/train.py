from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.data.analysis import load_data
from src.data.preprocessing import prepare_features, split_data
from src.features.engineering import add_engineered_features
from src.models.builders import (
    BASE_NUMERICAL_FEATURES,
    ENGINEERED_NUMERICAL_FEATURES,
    build_gradient_boosting_model,
    build_logistic_model,
    build_random_forest_model,
)
from src.utils.config import (
    BASELINE_MODEL_PATH,
    ENGINEERED_MODEL_PATH,
    GRADIENT_BOOSTING_MODEL_PATH,
    MODELS_DIR,
    RANDOM_FOREST_MODEL_PATH,
)


def train_baseline_model():
    df = load_data()

    X, y = prepare_features(df)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y)

    model = build_logistic_model(
        BASE_NUMERICAL_FEATURES
    )

    print("Training baseline Logistic Regression model...")

    model.fit(X_train, y_train)

    print("Baseline training completed.")

    return (
        model,
        X_validation,
        X_test,
        y_validation,
        y_test,
    )


def train_engineered_model():
    df = load_data()

    X, y = prepare_features(df)
    X = add_engineered_features(X)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y)

    numerical_features = (
        BASE_NUMERICAL_FEATURES
        + ENGINEERED_NUMERICAL_FEATURES
    )

    model = build_logistic_model(
        numerical_features
    )

    print("Training engineered Logistic Regression model...")

    model.fit(X_train, y_train)

    print("Engineered model training completed.")

    return (
        model,
        X_validation,
        X_test,
        y_validation,
        y_test,
    )


def train_random_forest_model():
    df = load_data()

    X, y = prepare_features(df)
    X = add_engineered_features(X)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y)

    numerical_features = (
        BASE_NUMERICAL_FEATURES
        + ENGINEERED_NUMERICAL_FEATURES
    )

    model = build_random_forest_model(
        numerical_features
    )

    print("Training Random Forest model...")

    model.fit(X_train, y_train)

    print("Random Forest training completed.")

    return (
        model,
        X_validation,
        X_test,
        y_validation,
        y_test,
    )


def train_gradient_boosting_model():
    df = load_data()

    X, y = prepare_features(df)
    X = add_engineered_features(X)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y)

    numerical_features = (
        BASE_NUMERICAL_FEATURES
        + ENGINEERED_NUMERICAL_FEATURES
    )

    model = build_gradient_boosting_model(
        numerical_features
    )

    print("Training Gradient Boosting model...")

    model.fit(X_train, y_train)

    print("Gradient Boosting training completed.")

    return (
        model,
        X_validation,
        X_test,
        y_validation,
        y_test,
    )


def save_model(
    model: Pipeline,
    model_path: Path,
) -> None:
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    print(f"Model saved: {model_path}")


def main() -> None:
    baseline_model, _, _, _, _ = (
        train_baseline_model()
    )

    save_model(
        baseline_model,
        BASELINE_MODEL_PATH,
    )

    engineered_model, _, _, _, _ = (
        train_engineered_model()
    )

    save_model(
        engineered_model,
        ENGINEERED_MODEL_PATH,
    )

    random_forest_model, _, _, _, _ = (
        train_random_forest_model()
    )

    save_model(
        random_forest_model,
        RANDOM_FOREST_MODEL_PATH,
    )

    gradient_boosting_model, _, _, _, _ = (
        train_gradient_boosting_model()
    )

    save_model(
        gradient_boosting_model,
        GRADIENT_BOOSTING_MODEL_PATH,
    )


if __name__ == "__main__":
    main()