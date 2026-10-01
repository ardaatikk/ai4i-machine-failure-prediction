import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.builders import (
    BASE_NUMERICAL_FEATURES,
    ENGINEERED_NUMERICAL_FEATURES,
    build_gradient_boosting_model,
    build_logistic_model,
    build_random_forest_model,
)

from src.models.config import (
    FINAL_MODEL_NAME,
    FINAL_MODEL_THRESHOLD,
    RANDOM_FOREST_THRESHOLD,
)


def test_final_model_configuration() -> None:
    assert FINAL_MODEL_NAME == "Random Forest"

    assert (
        FINAL_MODEL_THRESHOLD
        == RANDOM_FOREST_THRESHOLD
    )

    assert 0.0 < FINAL_MODEL_THRESHOLD < 1.0


def test_logistic_builder_returns_pipeline() -> None:
    model = build_logistic_model(
        BASE_NUMERICAL_FEATURES
    )

    assert isinstance(model, Pipeline)
    assert isinstance(
        model.named_steps["classifier"],
        LogisticRegression,
    )


def test_random_forest_builder_returns_pipeline() -> None:
    numerical_features = (
        BASE_NUMERICAL_FEATURES
        + ENGINEERED_NUMERICAL_FEATURES
    )

    model = build_random_forest_model(
        numerical_features
    )

    assert isinstance(model, Pipeline)
    assert isinstance(
        model.named_steps["classifier"],
        RandomForestClassifier,
    )


def test_gradient_boosting_builder_returns_pipeline() -> None:
    numerical_features = (
        BASE_NUMERICAL_FEATURES
        + ENGINEERED_NUMERICAL_FEATURES
    )

    model = build_gradient_boosting_model(
        numerical_features
    )

    assert isinstance(model, Pipeline)
    assert isinstance(
        model.named_steps["classifier"],
        GradientBoostingClassifier,
    )
    
def test_gradient_boosting_can_fit_and_predict() -> None:
    numerical_features = (
        BASE_NUMERICAL_FEATURES
        + ENGINEERED_NUMERICAL_FEATURES
    )

    model = build_gradient_boosting_model(
        numerical_features
    )

    X = pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L", "M", "H"],
            "Air temperature [K]": [
                298.0, 299.0, 300.0,
                301.0, 302.0, 303.0,
            ],
            "Process temperature [K]": [
                308.0, 309.0, 310.0,
                311.0, 312.0, 313.0,
            ],
            "Rotational speed [rpm]": [
                1400, 1500, 1600,
                1300, 1700, 1800,
            ],
            "Torque [Nm]": [
                30.0, 35.0, 40.0,
                50.0, 55.0, 60.0,
            ],
            "Tool wear [min]": [
                10, 20, 30,
                100, 150, 200,
            ],
            "Temperature difference [K]": [
                10.0, 10.0, 10.0,
                10.0, 10.0, 10.0,
            ],
            "Power proxy": [
                42000.0,
                52500.0,
                64000.0,
                65000.0,
                93500.0,
                108000.0,
            ],
            "Tool wear torque interaction": [
                300.0,
                700.0,
                1200.0,
                5000.0,
                8250.0,
                12000.0,
            ],
        }
    )

    y = [0, 0, 0, 1, 1, 1]

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (6, 2)
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()