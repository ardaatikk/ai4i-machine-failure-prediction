from src.data.analysis import load_data
from src.data.preprocessing import (
    FAILURE_TYPE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    prepare_features,
    split_data,
)
from src.models.builders import (
    BASE_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


EXPECTED_BASE_NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def test_base_feature_contract() -> None:
    assert BASE_NUMERICAL_FEATURES == (
        EXPECTED_BASE_NUMERICAL_FEATURES
    )

    assert CATEGORICAL_FEATURES == ["Type"]


def test_prepare_features_removes_leakage() -> None:
    df = load_data()

    X, y = prepare_features(df)

    forbidden_columns = (
        IDENTIFIER_COLUMNS
        + FAILURE_TYPE_COLUMNS
        + [TARGET_COLUMN]
    )

    for column in forbidden_columns:
        assert column not in X.columns

    assert y.name == TARGET_COLUMN


def test_prepare_features_has_expected_columns() -> None:
    df = load_data()

    X, _ = prepare_features(df)

    expected_columns = (
        CATEGORICAL_FEATURES
        + BASE_NUMERICAL_FEATURES
    )

    assert set(X.columns) == set(expected_columns)
    assert X.shape[1] == 6


def test_split_sizes() -> None:
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

    assert len(X_train) == 7000
    assert len(X_validation) == 1500
    assert len(X_test) == 1500

    assert len(y_train) == 7000
    assert len(y_validation) == 1500
    assert len(y_test) == 1500


def test_split_preserves_failure_rate() -> None:
    df = load_data()

    X, y = prepare_features(df)

    (
        _,
        _,
        _,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y)

    overall_rate = y.mean()

    assert abs(y_train.mean() - overall_rate) < 0.001
    assert abs(y_validation.mean() - overall_rate) < 0.001
    assert abs(y_test.mean() - overall_rate) < 0.001