import pandas as pd
import pytest

from src.features.engineering import add_engineered_features


def test_engineered_features_are_calculated_correctly() -> None:
    df = pd.DataFrame(
        {
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [100],
        }
    )

    result = add_engineered_features(df)

    assert result.loc[0, "Temperature difference [K]"] == pytest.approx(10.0)

    assert result.loc[0, "Power proxy"] == pytest.approx(60000.0)

    assert result.loc[
        0,
        "Tool wear torque interaction",
    ] == pytest.approx(4000.0)


def test_feature_engineering_does_not_mutate_input() -> None:
    df = pd.DataFrame(
        {
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [100],
        }
    )

    original_columns = df.columns.tolist()

    add_engineered_features(df)

    assert df.columns.tolist() == original_columns