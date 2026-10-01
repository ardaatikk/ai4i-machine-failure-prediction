import pandas as pd


AIR_TEMPERATURE_COLUMN = "Air temperature [K]"
PROCESS_TEMPERATURE_COLUMN = "Process temperature [K]"
ROTATIONAL_SPEED_COLUMN = "Rotational speed [rpm]"
TORQUE_COLUMN = "Torque [Nm]"
TOOL_WEAR_COLUMN = "Tool wear [min]"


def add_engineered_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df["Temperature difference [K]"] = (
        df[PROCESS_TEMPERATURE_COLUMN]
        - df[AIR_TEMPERATURE_COLUMN]
    )

    df["Power proxy"] = (
        df[ROTATIONAL_SPEED_COLUMN]
        * df[TORQUE_COLUMN]
    )

    df["Tool wear torque interaction"] = (
        df[TOOL_WEAR_COLUMN]
        * df[TORQUE_COLUMN]
    )

    return df