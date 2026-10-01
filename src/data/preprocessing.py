import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Machine failure"

IDENTIFIER_COLUMNS = [
    "UDI",
    "Product ID",
]

FAILURE_TYPE_COLUMNS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:

    columns_to_drop = (
        IDENTIFIER_COLUMNS
        + FAILURE_TYPE_COLUMNS
        + [TARGET_COLUMN]
    )

    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN]

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    validation_ratio = VALIDATION_SIZE / (1 - TEST_SIZE)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=validation_ratio,
        random_state=RANDOM_STATE,
        stratify=y_train_validation,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )