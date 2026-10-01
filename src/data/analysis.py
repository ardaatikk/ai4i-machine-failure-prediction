import pandas as pd

from src.utils.config import RAW_DATA_FILE


TARGET_COLUMN = "Machine failure"

FAILURE_COLUMNS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_data() -> pd.DataFrame:
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_FILE}\n"
            "Run 'python -m src.data.download' first."
        )

    return pd.read_csv(RAW_DATA_FILE)


def analyze_data(df: pd.DataFrame) -> None:
    print("\n=== DATASET SHAPE ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n=== COLUMNS ===")
    for column in df.columns:
        print(f"- {column}")

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

    print("\n=== TARGET DISTRIBUTION ===")
    print(df[TARGET_COLUMN].value_counts())

    print("\n=== TARGET DISTRIBUTION (%) ===")
    print(
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\n=== FAILURE TYPES ===")
    print(df[FAILURE_COLUMNS].sum())

    print("\n=== NUMERICAL SUMMARY ===")
    print(df.describe())


def main() -> None:
    df = load_data()
    analyze_data(df)


if __name__ == "__main__":
    main()