import joblib

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.analysis import load_data
from src.data.preprocessing import prepare_features, split_data
from src.utils.config import MODELS_DIR

CATEGORICAL_FEATURES = [
    "Type",
]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

MODEL_FILENAME = "baseline_logistic_regression.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ]
    )
    
def build_model() -> Pipeline:
    preprocessor = build_preprocessor()

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    
def train_model() -> Pipeline:
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

    model = build_model()

    print("Training baseline Logistic Regression model...")

    model.fit(X_train, y_train)

    print("Training completed.")

    return (
        model,
        X_validation,
        X_test,
        y_validation,
        y_test,
    )

def save_model(model: Pipeline) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved: {MODEL_PATH}")
    
def main() -> None:
    model, _, _, _, _ = train_model()
    save_model(model)


if __name__ == "__main__":
    main()