from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from api.schemas import PredictionRequest
from src.features.engineering import add_engineered_features


def load_inference_model(
    model_path: Path,
) -> Pipeline:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run 'python -m src.models.train' first."
        )

    return joblib.load(model_path)


def request_to_dataframe(
    request: PredictionRequest,
) -> pd.DataFrame:
    data = {
        "Type": [request.product_quality],
        "Air temperature [K]": [
            request.air_temperature_k
        ],
        "Process temperature [K]": [
            request.process_temperature_k
        ],
        "Rotational speed [rpm]": [
            request.rotational_speed_rpm
        ],
        "Torque [Nm]": [
            request.torque_nm
        ],
        "Tool wear [min]": [
            request.tool_wear_min
        ],
    }

    dataframe = pd.DataFrame(data)

    return add_engineered_features(dataframe)