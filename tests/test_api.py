import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.models.config import (
    FINAL_MODEL_NAME,
    FINAL_MODEL_THRESHOLD,
)
from api.inference import (
    load_inference_model,
    request_to_dataframe,
)
from api.schemas import PredictionRequest
from src.utils.config import FINAL_MODEL_PATH


NORMAL_REQUEST = {
    "machine_type": "L",
    "air_temperature_k": 301.5,
    "process_temperature_k": 309.8,
    "rotational_speed_rpm": 1549,
    "torque_nm": 38.5,
    "tool_wear_min": 81,
}

FAILURE_REQUEST = {
    "machine_type": "L",
    "air_temperature_k": 303.3,
    "process_temperature_k": 311.3,
    "rotational_speed_rpm": 1350,
    "torque_nm": 48.1,
    "tool_wear_min": 32,
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == FINAL_MODEL_NAME


def test_predict_normal_sample(client) -> None:
    response = client.post(
        "/predict",
        json=NORMAL_REQUEST,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 0
    assert data["label"] == "normal"
    assert data["failure_probability"] == pytest.approx(
        0.0
    )
    assert data["threshold"] == FINAL_MODEL_THRESHOLD
    assert data["model"] == FINAL_MODEL_NAME


def test_predict_failure_sample(client) -> None:
    response = client.post(
        "/predict",
        json=FAILURE_REQUEST,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 1
    assert data["label"] == "failure"

    assert data["failure_probability"] == pytest.approx(
        0.9433333333333334
    )

    assert data["threshold"] == FINAL_MODEL_THRESHOLD
    assert data["model"] == FINAL_MODEL_NAME


def test_prediction_probability_is_valid(client) -> None:
    response = client.post(
        "/predict",
        json=FAILURE_REQUEST,
    )

    probability = response.json()[
        "failure_probability"
    ]

    assert 0.0 <= probability <= 1.0


def test_invalid_machine_type_is_rejected(client) -> None:
    request = FAILURE_REQUEST.copy()
    request["machine_type"] = "INVALID"

    response = client.post(
        "/predict",
        json=request,
    )

    assert response.status_code == 422


def test_negative_torque_is_rejected(client) -> None:
    request = FAILURE_REQUEST.copy()
    request["torque_nm"] = -1.0

    response = client.post(
        "/predict",
        json=request,
    )

    assert response.status_code == 422


def test_negative_tool_wear_is_rejected(client) -> None:
    request = FAILURE_REQUEST.copy()
    request["tool_wear_min"] = -1

    response = client.post(
        "/predict",
        json=request,
    )

    assert response.status_code == 422


def test_missing_required_field_is_rejected(client) -> None:
    request = FAILURE_REQUEST.copy()
    request.pop("torque_nm")

    response = client.post(
        "/predict",
        json=request,
    )

    assert response.status_code == 422
    
def test_api_matches_direct_model_inference(
    client,
) -> None:
    response = client.post(
        "/predict",
        json=FAILURE_REQUEST,
    )

    assert response.status_code == 200

    api_probability = response.json()[
        "failure_probability"
    ]

    request = PredictionRequest(
        **FAILURE_REQUEST
    )

    model_input = request_to_dataframe(
        request
    )

    model = load_inference_model(
        FINAL_MODEL_PATH
    )

    direct_probability = float(
        model.predict_proba(
            model_input
        )[0, 1]
    )

    assert api_probability == pytest.approx(
        direct_probability
    )