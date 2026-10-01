from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.inference import (
    load_inference_model,
    request_to_dataframe,
)
from api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.models.config import (
    FINAL_MODEL_NAME,
    FINAL_MODEL_THRESHOLD,
)
from src.utils.config import FINAL_MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_inference_model(
        FINAL_MODEL_PATH
    )

    yield


app = FastAPI(
    title="Predictive Maintenance ML API",
    description=(
        "Predict machine failure risk from operational "
        "and sensor measurements."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model=FINAL_MODEL_NAME,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    model_input = request_to_dataframe(request)

    failure_probability = float(
        app.state.model.predict_proba(
            model_input
        )[0, 1]
    )

    prediction = int(
        failure_probability
        >= FINAL_MODEL_THRESHOLD
    )

    label = (
        "failure"
        if prediction == 1
        else "normal"
    )

    return PredictionResponse(
        prediction=prediction,
        label=label,
        failure_probability=failure_probability,
        threshold=FINAL_MODEL_THRESHOLD,
        model=FINAL_MODEL_NAME,
    )