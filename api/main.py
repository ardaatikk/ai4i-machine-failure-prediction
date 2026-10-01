from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.inference import (
    load_inference_model,
    request_to_dataframe,
)
from api.schemas import (
    EndpointLinks,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RootResponse,
)
from src.models.config import (
    FINAL_MODEL_NAME,
    FINAL_MODEL_THRESHOLD,
)
from src.utils.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    FINAL_MODEL_PATH,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_inference_model(
        FINAL_MODEL_PATH
    )

    yield


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

@app.get(
    "/",
    response_model=RootResponse,
)
def root() -> RootResponse:
    return RootResponse(
        name=API_TITLE,
        version=API_VERSION,
        model=FINAL_MODEL_NAME,
        endpoints=EndpointLinks(
            health="/health",
            predict="/predict",
            docs="/docs",
        ),
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