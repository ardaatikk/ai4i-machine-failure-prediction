from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product_quality": "L",
                "air_temperature_k": 303.3,
                "process_temperature_k": 311.3,
                "rotational_speed_rpm": 1350,
                "torque_nm": 48.1,
                "tool_wear_min": 32,
            }
        }
    )
    
    product_quality: Literal["L", "M", "H"] = Field(
        ...,
        description=(
            "Product quality type: "
            "L (low), M (medium), or H (high)."
        ),
    )

    air_temperature_k: float = Field(
        ...,
        gt=0,
        description="Air temperature in Kelvin.",
    )

    process_temperature_k: float = Field(
        ...,
        gt=0,
        description="Process temperature in Kelvin.",
    )

    rotational_speed_rpm: int = Field(
        ...,
        gt=0,
        description="Rotational speed in RPM.",
    )

    torque_nm: float = Field(
        ...,
        ge=0,
        description="Torque in Newton-metres.",
    )

    tool_wear_min: int = Field(
        ...,
        ge=0,
        description="Tool wear in minutes.",
    )


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1] = Field(
        ...,
        description=(
            "Binary prediction: "
            "0 for normal operation, 1 for predicted failure."
        ),
    )

    label: Literal["normal", "failure"] = Field(
        ...,
        description="Human-readable prediction label.",
    )

    failure_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Predicted probability of machine failure.",
    )

    threshold: float = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Decision threshold used to convert the "
            "failure probability into a binary prediction."
        ),
    )

    model: str = Field(
        ...,
        description="Machine learning model used for prediction.",
    )

class HealthResponse(BaseModel):
    status: Literal["healthy"]
    model: str
    
class EndpointLinks(BaseModel):
    health: str
    predict: str
    docs: str
    
class RootResponse(BaseModel):
    name: str
    version: str
    model: str
    endpoints: EndpointLinks