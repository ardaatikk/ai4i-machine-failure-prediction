from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    machine_type: Literal["L", "M", "H"]

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
    prediction: int
    label: str
    failure_probability: float
    threshold: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str