"""FastAPI application serving the census income classification model."""
import os
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from ml.data import process_data
from ml.model import inference, load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

app = FastAPI(
    title="Census Income Classification API",
    description=(
        "Predicts whether a person earns more than $50K per year "
        "based on census data."
    ),
    version="1.0.0",
)

model = load_model(os.path.join(MODEL_DIR, "model.pkl"))
encoder = load_model(os.path.join(MODEL_DIR, "encoder.pkl"))
lb = load_model(os.path.join(MODEL_DIR, "lb.pkl"))


class CensusRecord(BaseModel):
    """A single census record used as input for model inference."""

    age: int = Field(..., examples=[45])
    workclass: str = Field(..., examples=["State-gov"])
    fnlgt: int = Field(..., examples=[2334])
    education: str = Field(..., examples=["Bachelors"])
    education_num: int = Field(
        ..., alias="education-num", examples=[13]
    )
    marital_status: str = Field(
        ..., alias="marital-status", examples=["Never-married"]
    )
    occupation: str = Field(..., examples=["Prof-specialty"])
    relationship: str = Field(..., examples=["Wife"])
    race: str = Field(..., examples=["Black"])
    sex: str = Field(..., examples=["Female"])
    capital_gain: int = Field(
        ..., alias="capital-gain", examples=[2174]
    )
    capital_loss: int = Field(..., alias="capital-loss", examples=[0])
    hours_per_week: int = Field(
        ..., alias="hours-per-week", examples=[60]
    )
    native_country: str = Field(
        ..., alias="native-country", examples=["Cuba"]
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "age": 45,
                    "workclass": "State-gov",
                    "fnlgt": 2334,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Prof-specialty",
                    "relationship": "Wife",
                    "race": "Black",
                    "sex": "Female",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 60,
                    "native-country": "Cuba",
                }
            ]
        },
    )


class PredictionResponse(BaseModel):
    """Response returned by the inference endpoint."""

    prediction: Literal["<=50K", ">50K"]


@app.get("/")
async def root() -> dict:
    """Return a welcome message."""
    return {
        "message": (
            "Welcome to the Census Income Classification API! "
            "Send a POST request to /predict to get a prediction."
        )
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(record: CensusRecord) -> PredictionResponse:
    """Run model inference on a single census record."""
    data = pd.DataFrame([record.model_dump(by_alias=True)])
    X, _, _, _ = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X)
    prediction = lb.inverse_transform(preds)[0]
    return PredictionResponse(prediction=prediction)
