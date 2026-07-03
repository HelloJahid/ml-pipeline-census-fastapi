"""Unit tests for the machine learning model functions."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

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


@pytest.fixture(scope="module")
def data():
    """Load a sample of the cleaned census data."""
    df = pd.read_csv("data/census.csv")
    return df.sample(n=1000, random_state=42)


@pytest.fixture(scope="module")
def processed(data):
    """Process the sample data for training."""
    X, y, encoder, lb = process_data(
        data, categorical_features=CAT_FEATURES, label="salary",
        training=True
    )
    return X, y, encoder, lb


@pytest.fixture(scope="module")
def model(processed):
    """Train a model on the processed sample."""
    X, y, _, _ = processed
    return train_model(X, y)


def test_process_data(processed, data):
    """process_data should return arrays of matching length."""
    X, y, encoder, lb = processed
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == len(data)
    assert X.shape[0] == y.shape[0]


def test_train_model(model):
    """train_model should return a fitted RandomForestClassifier."""
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_")


def test_inference(model, processed):
    """inference should return binary predictions, one per row."""
    X, y, _, _ = processed
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics():
    """compute_model_metrics should return three floats in [0, 1]."""
    y = np.array([1, 1, 0, 0, 1])
    preds = np.array([1, 0, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    for metric in (precision, recall, fbeta):
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
