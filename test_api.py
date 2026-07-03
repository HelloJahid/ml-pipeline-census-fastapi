"""Unit tests for the FastAPI application."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root():
    """GET on the root must return 200 and the welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert "Welcome" in body["message"]


def test_post_predict_less_than_50k():
    """POST with a low-income profile must predict <=50K."""
    payload = {
        "age": 23,
        "workclass": "Private",
        "fnlgt": 122272,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_post_predict_greater_than_50k():
    """POST with a high-income profile must predict >50K."""
    payload = {
        "age": 47,
        "workclass": "Private",
        "fnlgt": 200000,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"
