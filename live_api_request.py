"""Script that sends a POST request to the live deployed API.

Prints both the status code and the model inference result,
as required by the rubric. Update LIVE_URL after deployment.
"""
import requests

# TODO: replace with your deployed app URL (Render/Heroku).
LIVE_URL = "https://ml-pipeline-census-fastapi.onrender.com/predict"

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


def main() -> None:
    """POST the payload to the live API and print the outcome."""
    response = requests.post(LIVE_URL, json=payload, timeout=30)
    print(f"Status code: {response.status_code}")
    print(f"Model inference result: {response.json()}")


if __name__ == "__main__":
    main()
