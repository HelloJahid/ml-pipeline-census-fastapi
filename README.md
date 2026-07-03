# Deploying a Machine Learning Model with FastAPI

Udacity Machine Learning DevOps Engineer Nanodegree — Project: ML Pipeline with CI/CD.

**GitHub repository:** https://github.com/HelloJahid/ml-pipeline-census-fastapi

This project trains a Random Forest classifier on publicly available Census Bureau data to predict whether a person earns more than $50K per year. The model is validated on data slices, served through a FastAPI application, tested with pytest, linted with flake8, integrated with GitHub Actions CI, and deployed to a cloud application platform with continuous delivery.

## Project Structure

```
.
├── .github/workflows/ci.yml   # CI: flake8 + pytest on push (Python 3.13)
├── data/census.csv             # Cleaned census dataset
├── ml/
│   ├── data.py                 # Data processing (one-hot encoding, label binarizer)
│   └── model.py                # Train, inference, metrics, slice metrics, save/load
├── model/                      # Trained model, encoder and label binarizer (.pkl)
├── screenshots/                # Rubric screenshots
├── main.py                     # FastAPI application (GET / and POST /predict)
├── train_model.py              # Training script, writes slice_output.txt
├── test_model.py               # Unit tests for the ML functions
├── test_api.py                 # Unit tests for the API
├── live_api_request.py         # POST request to the live deployed API
├── sanitycheck.py              # Udacity test-case sanity checker
├── slice_output.txt            # Model metrics per categorical feature slice
├── model_card.md               # Model card
├── Procfile                    # Process declaration for deployment
├── requirements.txt
└── README.md
```

## Environment Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python train_model.py
```

This loads `data/census.csv`, performs a stratified 80/20 train-test split, trains the model, saves `model/model.pkl`, `model/encoder.pkl` and `model/lb.pkl`, prints overall test metrics, and writes per-slice metrics for every categorical feature to `slice_output.txt`.

## Testing and Linting

```bash
pytest test_model.py test_api.py -v
flake8 main.py train_model.py test_model.py test_api.py ml/ --max-line-length=100
```

Seven tests cover the ML functions (process_data, train_model, inference, compute_model_metrics) and the API (GET root, POST predicting <=50K, POST predicting >50K). The same commands run automatically in GitHub Actions on every push.

## Running the API Locally

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive documentation, which includes a complete example request body.

- `GET /` returns a welcome message.
- `POST /predict` accepts a census record (hyphenated field names handled through Pydantic aliases) and returns the predicted salary class.

## Model Performance

On the held-out test set: precision 0.7847, recall 0.6135, F1 0.6886. See `model_card.md` for full details and `slice_output.txt` for per-slice metrics.

## Deployment

The app is deployed from this repository with continuous delivery enabled, so a new version deploys only after CI passes. Query the live API with:

```bash
python live_api_request.py
```

which prints both the status code and the model inference result.
