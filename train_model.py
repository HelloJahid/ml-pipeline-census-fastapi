"""Script to train the machine learning model on the census data."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    save_model,
    train_model,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "census.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def run_training():
    """Train the model, evaluate it and save all artifacts."""
    # Load the cleaned data.
    data = pd.read_csv(DATA_PATH)

    # Train-test split.
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )

    # Process the training data.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary",
        training=True
    )

    # Process the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train and save a model.
    model = train_model(X_train, y_train)
    save_model(model, os.path.join(MODEL_DIR, "model.pkl"))
    save_model(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
    save_model(lb, os.path.join(MODEL_DIR, "lb.pkl"))

    # Overall performance on the test set.
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(
        f"Overall test metrics -> precision: {precision:.4f}, "
        f"recall: {recall:.4f}, fbeta: {fbeta:.4f}"
    )

    # Performance on slices of every categorical feature,
    # written to slice_output.txt.
    output_path = os.path.join(BASE_DIR, "slice_output.txt")
    with open(output_path, "w") as f:
        f.write("Model performance on slices of categorical features\n")
        f.write("(computed on the held-out test set)\n")
        f.write("=" * 70 + "\n")
        for feature in cat_features:
            results = compute_slice_metrics(
                test, feature, model, encoder, lb, cat_features
            )
            for row in results:
                line = (
                    f"[{row['feature']} = {row['value']}] "
                    f"n={row['n_samples']} "
                    f"precision={row['precision']:.4f} "
                    f"recall={row['recall']:.4f} "
                    f"fbeta={row['fbeta']:.4f}"
                )
                f.write(line + "\n")
            f.write("-" * 70 + "\n")
    print(f"Slice metrics written to {output_path}")

    return precision, recall, fbeta


if __name__ == "__main__":
    run_training()
