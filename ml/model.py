"""Model training, inference, persistence and evaluation functions."""
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """Serialise a model or encoder to disk using pickle.

    Inputs
    ------
    model : object
        Trained model, encoder or label binarizer.
    path : str
        Destination file path.
    """
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path):
    """Load a pickled model or encoder from disk.

    Inputs
    ------
    path : str
        Path to the pickled object.
    Returns
    -------
    object
        The deserialised model, encoder or label binarizer.
    """
    with open(path, "rb") as file:
        return pickle.load(file)


def compute_slice_metrics(
    df, feature, model, encoder, lb, categorical_features, label="salary"
):
    """Compute model metrics on slices of the data.

    For a given categorical feature, computes precision, recall and
    F1 for every unique value of that feature (value held fixed).

    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the features and label.
    feature : str
        Name of the categorical feature to slice on.
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer.
    categorical_features : list[str]
        List of categorical feature names.
    label : str
        Name of the label column.
    Returns
    -------
    results : list[dict]
        One dict per unique value with precision, recall, fbeta
        and sample count.
    """
    results = []
    for value in sorted(df[feature].unique()):
        df_slice = df[df[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            df_slice,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        results.append(
            {
                "feature": feature,
                "value": value,
                "n_samples": len(df_slice),
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            }
        )
    return results
