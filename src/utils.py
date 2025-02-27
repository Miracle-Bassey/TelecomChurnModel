import sys
import os


import numpy as np
import pickle

from src.exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def train_model(model, X_train, y_train, X_val, y_val):
    """Train the CatBoost model."""
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model


def make_predictions(model, X_train, X_val, X_test, X_raw):
    """Generate predictions for all datasets."""
    return {
        "train": model.predict(X_train),
        "validation": model.predict(X_val),
        "test": model.predict(X_test),
        "raw": model.predict(X_raw),
        "train_proba": model.predict_proba(X_train)[:, 1],
        "validation_proba": model.predict_proba(X_val)[:, 1],
        "test_proba": model.predict_proba(X_test)[:, 1],
        "raw_proba": model.predict_proba(X_raw)[:, 1],
    }


def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if y_pred_proba is not None:
        metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
    return metrics


def compute_probability_stats(y_train_proba, y_val_proba, y_test_proba, y_raw_proba):
    """Compute standard deviation of probability predictions."""
    return {
        "train_std": np.std(y_train_proba),
        "validation_std": np.std(y_val_proba),
        "test_std": np.std(y_test_proba),
        "raw_std": np.std(y_raw_proba),
    }


def compute_class_distribution(y_raw):
    """Compute class distribution in the raw dataset."""
    return {"class_0": (y_raw == 0).sum(), "class_1": (y_raw == 1).sum()}


def evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, X_raw, model):
    """
    Train and evaluate a single CatBoost model using modular helper functions.

    Returns:
        dict: Model evaluation metrics.
        dict: Probability standard deviations.
        np.ndarray: Raw dataset predictions.
        dict: Raw data class distribution.
        CatBoostClassifier: Trained model.
    """
    try:
        logging.info("Training CatBoost model...")
        model = train_model(model, X_train, y_train, X_val, y_val)

        logging.info("Generating predictions...")
        preds = make_predictions(model, X_train, X_val, X_test, X_raw)

        logging.info("Computing evaluation metrics...")
        report = {
            "train": compute_metrics(y_train, preds["train"], preds["train_proba"]),
            "validation": compute_metrics(y_val, preds["validation"], preds["validation_proba"]),
            "test": compute_metrics(y_test, preds["test"], preds["test_proba"]),
        }

        logging.info("Computing probability distribution statistics...")
        probability_stats = compute_probability_stats(
            preds["train_proba"], preds["validation_proba"], preds["test_proba"], preds["raw_proba"]
        )

        logging.info("Computing class distribution for raw data...")
        raw_class_distribution = compute_class_distribution(preds["raw"])

        logging.info(f"Validation Recall: {report['validation']['recall']} | Test Recall: {report['test']['recall']}")
        logging.info(f"Raw Data Predictions: {raw_class_distribution}")

        return report, probability_stats, preds["raw"], preds["raw_proba"], raw_class_distribution, model

    except Exception as e:
        raise CustomException(e, sys)




def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)