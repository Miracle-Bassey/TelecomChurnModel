import sys
import os

import pandas as pd
import numpy as np
import pickle

from src.exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids):
    """
    Evaluate multiple classification models with hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        models (dict): Dictionary of models to evaluate.
        param_grids (dict): Dictionary of parameter grids for hyperparameter tuning.

    Returns:
        dict: A dictionary containing the evaluation metrics for each model.
        dict: A dictionary containing the best parameters for each model.
        dict: A dictionary containing the best trained model for each model.
    """
    try:
        report = {}  # Store model performance
        best_params = {}  # Store best hyperparameters
        trained_models = {}  # Store trained models

        for model_name, model in models.items():
            print(f"Training {model_name}...")

            # Hyperparameter tuning using RandomizedSearchCV if parameters exist
            if model_name in param_grids:
                print(f"Performing RandomizedSearchCV for {model_name}...")
                search = RandomizedSearchCV(
                    model,
                    param_distributions=param_grids[model_name],
                    n_iter=10,  # Number of parameter settings to sample
                    random_state=42,  # For reproducibility
                    n_jobs=-1,  # Use all available cores
                    cv=3  # 3-fold cross-validation
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_  # Get the best model

                # Save best parameters
                best_params[model_name] = search.best_params_

            # Train the best model
            model.fit(X_train, y_train)
            trained_models[model_name] = model  # Store trained model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Compute AUC correctly (some models need predict_proba)
            if hasattr(model, "predict_proba"):
                y_test_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
                test_auc = roc_auc_score(y_test, y_test_pred_proba)
            else:
                test_auc = roc_auc_score(y_test, y_test_pred)

            # Evaluate performance
            report[model_name] = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'train_precision': precision_score(y_train, y_train_pred),
                'train_recall': recall_score(y_train, y_train_pred),
                'train_f1': f1_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'test_precision': precision_score(y_test, y_test_pred),
                'test_recall': recall_score(y_test, y_test_pred),
                'test_f1': f1_score(y_test, y_test_pred),
                'test_auc': test_auc  # Store correct AUC
            }

        return report, best_params, trained_models

    except Exception as e:
        raise CustomException(e,sys)