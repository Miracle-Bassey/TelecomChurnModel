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
    """
    try:
        # Dictionary to store evaluation results
        report = {}
        # Dictionary to store the best parameters for each model
        best_params = {}

        for model_name, model in models.items():
            print(f"Training {model_name}...")

            # Hyperparameter tuning using RandomizedSearchCV if a param_grid is defined
            if model_name in param_grids:
                print("Performing RandomizedSearchCV...")
                search = RandomizedSearchCV(
                    model,
                    param_distributions=param_grids[model_name],
                    n_iter=10,  # Number of parameter settings to sample
                    random_state=42,  # For reproducibility
                    n_jobs=-1,  # Use all available cores
                    cv=3  # 3-fold cross-validation
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_

                # Save the best parameters
                best_params[model_name] = search.best_params_

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate Train and Test dataset
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            train_auc = roc_auc_score(y_train, y_train_pred)

            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_auc = roc_auc_score(y_test, y_test_pred)

            # Store results in the report
            report[model_name] = {
                'train_accuracy': train_accuracy,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'train_auc': train_auc,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc
            }


        return report, best_params

    except Exception as e:
        raise CustomException(e, sys)