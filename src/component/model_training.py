import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        """Trains multiple models and selects the best one based on performance metrics."""
        logging.info("Initiating model trainer")
        try:
            # Split dataset into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features
                train_array[:, -1],   # Target
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Compute class weights
            class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train), y=y_train)
            class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
            scale_pos_weight = class_weight_dict[0] / class_weight_dict[1]  # Adjust for CatBoost

            # Define models with class weights
            models = {
                "Logistic Regression": LogisticRegression(class_weight=class_weight_dict, random_state=42),
                "Random Forest": RandomForestClassifier(class_weight=class_weight_dict, random_state=42),
                "CatBoost": CatBoostClassifier(scale_pos_weight=scale_pos_weight, verbose=0, random_seed=42),
                "SVC": SVC(class_weight=class_weight_dict, probability=True, random_state=42),
            }

            # Define hyperparameter grids for tuning
            param_grids = {
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs'],
                    'max_iter': [100, 200, 300]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200, 500],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': [None, 'sqrt', 'log2']
                },
                "CatBoost": {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8],
                    'l2_leaf_reg': [1, 3, 5],
                    'subsample': [0.6, 0.8, 1]
                },
                "SVC": {
                    'C': [0.1, 1, 10, 100, 200],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'],
                    'degree': [3, 4, 5]
                },
            }

            logging.info("Model training has commenced")
            model_report, model_param,trained_models = evaluate_models(X_train, y_train, X_test, y_test, models, param_grids)

            logging.info("Checking model performance report")
            if not isinstance(model_report, dict):
                raise CustomException(Exception, sys)

            # Select the best model based on highest performance score based off recall score
            best_model_name = max(model_report,key=lambda k: model_report[k]['test_recall'])
            best_model_score = model_report[best_model_name]['test_recall']
            best_model = trained_models[best_model_name]

            # Log all model recall scores
            logging.info("Model selection based on recall score")
            for model, scores in model_report.items():
                logging.info(f"Model: {model}, Test Recall: {scores['test_recall']}")

            if best_model_score < 0.8:
                raise CustomException(Exception,sys)

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            auc = roc_auc_score(y_test, predicted)
            return auc

        except Exception as e:
            raise CustomException(e, sys)
