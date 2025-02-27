import sys
import os
from dataclasses import dataclass


import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , evaluate_model


from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, val_array, test_array, raw_array):
        """Trains a single CatBoost model with best parameters gotten from notebok model_triner and logs performance variance."""
        logging.info("Initiating CatBoost model trainer")

        try:
            # regrouping Train, Val, Test, and Raw sets from target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_val, y_val = val_array[:, :-1], val_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            X_raw, y_raw = raw_array[:, :-1], raw_array[:, -1]

            # Compute class weights
            class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train), y=y_train)
            class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
            scale_pos_weight = class_weight_dict[0] / class_weight_dict[1]  # Adjust for CatBoost

            # Initialize CatBoost with best parameters
            catboost_model = CatBoostClassifier(
                scale_pos_weight=scale_pos_weight,
                verbose=0,
                subsample=0.8,
                learning_rate=0.05,
                l2_leaf_reg=5,
                iterations=300,
                depth=6,
                random_seed=42
            )


            # Train and Evaluate Model
            logging.info("Training CatBoost model with best parameters")
            report, probability_stats, raw_pred, raw_pred_proba, raw_class_distribution, model = evaluate_model(
                X_train, y_train, X_val, y_val, X_test, y_test, X_raw, catboost_model
            )

            variance_scores = {
                "accuracy_variance": [report["train"]["accuracy"] - report["validation"]["accuracy"],
                                      report["validation"]["accuracy"] - report["test"]["accuracy"]],
                "precision_variance": [report["train"]["precision"] - report["validation"]["precision"],
                                       report["validation"]["precision"] - report["test"]["precision"]],
                "recall_variance": [report["train"]["recall"] - report["validation"]["recall"],
                                    report["validation"]["recall"] - report["test"]["recall"]],
                "f1_variance": [report["train"]["f1"] - report["validation"]["f1"],
                                report["validation"]["f1"] - report["test"]["f1"]],
                "auc_variance": [report["train"]["auc"] - report["validation"]["auc"],
                                 report["validation"]["auc"] - report["test"]["auc"]]
            }

            logging.info(f"Model validation and test Variance Scores: {variance_scores}")


            # Save the trained model
            save_object(self.model_trainer_config.trained_model_file_path, model)
            logging.info(f"Trained model saved to {self.model_trainer_config.trained_model_file_path}")

            return report, probability_stats, variance_scores, raw_pred, raw_pred_proba, raw_class_distribution, model

        except Exception as e:
            raise CustomException(e, sys)



