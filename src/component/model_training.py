import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import  RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):
        logging.info("Initiating model trainer")
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1], # Determinant
                train_array[:,-1], #Target
                test_array[:,:-1],
                test_array[:,-1]
            )
            # Compute class weight on y_trained data
            class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train), y=y_train)

            class_weight_dict = {cls: weight for cls,
            weight in zip(np.unique(y_train), class_weights)}

            # Compute scale_pos_weight for CatBoost
            scale_pos_weight = class_weight_dict[0] / class_weight_dict[1]  # Adjust based on class 0 & 1 ratio

            # list models
            models = {
                "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
                "Random Forest Classifier": RandomForestClassifier(class_weight='balanced', random_state=42),
                "CatBoosting Classifier": CatBoostClassifier(scale_pos_weight=scale_pos_weight, verbose=0, random_seed=42),
                "SVC": SVC(class_weight='balanced', probability=True, random_state=42),
            }

            # hyperparameter tuning

            param_grids = {
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1, 10],               # Regularization strength
                    'solver': ['liblinear', 'lbfgs'],       # Optimization algorithms
                    'max_iter': [100, 200, 300]             # Maximum iterations for convergence
                },
                "Random Forest Classifier": {
                    'n_estimators': [50, 100, 200, 500],    # Number of trees
                    'max_depth': [None, 10, 20, 30],         # Depth of trees
                    'min_samples_split': [2, 5, 10],         # Minimum samples to split a node
                    'min_samples_leaf': [1, 2, 4],           # Minimum samples at leaf node
                    'max_features': [None, 'sqrt', 'log2'] # Maximum features to consider at split
                },
                "CatBoosting Classifier": {
                    'iterations': [100, 200, 300],           # Number of boosting iterations
                    'learning_rate': [0.01, 0.05, 0.1],      # Step size for each boosting round
                    'depth': [4, 6, 8],                       # Depth of trees
                    'l2_leaf_reg': [1, 3, 5],                 # L2 regularization coefficient
                    'subsample': [0.6, 0.8, 1]                # Fraction of samples used for fitting each tree
                },
                "SVC": {
                    'C': [0.1, 1, 10, 100,200],                   # Regularization parameter
                    'kernel': ['linear', 'rbf'],              # Kernel type
                    'gamma': ['scale', 'auto'],               # Kernel coefficient
                    'degree': [3, 4, 5]                       # Degree of the polynomial kernel
                },

            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models,param_grids=param_grids)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.7:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            auc=roc_auc_score(y_test,predicted)
            return  auc




        except Exception as e:
            raise CustomException(e,sys)

