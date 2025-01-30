import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from scipy.stats import boxcox

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

# Define the named function at the module level
def selective_binary_transform_wrapper(X):
    """Wrapper function for selective_binary_transform."""
    return DataTransformation.selective_binary_transform(X, columns_to_transform=['Tariff Plan', 'Status'])

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def binary_change(x):
        """Convert 1 to 0 and 2 to 1 for binary columns that need it."""
        return np.where(x == 2, 1, np.where(x == 1, 0, x))

    @staticmethod
    def selective_binary_transform(X, columns_to_transform):
        """Apply binary_change only to specified columns."""
        X_transformed = X.copy()
        for i, col in enumerate(columns_to_transform):
            X_transformed[:, i] = DataTransformation.binary_change(X_transformed[:, i])
        return X_transformed

    @staticmethod
    def reflect_transform(x):
        """Apply reflection transformation to 'Subscription Length'."""
        return np.max(x, axis=0) - x

    @staticmethod
    def compute_call_efficiency(X):
        """Compute Call Efficiency = Seconds of Use / Frequency of Use."""
        return (X[:, 0] / np.clip(X[:, 1], 1e-6, None)).reshape(-1, 1)  # Avoid division by zero

    @staticmethod
    def boxcox_transform(x):
        """Apply Box-Cox transformation ensuring positive values."""
        x = np.clip(x, 1e-6, None)  # Ensure values are positive (>0)
        return np.column_stack([boxcox(x[:, i])[0] for i in range(x.shape[1])])

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            # Define feature groups
            binary_columns = ['Tariff Plan', 'Status', 'Complaints']
            ordinal_columns = ['Age Group', 'Charge Amount']
            log_transform_columns = ['Call Failure', 'Frequency of use', 'Frequency of SMS', 'Age']
            boxcox_transform_columns = ['Seconds of Use', 'Distinct Called Numbers', 'Customer Value']
            reflect_transform_columns = ['Subscription Length']

            # Binary pipeline
            binary_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("selective_binary", FunctionTransformer(
                        selective_binary_transform_wrapper,  # Use the global function
                        validate=False
                    ))
                ]
            )

            # Ordinal pipeline
            ordinal_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", RobustScaler())
                ]
            )

            # Call efficiency pipeline
            call_efficiency_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("call_eff", FunctionTransformer(self.compute_call_efficiency, validate=False)),
                    ("log", FunctionTransformer(np.log1p, validate=False)),
                    ("scaler", RobustScaler())
                ]
            )

            # Log transform pipeline
            log_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("log_transform", FunctionTransformer(np.log1p, validate=False)),
                    ("scaler", RobustScaler())
                ]
            )

            # Box-Cox transform pipeline
            boxcox_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("boxcox_transform", FunctionTransformer(self.boxcox_transform, validate=False)),
                    ("scaler", RobustScaler())
                ]
            )

            # Reflect and log transform pipeline
            reflect_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("reflect_transform", FunctionTransformer(self.reflect_transform, validate=False)),
                    ("log_transform", FunctionTransformer(np.log1p, validate=False)),
                    ("scaler", RobustScaler())
                ]
            )

            logging.info(f"Binary columns: {binary_columns}")
            logging.info(f"Ordinal columns: {ordinal_columns}")
            logging.info(f"Log transform columns: {log_transform_columns}")
            logging.info(f"Box-Cox transform columns: {boxcox_transform_columns}")
            logging.info(f"Reflect transform columns: {reflect_transform_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("binary_pipeline", binary_pipeline, binary_columns),
                    ("ordinal_pipeline", ordinal_pipeline, ordinal_columns),
                    ("call_efficiency_pipeline", call_efficiency_pipeline, ['Seconds of Use', 'Frequency of use']),
                    ("log_pipeline", log_pipeline, log_transform_columns),
                    ("boxcox_pipeline", boxcox_pipeline, boxcox_transform_columns),
                    ("reflect_pipeline", reflect_pipeline, reflect_transform_columns)
                ],
                remainder='drop'  # Drop columns not explicitly transformed
            )

            logging.info("Preprocessing object created successfully")
            return preprocessor

        except Exception as e:
            logging.error("Error in creating preprocessing object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function initiates the data transformation process.
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Churn"  # Replace with your actual target column name
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.error("Error in data transformation")
            raise CustomException(e, sys)