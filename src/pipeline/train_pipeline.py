import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_training import ModelTrainer


class TrainPipeline:
    """Class to handle the full classification training pipeline from data ingestion to model training."""

    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        """Executes the full training pipeline."""
        try:
            logging.info("Starting the training pipeline...")

            # Data Ingestion
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion Complete: Train - {train_data_path}, Test - {test_data_path}")

            #  Data Transformation
            train_array, test_array, preprocessor_path = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            logging.info(f"Data Transformation Complete: Preprocessor saved at {preprocessor_path}")

            #  Model Training
            best_model_auc = self.model_trainer.initiate_model_trainer(train_array, test_array, preprocessor_path)
            logging.info(f"Training Complete. Best model AUC: {best_model_auc}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()