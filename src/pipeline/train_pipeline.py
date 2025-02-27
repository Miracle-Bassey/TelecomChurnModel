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
            train_data_path,val_data_path,test_data_path,raw_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion Complete: Train - {train_data_path}, \n Validation- {val_data_path}, \nTest - {test_data_path} \n and Raw- {raw_data_path}")

            #  Data Transformation
            train_array,val_array,test_array,raw_array,preprocessor_path = self.data_transformation.initiate_data_transformation(train_data_path,val_data_path,test_data_path,raw_data_path)
            logging.info(f"Data Transformation Complete: Preprocessor saved at {preprocessor_path}")

            #  Model Training
            report, probability_stats, variance_scores, raw_pred, raw_pred_proba, raw_class_distribution, model = self.model_trainer.initiate_model_trainer(train_array,val_array,test_array,raw_array)
            logging.info(f"Training Complete. Report: {report} \n"
                         f"prob stats : {probability_stats} \n"
                         f"Variance scores : {variance_scores} \n"
                         f"raw prediction: {raw_pred} \n "
                         f"raw prediction probabilities: {raw_pred_proba} \n")


        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()