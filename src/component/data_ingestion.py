import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


#create an input to store data

@dataclass
class DataIngestionConfig:
    """
    inputs given to data ingestion component, to know where to save the different train, test and raw data
    """
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            df=pd.read_csv(notebook/data/customer_churn.csv)
            logging.info("Read data from csv to dataframe")

            # create data directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # create test split
            logging.info("Train test split initiated")

        except Exception as e:



