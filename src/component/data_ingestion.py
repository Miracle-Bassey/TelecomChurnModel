import os
import sys

from src.component.model_training import ModelTrainer
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.component.data_transformation import DataTransformation
from src.component.data_transformation import DataTransformationConfig

from src.component.model_training import ModelTrainerConfig
from src.component.model_training import ModelTrainer


#create an input to store data

@dataclass
class DataIngestionConfig:
    """
    inputs given to data ingestion component, to know where to save the different train, test and raw data
    """
    train_data_path: str=os.path.join("artifacts","train.csv")
    validation_data_path: str=os.path.join("artifacts","validation.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            df=pd.read_csv("notebook/data/customer_churn.csv")
            logging.info("Read data from csv to dataframe")

            # Drop duplicates
            df = df.drop_duplicates()
            logging.info('Dropped duplicates from the dataset')

            # create data directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # perform stratified train-test split
            logging.info("Train test split initiated")
            target_column ='Churn'
            # Stratified split to maintain the same proportion(distribution) of each class
            # Split into Train (60%) and Temp (40%)
            train_set, X_temp = train_test_split(df, test_size=0.4, random_state=42, stratify=df[target_column])

            # Split Temp into Validation (20%) and Test+Raw (20%)
            val_set, X_temp = train_test_split(X_temp, test_size=0.5, random_state=42, stratify=X_temp[target_column])

            #Split Temp into Test (10%) and Raw (10%)
            test_set, raw_set = train_test_split(X_temp, test_size=0.5, random_state=42, stratify=X_temp[target_column])

            logging.info("Saving Split sets in artifacts")
            # save train and test sets to their respective directories
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            raw_set.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            val_set.to_csv(self.ingestion_config.validation_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.validation_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,val_data,test_data,raw_data =obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,val_arr,test_arr,raw_arr,preprocessor_path = data_transformation.initiate_data_transformation(train_data,val_data,test_data,raw_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,val_arr,test_arr,raw_arr))

