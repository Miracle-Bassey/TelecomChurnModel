import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path ='artifacts\model.pkl'
            preprocessor_path ='artifacts\preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds= model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 call_failure: int,
                 complaints: int,
                 subscription_length: int,
                 charge_amount: int,
                 seconds_of_use: int,
                 frequency_of_use: int,
                 frequency_of_sms: int,
                 distinct_called_numbers: int,
                 age_group: int,
                 tariff_plan: int,
                 status: int,
                 age: int,
                 customer_value: float):
        self.call_failure = call_failure
        self.complaints = complaints
        self.subscription_length = subscription_length
        self.charge_amount = charge_amount
        self.seconds_of_use = seconds_of_use
        self.frequency_of_use = frequency_of_use
        self.frequency_of_sms = frequency_of_sms
        self.distinct_called_numbers = distinct_called_numbers
        self.age_group = age_group
        self.tariff_plan = tariff_plan
        self.status = status
        self.age = age
        self.customer_value = customer_value


    def get_data_as_df(self):
        try:
            """ creates a dictionary of, and then turn that to a df"""
            custom_data_input_dict = {
                'Call Failure': self.call_failure,
                'Complaints': self.complaints,
                'Subscription Length': self.subscription_length,
                'Charge Amount': self.charge_amount,
                'Seconds of Use': self.seconds_of_use,
                'Frequency of use': self.frequency_of_use,
                'Frequency of SMS': self.frequency_of_sms,
                'Distinct Called Numbers': self.distinct_called_numbers,
                'Age Group': self.age_group,
                'Tariff Plan': self.tariff_plan,
                'Status': self.status,
                'Age': self.age,
                'Customer Value': self.customer_value

            }

            return pd.DataFrame(custom_data_input_dict, index=[0])
        except Exception as e:
            raise CustomException(e, sys)


