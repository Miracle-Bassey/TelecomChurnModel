import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler

from src.pipeline.pred_pipeline import CustomData,PredictionPipeline
from src.logger import logging

application=Flask(__name__)

app = application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_churn',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home2.html') # the data fields needto make the prediction
    else:
        data=CustomData(
            call_failure= int(request.form.get('Call Failure')),
            complaints= int(request.form.get('Complaints')),
            subscription_length= int(request.form.get('Subscription Length')),
            charge_amount= int(request.form.get('Charge Amount')),
            seconds_of_use= int(request.form.get('Seconds of Use')),
            frequency_of_use= int(request.form.get('Frequency of use')),
            frequency_of_sms= int(request.form.get('Frequency of SMS')),
            distinct_called_numbers=int(request.form.get('Distinct Called Numbers')),
            age_group= int(request.form.get('Age Group')),
            tariff_plan= int(request.form.get('Tariff Plan')),
            status= int(request.form.get('Status')),
            age= int(request.form.get('Age')),
            customer_value=float(request.form.get('Customer Value')),
        )

        logging.info("Preparing Data into Dataframe for Prediction")
        pred_df=data.get_data_as_df()
        print(pred_df)

        logging.info("Prediction Started")
        prediction_pipeline=PredictionPipeline()
        results=prediction_pipeline.predict(pred_df)
        logging.info("Prediction Completed")
        return render_template('home.html',results=results[0])


if __name__ == '__main__':
    app.run(host="0.0.0.0")


