"""
Trains a baseline (no clusters) Facebook Prophet forecasting model on aggregated total ATM transactions.
"""

import cloudpickle

import mlflow
import mlflow.pyfunc

import numpy as np  # Might not need this
import pandas as pd

import prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def load_context(self, context):
        from prophet import Prophet

        return

    def predict(self, context, model_input):
        future = self.model.make_future_dataframe(periods=model_input["periods"][0])
        return self.model.predict(future)[["ds", "yhat"]]


def train(df, df_test):

    CONDA_ENV = {
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            "prophet={}".format(prophet.__version__),
            "cloudpickle={}".format(cloudpickle.__version__),
        ],
        "name": "baselinemodel_env",
    }

    m = Prophet()
    m.fit(df)

    # Perform cross-validation and get metrics.
    # Starting 180 days into the data, splits using on 7 day intervals, and predicts the next 30 days.


    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    resid = (df_test.set_index('ds')['y'] - forecast.set_index('ds')['yhat']).dropna()

    for step, (date, residual) in enumerate(resid.items()):
        mlflow.log_metric(key="residual_daily", value=residual, step=step+1)

    mse = (resid ** 2).mean()
    mae = resid.abs().mean()
    rmse = mse ** .5

    mlflow.log_metric(key="mse", value=mse)
    mlflow.log_metric(key="rmse", value=rmse)
    mlflow.log_metric(key="mae", value=mae)


    # Log parameters
    mlflow.log_param("start_date", df["ds"].min())
    mlflow.log_param("end_date", df["ds"].max())

    # Log metrics

    # Log artifacts
    mlflow.log_text(forecast.to_csv(index=False), "forecast.csv")

    # Log model
    mlflow.pyfunc.log_model(
        "prophet_model",
        conda_env=CONDA_ENV,
        python_model=ProphetWrapper(m),
        registered_model_name="forecast-baseline-model",
    )

    run_id = mlflow.active_run().info.run_id
    print("Logged model with URI: runs:/{run_id}/prophet_model".format(run_id=run_id))


if __name__ == "__main__":
    train()
