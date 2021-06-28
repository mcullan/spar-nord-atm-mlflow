"""Uses clusters of ATMs to train a separate Prophet forecasting model on each. The final model uses
the per-cluster predictions and sums them up.

TODO: Compute variance / confidence intervals for sum of forecasts?
"""

import cloudpickle

import numpy as np
import pandas as pd

import mlflow
import mlflow.pyfunc

import prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

from sklearn.pipeline import Pipeline
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn
import tslearn

conda_env = {
    "channels": ["defaults", "conda-forge"],
    "dependencies": [
        "prophet={}".format(prophet.__version__),
        "cloudpickle={}".format(cloudpickle.__version__),
        "tslearn={}".format(tslearn.__version__),
        "sklearn={}".format(sklearn.__version__),
    ],
    "name": "clustermodel_env",
}


class TSClusterProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, models_dict):
        self.models_dict = models_dict
        super().__init__()

    def load_context(self, context):
        from prophet import Prophet

    def predict(self, context, model_input):

        forecasts = {}
        for cluster_label, m in self.models_dict.items():
            future = m.make_future_dataframe(periods=30)
            cluster_forecast = m.predict(future)
            forecasts[cluster_label] = cluster_forecast[["ds", "yhat"]]

        forecast = pd.concat(list(forecasts.values())).groupby("ds").sum().sum(axis=1)
        return forecast


def train(df_train, df_test):
    """Store each cluster's model, forecast, and cross-validation matrix in a dictionary."""
    models = {}
    forecasts = {}

    mlflow.log_param("start_date", df_train["ds"].min())
    mlflow.log_param("end_date", df_train["ds"].max())

    # TODO: Improve
    for cluster_label, df_ in df_train.groupby("cluster"):

        df_ = (
            df_[["ds", "y"]].groupby(pd.Grouper(key="ds", freq="D")).sum().reset_index()
        )

        m = Prophet().fit(df_)

        # Should I be doing this in the training step? I'm only generating a forecast once,
        # so it might make sense to do this instead of making predictions separately.
        # Also, could probably just `make_future_dataframe` once, but I doubt it's slowing me down much.
        future = m.make_future_dataframe(periods=30, include_history=False)
        cluster_forecast = m.predict(future)
        cluster_forecast["cluster"] = cluster_label

        forecasts[cluster_label] = cluster_forecast

    forecast = pd.concat(list(forecasts.values())).groupby("ds")["yhat"].sum()


    resid = (df_test.set_index('ds')['y'] - forecast).dropna()
    # print("\n\n\n\n\n\n ENTER LOGGING")
    for step, (date, residual) in enumerate(resid.items()):
        print(f"\n\n\n\n\nLOGGING {step+1}: {residual}\n\n\n")
        print()
        mlflow.log_metric(key="residual_daily", value=residual, step=(step+1))

    mse = (resid ** 2).mean()
    mae = resid.abs().mean()
    rmse = mse ** .5

    mlflow.log_metric(key="mse", value=mse)
    mlflow.log_metric(key="rmse", value=rmse)
    mlflow.log_metric(key="mae", value=mae)


    # Aggregating yhat and y from individual cross validation results.
    # Can I do this with other computed values?

    # Performance metrics: Mean Squared Error(MSE), Root Mean Squared Error (RMSE) , and Mean Absolute Error (MAE)
    # df_p_all = (
    #     df_cv_all.groupby("horizon")
    #     .agg(
    #         mse=("se", "mean"),
    #         rmse=("se", lambda x: x.mean() ** 0.5),
    #         mae=("ae", "mean"),
    #     )
    #     .rolling(window=3)
    #     .mean()
    #     .dropna()
    #     .reset_index()
    # )

    # Need to double check what it means to just use the first row here.
    # I might want to do a sequence for this? I think MLflow has a thing for that.
    # Also, TODO: Is there like a cumulative sum metric?
    #
    # mlflow.log_metric("rmse", df_p_all.loc[0, "rmse"])

    mlflow.log_text(forecast.to_csv(index=False), "forecast.csv")
    # mlflow.log_text(df_p_all.to_csv(index=False), "df_p.csv")

    mlflow.pyfunc.log_model(
        "prophet_model",
        conda_env=conda_env,
        python_model=TSClusterProphetWrapper(models),
        registered_model_name="forecast-cluster-model",
    )

    # Log separately that it's based on a certain number of clusters. Not sure if this is useful yet.
    mlflow.pyfunc.log_model(
        "prophet_model_numbered",
        conda_env=conda_env,
        python_model=TSClusterProphetWrapper(models),
        registered_model_name=f"forecast-cluster-model-{df_train['cluster'].nunique()}",
    )

    run_id = mlflow.active_run().info.run_id
    print("Logged model with URI: runs:/{run_id}/prophet_model".format(run_id=run_id))


if __name__ == "__main__":
    train()
