"""
Trains a Facebook Prophet forecasting model on aggregated transaction data.

TODO:
Imports individual model training methods from scripts in this directory.
This seemed like an effective way of separating different models and adding new ones, but it
means I have to edit this file every time I want to add new model types.
How could I improve this? Should I just be keeping them in separate git branches?
That seems inconvenient if I plan on using both.
"""

import click
import cloudpickle

import mlflow
import mlflow.pyfunc

import pandas as pd

import prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics


@click.command(help="Trains a Prophet model to predict ATM demand")
@click.option("--df_prophet_train_uri")
@click.option("--model_name")
@click.option("--cluster_model_uri")
@click.option("--train_split_date")
def train_prophet(df_prophet_train_uri, model_name, cluster_model_uri, train_split_date):
    """Trains Prophet forecasting model. Actual training/model spec is contained in individual scripts for different model types."""

    train_prophet_params = {
        "df_prophet_train_uri": df_prophet_train_uri,
        "cluster_model_uri": cluster_model_uri,
        "model_name": model_name,
        "train_split_date": train_split_date,
    }
    print("\n\n\n\n TRAIN PROPHET PARAMS")
    print(train_prophet_params)
    print("\n\n\n\n TRAIN PROPHET PARAMS")


    df_all = pd.read_csv(df_prophet_train_uri)
    df_all["ds"] = pd.to_datetime(df_all["ds"])

    print(df_all.head())
    print(train_split_date)
    train_split = df_all["ds"] < pd.to_datetime(train_split_date)
    df = df_all[train_split]
    df_test = df_all[~train_split]

    #
    # df_prophet_test_uri = df_prophet_train_uri.replace(
    #     "/df_prophet_train.csv", "/df_prophet_test.csv")
    # df_test = pd.read_csv(df_prophet_test_uri)
    # df_test["ds"] = pd.to_datetime(df_test["ds"])

    with mlflow.start_run() as mlrun:
        if model_name == "baseline":
            from baseline_model import train

            mlflow.log_param("n_clusters", None)

            train(df, df_test)

        elif model_name == "cluster":
            from prophet_cluster import train

            atm_cluster_uri = cluster_model_uri.replace(
                "/cluster_model", "/atm_cluster.csv"
            )

            atm_cluster = pd.read_csv(atm_cluster_uri)
            n_clusters = atm_cluster["cluster"].nunique()
            mlflow.log_param("n_clusters", n_clusters)

            df = df.merge(atm_cluster)

            df_test = df_test.groupby('ds')['y'].sum().reset_index()
            train(df, df_test)


if __name__ == "__main__":
    train_prophet()
