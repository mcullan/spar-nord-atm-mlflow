"""
Workflow for downloading and processing data, then training the forecasting model
and, if necessary, the clustering model.
"""

import os

import click
import mlflow
from mlflow.entities import RunStatus
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils import mlflow_tags
from mlflow.utils.logging_utils import eprint

from util.caching import _get_or_run


@click.command("""Workflow for downloading and processing data, then training the forecasting model
 and, if necessary, the clustering model.""")
@click.option("--cluster_max_iter", default=10, type=int)
@click.option("--n_clusters", default=3, type=int)
@click.option("--model_name", default="baseline", type=str)
@click.option("--train_split_date", default="2017-09-01", type=str)
@click.option("--raw_data_uri", default="/Users/miccull/Downloads/open_dataset.zip")
def workflow(cluster_max_iter, n_clusters, model_name, train_split_date, raw_data_uri):
    """Workflow for downloading and processing data, then training the forecasting model
    and, if necessary, the clustering model."""

    print(train_split_date)
    if model_name == "baseline":
        aggregate_atms = 1
    elif model_name == "cluster":
        print("model name == cluster")
        aggregate_atms = 0


    with mlflow.start_run() as active_run:

        load_raw_data_run = _get_or_run("load_raw_data", {})
        atm_csv_uri = os.path.join(
            load_raw_data_run.info.artifact_uri, "atm_data/atm_data.csv"
        )

        process_data_run = _get_or_run(
            "process_data",
            {
                "model_name": model_name,
                "atm_csv_uri": atm_csv_uri,
                "aggregate_atms": aggregate_atms            },
        )

        df_prophet_uri = os.path.join(
            process_data_run.info.artifact_uri, "df_prophet.csv"
        )

        if model_name == "cluster":
            train_cluster_run = _get_or_run(
                "train_cluster",
                {
                    "n_clusters": n_clusters,
                    "df_prophet_train_uri": df_prophet_uri,
                    "max_iter": str(cluster_max_iter),
                    "train_split_date": train_split_date,
                },
            )

            cluster_model_uri = os.path.join(
                train_cluster_run.info.artifact_uri, "cluster_model"
            )
        else:
            cluster_model_uri = ""


        train_prophet_params = {
            "df_prophet_train_uri": df_prophet_uri,
            "cluster_model_uri": cluster_model_uri,
            "model_name": model_name,
            "train_split_date": train_split_date,
        }
        
        _get_or_run("train_prophet", train_prophet_params, use_cache=False)


if __name__ == "__main__":
    workflow()
