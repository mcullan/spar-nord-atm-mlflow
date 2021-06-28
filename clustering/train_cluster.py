"""
Trains a TimeSeriesKMeans model on the total number of daily(or given frequency) transactions at each ATM.
TODO: I had dynamic time warping before, but now that I'm filling in dates I probably no longer need this.
Might run faster and even perform better if I use a different thing.

TODO: Could/should this use an sklearn wrapper instead of a Pyfunc wrapper? Unclear if tslearn is an issue for that.
May be faster/easier though.

TODO: Should I make this its own MLProject file? I think I can make a separate one in this directory without interfering
with the one in the base directory.
"""
import click

import mlflow
import mlflow.pyfunc  # Is this necessary? Separate module from base MLflow?

import pandas as pd

import sklearn
from sklearn.pipeline import Pipeline
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from sklearn.base import BaseEstimator, TransformerMixin

import tslearn


CONDA_ENV = {
    "channels": ["defaults", "conda-forge"],
    "dependencies": [
        "tslearn={}".format(tslearn.__version__),
        "sklearn={}".format(sklearn.__version__),
    ],
    "name": "clustermodel_env",
}


class DemandPivotTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Converts long-form DF with (atm_id, ds, count) columns to a matrix in which
        each row is a separate time series.
        """
        print(X.head)

        demand_pivot = X.pivot(index="ds", columns="atm_id", values="y")
        ts = demand_pivot.fillna(0).T.values.reshape(
            (demand_pivot.shape[1], demand_pivot.shape[0], 1)
        )

        return ts


class TSClusterWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def load_context(self, context):
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance
        from tslearn.clustering import TimeSeriesKMeans
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.pipeline import Pipeline

        # TODO: Is this the right way to do this? Is it necessary?
        class DemandPivotTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                demand_pivot = X.pivot(index="ds", columns="atm_id", values="y")
                ts = demand_pivot.fillna(0).T.values.reshape(
                    (demand_pivot.shape[1], demand_pivot.shape[0], 1)
                )

                return ts

        return

    def predict(self, context, model_input):

        y_pred = self.model.predict(model_input)
        atm_clusters = (
            pd.Series(
                y_pred,
                index=model_input.pivot(
                    index="ds", columns="atm_id", values="y"
                ).columns,
            )
            .rename(f"cluster")
            .reset_index()
        )

        return atm_clusters


def get_cluster_model(df, n_clusters, max_iter=25):
    """Given an (atm_id, ds, counts) DataFrame, transforms to appropriate data shape to fit
    a TimeSeriesKMeans model. Returns the fitted model as well as the (atm_id, cluster) DataFrame."""
    pipe = Pipeline(
        [
            ("pivot", DemandPivotTransformer()),
            ("scaler", TimeSeriesScalerMeanVariance()),
            (
                "cluster",
                TimeSeriesKMeans(
                    n_clusters=n_clusters,
                    metric="softdtw",  # TODO: Other metrics
                    metric_params={"gamma": 0.01},  # TODO: Hyperparameter search
                    max_iter=max_iter,
                ),
            ),
        ]
    ).fit(df)

    # Predicted clusters.
    y_pred = pipe.predict(df)

    # Creates DataFrame matching each atm_id with its predicted cluster.
    atm_clusters = (
        pd.Series(
            y_pred, index=df.pivot(index="ds", columns="atm_id", values="y").columns
        )
        .rename(f"cluster")
        .reset_index()
    )

    return pipe, atm_clusters


@click.command(
    help="Creates clusters of ATMs based on time-series data describing daily transactions at each ATM."
)
@click.option("--df_prophet_train_uri")
@click.option("--n_clusters")
@click.option("--max_iter")
@click.option("--train_split_date")
def train_cluster(df_prophet_train_uri, n_clusters, max_iter, train_split_date):
    """Fits a TimeSeriesKMeans model to daily(or other frequency) ATM transaction counts."""

    n_clusters = int(n_clusters)
    max_iter = int(max_iter)

    df_all = pd.read_csv(df_prophet_train_uri)
    df_all["ds"] = pd.to_datetime(df_all["ds"])

    train_split = df_all["ds"] < train_split_date
    df = df_all[train_split]
    df_test = df_all[~train_split]
    print("data processed")
    if int(n_clusters) < 2:
        raise ValueError(
            f"Parameter n_clusters must be at least 2. The value {n_clusters} was passed."
        )

    # TODO: do I need to name the run object if I don't reference it by name?
    with mlflow.start_run() as mlrun:
        run_id = mlflow.active_run().info.run_id

        mlflow.log_param("start_date", df["ds"].min())
        mlflow.log_param("end_date", df["ds"].max())

        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("max_iter", max_iter)

        cluster_model, atm_cluster = get_cluster_model(df, n_clusters, max_iter)

        mlflow.log_text(atm_cluster.to_csv(index=False), "atm_cluster.csv")

        mlflow.pyfunc.log_model(
            "cluster_model",
            conda_env=CONDA_ENV,
            python_model=TSClusterWrapper(cluster_model),
            registered_model_name=f"atm-cluster-model",
        )

        # Saving a seprately named model for each number of clusters. Should I cut one of these?
        mlflow.pyfunc.log_model(
            "cluster_model_n",
            conda_env=CONDA_ENV,
            python_model=TSClusterWrapper(cluster_model),
            registered_model_name=f"atm-cluster-model_{n_clusters}",
        )

        print(
            "Logged model with URI: runs:/{run_id}/cluster_model".format(run_id=run_id)
        )


if __name__ == "__main__":
    train_cluster()
