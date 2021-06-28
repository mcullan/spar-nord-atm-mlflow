"""
Trains baseline and clustering-based Prophet models on ATM Data.
"""

import mlflow
import pandas as pd
import os
import shutil


# We want this script to start everything fresh, so we delete any existing runs
if os.path.exists('./mlruns.db'):
    os.remove('./mlruns.db')

if os.path.exists('./mlruns') and os.path.isdir('./mlruns'):
    shutil.rmtree('./mlruns')

mlflow.set_tracking_uri("sqlite:///mlruns.db")

for date in pd.date_range(start="2017-06-01", end="2017-11-30"):
    # BASELINE: No Clustering
    print(date)
    mlflow.projects.run(uri=".", entry_point="main", parameters={"model_name": "baseline", "train_split_date":date})

    # CLUSTER: Trains TS clustering model, then fits Prophet to each cluster and sums forecasts.
    for n_clusters in range(2, 5):
        mlflow.projects.run(
            uri=".",
            entry_point="main",
            parameters={
                "cluster_max_iter": 10,
                "n_clusters": n_clusters,
                "model_name": "cluster",
                "train_split_date": date
            },
        )
