# MLFlow Case Study: Spar Nord Bank ATM Demand

To get some experience using [MLFlow](mlflow.org) for time series analysis projects, I recently started working on a case study with Spar Nord Bank's [dataset](https://sparnordopenbanking.com/OpenData) of all ATM transactions throughout 2017. The data contains 2.5M individual transactions across 108 different ATMs in Denmark. Each transaction encodes details such as the current weather conditions, whether the customer was a Spar Nord account holder, and whether the ATM was in a bank branch. Notably, there are no transaction amounts listed.

I wanted to build a forecasting model to predict demand in terms of cash flow, but I decided to predict the number of transactions since the values were not available. I wanted to compare a few candidate models, and decided on the following scheme:

1. A default forecasting model from the Prophet library as a baseline.
2. Clustering the time series, then training a separate Prophet model on each cluster. Sum values across within-cluster forecasts to get the overall forecast. I used [tslearn](https://tslearn.readthedocs.io/en/stable/user_guide/clustering.html) to perform time series K-Means clustering.

## MLflow

There is much I could have done to further develop these models, but this approach highlights one of the benefits of MLFlow: using and managing modular machine learning models and components. If I found that some clustering-based forecasting model was very useful in predicting demand, then the clusters could be useful for other analyses or visualizations. MLFlow makes it easy to decouple the steps in complicated machine learning workflows with [MLFlow Projects](https://www.mlflow.org/docs/latest/projects.html), and also makes it easy to track and deploy trained models with the [MLFlow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html#registry).

### Workflow:

To keep preprocessing and training steps organized, I used MLFlow's [MLProject](https://www.mlflow.org/docs/latest/projects.html#specifying-projects) file specification. This lets us describe complicated workflows that we can run from the command line with the `mlflow` CLI, and which MLFlow knows how to track separate runs/steps.

The entire MLProject file is shown below. Here's what's going on in there:

```
# MLProject
name: atm_demand

conda_env: conda.yaml

entry_points:
  load_raw_data:
    parameters:
      uri: {type: uri, default: http://localhost:8000/open_dataset.zip}
    command: "python load_raw_data.py --uri {uri}"

  process_data:
    parameters:
      aggregate_atms: {type:int, default: 1}
      model_name: {type:string, default: baseline}
      atm_csv_uri: {type: path, default: /Users/miccull/projects/atm-data/atm-dataset/atm_data.csv}
    command: "python process_data.py --atm_csv_uri {atm_csv_uri} --aggregate_atms {aggregate_atms} --model_name {model_name}"

  train_cluster:
    parameters:
      df_prophet_train_uri: path
      n_clusters: {type: float, default: 3}
      max_iter: {type: float, default: 25}
      train_split_date: {type:string, default: 2017-06-01}
    command: "python clustering/train_cluster.py --df_prophet_train_uri {df_prophet_train_uri} --n_clusters {n_clusters} --max_iter {max_iter} --train_split_date {train_split_date}"

  train_prophet:
    parameters:
      df_prophet_train_uri: path
      model_name: {type:string, default: baseline}
      cluster_model_uri: {type: string, default: ""}
      train_split_date: {type:string, default: 2017-06-01}
    command: "python forecasting/train_prophet.py --df_prophet_train_uri {df_prophet_train_uri} --cluster_model_uri {cluster_model_uri} --model_name {model_name} --train_split_date {train_split_date}"

  main:
    parameters:
      cluster_max_iter: {type: float, default: 25}
      model_name: {type: string, default: baseline}
      train_split_date: {type:string, default: 2017-06-01}
      n_clusters: {type: float, default: 1}
    command: "python main.py --model_name {model_name} --cluster_max_iter {cluster_max_iter} --n_clusters {n_clusters} --train_split_date {train_split_date}"

```



* `conda_env` : This points to a conda environment file in the same directory, which MLflow will use to create a virtual environment for all of the steps in this project. We could also specify a Docker image here, but I found out that Prophet can be difficult to use with Docker because it is very resource intensive and didn't use it here.

* `entry_points`: In here we list the separate steps we need to carry out. For me, these are:

  * `load_raw_data`: Pretty much what it sounds like. Downloads the original zip file, unzips the contents, combines two CSVs of ATM transactions into a single CSV, then saves the data for future use. Note that we have a parameter listed under this entrypoint. It points to the URI of the zip file.

  * `process_data`: The raw data comes in as individual transactions. For forecasting, we need to aggregate these to regular intervals. I chose daily aggregates, but it could have been interesting to try aggregating at the hourly level as well. This grabs the (already downloaded) raw data and aggregates it, then saves it as a new CSV. We have a few parameters here, but I want to point out this crucial one:

    * `aggregate_atms` : By default, this is true. This means that we add up the number of transactions over all ATMs for each day. If we want to train the cluster model, however, we need separate counts for each ATM, so we would need to set this to false (or in this case, 0) and keep the ATMs separate.



  * `train_cluster`: Here, we're training a time series K-Means clustering model to assign each ATM to a particular cluster. A few parameters to note:

    * `df_prophet_train_uri`: This points to the data output from the `process_data` step.
    * `n_clusters`: How many clusters should the model fit?
    * `max_iter` : For how many iterations (maximum) should the algorithm try to improve the clustering? For testing the overall framework of the code, it's helpful to set this number very low.
    * `train_split_date`: The last step, `process_data`, outputs a dataset for the entire year of 2017. I want to try different train_test split points in this experiment, so I specify in this parameter where to end the training set. If we did this in the previous step, we would need to save many copies of the processed dataset.

  * `train_prophet`: Finally, we train the actual forecasting model. For our baseline model, this will just predict the overall forecasts for each day. Our other models with use a clustering model trained in the previous step to segment the data, then make a forecast for each cluster. Some parameters:

    * `df_prophet_train_uri`: See above.
    * `model_name:` This tells us if we should train the baseline model, which doesn't use clustering as a preprocessing step, or the per-cluster model.
    * `cluster_model_uri`: This points to a trained clustering model, which we need if we're training a forecast on each cluster.
    * `train_split_date`: See above.

  * `main`: This lets us run all the steps above from a single command using `mlflow run`.

**Syntax for MLFlow CLI**: If I wanted to run the entire experiment to train a model with 3 clusters, I could use the following command. This assumes we're currently in the same directory as the `MLProject` file.

```
mlflow run . -e main -P n_clusters=3 -P model_name=cluster
```

#### Caching results

Another benefit of separating our steps like this: we can cache the results of intermediate steps so we don't have to repeat them! For instance, we don't have to re-run the expensive preprocessing steps if we want to train new models. To implement this, I adapted code from an [official MLFlow example repository](), which showcases some experimental caching functionality.

So, if I had already run the command shown above, which trains clustering and forecast models for 3 different clusters, and I ran the following:

```
mlflow run . -e main -P n_clusters=4 -P model_name=cluster
```

Then MLFlow wouldn't have to re-run the `load_raw_data` or `process_data` steps, since those are the same for these two models. It would use results from last time and go on to the model training steps.

## Saving trained models

After we train models, there are a few ways we can save them using MLFlow. Here are the two major ways:

* Use `log_model` to save the model as an "artifact" of a run
* Save the model to an `MLFlow Model Registry`

The Model Registry lets us be more organized in developing and maintaining ML models throughout their lifecycle. For instance, let's say we deploy one of these forecasting models so that it can serve predictions on demand. We save a trained model to the registry and set its model stage to "production". Then we create an MLFlow server that serves the forecasting model labeled "production". In the future, if we improve upon our model and want to replace it, we can simply set a newly registered model to "production", and our prediction service will switch over to the new model.

There are some requirements for using the model registry in an MLFlow project. We need our tracking server to work with a database backend, and not just flat files. In my project, I'm using  `SQLite` to get the best of both worlds, but in production I would use a Postgres or MySQL backend.

## Running the experiment:

We can run entry points of our MLProject from the command line, but we can also call them through Python scripts. I wrote a script to perform a medium-scaled version of an experiment with this data and some candidate models. If we run `python fresh_run.py`, then the following will happen:

1. All existing runs and associated data are erased so the project can be re-run from scratch.
2. For a range of training set end dates, train a baseline model (no clusters), and models with 2, 3, and 4 clusters, respectively. By doing this for a long range of dates, we get a sense of how the models improve as we get more data.

## Visualizations

Running our experiment produces an entire database of the results of MLFlow runs. We can see how each model scores, how long each run took to complete, which parameters were used as inputs, and so forth. I'm currently working on creating a dashboard to visualize the results of this experiment, so stay tuned for that!
