"""
https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/main.py

Attempts caching between steps. Pulled from MLFlow multi-step workflow example.
"""

import click
import os


import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint

from mlflow.tracking.fluent import _get_experiment_id


def _already_ran(entry_point_name, parameters, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    print(f"\n\n\n\n\n\n\n\n\n{entry_point_name}:")
    for param in parameters:
        print(f"{param}: {type(param)}")

    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        print("\n\n\n\n\n\n\n")
        for param_key, param_value in parameters.items():
            run_value = full_run.data.params.get(param_key)
            if type(param_value) == int:
                try:
                    run_value = int(run_value)
                    print("changed")

                except:
                    print("failed")
            if type(param_value) == str and type(run_value) == str:
                try:
                    run_value = run_value.strip("'")
                    print("changed")

                except:
                    print("failed")

            print(f"{param_key} SAVED: {param_value}: {type(param_value)}")
            print(f"{param_key} RUN: {run_value}: {type(run_value)}")
            print(f"EQUAL: {param_value == run_value}")
            print(f"TYPE EQUAL: {type(param_value) == type(run_value)}")
            print("\n\n\n\n\n\n")



            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.to_proto().status != RunStatus.FINISHED:
            eprint(
                (
                    "Run matched, but is not FINISHED, so skipping "
                    "(run_id=%s, status=%s)"
                )
                % (run_info.run_id, run_info.status)
            )
            continue

        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


def _get_or_run(entrypoint, parameters, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters)
    if use_cache and existing_run:
        print(
            "Found existing run for entrypoint=%s and parameters=%s"
            % (entrypoint, parameters)
        )
        return existing_run
    print(
        "Launching new run for entrypoint=%s and parameters=%s"
        % (entrypoint, parameters)
    )
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


if __name__ == "__main__":
    workflow()
