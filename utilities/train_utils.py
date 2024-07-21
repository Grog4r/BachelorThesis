import os
import pickle
from datetime import datetime
from typing import Any

import mlflow
import pandas as pd
from dotenv import load_dotenv


def log_model_artifact(model: Any, run_artifact_uri: str, mlflow_run_name: str) -> None:
    """Logs a model artifact to mlflow

    :param model: The model to log
    :param run_artifact_uri: The artifact uri
    :param mlflow_run_name: The mlflow run name
    """
    model_save_path = f"{run_artifact_uri}/{mlflow_run_name}.pickle"
    with open(model_save_path, "wb") as file:
        pickle.dump(model, file)
    mlflow.log_artifact(model_save_path)


def log_feature_importances(model: Any, features: list[str]) -> None:
    """Logs the feature importances as a dict to mlflow

    :param model: The model to log the feature importances for
    :param features: The features to log the importances for
    """
    if model.__class__.__name__ == "LinearRegression":
        feature_importances = model.coef_
    else:
        feature_importances = model.feature_importances_

    feature_importances_dict = {
        feature: f"{feature_importance:.2f}"
        for feature, feature_importance in zip(features, feature_importances)
    }
    mlflow.log_dict(feature_importances_dict, "feature_importances.json")


def get_mlflow_context(
    mlflow_run_name: str | None = None, mlflow_experiment: str | None = None
) -> mlflow.ActiveRun:
    """Gets the context for mlflow

    :param mlflow_run_name: The name of the mlflow run, defaults to None
    :param mlflow_experiment: The name of the mlflow experiment, defaults to None
    :return: The active mlflow run context
    """
    load_dotenv()

    MLFLOW_TRACKING_URL = os.environ.get("MLFLOW_TRACKING_URL")
    print(MLFLOW_TRACKING_URL)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)

    if mlflow_experiment is None:
        mlflow_experiment = "Defalut"
    if mlflow.get_experiment_by_name(mlflow_experiment) is None:
        mlflow.create_experiment(mlflow_experiment)
        print(f"Created experiment {mlflow_experiment}")
    mlflow.set_experiment(mlflow_experiment)

    if mlflow_run_name is None:
        mlflow_run_name = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    return mlflow.start_run(run_name=mlflow_run_name)


def log_df_to_mlflow(
    df: pd.DataFrame, dataset_path: str, dataset_name: str, targets: str, context: str
) -> None:
    """Logs a dataset to mlflow

    :param df: The dataframe to log
    :param dataset_path: The path to the dataset
    :param dataset_name: The name of the dataset
    :param targets: The target col of the dataset
    :param context: The context of the dataset
    """
    dataset = mlflow.data.from_pandas(
        df,
        source=dataset_path,
        name=dataset_name,
        targets=targets,
    )
    mlflow.log_input(dataset, context=context)


def training_dataset_path_to_test_dataset_path(training_dataset_path: str) -> str:
    """Generates the test dataset path from the training dataset path

    :param training_dataset_path: The path to the training dataset
    :return: The resulting test dataset path
    """
    suffix = training_dataset_path.split("|")[-1].split(".parquet")[0]
    return f"{suffix}.test.parquet"


def dataset_path_to_dataset_params(dataset_path: str) -> dict[str, Any]:
    """Analyzes a dataset path to retrieve the dataset parameters

    :param dataset_path: The path of the dataset
    :return: Dictionary of the dataset params and their values
    """
    dataset_path = dataset_path.split(".parquet")[0]
    dataset_params = {}
    for param in dataset_path.split("|"):
        key = param.split("=")[0]
        value = param.split("=")[1]
        dataset_params[key] = value
    return dataset_params
