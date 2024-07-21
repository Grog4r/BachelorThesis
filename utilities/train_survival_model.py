import os
from datetime import datetime
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lifelines.utils import concordance_index
from numpy.typing import ArrayLike
from sklearn.model_selection import GroupShuffleSplit
from sksurv.ensemble import RandomSurvivalForest
from sksurv.functions import StepFunction
from lifelines.fitters.coxph_fitter import CoxPHFitter
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv


load_dotenv()
if __name__ == "__main__" or os.environ.get("EXPERIMENT") == "True":
    from train_utils import get_mlflow_context, log_df_to_mlflow, log_model_artifact
else:
    from utilities.train_utils import (
        get_mlflow_context,
        log_df_to_mlflow,
        log_model_artifact,
    )


RANDOM_STATE = int(os.environ.get("RANDOM_STATE"))


def X_y_to_df(X: pd.DataFrame, y: ArrayLike) -> pd.DataFrame:
    """Converts an X and a y dataset into a single dataframe.

    :param X: The X DataFrame
    :param y: The y ArrayLike
    :return: The resulting DataFrame
    """
    return pd.concat([X, pd.DataFrame(y, index=X.index)], axis="columns")


def X_y_to_train_test(
    X: pd.DataFrame,
    y: ArrayLike,
    train_size: float = 0.7,
    time_column: str = "duration",
    event_column: str = "event",
) -> tuple[pd.DataFrame, pd.DataFrame, ArrayLike, ArrayLike]:
    """Converts an X and a y dataset into group shuffeled test and train datasets

    :param X: The X dataset
    :param y: The y dataset
    :param train_size: The size for the training data, defaults to 0.7
    :param time_column: The column where the time is in, defaults to "duration"
    :return: X_train, X_test, y_train, y_test
    """
    X = X.copy()

    gss = GroupShuffleSplit(
        n_splits=1, train_size=train_size, random_state=RANDOM_STATE
    )

    train_index, test_index = next(gss.split(X, y, groups=X["device_uuid"]))

    if "device_uuid" in X.columns:
        X.drop(columns=["device_uuid"], inplace=True)

    X_train = X.loc[train_index]
    y_train = y[train_index]

    X_test = X.loc[test_index]
    y_test = y[test_index]

    tr_min = y_train[time_column].min()
    tr_max = y_train[time_column].max()
    ts_max = y_test[time_column].max()

    if ts_max > tr_max:
        # Identify rows that will be clipped
        clipped_indices = (y_test[time_column] < tr_min) | (
            y_test[time_column] > tr_max
        )

        # Change the event_column to be False for these rows
        y_test[event_column][clipped_indices] = False

        y_test[time_column] = np.clip(y_test[time_column], tr_min, tr_max)

    return X_train, X_test, y_train, y_test


def df_to_X_y(
    df: pd.DataFrame,
    features: list[str],
    event_column: str = "event",
    time_column: str = "duration",
    keep_device_uuid: bool = True,
) -> tuple[pd.DataFrame, ArrayLike]:
    """Splits a dataframe into X and y by some given features and columns.

    :param df: The source DataFrame
    :param features: The features that go into X
    :param event_column: The column for events, defaults to "event"
    :param time_column: The column for time, defaults to "duration"
    :return: X, y
    """
    features = features.copy()
    if keep_device_uuid:
        if "device_uuid" not in features:
            features.append("device_uuid")
    else:
        if "device_uuid" in features:
            features.remove("device_uuid")

    df.reset_index(drop=True, inplace=True)

    features = list(set(df.columns).intersection(set(features)))

    X = df[features]
    y = Surv.from_dataframe(
        event=event_column,
        time=time_column,
        data=df[[event_column, time_column]],
    )

    return X, y


def print_censoring_metrics(
    y_train: ArrayLike, y_test: ArrayLike, event_column: str = "event"
) -> None:
    """Prints metrics about the amount of censoring in train and test data

    :param y_train: The train data
    :param y_test: The test data
    :param event_column: The event column, defaults to "event"
    """
    n_total_train = len(y_train)
    n_censored_train = (
        len(y_train[y_train[event_column] == False]) / n_total_train * 100
    )
    n_total_test = len(y_test)
    n_censored_test = len(y_test[y_test[event_column] == False]) / n_total_test * 100

    print(f"Train Size: {n_total_train}; Train Censored Ratio: {n_censored_train:.1f}%")
    print(f"TestSize: {n_total_test}; Test Censored Ratio: {n_censored_test:.1f}%")


def calculate_integrated_brier_score(
    y_train: ArrayLike,
    X_test: pd.DataFrame,
    y_test: ArrayLike,
    model: Any,
    model_class_name: str,
) -> float:
    """Calculates the integrated brier score for some model

    :param y_train: The y_train dataset
    :param X_test: The X_test dataset
    :param y_test: The y_test dataset
    :param model: The model to test
    :param model_class_name: The name of the model class
    :return: _description_
    """
    lower, upper = np.percentile(y_test["duration"], [10, 90])
    lower = int(lower)
    upper = int(upper)
    times = np.arange(lower, upper - 1)
    if model_class_name == "kaplan_meier_estimator":
        surv_probs = np.tile(model(times), (y_test.shape[0], 1))
        try:
            ibs = integrated_brier_score(y_train, y_test, surv_probs, times)
        except ValueError:
            print(y_train, y_test, surv_probs, times)
            raise
    else:
        try:
            if model_class_name == "CoxPHFitter":
                surv_probs = np.array(model.predict_survival_function(X_test, times)).T

            else:
                surv_probs = np.row_stack(
                    [
                        step_fn(times)
                        for step_fn in model.predict_survival_function(X_test)
                    ]
                )

            ibs = integrated_brier_score(y_train, y_test, surv_probs, times)
        except ValueError:
            return 1.0
    return ibs


def train_survival_model(
    model_class: Any,
    params: dict[str, Any],
    features: list[str],
    dataset: pd.DataFrame | None = None,
    dataset_path: str | None = None,
    mlflow_run_name: str | None = None,
    event_column: str = "event",
    time_column: str = "duration",
) -> None:
    """Trains a survival model

    :param model_class: The class of the model to train. Can be one of
        kaplan_meier_estimator, CoxPHFitter, RandomSurvivalForest
    :param params: The params for the model
    :param features: The features to train on
    :param dataset: The dataset to train the model on, if this is None, the dataset_path will be used, defaults to None
    :param dataset_path: The path to the dataset, if this is None and the dataset is also None the function will throw an exception, defaults to None
    :param mlflow_run_name: The name for the mlflow run, if this is None it will be
        generating a name automatically, defaults to None
    :param event_column: The column for the event, defaults to "event"
    :param time_column: The column for the time, defaults to "duration"

    :raises ValueError: When both the dataset and the dataset_path are None
    """
    if dataset is not None:
        df = dataset.copy()
    elif dataset_path is not None:
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError("Either dataset_path or dataset must be specified.")

    X, y = df_to_X_y(
        df, features=features, event_column=event_column, time_column=time_column
    )
    features.remove("device_uuid")
    (X_train, X_test, y_train, y_test) = X_y_to_train_test(
        X, y, time_column=time_column
    )

    print_censoring_metrics(y_train, y_test, event_column=event_column)

    for col in X_train.columns:
        if X_train[col].nunique() == 1:
            print(f"The column {col} only has 1 unique value!!!")
            X_train.drop(columns=[col])
            X_test.drop(columns=[col])

    model_class_name = model_class.__name__
    if mlflow_run_name is None:
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        mlflow_run_name = f"{model_class_name}_{now}"
        for key, value in params.items():
            mlflow_run_name += f"_{key}={value}"
        if dataset_path is not None:
            dataset_name = dataset_path.split("/")[-1].split(".")[0]
            mlflow_run_name += f"_{dataset_name}"

    mlflow_experiment = os.environ.get("MLFLOW_SURVIVAL_EXPERIMENT")
    with get_mlflow_context(
        mlflow_run_name=mlflow_run_name, mlflow_experiment=mlflow_experiment
    ) as run:
        if model_class_name == "kaplan_meier_estimator":
            # Kaplan Meier
            model = StepFunction(
                *model_class(y_train[event_column], y_train[time_column])
            )

            ibs = calculate_integrated_brier_score(
                y_train, X_test, y_test, model, model_class_name
            )
            c_index_censored = c_index_ipcw = 0.5
        else:
            # Cox PH and Random Survival Forest
            if model_class_name == "RandomSurvivalForest":
                model = model_class(**params, random_state=RANDOM_STATE)
            elif model_class_name == "CoxPHFitter":
                model = model_class(**params)
                df_train = X_y_to_df(X_train, y_train)
                model.fit(
                    df_train[[*features, event_column, time_column]],
                    duration_col=time_column,
                    event_col=event_column,
                )

                try:
                    ibs = calculate_integrated_brier_score(
                        y_train,
                        X_test,
                        y_test,
                        model,
                        model_class_name,
                    )
                except:
                    raise

                try:
                    c_index_censored = concordance_index(
                        test_df["duration"],
                        -model.predict_partial_hazard(test_df),
                        test_df["event"],
                    )
                except:
                    c_index_censored = 0.0

                try:
                    prediction = model.predict_partial_hazard(X_test)
                    c_index_ipcw = concordance_index_ipcw(
                        y_train,
                        y_test,
                        prediction,
                    )[0]
                except:
                    c_index_ipcw = c_index_censored
            else:
                model = model_class(**params)
                model.fit(X_train, y_train)

            ibs = calculate_integrated_brier_score(
                y_train, X_test, y_test, model, model_class_name
            )
            prediction = model.predict(X_test)
            c_index_censored = concordance_index_censored(
                y_test[event_column],
                y_test[time_column],
                prediction,
            )[0]
            try:
                c_index_ipcw = concordance_index_ipcw(
                    y_train,
                    y_test,
                    prediction,
                )[0]
            except:
                c_index_ipcw = c_index_censored

        metrics = {
            "ibs": ibs,
            "c_index_censored": c_index_censored,
            "c_index_ipcw": c_index_ipcw,
        }
        mlflow.log_metrics(metrics)

        mlflow.log_params(params)

        log_df_to_mlflow(
            X_y_to_df(X_train, y_train),
            dataset_path,
            "Survival Dataset Train",
            time_column,
            "training",
        )

        log_df_to_mlflow(
            X_y_to_df(X_test, y_test),
            dataset_path,
            "Survival Dataset Test",
            time_column,
            "testing",
        )

        run_artifact_uri = run.info.artifact_uri
        log_model_artifact(model, run_artifact_uri, mlflow_run_name)

        mlflow.set_tag(key="model_type", value=model_class_name)


def experiment_train_survival_model(
    model_class: Any,
    train_df_path: str,
    test_df_path: str,
    model_params: dict[str, Any],
    dataset_params: dict[str, Any],
    features: list[str],
    mlflow_run_name: str | None = None,
    event_column: str = "event",
    time_column: str = "duration",
) -> None:
    """Trains a survival model specifically for an experiment. For example the
        training and test dataset paths are separate here.

    :param model_class: The class of the model to train. Can be one of
        kaplan_meier_estimator, CoxPHFitter, RandomSurvivalForest
    :param train_df_path: The path to the train dataset
    :param test_df_path: The path to the test dataset
    :param model_params: The parameters for the model
    :param dataset_params: The parameters of the train dataset
    :param features: The features to train the model on
    :param mlflow_run_name: The name for the mlflow run, if this is None it will
        automatically generate a name, defaults to None
    :param event_column: The column for the event, defaults to "event"
    :param time_column: The column for the time, defaults to "duration"
    """
    model_class_name = model_class.__name__

    mlflow_experiment = os.environ.get("MLFLOW_SURVIVAL_EXPERIMENT")
    if mlflow_run_name is None:
        dataset_name = train_df_path.split("/")[-1].split(".parquet")[0]
        mlflow_run_name = f"{model_class_name}_{dataset_name}"
        for key, value in model_params.items():
            mlflow_run_name += f"_{key}={value}"

    # Check if the run already exists and if so, skip it
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URL"))
    experiment = mlflow.get_experiment_by_name(mlflow_experiment)
    if experiment is not None:
        mlflow_run = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"run_name='{mlflow_run_name}' AND status='FINISHED'",
        )
        if len(mlflow_run) > 0:
            print(f"Already did run {mlflow_run_name}. Skipping...")
            return

    print(f"Training model {model_class_name} for {train_df_path}")
    train_df = pd.read_parquet(train_df_path)
    test_df = pd.read_parquet(test_df_path)

    features = list(set(test_df.columns).intersection(set(features)))

    for feature in features:
        if train_df[feature].nunique() == 1:
            print(f"The column {feature} only has 1 unique value!!!")
            train_df.drop(columns=[feature], inplace=True)
            features.remove(feature)

    X_test = test_df[features]
    y_test = Surv.from_dataframe(
        event=event_column,
        time=time_column,
        data=test_df[[event_column, time_column]],
    )

    X_train = train_df[features]
    y_train = Surv.from_dataframe(
        event=event_column,
        time=time_column,
        data=train_df[[event_column, time_column]],
    )

    train_min = y_train[time_column].min()
    train_max = y_train[time_column].max()
    test_max = y_test[time_column].max()

    if test_max > train_max:
        # Identify rows that will be clipped
        clipped_indices = (y_test[time_column] < train_min) | (
            y_test[time_column] > train_max
        )

        # Change the event_column to be False for these rows
        y_test[event_column][clipped_indices] = False

        y_test[time_column] = np.clip(y_test[time_column], train_min, train_max)

    print_censoring_metrics(y_train, y_test, event_column=event_column)

    with get_mlflow_context(
        mlflow_run_name=mlflow_run_name, mlflow_experiment=mlflow_experiment
    ) as run:
        if model_class_name == "kaplan_meier_estimator":
            # Kaplan Meier
            model = StepFunction(
                *model_class(y_train[event_column], y_train[time_column])
            )

            ibs = calculate_integrated_brier_score(
                y_train,
                X_test,
                y_test,
                model,
                model_class_name,
            )
            c_index_censored = c_index_ipcw = 0.5

        # lifelines CoxPH
        elif model_class_name == "CoxPHFitter":
            model = model_class(**model_params)
            model.fit(
                train_df[[*features, event_column, time_column]],
                duration_col=time_column,
                event_col=event_column,
            )

            try:
                ibs = calculate_integrated_brier_score(
                    y_train,
                    X_test,
                    y_test,
                    model,
                    model_class_name,
                )
            except:
                raise

            try:
                c_index_censored = concordance_index(
                    test_df["duration"],
                    -model.predict_partial_hazard(test_df),
                    test_df["event"],
                )
            except:
                c_index_censored = 0.0

            try:
                prediction = model.predict_partial_hazard(X_test)
                c_index_ipcw = concordance_index_ipcw(
                    y_train,
                    y_test,
                    prediction,
                )[0]
            except:
                print(y_train, y_test, prediction)
                raise

        else:
            # sksurv CoxPH and Random Survival Forest
            if model_class_name == "RandomSurvivalForest":
                model = model_class(**model_params, random_state=RANDOM_STATE)
            else:
                model = model_class(**model_params)

            try:
                model.fit(X_train, y_train)
                ibs = calculate_integrated_brier_score(
                    y_train, X_test, y_test, model, model_class_name
                )
            except:
                ibs = 1.0

            try:
                prediction = model.predict(X_test)
                c_index_censored = concordance_index_censored(
                    y_test[event_column],
                    y_test[time_column],
                    prediction,
                )[0]
            except:
                c_index_censored = 0.0

            try:
                c_index_ipcw = concordance_index_ipcw(
                    y_train,
                    y_test,
                    prediction,
                )[0]
            except:
                c_index_ipcw = c_index_censored

        metrics = {
            "ibs": ibs,
            "c_index_censored": c_index_censored,
            "c_index_ipcw": c_index_ipcw,
        }
        mlflow.log_metrics(metrics)

        mlflow.log_params(dataset_params)
        mlflow.log_params(model_params)
        mlflow.log_param("model_class", model_class_name)
        model_class_translator = {
            "kaplan_meier_estimator": 0,
            "CoxPHSurvivalAnalysis": 1,
            "CoxPHFitter": 1,
            "RandomSurvivalForest": 2,
        }
        mlflow.log_param("model_class_id", model_class_translator[model_class_name])

        log_df_to_mlflow(
            train_df,
            train_df_path,
            "Survival Dataset Train",
            time_column,
            "training",
        )

        log_df_to_mlflow(
            test_df,
            test_df_path,
            "Survival Dataset Test",
            time_column,
            "testing",
        )

        run_artifact_uri = run.info.artifact_uri
        log_model_artifact(model, run_artifact_uri, mlflow_run_name)

        mlflow.set_tag(key="model_type", value=model_class_name)


if __name__ == "__main__":
    SURVIVAL_DATASET_PATH = os.environ.get("SURVIVAL_DATASET_PATH")

    features = [
        "batt_min",
        "batt_max",
        "batt_median",
        "daily_roc",
        # "temp_min",
        # "temp_max",
        "temp_median",
        # "radio_min",
        # "radio_max",
        "radio_median",
        "fw_version_v01.70",
        "fw_version_v01.66",
        "fw_version_v01.49",
        "battery_type_id_1.0",
        "battery_type_id_2.0",
        "device_model_code_0572 2620",
        "device_model_code_0572 2621",
        "device_model_code_0572 2622",
        "device_model_code_0572 2623",
    ]

    for model_class, params in [
        (kaplan_meier_estimator, {}),
        (CoxPHFitter, {}),
        (RandomSurvivalForest, {}),
    ]:
        train_survival_model(
            model_class,
            params=params,
            features=features,
            dataset_path=SURVIVAL_DATASET_PATH,
        )
