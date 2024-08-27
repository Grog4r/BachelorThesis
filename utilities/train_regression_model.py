import os
from datetime import datetime, timedelta
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

load_dotenv()
if __name__ == "__main__" or os.environ.get("EXPERIMENT") == "True":
    from generate_regression_dataset import calculate_rolling_metric
    from train_utils import (
        get_mlflow_context,
        log_df_to_mlflow,
        log_feature_importances,
        log_model_artifact,
    )
else:
    from utilities.generate_regression_dataset import calculate_rolling_metric
    from utilities.train_utils import (
        get_mlflow_context,
        log_df_to_mlflow,
        log_feature_importances,
        log_model_artifact,
    )

RANDOM_STATE = int(os.environ.get("RANDOM_STATE"))


def X_y_to_df(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Concatinates an X and a y dataset into a single dataset.

    :param X: The X DataFrame
    :param y: The y Series
    :return: The convatinated resulting dataset
    """
    return pd.concat([X, y], axis="columns")


def df_to_X_y(
    df: pd.DataFrame, features: list[str], target_column: str = "target"
) -> tuple[pd.DataFrame, pd.Series]:
    """Converts a DataFrame into X and y by using a list of features and a target column.

    :param df: The DataFrame to convert
    :param features: The list of features
    :param target_column: The column containng the target, defaults to "target"
    :return: X and y
    """
    X = df[features]
    y = df[target_column]
    return X, y


def train_regression_model(
    model_class,
    dataset_path: str,
    params: dict[str, Any],
    features: list[str],
    mlflow_run_name: str | None = None,
    target_column: str = "target",
) -> None:
    """Trains a regression model

    :param model_class: The class of the model to train, can be one of
        LinearRegression, DecisionTreeRegressor or XGBRegressor
    :param dataset_path: The path to the dataset
    :param params: The parameters for the model
    :param features: The features to train the model on
    :param mlflow_run_name: The name for the mlflow run, if this is None it will
        automatically generate a name, defaults to None
    :param target_column: The name of the target column, defaults to "target"
    """
    df = pd.read_parquet(dataset_path)

    X, y = df_to_X_y(df, features, target_column=target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    model_prefix = model_class.__name__
    if mlflow_run_name is None:
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        mlflow_run_name = f"{model_prefix}_{now}"
        for key, value in params.items():
            mlflow_run_name += f"_{key}={value}"
        dataset_name = dataset_path.split("/")[-1].split(".")[0]
        mlflow_run_name += f"_{dataset_name}"

    mlflow_experiment = os.environ.get("MLFLOW_REGRESSION_EXPERIMENT")
    with get_mlflow_context(
        mlflow_run_name=mlflow_run_name, mlflow_experiment=mlflow_experiment
    ) as run:
        random_state = {}
        if model_prefix in ["DecisionTreeRegressor", "XGBRegressor"]:
            random_state["random_state"] = RANDOM_STATE
        model = model_class(**params, **random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r_squared = r2_score(y_test, y_pred)
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r_squared": r_squared,
        }
        mlflow.log_metrics(metrics)

        mlflow.log_params(params)

        log_df_to_mlflow(
            X_y_to_df(X_train, y_train),
            dataset_path,
            "Regression Dataset Train",
            target_column,
            "training",
        )

        log_df_to_mlflow(
            X_y_to_df(X_test, y_test),
            dataset_path,
            "Regression Dataset Test",
            target_column,
            "testing",
        )

        run_artifact_uri = run.info.artifact_uri
        log_model_artifact(model, run_artifact_uri, mlflow_run_name)

        mlflow.set_tag(key="model_type", value=model_prefix)


def experiment_train_regression_model(
    model_class,
    train_df_path: str,
    test_df_path: str,
    model_params: dict[str, Any],
    dataset_params: dict[str, Any],
    features: list[str],
    mlflow_run_name: str | None = None,
    target_column: str = "target",
) -> None:
    """Trains a regression model specifically for an experiment. For example the
        training and test dataset paths are separate here.

    :param model_class: The class of the model to train, can be one of
        LinearRegression, DecisionTreeRegressor or XGBRegressor
    :param train_df_path: The path to the train dataset
    :param test_df_path: The path to the test dataset
    :param model_params: The parameters for the model
    :param dataset_params: The parameters of the train dataset
    :param features: The features to train the model on
    :param mlflow_run_name: The name for the mlflow run, if this is None it will
        automatically generate a name, defaults to None
    :param target_column: The name of the target column, defaults to "target"
    """
    model_class_name = model_class.__name__

    mlflow_experiment = os.environ.get("MLFLOW_REGRESSION_EXPERIMENT")
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
            filter_string=f"run_name='{mlflow_run_name}'",
        )
        if len(mlflow_run) > 0:
            print(f"Already did run {mlflow_run_name}. Skipping...")
            return

    print(f"Training model {model_class_name} for {train_df_path}")
    train_df = pd.read_parquet(train_df_path)
    test_df = pd.read_parquet(test_df_path)

    try:
        X_test = test_df[features]
        y_test = test_df[target_column]
    except KeyError:
        features = list(set(test_df.columns).intersection(set(features)))
        X_test = test_df[features]
        y_test = test_df[target_column]
    X_train = train_df[features]
    y_train = train_df[target_column]

    with get_mlflow_context(
        mlflow_run_name=mlflow_run_name, mlflow_experiment=mlflow_experiment
    ) as run:
        random_state = {}
        if model_class_name in ["DecisionTreeRegressor", "XGBRegressor"]:
            random_state["random_state"] = RANDOM_STATE
        model = model_class(**model_params, **random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r_squared = r2_score(y_test, y_pred)
        mdt, adt, med_dt, _, _, std = divergence_time_metrics(
            model,
            test_df,
            prediction_horizon=int(dataset_params["pred_hor"]),
        )
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r_squared": r_squared,
            "mdt": mdt,
            "adt": adt,
            "med_dt": med_dt,
            "std": std,
        }
        mlflow.log_metrics(metrics)

        mlflow.log_params(model_params)
        mlflow.log_params(dataset_params)
        mlflow.log_param("model_class", model_class_name)
        model_class_translator = {
            "LinearRegression": 0,
            "DecisionTreeRegressor": 1,
            "XGBRegressor": 2,
        }
        mlflow.log_param("model_class_id", model_class_translator[model_class_name])

        log_df_to_mlflow(
            train_df,
            train_df_path,
            "Regression Dataset Train",
            target_column,
            "training",
        )

        log_df_to_mlflow(
            test_df,
            test_df_path,
            "Regression Dataset Test",
            target_column,
            "testing",
        )

        run_artifact_uri = run.info.artifact_uri
        log_model_artifact(model, run_artifact_uri, mlflow_run_name)

        log_feature_importances(model, features)

        mlflow.set_tag(key="model_type", value=model_class_name)


def cross_validate_regression_model(
    clf: Any,
    df: pd.DataFrame,
    features: list[str],
    target_column: str = "target",
    scoring: dict | None = None,
    cv: int = 5,
) -> Any:
    device_uuids = df["device_uuid"].unique()
    # X, y = df_to_X_y(df, features, target_column=target_column)
    kf = KFold(n_splits=2)
    for train, test in kf.split(device_uuids):
        print(train, test)
        print(len(train), len(test))


def col_name_to_rolling_feature(col_name: str) -> tuple[str, str, int]:
    """Splits a rolling column name into the feature name, metric and rolling window
        size.

    :param col_name: The name of the column
    :return: The feature name, metric and rolling window size
    """
    feature_name = col_name.split("_rolling_")[0]
    metric = col_name.split("_rolling_")[1].split("_")[0]
    window = int(col_name.split("_")[-1])
    return feature_name, metric, window


def add_prediction_to_df(input_df: pd.DataFrame, prediction: float) -> pd.DataFrame:
    """Adds a prediction value as a new row to a given dataframe. This also calculates
        dependant features like battery_diff or rolling metrics.

    :param input_df: The input dataframe
    :param prediction: The prediction value
    :return: The resulting dataframe
    """
    new_row = input_df.tail(1).copy()
    rolling_columns = [col for col in new_row.columns if "rolling" in col]
    diff = prediction - new_row["battery_level_percent"]
    new_row["status_time"] += timedelta(days=1)
    new_row["battery_diff"] = diff
    new_row["battery_level_percent"] = prediction
    result_df = pd.concat([input_df, new_row], axis="index").reset_index(drop=True)
    for rolling_col in rolling_columns:
        feature_name, metric, window = col_name_to_rolling_feature(rolling_col)
        result_df = calculate_rolling_metric(
            result_df, window, metric_name=metric, column=feature_name
        )
    return result_df


def calculate_rolling_metric_for_new_row(
    input_list: list[dict], new_row: dict, window: int, metric: str, column: str
) -> None:
    """Calculates and updates the rolling metric for the new row.

    :param input_list: The list of row dictionaries
    :param new_row: The new row dictionary to update
    :param window: The window for the rolling metric
    :param metric: The name of the metric, must be one of 'mean', 'min', 'max', or 'median'
    :param column: The column to calculate the metric for
    """
    cycle_id = new_row["cycle_id"]
    relevant_rows = [row[column] for row in input_list if row["cycle_id"] == cycle_id]

    if len(relevant_rows) >= window:
        relevant_rows = relevant_rows[-window:]

    if metric == "mean":
        rolling_value = sum(relevant_rows) / len(relevant_rows)
    elif metric == "min":
        rolling_value = min(relevant_rows)
    elif metric == "max":
        rolling_value = max(relevant_rows)
    elif metric == "median":
        sorted_rows = sorted(relevant_rows)
        mid = len(sorted_rows) // 2
        if len(sorted_rows) % 2 == 0:
            rolling_value = (sorted_rows[mid - 1] + sorted_rows[mid]) / 2
        else:
            rolling_value = sorted_rows[mid]
    else:
        raise NotImplementedError(f"The metric '{metric}' is not implemented.")

    new_row[f"{column}_rolling_{metric}_{window}"] = rolling_value


def add_prediction_to_list(
    input_list: list[dict], last_row: dict, prediction: float
) -> dict:
    """Adds a prediction value as a new row dictionary to a list. Updates dependant
       features like battery_diff.

    :param input_list: The list of row dictionaries
    :param last_row: The last row dictionary in the input list
    :param prediction: The prediction value
    :return: The new row dictionary with updated values
    """
    new_row = last_row.copy()
    rolling_columns = [col for col in new_row if "rolling" in str(col)]
    diff = prediction - new_row["battery_level_percent"]
    new_row["status_time"] += timedelta(days=1)
    new_row["battery_diff"] = diff
    new_row["battery_level_percent"] = prediction

    # Update rolling metrics only for the new row
    for rolling_col in rolling_columns:
        feature_name, metric, window = col_name_to_rolling_feature(rolling_col)
        calculate_rolling_metric_for_new_row(
            input_list, new_row, window, metric, feature_name
        )

    return new_row


def iterative_prediction(
    model: Any,
    input_df: pd.DataFrame,
    n_predictions: int,
    prediction_horizon: int,
    test_df: pd.DataFrame | None = None,
    divergence_threshold: float = 10,
    divergence_window: int = 5,
    for_metric: bool = False,
) -> pd.DataFrame:
    """This calculates a iterative prediction for a model. Iterative means that some input features
       of the next row are dependent on the previous prediction.

    :param model: The model to use for the predictions
    :param input_df: The input dataframe
    :param n_predictions: The number of predictions to be computed
    :param prediction_horizon: The prediction horizon of the model.
    :param test_df: If this is not None, this will be used to stop the computation as
                    soon as the divergence threshold is reached. This makes calculating the MDT
                    faster, defaults to None
    :param divergence_threshold: The divergence threshold for the MDT computation,
                                 defaults to 10
    :param divergence_window: The window for the MDT computation, defaults to 5
    :param for_metric: If this is True, this will directly return the time it took the
                       until the divergence threshold was reached.
    :raises ValueError: Raises a ValueError if the number of datapoints in the input
                        dataframe is less than the prediction horizon. We need at least the amount of
                        prediction_horizon to be able to perform a iterative prediction
    :return: the resulting dataframe containing the input rows and n_predictions rows
             with predicted values.
    """
    if len(input_df) < prediction_horizon:
        raise ValueError(
            "The input dataset must have at least the number of datapoints as the "
            f"prediction horizon is long. ({len(input_df)} vs {prediction_horizon})"
        )

    input_df = input_df.copy()
    features = list(model.feature_names_in_)
    start_datetime = input_df["status_time"].iloc[-1]
    n_divergence_times = 0

    # Convert DataFrame to list of dicts for faster row-wise operations
    input_list = input_df.to_dict(orient="records")
    last_row = input_list[-1]

    # Preprocess divergence checks if test_df is provided
    if test_df is not None:
        test_values = (
            test_df["battery_level_percent"]
            .iloc[len(input_df) : len(input_df) + n_predictions]
            .to_numpy()
        )

    for i in range(n_predictions):
        input_datapoint = pd.DataFrame([last_row])[features]
        prediction = model.predict(input_datapoint)[0]
        if prediction < 0:
            break

        if test_df is not None:
            test_value = test_values[i]
            if abs(prediction - test_value) >= divergence_threshold:
                n_divergence_times += 1
                if n_divergence_times == divergence_window:
                    
                    break
            else:
                n_divergence_times = 0

        last_row = add_prediction_to_list(input_list, last_row, prediction)
        input_list.append(last_row)

    if for_metric:
        end_datetime = last_row["status_time"]
        return (end_datetime - start_datetime).days
    else:
        # Convert list of dicts back to DataFrame
        return pd.DataFrame(input_list)


def find_divergence_time(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    divergence_threshold: float = 10,
    col: str = "battery_level_percent",
    divergence_window: int = 5,
) -> datetime:
    """Finds the first time where the values of two dataframes have a certain
        distance for at least 5 days. This is used to determine the point where the
        iterative prediction gets too far away from the test dataframe.

    :param df1: The first dataframe
    :param df2: The second dataframe
    :param divergence_threshold: The threshold for the distance between the two
        dataframes, defaults to 10
    :param col: The column to find the distance between the two dataframes for
    :param divergence_window: The window for the rolling min. This is the time window
        the divergence needs to go on for to be counted. This is to compensate local
        divergence maxima because of outliers in the test data, defaults to 5
    :return: The found datetime
    """
    # Ensure the dataframes are aligned on the 'status_time' column
    columns = ["status_time", col]
    merged_df = pd.merge(
        df1[columns],
        df2[columns],
        on="status_time",
        suffixes=("_target", "_prediction"),
    )

    # Calculate the absolute difference between the 'battery_level_percent' columns
    merged_df["difference"] = (
        merged_df[f"{col}_target"] - merged_df[f"{col}_prediction"]
    ).abs()

    merged_df["difference"] = merged_df["difference"].rolling(divergence_window).min()

    # Find the first instance where the difference exceeds the threshold
    divergence_row = merged_df[merged_df["difference"] >= divergence_threshold].head(1)

    if not divergence_row.empty:
        return divergence_row["status_time"].iloc[0]
    else:
        return merged_df["status_time"].iloc[-1]


def divergence_time_metrics(
    model: Any,
    test_df: pd.DataFrame,
    prediction_horizon: int,
    input_size: int = -1,
    divergence_threshold: float = 10,
    divergence_window: int = 5,
    VERBOSE: bool = False,
) -> tuple[float, float, float, dict[str, list[float]], dict[int, float], float]:
    """Calculates divergence time metrics for a model on some test data.
        The metrics are:
        - Mean Divergence Time (MDT): The mean time the prediction stays within divergence_threshold % of the
            test data.
        - Adjusted Divergence Time (ADT): The MDT multiplied with a weighted standard
            divergence. This is to be able to differentiate between a model that has
            divergence times of [30, 30] or [0, 60].
        - Median Divergence Time (MEDDT): The median time the prediction stays within divergence_threshold %
            of the test data.

    :param model: The model to calculate the metric for.
    :param test_df: The test data to calculate the divergence time for.
    :param prediction_horizon: The prediction horizon of the model.
    :param input_size: The number of days to waint until starting the prediction. If
        this is -1 it will use three time points which from Q1, Q2 and Q3,
        defaults to -1
    :param divergence_threshold: The divergence threshold. If a prediction is at least
        this far away from the test data for divergence_window time steps, it will be
        considered as diverted, defaults to 10
    :param divergence_window: The number of time steps the prediction has to be diverted
        for to be considered as diverted, defaults to 5
    :return: MDT, ADT, MED_DT, device times, cycle times, standard deviation
    """
    device_times = {}
    cycle_times = {}

    divergence_timedeltas = []

    for cycle_id, group in test_df.groupby("cycle_id"):
        if input_size == -1:
            input_sizes = [int(len(group) * factor) for factor in [0.25, 0.5, 0.75]]
            input_sizes = [
                input_size
                for input_size in input_sizes
                if input_size > prediction_horizon
            ]
        else:
            input_sizes = [input_size]
        for in_size in input_sizes:
            n_predictions = len(group) - in_size
            if n_predictions <= 0:
                print("The cycle length is smaller than the prediction horizon.")
                continue
            divergence_timedelta = iterative_prediction(
                model,
                group.iloc[0:in_size],
                n_predictions,
                prediction_horizon,
                test_df=group,
                divergence_threshold=divergence_threshold,
                divergence_window=divergence_window,
                for_metric=True,
            )

            if VERBOSE:
                print(f"Cycle {cycle_id} divergence time: {divergence_timedelta}")

            device_uuid = group["device_uuid"].unique()[0]
            if device_uuid not in device_times:
                device_times[device_uuid] = [divergence_timedelta]
            else:
                device_times[device_uuid].append(divergence_timedelta)

            if cycle_id in cycle_times:
                cycle_times[cycle_id].append(divergence_timedelta)
            else:
                cycle_times[cycle_id] = [divergence_timedelta]

            divergence_timedeltas.append(divergence_timedelta)

    mdt = np.mean(divergence_timedeltas)

    med_dt = np.median(divergence_timedeltas)

    adt = adjusted_divergence_time(divergence_timedeltas)

    std = np.std(divergence_timedeltas)

    return mdt, adt, med_dt, device_times, cycle_times, std


def adjusted_divergence_time(
    divergence_timedeltas: list[float], std_weight: float = 0.3
) -> float:
    """This calculates an Adjusted Divergence Time (ADT), taking into account the
        standard divergence of the values.

    :param divergence_timedeltas: The measurements to calculate the ADT for
    :param std_weight: The weight for the standard divergence, defaults to 0.5
    :return: The ADT metric
    """
    return np.mean(divergence_timedeltas) - std_weight * np.std(divergence_timedeltas)


if __name__ == "__main__":
    REGRESSION_DATASET_PATH = os.environ.get("REGRESSION_DATASET_PATH")

    features = [
        "battery_level_percent",
        "radio_level_percent",
        "air_temperature",
        "battery_diff",
        "battery_level_percent_rolling_mean_5",
        "battery_level_percent_rolling_min_5",
        "battery_level_percent_rolling_max_5",
        "battery_level_percent_rolling_median_5",
        "battery_diff_rolling_mean_5",
        "battery_diff_rolling_min_5",
        "battery_diff_rolling_max_5",
        "battery_diff_rolling_median_5",
        "battery_level_percent_rolling_mean_20",
        "battery_level_percent_rolling_min_20",
        "battery_level_percent_rolling_max_20",
        "battery_level_percent_rolling_median_20",
        "battery_diff_rolling_mean_20",
        "battery_diff_rolling_min_20",
        "battery_diff_rolling_max_20",
        "battery_diff_rolling_median_20",
        "v01.49",
        "v01.66",
        "v01.70",
        "1.0",
        "2.0",
        "0572 2620",
        "0572 2621",
        "0572 2622",
        "0572 2623",
    ]

    for model_class, params in [
        (LinearRegression, {}),
        (DecisionTreeRegressor, {}),
        (XGBRegressor, {}),
    ]:
        train_regression_model(
            model_class, REGRESSION_DATASET_PATH, params=params, features=features
        )
