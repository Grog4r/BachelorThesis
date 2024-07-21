import os
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv


def drop_insufficient_cycles(
    df: pd.DataFrame,
    duration_thresh: int = 25,
    batt_diff_thresh: float = 30,
) -> pd.DataFrame:
    """Drops all cycles with insufficient data as defined through the parameters.

    :param df: The DataFrame to clean
    :param duration_thresh: The time a cycle needs to be in days to be considered
        sufficient, defaults to 25
    :param batt_diff_thresh: The range the measurements of the cycle need to have,
        defaults to 30
    :return: The cleaned DataFrame.
    """
    df = df.copy()

    start_rows = df.shape[0]

    battery_metadata = (
        df.groupby(by="cycle_id")["battery_level_percent"]
        .agg(["min", "max"])
        .assign(range=lambda row: row["max"] - row["min"])
        .reset_index()
    )

    mask = battery_metadata["range"] < batt_diff_thresh
    to_drop = battery_metadata[mask]["cycle_id"]
    number_to_drop = len(to_drop)
    df = df[~df["cycle_id"].isin(to_drop)]

    df.reset_index(drop=True, inplace=True)

    time_metadata = (
        df.groupby(by="cycle_id")["status_time"]
        .agg(["min", "max"])
        .assign(duration=lambda row: row["max"] - row["min"])
        .reset_index()
    )

    mask = time_metadata["duration"] < timedelta(days=duration_thresh)
    to_drop = time_metadata[mask]["cycle_id"]
    number_to_drop += len(to_drop)
    df = df[~df["cycle_id"].isin(to_drop)]

    df.reset_index(drop=True, inplace=True)

    print(
        f"Dropped {start_rows-df.shape[0]} rows in {number_to_drop} cycles.\n"
        f"{len(df['cycle_id'].unique())} cycles left."
    )

    return df


def calculate_rolling_metric(
    df: pd.DataFrame,
    window: int,
    metric_name: str,
    column: str = "battery_level_percent",
) -> pd.DataFrame:
    """Calculates a rolling metric for a defined column

    :param df: The DataFrame to calculate the rolling metric for
    :param window: The window for the rolling metric
    :param metric_name: The name of the metric. Has to be one of 'mean', 'min', 'max' or
        'median'.
    :param column: The column to calculate the metric for, defaults to
        "battery_level_percent"
    :raises NotImplementedError: Raises a NotImplementedError if the metric is not
        implemented
    :return: The resulting DataFrame
    """
    df = df.copy()
    grouped = df.groupby(by="cycle_id")
    rolling = grouped[column].rolling(window=window, min_periods=1)

    if metric_name.strip().lower() == "mean":
        df[f"{column}_rolling_{metric_name}_{window}"] = rolling.mean().reset_index(
            level=0, drop=True
        )
        return df

    if metric_name.strip().lower() == "min":
        df[f"{column}_rolling_{metric_name}_{window}"] = rolling.min().reset_index(
            level=0, drop=True
        )
        return df

    if metric_name.strip().lower() == "max":
        df[f"{column}_rolling_{metric_name}_{window}"] = rolling.max().reset_index(
            level=0, drop=True
        )
        return df

    if metric_name.strip().lower() == "median":
        df[f"{column}_rolling_{metric_name}_{window}"] = rolling.median().reset_index(
            level=0, drop=True
        )
        return df

    raise NotImplementedError(f"The metric '{metric_name}' is not implemented.")


def calculate_rolling_metrics(
    df: pd.DataFrame,
    window: int,
    metric_column_names: list[tuple[str, str]],
) -> pd.DataFrame:
    """Calculates a number of rolling metrics for some columns

    :param df: The DataFrame to calculate rolling metrics for.
    :param window: The window for the rolling metrics
    :param metric_column_names: A list of (str, str) tuples specifying the metric and
        column names to be calculated
    :return: The resulting DataFrame
    """
    df = df.copy()
    for metric_name, column_name in metric_column_names:
        df = calculate_rolling_metric(
            df, window=window, metric_name=metric_name, column=column_name
        )
    return df


def generate_regression_target(
    df: pd.DataFrame,
    prediction_horizon: int = 10,
    remove_untargeted_rows: bool = True,
) -> pd.DataFrame:
    """Generates targets for the regression. The goal is to predict n days into the
    future.

    :param df: The DataFrame to generate targets for.
    :param prediction_horizon: The number of days to predict into the future,
        defaults to 10
    :param remove_untargeted_rows: Whether to remove the untargeted rows, defaults to True
    :return: The resulting targeted DataFrame
    """
    df = df.copy()
    df["target"] = df.groupby(by="cycle_id")["battery_level_percent"].shift(
        -prediction_horizon
    )
    if remove_untargeted_rows:
        df.dropna(subset=["target"], axis="index", inplace=True)
    return df


def one_hot_encode_column(
    df: pd.DataFrame,
    column: str,
    drop_raw_column: bool = True,
) -> pd.DataFrame:
    """One hot encodes a column in a DataFrame

    :param df: The DataFrame to be encoded
    :param column: The column to be encoded
    :param drop_raw_column: Wheter to drop the original column after, defaults to True
    :return: The DataFrame containing the one hot encoded column
    """
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    if drop_raw_column:
        df.drop(columns=[column], inplace=True)
    return df


def base_to_regression_dataset(
    df: pd.DataFrame,
    rolling_windows: list[int] | None = None,
    metric_column_names: list[tuple[str, str]] | None = None,
    prediction_horizon: int = 10,
    ohe_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Generates a regression dataset out of a given base dataset

    :param df: The base dataset
    :param rolling_windows: A list of the rolling windows to calculate metrics for. If
        this is None it defaults to [5, 20], defaults to None
    :param metric_column_names: A list of metric and column pairs for the rolling
        windows to be calculated. If this is None it defaults to calculate `mean`,
        `min`, `max` and `median` for the columns `battery_level_percent` and
        `battery_diff`, defaults to None
    :param prediction_horizon: The number of days the targets should be looking into the
        future. This ends up being the number of days the regression model should
        predict into the future, defaults to 10
    :param ohe_columns: The columns to one hot encode. If this is None it defaults to
        ["fw_version", "battery_type_id", "device_model_code"], defaults to None
    :return: The resulting DataFrame containing the regression dataset
    """
    df = df.copy()

    df = drop_insufficient_cycles(df)

    rolling_windows = [5, 50] if rolling_windows is None else rolling_windows
    metric_column_names = (
        [
            ("median", "battery_level_percent"),
            ("median", "battery_diff"),
        ]
        if metric_column_names is None
        else metric_column_names
    )
    for rolling_window in rolling_windows:
        df = calculate_rolling_metrics(
            df, window=rolling_window, metric_column_names=metric_column_names
        )

    ohe_columns = (
        ["fw_version", "battery_type_id", "device_model_code"]
        if ohe_columns is None
        else ohe_columns
    )
    for ohe_column in ohe_columns:
        df = one_hot_encode_column(df, column=ohe_column)

    df = generate_regression_target(df, prediction_horizon=prediction_horizon)

    return df


if __name__ == "__main__":
    load_dotenv()
    BASE_DATASET_PATH = os.environ.get("BASE_DATASET_PATH")
    if BASE_DATASET_PATH.startswith("/Documents"):
        BASE_DATASET_PATH = os.environ.get("HOME") + BASE_DATASET_PATH

    REGRESSION_DATASET_PATH = os.environ.get("REGRESSION_DATASET_PATH")
    print(REGRESSION_DATASET_PATH)

    if os.path.exists(BASE_DATASET_PATH):
        df = pd.read_parquet(BASE_DATASET_PATH)
    else:
        raise FileNotFoundError(
            f"The base dataset file could not be found in {BASE_DATASET_PATH}.\n"
            "Either correct the path in the .env file or make sure the base dataset"
            "file exists in the specified path."
        )

    regression_df = base_to_regression_dataset(df)
    print(regression_df.info())
    regression_df.to_parquet(REGRESSION_DATASET_PATH)
