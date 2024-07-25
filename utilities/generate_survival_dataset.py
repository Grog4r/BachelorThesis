import os

import pandas as pd
from dotenv import load_dotenv


def calculate_cycle_metrics(
    df: pd.DataFrame,
    threshold: float = 20.0,
) -> pd.DataFrame:
    """Calculates metrics for each cycle in a dataset. Each cycle will be converted into
        one row in the resulting DataFrame.
        The resulting DataFrame will have the following columns:
        - device_uuid: The uuid of the device associated with the cycle
        - cycle_id: The id of the cycle
        - duration: The duration of the cycle in days
        - batt_max: The maximum battery level of the cycle
        - daily_roc: The average daily charge reduction of the battery level
        - temp_max: The maximum value of the temperature
        - temp_median: The median value of the temperature
        - radio_max: The maximum value of the radio level
        - radio_median: The median value of the radio level
        - fw_version: A list of all firmware versions of the cycle
        - battery_type_id: A list of the battery types of the cycle
        - device_model_code: A list of the device model codes of the cycle

    :param df: The DataFrame to calculate metrics for
    :param threshold: The threshold until which we want the model to predict the duration
    :return: The resulting DataFrame
    """
    survival_entries = []
    for cycle_id, group in df.groupby("cycle_id"):
        device_uuid = group["device_uuid"].unique()[0]

        start = group["status_time"].min()
        end = group["status_time"].max()
        duration = (end - start).round("1D").days

        if group["battery_level_percent"].min() < threshold:
            # If the cycle is uncesored, the duration should be the time from the start until the 20% point
            sub_group = group[group["battery_level_percent"] >= threshold]
            end = sub_group["status_time"].max()
            duration = (end - start).round("1D").days

        batt_max = group["battery_level_percent"].max()
        batt_min = group["battery_level_percent"].min()
        batt_diff = batt_max - batt_min

        # daily_roc = batt_diff / duration

        temp_std = group["air_temperature"].std()
        temp_median = group["air_temperature"].median()

        radio_std = group["radio_level_percent"].std()
        radio_median = group["radio_level_percent"].median()

        fw_version = group["fw_version"].unique().tolist()
        battery_type_id = group["battery_type_id"].unique().tolist()[0]
        device_model_code = group["device_model_code"].unique().tolist()[0]

        entry = {
            "device_uuid": device_uuid,
            "cycle_id": cycle_id,
            "duration": duration,
            "batt_max": batt_max,
            "batt_min": batt_min,  # will be removed, this is only needed for labeling events
            "batt_diff": batt_diff,  # will be removed, this is only needed for dropping insufficient cycles
            "temp_std": temp_std,
            "temp_median": temp_median,
            "radio_std": radio_std,
            "radio_median": radio_median,
            "fw_version": fw_version,
            "battery_type_id": battery_type_id,
            "device_model_code": device_model_code,
        }
        survival_entries.append(entry)
    return pd.DataFrame(survival_entries)


def label_events(df: pd.DataFrame, event_max_min_thresh: int = 20) -> pd.DataFrame:
    """Labels if a correct battery exchange event occurs in a cycle or not. This is used
        by the survival models to see if a datapoint is right censored or not. The value
        of `event_max_min_thresh` is used as a threshold for determining whether an
        exchange should be counted as "correct".

    :param df: The survival metric DataFrame containing the cycles
    :param event_max_min_thresh: The maximum value for the minimum a cycle should have to be
        counted as an event, defaults to 20
    :return: The labeled DataFrame
    """
    df = df.copy()
    df["event"] = df["batt_min"] <= event_max_min_thresh
    return df


def one_hot_encode_feature(
    df: pd.DataFrame, feature: str, drop_raw_feature: bool = True,
) -> pd.DataFrame:
    """One-hot encodes a categorical feature in a DataFrame

    :param df: The DataFrame containing the categorical feature
    :param feature: The feature to be encoded
    :param drop_raw_feature: Whether or not to drop the raw feature column after
        encoding, defaults to True
    :return: The resulting DataFrame
    """
    df = df.copy()
    feature_expressions = set(df[feature].sum())

    for feature_expression in feature_expressions:
        df[f"{feature}_{feature_expression}"] = False

    for idx, row in df.iterrows():
        for feature_expression in feature_expressions:
            if feature_expression in row[feature]:
                df.at[idx, f"{feature}_{feature_expression}"] = True
    if drop_raw_feature:
        df.drop(columns=[feature], inplace=True)
    return df


def one_hot_encode_features(
    df: pd.DataFrame, features: list[str], drop_raw_features: bool = True,
) -> pd.DataFrame:
    """One-hot encodes a list of features.

    :param df: The DataFrame conaining the features.
    :param features: The list of features to be encoded. If this is None the features
        will be "fw_version", "battery_type_id" and "device_model_code",
        defaults to None
    :param drop_raw_features: Whether the raw feature columns should be dropped after
        encoding, defaults to True
    :return: The resulting DataFrame
    """
    df = df.copy()
    for feature in features:
        df = one_hot_encode_feature(df, feature, drop_raw_feature=drop_raw_features)
    return df


def categorize_features(
    df: pd.DataFrame,
    features_to_categorize: list[str],
):
    """Categorizes a list of features. Each feature expression will get a unique integer mapping.

    :param df: The DataFrame to categorize
    :param features_to_categorize: The list of features to categorize
    :return: The resulting DataFrame with the specified features turned into integer categories
    """
    df = df.copy()
    for feature in features_to_categorize:
        expressions = set(df[feature].tolist())
        map = {exp: i for i, exp in enumerate(expressions)}
        df[feature] = df[feature].map(map)
    return df


def drop_insufficient_cycles(
    df: pd.DataFrame, duration_thresh: int = 25, batt_diff_thresh: float = 30,
) -> pd.DataFrame:
    """Drops all the cycles that do not fit into the given thresholds for the duration
        and the battery difference.

    :param df: The DataFrame to drop insufficient cycles for
    :param duration_thresh: The threshold for the cycle duration, defaults to 25
    :param batt_diff_thresh: The threshold for the battery difference, defaults to 30
    :return: The DataFrame containing only sufficient cycles
    """
    df = df.copy()
    df = df[df["duration"] >= duration_thresh]
    df = df[df["batt_diff"] >= batt_diff_thresh]

    # Drop categorical features that only conain false
    only_false_categorical = []
    for col in df.columns:
        if df[col].dtype == bool and not df[col].any():
            only_false_categorical.append(col)

    df = df.drop(columns=only_false_categorical)

    return df


def base_to_survival_dataset(
    df: pd.DataFrame,
    duration_thresh: int = 25,
    batt_diff_thresh: int = 30,
    event_max_min_thresh: int = 20,
    features_to_ohe: list[str] | None = None,
    features_to_categorize: list[str] | None = None,
    drop_raw_features_after_ohe: bool = True,
) -> pd.DataFrame:
    """Convers a base dataset to a survival dataset. This will first calculate the cycle
        metrics, then label the events and then one hot encode categorical features.

    :param df: The DataFrame containing the base dataset
    :param duration_thresh: The threshold for the cycle duration that a cycle needs to
        be counted as sufficient, defaults to 25
    :param batt_diff_thresh: The threshold for the battery difference that a cycle needs
        to be counted as sufficient, defaults to 30
    :param event_max_min_thresh: The maximum value for the minimum of the battery level
        a cycle needs to be counted as an event, defaults to 20
    :param features_to_ohe: The list of features to be one-hot encoded. If this is None
        ["fw_version"] will be one-hot encoded, defaults to None
    :param features_to_categorize: The list of features to be categorized. If this is None
        ["battery_type_id", "de"] will be one-hot encoded, defaults to None
    :param drop_raw_features_after_ohe: Wheter to drop the raw features after one-hot
        encoding, defaults to True
    :return: The resulting survival dataset
    """
    try:
        df = df.copy()
        df = calculate_cycle_metrics(df)
        df = drop_insufficient_cycles(
            df, duration_thresh=duration_thresh, batt_diff_thresh=batt_diff_thresh
        )
        df = label_events(df, event_max_min_thresh=event_max_min_thresh)

        df.drop(
            columns=[
                # "batt_min",
                "batt_diff",
            ],
            inplace=True,
        )

        if features_to_ohe is None:
            features_to_ohe = ["fw_version"]
        df = one_hot_encode_features(
            df, features=features_to_ohe, drop_raw_features=drop_raw_features_after_ohe
        )

        if features_to_categorize is None:
            features_to_categorize = ["battery_type_id", "device_model_code"]
        df = categorize_features(df, features_to_categorize=features_to_categorize)
        return df
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    load_dotenv()
    BASE_DATASET_PATH = os.environ.get("BASE_DATASET_PATH")
    SURVIVAL_DATASET_PATH = os.environ.get("SURVIVAL_DATASET_PATH")

    SURVIVAL_DATASET_PATH = "/home/nkuechen/Documents/Thesis/code/thesis_code/data/my_datasets/survival_dataset.parquet"

    if os.path.exists(BASE_DATASET_PATH):
        df = pd.read_parquet(BASE_DATASET_PATH)
    else:
        raise FileNotFoundError(
            f"The base dataset file could not be found in {BASE_DATASET_PATH}.\n"
            "Either correct the path in the .env file or make sure the base dataset"
            "file exists in the specified path."
        )

    survival_df = base_to_survival_dataset(df)
    print(survival_df.info())
    survival_df.to_parquet(SURVIVAL_DATASET_PATH)
