import os
from datetime import timedelta
from time import time

import pandas as pd
import scipy.signal as ssig
from dotenv import load_dotenv

if __name__ == "__main__" or os.environ.get("EXPERIMENT") == "True":
    from augmentation import (  # pylint: disable=import-error
        add_fixed_warping_to_status_times,
        add_noise_to_devices,
        add_random_interval_warping_to_status_times,
        add_random_micro_warping_to_status_times,
    )
else:
    from utilities.augmentation import (  # pylint: disable=import-error
        add_fixed_warping_to_status_times,
        add_noise_to_devices,
        add_random_interval_warping_to_status_times,
        add_random_micro_warping_to_status_times,
    )

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR_PATH")


def load_raw_status_df() -> pd.DataFrame:
    """Loads the raw status DataFrame from `DATA_DIR/raw/devices_status_inovex.csv`.

    :return: The DataFrame conaining status metadata information about the devices like
        the battery level or the firmware version.
    """
    RAW_STATUS_PATH = os.path.join(DATA_DIR, "raw", "devices_status_inovex.csv")

    df = pd.read_csv(RAW_STATUS_PATH)
    df = df[
        [
            "device_uuid",
            "status_time",
            "battery_level_percent",
            "radio_level_percent",
            "fw_version",
        ]
    ]
    df["status_time"] = pd.to_datetime(df["status_time"])
    df.sort_values(by=["device_uuid", "status_time"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_raw_measurements_df() -> pd.DataFrame:
    """Loads the raw measurements DataFrame from
        `DATA_DIR/raw/measurements_inovex.parquet`.

    :return: The DataFrame conatining the temperature measurements for the devices.
    """
    RAW_MEASUREMENT_PATH = os.path.join(DATA_DIR, "raw", "measurements_inovex.parquet")
    columns_to_load = ["device_uuid", "timestamp", "measurement", "physical_extension"]

    df = pd.read_parquet(RAW_MEASUREMENT_PATH, columns=columns_to_load)
    df = df[df["physical_extension"] == "Air Temperature"]
    df.drop(columns=["physical_extension"], inplace=True)
    df["status_time"] = pd.to_datetime(df["timestamp"])
    df.drop(columns=["timestamp"], inplace=True)
    df.sort_values(by=["device_uuid", "status_time"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_raw_devices_df() -> pd.DataFrame:
    """Loads the raw devices DataFrame from `DATA_DIR/raw/devices_inovex.csv`.

    :return: The DataFrame containing information about the devices.
    """
    RAW_DEVICES_PATH = os.path.join(DATA_DIR, "raw", "devices_inovex.csv")

    df = pd.read_csv(RAW_DEVICES_PATH)
    df = df[["device_uuid", "battery_type_id", "device_model_code"]]
    df.drop_duplicates(inplace=True)
    return df


def load_and_merge_raw_dfs(DEBUG: bool = False) -> pd.DataFrame:
    """Loads the raw DataFrames through the functions above and merges them into one
        DataFrame containing all the necessary information.

    :return: The DataFrame containing all the necessary information.
    """
    df_measurements = load_raw_measurements_df()
    df_status = load_raw_status_df()

    df_status = df_status.sort_values(by=["status_time"])
    df_measurements = df_measurements.sort_values(["status_time"])

    df_merged = pd.merge_asof(
        df_status,
        df_measurements,
        on="status_time",
        by="device_uuid",
        direction="nearest",
    )

    all_devices = set(df_merged["device_uuid"].values)
    df_merged.dropna(axis="index", inplace=True)
    remaining_devices = set(df_merged["device_uuid"].values)
    removed_devices = all_devices - remaining_devices
    if DEBUG:
        print(
            f"Removed {len(removed_devices)} entries that had only NaN values. "
            f"{len(remaining_devices)} devices are remaining."
        )

    for device, group in df_merged.groupby(by="device_uuid"):
        time_frame = group["status_time"].max() - group["status_time"].min()
        if time_frame <= timedelta(days=50):
            if DEBUG:
                print(f"{device} is active for only {time_frame}. We will remove it!")
            remaining_devices.remove(device)

    df_merged = df_merged[df_merged["device_uuid"].isin(remaining_devices)]

    df_merged.rename({"measurement": "air_temperature"}, inplace=True, axis="columns")

    df_merged.sort_values(by=["device_uuid", "status_time"], inplace=True)

    df_devices = load_raw_devices_df()
    df_merged = pd.merge(df_merged, df_devices, on="device_uuid", how="left")

    df_merged.dropna(axis="index", inplace=True)

    return df_merged


def drop_insufficient_data(
    df: pd.DataFrame, column: str, count_thresh: int = 25, range_thresh: int = 20, DEBUG: bool = False,
) -> pd.DataFrame:
    """Drops all devices with insufficient data as defined through the parameters.

    :param df: The DataFrame to clean
    :param column: The column to check the thresholds for
    :param count_thresh: The number of measurements a device needs to have to stay,
        defaults to 25
    :param range_thresh: The range the measurements need to have.
        defaults to 20
    :return: The cleaned DataFrame.
    """
    df = df.copy()
    battery_metadata = (
        df.groupby(by=column)["battery_level_percent"]
        .agg(["min", "max", "count"])
        .assign(range=lambda row: row["max"] - row["min"])
        .reset_index()
    )

    mask = (battery_metadata["count"] < count_thresh) | (
        battery_metadata["range"] < range_thresh
    )
    to_drop = battery_metadata[mask][column]
    before = df.shape[0]
    df = df[~df[column].isin(to_drop)]
    after = df.shape[0]
    df.reset_index(drop=True, inplace=True)

    if DEBUG:
        print(
            f"Dropped {before-after} entries from {len(to_drop)} "
            f"{column} because they were insufficient."
        )

    return df


def sort_devices(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts a DataFrame by its devices and the status_time.

    :param df: The DataFrame to sort.
    :return: The sorted DataFrame.
    """
    return df.sort_values(by=["device_uuid", "status_time"]).reset_index(drop=True)


def calculate_daily_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the daily mean values for all numerical values in a raw DataFrame.

    :param df: The raw DataFrame.
    :return: The processed DataFrame.
    """
    # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[float, int]).columns
    non_numeric_columns = df.select_dtypes(exclude=[float, int]).columns

    # Calculate daily mean for numeric columns
    numeric_mean = (
        df.groupby(["device_uuid", df["status_time"].dt.date])[numeric_columns]
        .mean()
        .reset_index()
    )

    # Fill missing values in non-numeric columns with the most frequent value
    non_numeric_fill = df.groupby(["device_uuid", df["status_time"].dt.date])[
        non_numeric_columns
    ].agg(lambda x: x.value_counts().index[0])

    non_numeric_fill.drop(columns=["status_time"], inplace=True)

    # Merge results back together
    merged_df = pd.merge(
        numeric_mean, non_numeric_fill, on=["device_uuid", "status_time"]
    )
    merged_df["status_time"] = pd.to_datetime(merged_df["status_time"])
    return merged_df


def ffill_missing_days(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.copy()
        .set_index("status_time")
        .groupby("device_uuid")
        .resample("D")
        .max()
        .reset_index(0, drop=True)
        .reset_index()
        .ffill()
    )


def ffill_cycles(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.copy()
        .set_index("status_time")
        .groupby("cycle_id")
        .resample("D")
        .max()
        .reset_index(0, drop=True)
        .reset_index()
        .ffill()
    )


def smooth_df_using_median(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Smooths the values for the column `battery_level_percent` using a rolling median.

    :param df: The DataFrame to be smoothed.
    :param window: The rolling window to use, defaults to 5
    :return: The smoothed DataFrame.
    """

    def smooth_group_using_median(group: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Smooths the values for the column `battery_level_percent` for just one group.

            :param group: The group to be smoothed.
            :param window: The rolling window to use, defaults to 5
        :return: The smoothed group.
        """
        group = group.sort_values("status_time").reset_index(drop=True)
        smoothed_values = (
            group["battery_level_percent"]
            .rolling(window)
            .median()
            .shift(-1 * int(window / 2))
        )
        group["battery_level_percent"] = smoothed_values.combine_first(
            group["battery_level_percent"]
        )
        return group

    return (
        df.groupby("device_uuid")
        .apply(smooth_group_using_median, window=window)
        .reset_index(drop=True)
    )


def label_df_peaks(
    df: pd.DataFrame, height: int = 20, distance: int = 25
) -> pd.DataFrame:
    """Labels the peaks of a given DataFrame for all the devices.
        This works best when given a smoothed DataFrame of daily mean values.
        This works by calculating the first derivative of the column
        `battery_level_percent` and then running the find_peaks function from
        scipy.signal on that.
        This will also calculate the left and right borders of the peak.
        The middle of the peak will be in the column `peak_label`.
        The right and left borders will be in the columns `right_peak_border` and
        `left_peak_border`.

    :param df: The DataFrame to detect the peaks of.
    :param height: The height the peaks of the first derivative need to be,
        defaults to 20
    :param distance: The required distance between to peaks, defaults to 25
    :return: The labeled DataFrame.
    """

    def label_group_peaks(
        group: pd.DataFrame, height: int = 20, distance: int = 25
    ) -> pd.DataFrame:
        """This function applies the above described peak detection algorithm to a group
            that only contains one device.

        :param group: The DataFrame containing the group
        :param height: The height the peaks of the first derivative need to be,
            defaults to 20
        :param distance: The required distance between to peaks, defaults to 25
        :return: The labeled DataFrame for the group.
        """
        group = group.sort_values("status_time").reset_index(drop=True)
        peaks, _ = ssig.find_peaks(
            group["battery_diff"], height=height, distance=distance
        )

        peaks_left = [peak - 2 for peak in peaks if peak >= 2]
        peaks_right = [peak + 2 for peak in peaks if peak < len(group) - 2]

        group["peak_label"] = False
        group["left_peak_border"] = False
        group["right_peak_border"] = False
        group.loc[peaks, "peak_label"] = True
        group.loc[peaks_left, "left_peak_border"] = True
        group.loc[peaks_right, "right_peak_border"] = True

        return group

    df["battery_diff"] = df["battery_level_percent"].diff().fillna(0)
    return (
        df.groupby("device_uuid")
        .apply(label_group_peaks, height=height, distance=distance)
        .reset_index(drop=True)
    )


def extract_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts the battery cycles per device from a dataset with labeled peak borders.

    :param df: The dataset with labeled peaks
    :return: The dataset with added cycle ids.
    """
    df = df.copy()
    df = sort_devices(df)
    df["cycle_id"] = pd.NA

    cycle_id = 0

    for _, group in df.groupby("device_uuid"):
        in_cycle = True
        idx = None  # This is just to make pylint shut up about idx possibly being uninitialized 🤷
        for idx, row in group.iterrows():
            if row["left_peak_border"]:
                in_cycle = False

            elif row["right_peak_border"]:
                in_cycle = True
                cycle_id += 1

            if in_cycle:
                df.at[idx, "cycle_id"] = cycle_id
        if idx is not None:
            if idx >= 2:
                # Drop last two rows because they might not be labeled as a peak
                df.at[idx - 1, "cycle_id"] = pd.NA
                df.at[idx - 2, "cycle_id"] = pd.NA

        cycle_id += 1

    df = df.dropna(subset=["cycle_id"])
    df["cycle_id"] = df["cycle_id"].astype("int64")
    df.drop(columns=["peak_label", "left_peak_border", "right_peak_border"], inplace=True)
    return df


def filter_outliers(
    df: pd.DataFrame,
    cycle_id_col: str = "cycle_id",
    status_time_col: str = "status_time",
    days_threshold: int = 5,
) -> pd.DataFrame:
    """This function filters out outlier datapoints in the cycles.

    :param df: The dataframe
    :param cycle_id_col: The column for the cycle id, defaults to "cycle_id"
    :param status_time_col: The column for the status time, defaults to "status_time"
    :param days_threshold: The number of days a gap has to be to be removed,
        defaults to 5
    :return: The filtered dataframe
    """
    df = df.copy()

    for _, group in df.groupby(cycle_id_col):
        diff = group[status_time_col].diff()
        df.drop(diff[diff > timedelta(days=days_threshold)].index, inplace=True)

    return df


def load_base_dataset(
    saved_raw_merged_df_path: str | None = None,
    raw_merged_df: pd.DataFrame | None = None,
    device_subset: list[str] | None = None,
    device_uuid_count_thresh: int = 100,
    device_uuid_range_thresh: int = 20,
    add_noise: bool = False,
    max_noise: float = 5,
    add_noise_temperature: bool = False,
    max_noise_temperature: float = 5,
    random_warp_status_times: bool = False,
    random_max_time_warp_percent: float = 0.5,
    fixed_warp_status_times: bool = False,
    fixed_warping_percent: float = 0.2,
    random_interval_warp_status_times: bool = False,
    random_interval_warp_n_intervals: int = 5,
    random_interval_max_time_warp_percent: float = 0.3,
    median_smoothing_window: int = 5,
    peak_height: int = 20,
    peak_distance: int = 25,
    cycle_id_count_thresh: int = 25,
    cycle_id_range_thresh: int = 20,
    do_inter_cycle_ffill: bool = False,
    do_intra_cycle_ffill: bool = True,
    DEBUG: bool = False,
) -> pd.DataFrame:
    """Performs all the necessary steps to load the base dataset from the raw data.
    The returned dataset can then be further processed to generate either the survival
        dataset or the regression dataset.

    :param saved_raw_merged_df_path: If this is set, instead of loading and merging the
        raw datasets, we will load it from the specified path, defaults to None
    :param raw_merged_df: If this is set, instead of loading and merging the
        raw datasets, we will use this raw dataset, defaults to None
    :param device_subset: If this is set the resulting dataset will only contain entries
        from the specified devices, defaults to None
    :param device_uuid_count_thresh: The minimum number of rows a device in the raw
        dataset should have to be further preprocessed, defaults to 100
    :param device_uuid_range_thresh: The minimum range of battery min and max a device
        in the raw dataset should have to be further preprocessed, defaults to 20
    :param add_noise: Wheter or not to add noise to battery_level_percent to the raw dataset, defaults to False
    :param max_noise: The absolute maximum value of noise to add to the raw dataset,
        defaults to 5
    :param add_noise_temperature: Wheter or not to add noise to the temperature to the raw dataset, defaults to False
    :param max_noise_temperature: The absolute maximum value of noise to add to the temperature,
        defaults to 5
    :param random_warp_status_times: Wheter or not to randomly warp the raw dataset
        status times by a percent value, defaults to False
    :param random_max_time_warp_percent: The maximum percentage of the mean time
        difference to warp, defaults to 0.5
    :param fixed_warp_status_times: Wheter or not to warp the status times by a fixed
        amount, defaults to False
    :param fixed_warping_percent: The amount of fixed warping to do, defaults to 0.2
    :param random_interval_warp_status_times: Wheter or not to warp the status times in
        n random intervals for random amounts, defaults to False
    :param random_interval_warp_n_intervals: The number of random intervals to warp,
        defaults to 5
    :param random_interval_max_time_warp_percent: The maximum warp amount per interval,
        defaults to 0.3
    :param median_smoothing_window: The window to use for the median smoothing,
        defaults to 5
    :param peak_height: The minimum height a peak should have (Peaks are detected using
        the gradient of the battery levels, so the height of the gradient),
        defaults to 20
    :param peak_distance: The minimum distance two peak should have to each other,
        defaults to 25
    :param cycle_id_count_thresh: The minimum number of rows a cycle should have to be
        further preprocessed, defaults to 25
    :param cycle_id_range_thresh: The minimum range of battery min and max a cycle
        should have to be further preprocessed, defaults to 20
    :param do_inter_cycle_ffill: Whether or not to perform an ffill before splitting the devices into cycles.
        This is not recommended since it might change the cycles, defaults to False
    :param do_intra_cycle_ffill: Whether or not to perform an ffill after splitting the devices into cycles.
        This is the recommended way, since this will only ffill in cycles and will also remove outlier points
        that might artifically enlongate the cycles, defaults to True
    :param DEBUG: Enable debug output, defaults to False
    :return: The base dataset.
    """
    if DEBUG:
        start_time = time()

    if saved_raw_merged_df_path is not None:
        df = pd.read_parquet(saved_raw_merged_df_path)
    elif raw_merged_df is not None:
        df = raw_merged_df.copy()
    else:
        df = load_and_merge_raw_dfs()

    if DEBUG:
        print(f"-> Done loading raw dataset. This took {time()-start_time:.1f} s.")
        df.info()
        start_time = time()

    if device_subset is not None:
        df = df[df["device_uuid"].isin(device_subset)]

    df = drop_insufficient_data(
        df,
        column="device_uuid",
        count_thresh=device_uuid_count_thresh,
        range_thresh=device_uuid_range_thresh,
    )
    if DEBUG:
        print(
            f"-> Done dropping insufficient devices. This took {time()-start_time:.1f} s."
        )
        df.info()
        start_time = time()

    if add_noise:
        df = add_noise_to_devices(df, max_deviation=max_noise)
        if DEBUG:
            print(f"-> Done adding noise. This took {time()-start_time:.1f} s.")
            df.info()
            start_time = time()

    if add_noise_temperature:
        df = add_noise_to_devices(
            df, max_deviation=max_noise_temperature, column_to_noise="air_temperature"
        )
        if DEBUG:
            print(
                f"-> Done adding temperature noise. This took {time()-start_time:.1f} s."
            )
            df.info()
            start_time = time()

    if random_warp_status_times:
        df = add_random_micro_warping_to_status_times(
            df, max_time_warp_percent=random_max_time_warp_percent
        )
        if DEBUG:
            print(f"-> Done random time warping. This took {time()-start_time:.1f} s.")
            df.info()
            start_time = time()

    if fixed_warp_status_times:
        df = add_fixed_warping_to_status_times(
            df, warping_percent=fixed_warping_percent
        )
        if DEBUG:
            print(f"-> Done fixed time warping. This took {time()-start_time:.1f} s.")
            df.info()
            start_time = time()

    if random_interval_warp_status_times:
        df = add_random_interval_warping_to_status_times(
            df,
            n_intervals=random_interval_warp_n_intervals,
            max_warping_percent=random_interval_max_time_warp_percent,
        )
        if DEBUG:
            print(
                f"-> Done random interval time warping. This took {time()-start_time:.1f} s."
            )
            df.info()
            start_time = time()

    df = calculate_daily_mean(df)
    if DEBUG:
        print(f"-> Done calculating daily means. This took {time()-start_time:.1f} s.")
        df.info()
        start_time = time()

    if do_inter_cycle_ffill:
        df = ffill_missing_days(df)

    df = smooth_df_using_median(df, window=median_smoothing_window)
    if DEBUG:
        print(f"-> Done smoothing data. This took {time()-start_time:.1f} s.")
        df.info()
        start_time = time()

    df = label_df_peaks(df, height=peak_height, distance=peak_distance)
    if DEBUG:
        print(f"-> Done labeling peaks. This took {time()-start_time:.1f} s.")
        df.info()
        start_time = time()

    df = extract_cycles(df)
    if DEBUG:
        print(f"-> Done extracting cycles. This took {time()-start_time:.1f} s.")
        df.info()
        start_time = time()

    if do_intra_cycle_ffill:
        df = filter_outliers(df)
        df = ffill_cycles(df)

    df = drop_insufficient_data(
        df,
        column="cycle_id",
        count_thresh=cycle_id_count_thresh,
        range_thresh=cycle_id_range_thresh,
    )
    if DEBUG:
        print(
            "-> Done dropping insufficient cycles. "
            f"This took {time()-start_time:.1f} s."
        )
        df.info()

    return df


def save_dataset(df: pd.DataFrame, path: str, form: str = "parquet") -> None:
    """Saves a dataset in the given path under the given format.

    :param df: The dataset DataFrame to save
    :param path: The path to save the dataset to
    :param form: The format to save the dataset in. Can be either "parquet" or "csv",
        defaults to "parquet"
    :raises NotImplementedError: Raises an Error when trying to call this function with
        an unsupported format.
    """
    if form.strip().lower() == "parquet":
        df.to_parquet(path)
    elif form.strip().lower() == "csv":
        df.to_csv(path, index=False)
    else:
        raise NotImplementedError(
            f"The format {form} could not be recognized. Please use either parquet or CSV."
        )


if __name__ == "__main__":
    RAW_DATASET_PATH = os.environ.get("RAW_MERGED_DATASET_PATH")
    raw_df = load_and_merge_raw_dfs()
    raw_df.to_parquet(RAW_DATASET_PATH)

    BASE_DATASET_PATH = os.getenv("BASE_DATASET_PATH")
    print(BASE_DATASET_PATH)

    base_df = load_base_dataset(raw_merged_df=raw_df)
    print(base_df.info())
    base_df.to_parquet(BASE_DATASET_PATH)
