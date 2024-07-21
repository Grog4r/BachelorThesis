import random
import uuid
from datetime import timedelta

import numpy as np
import pandas as pd


def add_noise_to_devices(
    df: pd.DataFrame,
    max_deviation: float = 5,
    column_to_noise: str = "battery_level_percent",
) -> pd.DataFrame:
    """Adds noise to a column

    :param df: The DataFrame to add noise to
    :param max_deviation: The absolute maximum value to add, defaults to 5
    :param column_to_noise: The column to add the noise to, defaults to "battery_level_percent".
        If this is a feature column this will calculate the max_deviation as the percentage of
        the absolute difference of min and max.
    :return: The DataFrame containing the noisy column
    """
    df = df.copy()
    if column_to_noise != "battery_level_percent":
        noised_groups = []
        for _, group in df.groupby(by="device_uuid"):
            group = group.copy()
            min = group[column_to_noise].min()
            max = group[column_to_noise].max()
            diff = abs(max - min)
            max_deviation_value = max_deviation * 0.01 * diff
            noise = np.random.uniform(
                -max_deviation_value, max_deviation_value, group.shape[0]
            )
            group[column_to_noise] += noise
            noised_groups.append(group)
        df = pd.concat(noised_groups).reset_index(drop=True)
    else:
        noise = np.random.uniform(-max_deviation, max_deviation, df.shape[0])
        df[column_to_noise] += noise

    unique_uuids = {
        device_uuid: f"{device_uuid}_{uuid.uuid1()}"
        for device_uuid in df["device_uuid"].unique()
    }
    df["device_uuid"] = df["device_uuid"].map(unique_uuids)

    return df


def add_random_micro_warping_to_status_times(
    df: pd.DataFrame, max_time_warp_percent: float = 0.5
) -> pd.DataFrame:
    """Adds random time warping to the status_times of a DataFrame

    :param df: The DataFrame to add random time warping to
    :param max_time_warp_percent: The maximum percentage of time warping to add. This is
        relative to the mean time difference between the rows in one device,
        defaults to 0.5
    :return: The time warped DataFrame
    """

    df = df.copy()
    df["status_time"] = pd.to_datetime(df["status_time"])

    warped_devices = []
    unique_uuids = {
        device_uuid: f"{device_uuid}_{uuid.uuid1()}"
        for device_uuid in df["device_uuid"].unique()
    }

    for device_uuid, group in df.groupby(by="device_uuid"):
        mean_timedelta = group["status_time"].diff().mean()
        max_time_warp = mean_timedelta * max_time_warp_percent

        # Generate the random warping
        time_warp = np.random.uniform(
            -max_time_warp.total_seconds(), max_time_warp.total_seconds(), len(group)
        )
        time_warp = pd.to_timedelta(time_warp, unit="s")

        # Apply the warping
        group = group.copy()
        group["status_time"] = group["status_time"] + time_warp
        group["device_uuid"] = unique_uuids[device_uuid]

        warped_devices.append(group)

    return pd.concat(warped_devices).reset_index(drop=True)


def add_fixed_warping_to_status_times(
    df: pd.DataFrame, warping_percent: float = 0.2
) -> pd.DataFrame:
    """Adds a fixed amount of time warping to an entire device.
    DEPRECATED! THIS IS NOT USED!

    :param df: The DataFrame to add the fixed amount of time warping to
    :param warping_percent: The amount of warping, defaults to 0.2
    :return: The time warped DataFrame
    """

    def add_fixed_warping_to_group(group: pd.Series) -> pd.Series:
        """Adds a fixed amount of time warping to a group.

        :param group: The group to add the time warping to
        :return: The warped group
        """
        device_uuid = group["device_uuid"].unique()[0]
        new_device_uuid = f"{device_uuid}_{uuid.uuid1()}"

        time_diff = group["status_time"].diff().fillna(timedelta(0))
        warped_time_diff = time_diff * warping_percent

        group["status_time"] = group["status_time"] + warped_time_diff.cumsum()
        group["device_uuid"] = new_device_uuid
        return group

    df = df.copy()
    df = df.groupby("device_uuid").apply(add_fixed_warping_to_group)

    return df.reset_index(drop=True)


def add_random_interval_warping_to_status_times(
    df: pd.DataFrame, n_intervals: int = 5, max_warping_percent: float = 0.3
) -> pd.DataFrame:
    """Adds a random amount of time warping to n intervals per device.
    DEPRECATED! THIS IS NOT USED!

    :param df: The DataFrame to add the interval time warping to.
    :param n_intervals: The number of intervals per device, defaults to 5
    :param max_warping_percent: The absolute maximum amount of warping to do,
        defaults to 0.3
    :return: The warped DataFrame
    """

    def add_random_interval_warping_to_group(group: pd.Series) -> pd.Series:
        """Adds a random amount of time warping to n intervals in the given group

        :param group: The group to add the time warping to
        :return: The warped group
        """
        device_uuid = group["device_uuid"].unique()[0]
        new_device_uuid = f"{device_uuid}_{uuid.uuid1()}"

        random_split_points = sorted(
            [0]
            + [random.choice(range(0, len(group) - 1)) for _ in range(n_intervals - 1)]
            + [len(group)]
        )

        split_diffs = np.diff(random_split_points)

        warping_percents = [
            random.uniform(-1, 1) * max_warping_percent for _ in range(n_intervals)
        ]

        warping_percents_list = []
        for i, split_diff in enumerate(split_diffs):
            warping_percents_list += [warping_percents[i]] * split_diff

        time_diff = group["status_time"].diff().fillna(timedelta(0))

        warped_time_diff = time_diff * warping_percents_list

        group["status_time"] = group["status_time"] + warped_time_diff.cumsum()
        group["device_uuid"] = new_device_uuid
        return group

    df = df.copy()
    df = df.groupby("device_uuid").apply(add_random_interval_warping_to_group)

    return df.reset_index(drop=True)
