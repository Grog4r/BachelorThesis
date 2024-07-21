import json
import os
import random

import pandas as pd
from dotenv import load_dotenv

os.environ["EXPERIMENT"] = "True"
import preprocess_raw_data as prep_data


def merge_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """Merges a list of DataFrames into one DataFrame by the indices.

    :param datasets: The list of DataFrames to merge
    :return: The resulting DataFrame
    """
    return pd.concat(datasets, axis="index").reset_index(drop=True)


if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()
    RANDOM_STATE = int(os.environ.get("RANDOM_STATE"))

    RAW_MERGED_DATASET_PATH = os.environ.get("RAW_MERGED_DATASET_PATH")

    EXPERIMENT_BASE_DATASETS_PATH = os.environ.get("EXPERIMENT_BASE_DATASETS_PATH")
    # Make sure the destination directory exists
    if not os.path.exists(EXPERIMENT_BASE_DATASETS_PATH):
        os.makedirs(EXPERIMENT_BASE_DATASETS_PATH)

    # Read the experiment config
    with open(
        os.environ.get("EXPERIMENT_CONFIG_PATH"), "r", encoding="utf-8"
    ) as config_file:
        base_config = json.loads(config_file.read())["augmentation"]

    print()
    print("########################################")
    print("BASE DATASETS")
    print("########################################")
    print()

    clean_base_dataset = prep_data.load_base_dataset(
        saved_raw_merged_df_path=RAW_MERGED_DATASET_PATH,
        cycle_id_count_thresh=25,
        cycle_id_range_thresh=20,
    )

    all_devices = clean_base_dataset["device_uuid"].unique().tolist()

    # Choose 20 random test devices
    n_test_devices = 20
    random.seed(RANDOM_STATE)
    test_devices = random.sample(all_devices, n_test_devices)

    # Save the test dataset
    test_dataset = clean_base_dataset[
        clean_base_dataset["device_uuid"].isin(test_devices)
    ].copy()
    test_dataset_path = f"{EXPERIMENT_BASE_DATASETS_PATH}/test.parquet"
    test_dataset.to_parquet(test_dataset_path)

    # Create a base dataset for different number of training devices
    all_train_devices = list(set(all_devices) - set(test_devices))
    for n_devices in base_config["n_devices"]:
        if n_devices == -1:
            n_devices = len(all_train_devices)

        # Select the random training devices
        random.seed(RANDOM_STATE)
        train_devices = random.sample(all_train_devices, k=n_devices)

        # Generate the augmented data
        for n_augmentation in base_config["n_augmentation"]:

            if n_augmentation == 0:
                dataset_name = f"n_dev={n_devices}|n_aug={n_augmentation}"
                dataset_path = f"{EXPERIMENT_BASE_DATASETS_PATH}/{dataset_name}.parquet"
                clean_base_dataset[
                    clean_base_dataset["device_uuid"].isin(train_devices)
                ].copy().to_parquet(dataset_path)
                continue

            for max_jittering_battery_level in base_config[
                "max_jittering_battery_level"
            ]:
                for max_jittering_measurement_interval in base_config[
                    "max_jittering_measurement_interval"
                ]:
                    for max_jittering_air_temperature in base_config[
                        "max_jittering_air_temperature"
                    ]:
                        dataset_name = (
                            f"n_dev={n_devices}|"
                            f"n_aug={n_augmentation}|"
                            f"noise={max_jittering_battery_level}|"
                            f"noise_temperature={max_jittering_air_temperature}|"
                            f"rand_warp={max_jittering_measurement_interval}|"
                        )
                        if os.path.exists(
                            os.path.join(
                                EXPERIMENT_BASE_DATASETS_PATH,
                                f"{dataset_name}.parquet",
                            )
                        ):
                            print(f"Dataset {dataset_name} already exists. Skipping.")
                            continue
                        print(f"Generating dataset {dataset_name} ...")
                        # Reinit the train_datasets list
                        train_datasets = []
                        # Add the unaugmented data once
                        train_datasets.append(
                            clean_base_dataset[
                                clean_base_dataset["device_uuid"].isin(train_devices)
                            ].copy()
                        )

                        # Add n_augmentation times augmented data
                        for _ in range(n_augmentation):
                            train_datasets.append(
                                prep_data.load_base_dataset(
                                    saved_raw_merged_df_path=RAW_MERGED_DATASET_PATH,
                                    device_subset=train_devices,
                                    add_noise=max_jittering_battery_level != 0,
                                    max_noise=max_jittering_battery_level,
                                    add_noise_temperature=max_jittering_air_temperature
                                    != 0,
                                    max_noise_temperature=max_jittering_air_temperature,
                                    random_warp_status_times=max_jittering_measurement_interval
                                    != 0,
                                    random_max_time_warp_percent=max_jittering_measurement_interval,
                                    cycle_id_count_thresh=25,
                                    cycle_id_range_thresh=20,
                                )
                            )

                        # Create resulting dataset and save it
                        result_dataset = merge_datasets(train_datasets)
                        dataset_path = (
                            f"{EXPERIMENT_BASE_DATASETS_PATH}/{dataset_name}.parquet"
                        )
                        print(f"Saving to {dataset_path}")
                        result_dataset.to_parquet(dataset_path)
