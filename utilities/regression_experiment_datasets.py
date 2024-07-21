import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import generate_regression_dataset as gen_reg
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

# Load the environment variables
load_dotenv()
EXPERIMENT_BASE_DATASETS_PATH = os.environ.get("EXPERIMENT_BASE_DATASETS_PATH")
EXPERIMENT_REGRESSION_DATASETS_PATH = os.environ.get(
    "EXPERIMENT_REGRESSION_DATASETS_PATH"
)

# Read the config file
with open(
    os.environ.get("EXPERIMENT_CONFIG_PATH"), "r", encoding="utf-8"
) as config_file:
    REG_CONFIG = json.loads(config_file.read())["regression"]


def preprocess_dataset(dataset_name: str) -> None:
    print(dataset_name)
    base_path = f"{EXPERIMENT_BASE_DATASETS_PATH}/{dataset_name}"
    df = pd.read_parquet(base_path)

    # configure the rolling window sizes and metrics to generate for the different cols
    rolling_windows = [5, 50]
    metric_column_names = [
        ("median", "battery_level_percent"),
        ("median", "battery_diff"),
    ]

    # Generate the regression dataset for each value of prediction horizon
    for prediction_horizon in REG_CONFIG["prediction_horizon"]:

        # Set the dataset save path
        if "test" not in dataset_name:
            dataset_base_name = dataset_name.split(".parquet")[0]
            reg_path = os.path.join(
                EXPERIMENT_REGRESSION_DATASETS_PATH,
                f"{dataset_base_name}|pred_hor={prediction_horizon}.parquet",
            )
        else:
            reg_path = os.path.join(
                EXPERIMENT_REGRESSION_DATASETS_PATH,
                f"pred_hor={prediction_horizon}.test.parquet",
            )

        if os.path.exists(reg_path):
            print(f"Dataset {dataset_name} already exists. Skipping...")
            return

        reg_df = gen_reg.base_to_regression_dataset(
            df,
            prediction_horizon=prediction_horizon,
            rolling_windows=rolling_windows,
            metric_column_names=metric_column_names,
        )

        # Save the dataset
        reg_df.to_parquet(reg_path)


def main() -> None:
    # Make sure the destination directory exists
    if not os.path.exists(EXPERIMENT_REGRESSION_DATASETS_PATH):
        os.makedirs(EXPERIMENT_REGRESSION_DATASETS_PATH)

    print()
    print("########################################")
    print("REGRESSION DATASETS")
    print("########################################")
    print()

    # Detect test and training datasets
    dataset_names = os.listdir(EXPERIMENT_BASE_DATASETS_PATH)

    with ThreadPoolExecutor() as executor:
        executor.map(preprocess_dataset, dataset_names)


if __name__ == "__main__":
    main()
