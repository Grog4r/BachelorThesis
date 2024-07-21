import os

os.environ["EXPERIMENT"] = "True"

from typing import Any

import pandas as pd
import train_regression_model as train_reg
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from train_utils import (
    dataset_path_to_dataset_params,
    training_dataset_path_to_test_dataset_path,
)
from xgboost import XGBRegressor

if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()
    EXPERIMENT_REGRESSION_DATASETS_PATH = os.environ.get(
        "EXPERIMENT_REGRESSION_DATASETS_PATH"
    )

    print()
    print("########################################")
    print("REGRESSION TRAINING")
    print("########################################")
    print()

    # Get all the training dataset paths
    all_datasets = os.listdir(EXPERIMENT_REGRESSION_DATASETS_PATH)
    training_dataset_paths = [
        dataset for dataset in all_datasets if "test" not in dataset
    ]

    for training_dataset_path in training_dataset_paths:
        print(training_dataset_path)
        # Find the matching test dataset to the training dataset
        test_dataset_path = training_dataset_path_to_test_dataset_path(
            training_dataset_path
        )
        absolute_test_dataset_path = os.path.join(
            EXPERIMENT_REGRESSION_DATASETS_PATH, test_dataset_path
        )
        if not os.path.exists(absolute_test_dataset_path):
            raise FileNotFoundError(
                f"The Test dataset {absolute_test_dataset_path} could not be found."
            )

        # Resolve the dataset params from the training dataset path
        dataset_params = dataset_path_to_dataset_params(training_dataset_path)

        absolute_training_dataset_path = os.path.join(
            EXPERIMENT_REGRESSION_DATASETS_PATH, training_dataset_path
        )

        # Set the features for training
        all_columns = pd.read_parquet(absolute_training_dataset_path).columns
        features = all_columns.drop(
            [
                "status_time",
                "device_uuid",
                "target",
                "cycle_id",
                "left_peak_border",
                "right_peak_border",
                "peak_label",
                "2.0",
            ], errors="ignore"
        )

        # Train all the models
        for model_class, model_params in [
            (LinearRegression, {"n_jobs": -1}),
            (DecisionTreeRegressor, {}),
            (XGBRegressor, {}),
        ]:
            train_reg.experiment_train_regression_model(
                model_class=model_class,
                train_df_path=absolute_training_dataset_path,
                test_df_path=absolute_test_dataset_path,
                model_params=model_params,
                dataset_params=dataset_params,
                features=features,
            )
