import os

import pandas as pd
from dotenv import load_dotenv
from lifelines.fitters.coxph_fitter import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.nonparametric import kaplan_meier_estimator

os.environ["EXPERIMENT"] = "True"

import train_survival_model as train_surv
from train_utils import dataset_path_to_dataset_params

if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()
    EXPERIMENT_SURVIVAL_DATASETS_PATH = os.environ.get(
        "EXPERIMENT_SURVIVAL_DATASETS_PATH"
    )

    print()
    print("########################################")
    print("SURVIVAL TRAINING")
    print("########################################")
    print()

    all_datasets = os.listdir(EXPERIMENT_SURVIVAL_DATASETS_PATH)

    training_dataset_paths = [
        dataset for dataset in all_datasets if "test" not in dataset
    ]

    # Get all the training dataset paths
    for training_dataset_path in training_dataset_paths:
        print(training_dataset_path)
        test_dataset_path = "test.parquet"
        absolute_test_dataset_path = os.path.join(
            EXPERIMENT_SURVIVAL_DATASETS_PATH, test_dataset_path
        )
        if not os.path.exists(absolute_test_dataset_path):
            raise FileNotFoundError(
                f"The Test dataset {absolute_test_dataset_path} could not be found."
            )

        # Resolve the dataset params from the training dataset path
        dataset_params = dataset_path_to_dataset_params(training_dataset_path)

        absolute_training_dataset_path = os.path.join(
            EXPERIMENT_SURVIVAL_DATASETS_PATH, training_dataset_path
        )

        # Set the features for training
        all_columns = set(
            pd.read_parquet(absolute_training_dataset_path).columns.tolist()
        )
        print(all_columns)

        columns_to_drop = {
            "device_uuid",
            "cycle_id",
            "duration",
            "event",
            "batt_diff",
            "batt_min"
            "batt_median",
            "daily_roc",
            "temp_diff",
            "temp_max",
            "radio_diff",
            "battery_type_id_2.0",
        }
        features = list(all_columns - columns_to_drop)

        # Train all the models
        for model_class, model_params in [
            (kaplan_meier_estimator, {}),
            # (CoxPHSurvivalAnalysis, {"alpha": 0.01}),
            # The lifelines CoxPH model is better for some reason
            (CoxPHFitter, {"penalizer": 0.01}),
            (RandomSurvivalForest, {}),
        ]:
            train_surv.experiment_train_survival_model(
                model_class=model_class,
                train_df_path=absolute_training_dataset_path,
                test_df_path=absolute_test_dataset_path,
                model_params=model_params,
                dataset_params=dataset_params,
                features=features,
            )
