import os
import pickle
import random
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import utilities.generate_regression_dataset as gen_reg
import utilities.preprocess_raw_data as prep_data
import utilities.train_regression_model as train_reg
import utilities.train_utils as train_utils
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def load_model(artifact_uri: str) -> Any:
    """Loads a model from the specified artifact uri.

    :param artifact_uri: The artifact uri
    :return: The loaded model
    """
    for file_name in os.listdir(artifact_uri):
        if file_name.endswith(".pickle"):
            return pickle.load(open(f"{artifact_uri}/{file_name}", "rb"))


def get_runs_df() -> pd.DataFrame:
    load_dotenv()
    EXPERIMENT = os.environ.get("EXPERIMENT_NUMBER")
    print(f"The experiment number is {EXPERIMENT}")
    return pd.read_csv(f"./data/runs/reg_runs_{EXPERIMENT}.csv")


def augmented_base_dataset(
    raw_merged_df: pd.DataFrame,
    n_aug: int,
    device_subset: list[str],
    params: dict[str, Any] | None = None,
    VERBOSE: bool = False,
) -> pd.DataFrame:
    """Augments a base dataset with the specified parameters

    :param raw_merged_df: The merged raw DataFrame to use for the base dataset
    :param n_aug: The number of augmentation steps to perform
    :param device_subset: The subset of devices to use for the dataset
    :param params: The dataset parameters, defaults to None
    :param VERBOSE: If the augmentation should print more information, defaults to False
    :return: The augmented base dataset
    """
    if params is None:
        params = {}

    resulting_dfs = []

    clean_df = prep_data.load_base_dataset(
        raw_merged_df=raw_merged_df,
        device_subset=device_subset,
    )
    resulting_dfs.append(clean_df)

    for i in range(n_aug):
        if VERBOSE:
            print(f"Augmentation step {i+1}/{n_aug}.")
        aug_df = prep_data.load_base_dataset(
            raw_merged_df=raw_merged_df,
            device_subset=device_subset,
            **params,
        )
        resulting_dfs.append(aug_df)

    result_df = pd.concat(resulting_dfs, axis="index").reset_index(drop=True)
    return result_df


def cross_validate_regression_model(
    raw_merged_df: pd.DataFrame,
    model: Any,
    n_aug: int,
    n_dev: int,
    pred_hor: int,
    train_df_params: dict[str, Any],
    features: list[str],
    all_device_uuids: list[str],
    by_metric: str,
    trains_in: Any | None = None,
    tests_in: Any | None = None,
    target_column: str = "target",
    n_splits: int = 4,
    shuffle: bool = True,
    mlflow_experiment: str | None = None,
    VERBOSE: bool = False,
) -> tuple[float, list[str], list[str]]:
    """Cross-validates a regression model and returns the mean metric.

    :param raw_merged_df: The merged raw DataFrame to use for the base dataset
    :param model: The model to cross-validate
    :param n_aug: The number of augmentation steps for the base dataset
    :param n_dev: The number of training devices for the dataset
    :param pred_hor: The prediction horizon for the regression model
    :param train_df_params: The dataset parameters
    :param features: The features
    :param all_device_uuids: A list of all the device UUIDs
    :param by_metric: The metric to be used for the cross validation
    :param trains_in: A list of the train split indices, if this is None it will create a new split.
        This is used so that the augmented and the unaugmented models are trained and evaluated on the same devices,
        defaults to None
    :param tests_in: A list of the test split indices, if this is None it will create a new split.
        This is used so that the augmented and the unaugmented models are trained and evaluated on the same devices,
        defaults to None
    :param target_column: The column where the target is in, defaults to "target"
    :param n_splits: The number of splits that the cross validation should do, defaults to 4
    :param shuffle: Whether or not the devices should be shuffeled in the splits, defaults to True
    :param VERBOSE: Whether or not to print additional information, defaults to False
    :raises NotImplementedError: Raises a NotImplementedError if the metric is not in the set of allowed metrics
    :return: The mean cv metric, the train split list and the test split list
    """
    load_dotenv()
    if mlflow_experiment is None:
        mlflow_experiment = os.environ.get("MLFLOW_REGRESSION_CV")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URL"))

    with train_utils.get_mlflow_context(mlflow_experiment=mlflow_experiment) as run:
        mlflow.log_params(train_df_params)
        mlflow.log_param("n_aug", n_aug)
        mlflow.log_param("n_dev", n_dev)
        mlflow.log_param("pred_hor", pred_hor)
        mlflow.log_param("model_class", model.__class__.__name__)
        mlflow.log_param("by_metric", by_metric)

        if by_metric not in ["metrics.mdt", "metrics.med_dt"]:
            raise NotImplementedError

        kf = KFold(n_splits=n_splits, shuffle=shuffle)
        mdts = []
        med_dts = []
        stds = []
        if trains_in is None or tests_in is None:
            print("!! Creating new split...")
            split = kf.split(all_device_uuids)
        else:
            if VERBOSE:
                print("trains and tests from last split recieved :)")
            split = zip(trains_in, tests_in)

        trains = []
        tests = []
        for i, (train, test) in enumerate(split):
            print(f"###Split {i+1}/{n_splits}!")

            if len(train) > n_dev:
                if VERBOSE:
                    print(f"Selecting a subset of {n_dev} training devices.")
                random.seed(os.environ.get("RANDOM_STATE"))
                train = random.sample(list(train), k=n_dev)

            if VERBOSE:
                print("####", train, test)

            trains.append(train)
            tests.append(test)

            train_uuids = all_device_uuids[train]
            test_uuids = all_device_uuids[test]

            if VERBOSE:
                print(
                    f"{len(train_uuids)} train devices; {len(test_uuids)} test devices."
                )

            if VERBOSE:
                print("Generating train dataset.")

            train_base_df = augmented_base_dataset(
                raw_merged_df, n_aug, train_uuids, train_df_params, VERBOSE=VERBOSE
            )
            train_reg_df = gen_reg.base_to_regression_dataset(
                train_base_df, prediction_horizon=pred_hor
            )

            if VERBOSE:
                print("Generating test dataset.")
            test_base_df = augmented_base_dataset(
                raw_merged_df, 0, test_uuids, VERBOSE=VERBOSE
            )
            test_reg_df = gen_reg.base_to_regression_dataset(
                test_base_df, prediction_horizon=pred_hor
            )

            features = list(
                set(test_reg_df.columns)
                .intersection(set(features))
                .intersection(set(train_reg_df))
            )

            X_train, y_train = train_reg.df_to_X_y(
                train_reg_df, features, target_column=target_column
            )

            model.fit(X_train, y_train)
            mdt, _, med_dt, _, _, std = train_reg.divergence_time_metrics(
                model,
                test_reg_df,
                prediction_horizon=pred_hor,
            )

            mdts.append(mdt)
            med_dts.append(med_dt)
            stds.append(std)
            print(f"{mdt=}; {med_dt=}; {std=}")

        mean_mdt = np.mean(mdts)
        mean_med_dt = np.mean(med_dts)
        mean_std = np.mean(stds)

        mlflow.log_metrics({"mdt": mean_mdt, "med_dt": mean_med_dt, "std": mean_std})

    if by_metric == "metrics.mdt":
        score = mean_mdt
    else:
        score = mean_med_dt

    return score, trains, tests


def cross_validate_best_models_by_metric(
    raw_merged_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    by_metric: str,
    model_class: str,
    all_device_uuids: list[str],
    n_best_models_to_check: int,
    counter_start: int = 0,
    n_dev_subset: list[int] | None = None,
    n_splits: int = 4,
    shuffle: bool = True,
    VERBOSE: bool = False,
) -> dict[str, float]:
    """Performs a cross-validation on all the best models for each number of training devices
        and compares them to the best unaugmented models.

    :param raw_merged_df: The merged raw DataFrame to use for the base dataset
    :param runs_df: The DataFrame containing the run logs
    :param by_metric: The metric to use for the cross-validation
    :param model_class: The model class to cross validate
    :param n_best_models_to_check: The number of best models to check
    :param n_dev_subset: The subset of n_dev values to check, if this is None it will check all,
        defaults to None
    :param n_splits: The number of splits to do for the cross-validation, defaults to 4
    :param shuffle: Whether or not to shuffle the devices in the splits, defaults to True
    :param VERBOSE: Whether or not to print additional information, defaults to False
    :return: A dictionary with all the cross-validation scores for the different validated models
    """
    all_scores = {}

    if n_dev_subset is not None:
        runs_df = runs_df[runs_df["params.n_dev"].isin(n_dev_subset)]

    for n_dev, sub_group in runs_df.sort_values(by=["params.n_dev"]).groupby(
        "params.n_dev"
    ):
        trains_out = None
        tests_out = None

        counter = counter_start
        while counter < n_best_models_to_check:
            print(
                f"# Calculating the CV for model class {model_class} by {by_metric} with {n_dev} training devices."
            )
            for with_aug in [False, True]:
                if with_aug:
                    df = sub_group[sub_group["params.n_aug"] != 0]
                else:
                    if counter > 0:
                        continue
                    df = sub_group[sub_group["params.n_aug"] == 0]

                best = df.sort_values(by=[by_metric], ascending=False).iloc[counter]

                best_metric = best[by_metric]
                print(
                    f"## Best model WITH{'OUT' if not with_aug else ''} augmentation has had a {by_metric} of {best_metric:.3f}"
                )

                # Get best Model
                best_artifact_uri = best["artifact_uri"]
                model = load_model(best_artifact_uri)
                features = list(model.feature_names_in_)

                if model_class == "LinearRegression":
                    model = LinearRegression(n_jobs=8)
                elif model_class == "DecisionTreeRegressor":
                    model = DecisionTreeRegressor()
                elif model_class == "XGBRegressor":
                    model = XGBRegressor()

                # Get best dataset parameters
                n_aug = int(best["params.n_aug"])
                pred_hor = best["params.pred_hor"]

                max_jittering_battery_level = best["params.noise"]
                max_jittering_air_temperature = best["params.noise_temperature"]
                max_jittering_measurement_interval = best["params.rand_warp"]

                train_df_aug_params = {
                    "add_noise": max_jittering_battery_level != 0,
                    "max_noise": max_jittering_battery_level,
                    "add_noise_temperature": max_jittering_air_temperature != 0,
                    "max_noise_temperature": max_jittering_air_temperature,
                    "random_warp_status_times": max_jittering_measurement_interval != 0,
                    "random_max_time_warp_percent": max_jittering_measurement_interval,
                }

                # Do the cross validation
                score, trains_out, tests_out = cross_validate_regression_model(
                    raw_merged_df,
                    model,
                    n_aug=n_aug,
                    n_dev=n_dev,
                    pred_hor=pred_hor,
                    train_df_params=train_df_aug_params,
                    features=features,
                    all_device_uuids=all_device_uuids,
                    by_metric=by_metric,
                    trains_in=trains_out,
                    tests_in=tests_out,
                    n_splits=n_splits,
                    shuffle=shuffle,
                    VERBOSE=VERBOSE,
                )
                all_scores[
                    f"{model_class}_{n_dev}_{'with_aug' if with_aug else 'no_aug'}"
                ] = score
                print(f"## CV-Metrics: {by_metric}: {score:.3f} ({best_metric:.3f})")
            counter += 1
    return all_scores


def main():
    raw_merged_df = pd.read_parquet("data/my_datasets/raw_merged.parquet")
    runs_df = get_runs_df()
    all_device_uuids = raw_merged_df["device_uuid"].unique()

    for metric in ["metrics.mdt", "metrics.med_dt"]:
        for model_class in [
            "LinearRegression",
            "DecisionTreeRegressor",
            "XGBRegressor",
        ]:
            cross_validate_best_models_by_metric(
                raw_merged_df,
                runs_df,
                by_metric=metric,
                model_class=model_class,
                all_device_uuids=all_device_uuids,
                counter_start=0,
                n_best_models_to_check=8,
                n_dev_subset=[10, 20, 40, 63],
            )


if __name__ == "__main__":
    main()
