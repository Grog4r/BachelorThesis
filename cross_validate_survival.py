import os
import random
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import utilities.generate_survival_dataset as gen_surv
import utilities.preprocess_raw_data as prep_data
import utilities.train_survival_model as train_surv
import utilities.train_utils as train_utils
from dotenv import load_dotenv
from lifelines.fitters.coxph_fitter import CoxPHFitter
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.functions import StepFunction
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator


def get_runs_df() -> pd.DataFrame:
    load_dotenv()
    EXPERIMENT = os.environ.get("EXPERIMENT_NUMBER")
    print(f"The experiment number is {EXPERIMENT}")
    return pd.read_csv(f"./data/runs/surv_runs_{EXPERIMENT}.csv")


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


def ibs_scorer(model, X_test, y):
    (y_train, y_test) = y
    lower, upper = np.percentile(y_test["duration"], [10, 90])
    lower = int(lower)
    upper = int(upper)
    times = np.arange(lower, upper - 1)
    if model.__class__.__name__ == "CoxPHFitter":
        surv_probs = np.array(model.predict_survival_function(X_test, times)).T
    elif model.__class__.__name__ == "RandomSurvivalForest":
        surv_probs = np.row_stack(
            [step_fn(times) for step_fn in model.predict_survival_function(X_test)]
        )
    try:
        return -integrated_brier_score(y_train, y_test, surv_probs, times)
    except:
        print(y_train)
        print(y_test)
        print(surv_probs)
        print(times)
        raise


def cross_validate_survival_model(
    raw_merged_df: pd.DataFrame,
    model_class: str,
    n_aug: int,
    n_dev: int,
    train_df_params: dict[str, Any],
    all_device_uuids: list[str],
    by_metric: str,
    trains_in: Any | None = None,
    tests_in: Any | None = None,
    event_column: str = "event",
    time_column: str = "duration",
    n_splits: int = 4,
    shuffle: bool = True,
    mlflow_experiment: str | None = None,
    VERBOSE: bool = False,
) -> tuple[float, float]:
    """Cross-validates a survival model and returns the mean metric.

    :param raw_merged_df: The merged raw DataFrame to use for the base dataset
    :param model: The model class name to cross-validate
    :param n_aug: The number of augmentation steps for the base dataset
    :param n_dev: The number of training devices for the dataset
    :param train_df_params: The dataset parameters
    :param all_device_uuids: A list of all the device UUIDs
    :param by_metric: The metric to be used for the cross validation
    :param trains_in: A list of the train split indices, if this is None it will create a new split.
        This is used so that the augmented and the unaugmented models are trained and evaluated on the same devices,
        defaults to None
    :param tests_in: A list of the test split indices, if this is None it will create a new split.
        This is used so that the augmented and the unaugmented models are trained and evaluated on the same devices,
        defaults to None
    :param event_column: The column where the event is in, defaults to "event"
    :param time_column: The column where the time is in, defaults to "duration"
    :param n_splits: The number of splits that the cross validation should do, defaults to 4
    :param shuffle: Whether or not the devices should be shuffeled in the splits, defaults to True
    :param VERBOSE: Whether or not to print additional information, defaults to False
    :raises NotImplementedError: Raises a NotImplementedError if the metric is not in the set of allowed metrics
    :return: The mean cv metric, the train split list and the test split list
    """
    load_dotenv()
    if mlflow_experiment is None:
        os.environ.get("MLFLOW_SURVIVAL_CV")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URL"))

    with train_utils.get_mlflow_context(mlflow_experiment=mlflow_experiment) as run:
        mlflow.log_params(train_df_params)
        mlflow.log_param("n_aug", n_aug)
        mlflow.log_param("n_dev", n_dev)
        mlflow.log_param("model_class", model_class)
        mlflow.log_param("by_metric", by_metric)

        if by_metric not in ["metrics.ibs", "metrics.c_index_ipcw"]:
            raise NotImplementedError

        kf = KFold(n_splits=n_splits, shuffle=shuffle)
        cis = []
        ibs_s = []

        if trains_in is None or tests_in is None:
            print("!! Creating new split...")
            split = kf.split(all_device_uuids)
        else:
            if VERBOSE:
                print("trains and tests from last split recieved :)")
            split = zip(trains_in, tests_in)

        trains = []
        tests = []
        coefs_list = []
        exp_coefs_list = []
        perm_imp_ibs_list = []
        for i, (train, test) in enumerate(split):
            print(f"###Split {i+1}/{n_splits}!")

            all_train_uuids_censored = True
            while all_train_uuids_censored:
                if len(train) > n_dev:
                    if VERBOSE:
                        print(f"Selecting a subset of {n_dev} training devices.")
                    random.seed(os.environ.get("RANDOM_STATE"))
                    train = random.sample(list(train), k=n_dev)
                train_uuids = all_device_uuids[train]
                train_base_df = augmented_base_dataset(
                    raw_merged_df, n_aug, train_uuids, train_df_params, VERBOSE=VERBOSE
                )
                all_train_uuids_censored = all(
                    train_base_df.groupby(by="cycle_id")["battery_level_percent"].min()
                    < 20
                )

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

            train_surv_df = gen_surv.base_to_survival_dataset(train_base_df)

            if VERBOSE:
                print("Generating test dataset.")
            test_base_df = augmented_base_dataset(
                raw_merged_df, 0, test_uuids, VERBOSE=VERBOSE
            )
            test_surv_df = gen_surv.base_to_survival_dataset(test_base_df)

            features = list(
                test_surv_df.drop(
                    columns=["device_uuid", "cycle_id", event_column, time_column]
                ).columns
            )
            features = list(set(features).intersection(set(train_surv_df.columns)))

            low_var_features = []
            event = train_surv_df[event_column].astype(bool)
            for feature in features:
                var1 = train_surv_df.loc[event, feature].var()
                var2 = train_surv_df.loc[~event, feature].var()
                if var1 <= 0.15 or var2 <= 0.15:
                    low_var_features.append(feature)

            for feature in low_var_features:
                features.remove(feature)

            if "device_uuid" in features:
                features.remove("device_uuid")

            X_train, y_train = train_surv.df_to_X_y(
                train_surv_df, features, keep_device_uuid=False
            )
            X_test, y_test = train_surv.df_to_X_y(
                test_surv_df, features, keep_device_uuid=False
            )
            train_min = y_train[time_column].min()
            train_max = y_train[time_column].max()
            test_max = y_test[time_column].max()

            if test_max > train_max:
                # Identify rows that will be clipped
                clipped_indices = (y_test[time_column] < train_min) | (
                    y_test[time_column] > train_max
                )

                # Change the event_column to be False for these rows
                y_test[event_column][clipped_indices] = False

                # If you need to clip the values as well
                y_test[time_column] = np.clip(y_test[time_column], train_min, train_max)

            if model_class == "CoxPHFitter":
                model = CoxPHFitter(penalizer=0.01)
                model.fit(
                    train_surv_df[[*features, event_column, time_column]],
                    duration_col=time_column,
                    event_col=event_column,
                )

                ibs = train_surv.calculate_integrated_brier_score(
                    y_train,
                    X_test,
                    y_test,
                    model,
                    model.__class__.__name__,
                )

                prediction = model.predict_partial_hazard(X_test)

                coefs = model.params_.to_dict()
                exp_coefs = model.hazard_ratios_.to_dict()
                y = (y_train, y_test)
                perm_imp_ibs = {
                    k: v
                    for k, v in zip(
                        X_test.columns,
                        permutation_importance(
                            model, X_test, y, n_repeats=10, scoring=ibs_scorer
                        )["importances_mean"],
                    )
                }
                coefs_list.append(coefs)
                exp_coefs_list.append(exp_coefs)
                perm_imp_ibs_list.append(perm_imp_ibs)

            elif model_class == "RandomSurvivalForest":
                model = RandomSurvivalForest()
                model.fit(X_train, y_train)
                ibs = train_surv.calculate_integrated_brier_score(
                    y_train,
                    X_test,
                    y_test,
                    model,
                    model.__class__.__name__,
                )
                prediction = model.predict(X_test)

                y = (y_train, y_test)
                perm_imp_ibs = {
                    k: v
                    for k, v in zip(
                        X_test.columns,
                        permutation_importance(
                            model, X_test, y, n_repeats=10, scoring=ibs_scorer
                        )["importances_mean"],
                    )
                }
                perm_imp_ibs_list.append(perm_imp_ibs)

            elif model_class == "kaplan_meier_estimator":
                model = StepFunction(
                    *kaplan_meier_estimator(y_train[event_column], y_train[time_column])
                )

                tr_max = y_train[time_column].max()
                ts_max = y_test[time_column].max()
                to_remove = []
                if ts_max > tr_max:
                    for i, test_entry in enumerate(y_test):
                        if test_entry[time_column] >= tr_max:
                            to_remove.append(i)

                    y_test = np.delete(y_test, to_remove, axis=0)

                ibs = train_surv.calculate_integrated_brier_score(
                    y_train,
                    X_test,
                    y_test,
                    model,
                    model_class,
                )
                cis.append(0.5)

            ibs_s.append(ibs)

            if model_class != "kaplan_meier_estimator":
                try:
                    tr_max = y_train[time_column].max()
                    ts_max = y_test[time_column].max()

                    to_remove = []
                    if ts_max > tr_max:
                        for i, test_entry in enumerate(y_test):
                            if test_entry[time_column] >= tr_max:
                                to_remove.append(i)

                        y_test = np.delete(y_test, to_remove, axis=0)
                        try:
                            prediction = np.delete(prediction, to_remove, axis=0)
                        except:
                            prediction = prediction.drop(labels=to_remove)

                    c_index_ipcw = concordance_index_ipcw(
                        y_train,
                        y_test,
                        prediction,
                    )[0]
                    cis.append(c_index_ipcw)
                except:
                    print("ERROR WITH THE C_INDEX_IPCW! Leaving out test.")
                    print(f"{n_aug=}, {n_dev=}, {by_metric=}")
                    print(model.__class__.__name__)
                    print(train_df_params)
                    print(y_train)
                    print(y_test)
            else:
                c_index_ipcw = 0.5
                cis.append(c_index_ipcw)

            print(f"{c_index_ipcw=}; {ibs=}")

        mean_cis = np.mean(cis)
        mean_ibs_s = np.mean(ibs_s)

        mlflow.log_metrics(metrics={"c_index_ipcw": mean_cis, "ibs": mean_ibs_s})

        mlflow.log_dict(perm_imp_ibs_list, "dir/perm_imp_ibs.json")
        mlflow.log_dict(coefs_list, "dir/coefs.json")
        mlflow.log_dict(exp_coefs_list, "dir/exp_coefs.json")

        if by_metric == "metrics.ibs":
            score = mean_ibs_s
        elif by_metric == "metrics.c_index_ipcw":
            score = mean_cis

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

        if (
            model_class == "kaplan_meier_estimator"
            and by_metric == "metrics.c_index_ipcw"
        ):
            continue

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

                best = df.sort_values(
                    by=[by_metric], ascending=by_metric == "metrics.ibs"
                ).iloc[counter]

                best_metric = best[by_metric]
                print(
                    f"# Best model WITH{'OUT' if not with_aug else ''} augmentation has a {by_metric} of {best_metric:.3f}"
                )

                # Get best dataset parameters
                n_aug = int(best["params.n_aug"])

                max_jittering_battery_level = best["params.noise"]
                max_jittering_air_temperature = best["params.noise_temperature"]
                max_jittering_measurement_interval = best["params.rand_warp"]

                train_df_params = {
                    "add_noise": max_jittering_battery_level != 0,
                    "max_noise": max_jittering_battery_level,
                    "add_noise_temperature": max_jittering_air_temperature != 0,
                    "max_noise_temperature": max_jittering_air_temperature,
                    "random_warp_status_times": max_jittering_measurement_interval != 0,
                    "random_max_time_warp_percent": max_jittering_measurement_interval,
                }

                # Do the cross validation
                score, trains_out, tests_out = cross_validate_survival_model(
                    raw_merged_df,
                    model_class,
                    n_dev=n_dev,
                    n_aug=n_aug,
                    train_df_params=train_df_params,
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
                print(f"# CV-Metrics: {by_metric}: {score:.3f} ({best_metric:.3f})")
            counter += 1
    return all_scores


def main():
    raw_merged_df = pd.read_parquet("data/my_datasets/raw_merged.parquet")
    runs_df = get_runs_df()
    all_device_uuids = raw_merged_df["device_uuid"].unique()

    for metric in ["metrics.c_index_ipcw", "metrics.ibs"]:
        for model_class in [
            "kaplan_meier_estimator",
            "CoxPHFitter",
            "RandomSurvivalForest",
        ]:
            print(f"Cross Validating {model_class} by {metric}!")
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
