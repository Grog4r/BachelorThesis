# BA Niklas

## File and directory explanation

### `./data/` directory

- [./data/aug_effect/](./data/aug_effect/) contains the experiment run CSV files for the augmentation effect experiments.
- [./data/clean/](./data/clean/) contains the clean datasets.
- [./data/cross_validation/](./data/cross_validation/) contains the experiment run CSV files for the cross validation experiments.
- [./data/cross_validation/download_cross_validation_csvs.py](./data/cross_validation/download_cross_validation_csvs.py) will download the run infos for the cross validation experiments.
- [./data/experiment_datasets/](./data/experiment_datasets/) contains the experiment datasets.
- [./data/experiment_datasets/config.json](./data/experiment_datasets/config.json) contains configuration for the experiment datasets.
- [./data/my_datasets/](./data/my_datasets/) contains some useful datasets created in the scripts.
- [./data/raw/](./data/raw/) contains the raw datasets.
- [./data/runs/](./data/runs/) contains the experiment run CSV files from the experiments without cross validation.
- [./data/runs/download_run_csvs.py](./data/runs/download_run_csvs.py) will download the experiment run infos with the `EXPERIMENT_NUMBER` from the [.env](.env) file.

### `./utilities/` directory

- [./utilities/augmentation.py](./utilities/augmentation.py) has all the functionality for augmenting datasets.
- [./utilities/base_experiment_datasets.py](./utilities/base_experiment_datasets.py) creates the base datasets for the experiments.
- [./utilities/generate_regression_dataset.py](./utilities/generate_regression_dataset.py) has all the functionality for generating the regression datasets.
- [./utilities/generate_survival_dataset.py](./utilities/generate_survival_dataset.py) has all the functionality for generating the survival datasets.
- [./utilities/latex_figures.py](./utilities/generate_survival_dataset.py) some helper classes to generate figures and subfigures for latex.
- [./utilities/latex_tables.py](./utilities/generate_survival_dataset.py) some helper classes to compare runs and generate tables for latex.
- [./utilities/plotting.py](./utilities/plotting.py) has functions that are used for plotting different things.
- [./utilities/preprocess_raw_data.py](./utilities/preprocess_raw_data.py) has all the functionality for preprocessing the raw data into base datasets.
- [./utilities/regression_experiment_datasets.py](./utilities/regression_experiment_datasets.py) creates the regression datasets for the experiments.
- [./utilities/regression_experiment_training.py](./utilities/regression_experiment_training.py) trains the regression models for the experiments.
- [./utilities/run_experiments.sh](./utilities/run_experiments.sh) executes all the python scripts for the experiments.
- [./utilities/survival_experiment_datasets.py](./utilities/survival_experiment_datasets.py) creates the survival dataset for the experiments.
- [./utilities/survival_experiment_training.py](./utilities/survival_experiment_training.py) trains the survival models for the experiments.
- [./utilities/train_regression_model.py](./utilities/train_regression_model.py) has all the functionality for training the regression models.
- [./utilities/train_survival_model.py](./utilities/train_survival_model.py) has all the functionality for training the survival models.
- [./utilities/train_utils.py](./utilities/train_utils.py) has some utilities for training models.

### `./` directory

- [.env](.env) contains environment variables used for the experiments.
- [cross_validate_regression.py](cross_validate_regression.py) is the script that cross validates the n best regression models from the experiments. This also has all the cross validation functionality for the regression.
- [cross_validate_survival.py](cross_validate_survival.py) is the script that cross validates the n best survival models from the experiments. This also has all the cross validation functionality for the survival.
- [evaluate_cross_validation_regression.ipynb](evaluate_cross_validation_regression.ipynb) is the evaluation notebook for the cross validation of the regression experiments.
- [evaluate_cross_validation_survival.ipynb](evaluate_cross_validation_survival.ipynb) is the evaluation notebook for the cross validation of the survival experiments.
- [evaluate_regression_experiment.ipynb](evaluate_regression_experiment.ipynb) is the evaluation notebook for the regression experiment.
- [evaluate_survival_experiment.ipynb](evaluate_survival_experiment.ipynb) is the evaluation notebook for the survival experiment.
- [find_best_reg_aug_params.ipynb](find_best_reg_aug_params.ipynb) is the notebook that tests and finds the best augmentation parameters for the regression models.
- [find_best_rsurv_aug_params.ipynb](find_best_surv_aug_params.ipynb) is the notebook that tests and finds the best augmentation parameters for the survival models.
- The other notebooks all contain some testing code for different parts of the code or they generate plots and other things for the written Thesis.

## Environment

To create a new environment use conda: `conda create -n ba_niklas python=3.11`. Then activate it: `conda activate ba_niklas`.

You can then install the required packages by running `pip install -r requirements.txt`.

## Experiments

### Dataset Download and Unpacking

When first downloading the repository the directories [./data/raw/](./data/raw/) and [./data/clean/](./data/clean/) will be empty. To get these files you will need to first download the following ZIP archive from GDrive into the root repository directory: [DeKiOpsLoggerData.zip](https://drive.google.com/file/d/1mRFIQrTcjMtPeZojhUipEChUGF7mm8UJ/view?usp=drive_link).

You will then need to run the [setup_data.sh](./setup_data.sh) script. Make sure it is executable (`chmod +x setup_data.sh`). The script will ask you for an AES-256-CBC password. You will need to ask [Niklas Küchen](mailto:niklas.kuechen@inovex.de) or [Muhammad Daniel Bin Mohd Khir](mailto:daniel.binmohdkhir@inovex.de) for that password.

### Basic Experiment Execution

First the path to the data directory should be filled into the [.env](.env) file under `DATA_DIR_PATH`.

The main code used for the experiments is in the [utilities](./utilities/) directory.

To start a new experiment the configuration file [./data/experiment_datasets/config.json](./data/experiment_datasets/config.json) has to be set with the wished configuration. Then a new unused experiment id has to be set in the [.env](.env) file under `EXPERIMENT_NUMBER`.

MLflow needs to be started by using `mlflow ui --backend-store-uri SAVE_URL` where `SAVE_URL` needs to be set to a valid path where the MLflow files should be logged to. The same URL needs to be filled into the [.env](.env) file under `MLFLOW_TRACKING_URL`.

Then the experiment can be started by running the [./utilities/run_experiments.sh](./utilities/run_experiments.sh) bash script. The script will do the following:

- The base datasets will be generated
- The regression datasets will be generated
- The regression models will be trained
- The survival datasets will be generated
- The survival models will be trained

Note that executing the experiments may take multiple hours depending on the number of different options set in the configuration file [./data/experiment_datasets/config.json](./data/experiment_datasets/config.json).

After this the run infos from the experiments can be downloaded by using the [./data/runs/download_run_csvs.py](./data/runs/download_run_csvs.py) script. The generated commands will be shown in the terminal and need to be accepted by pressing enter.

Then as a last step the evaluation notebooks can be executed. Note that the variable `EXPERIMENT` at the top of the notebooks need to be set to the number of the experiment you want to evaluate (e.g. the value of the `EXPERIMENT_NUMBER` from the [.env](.env) file).

### Cross Validation

To cross validate the best runs of the basic experiments, the scripts [cross_validate_regression.py](cross_validate_regression.py) and [cross_validate_survival.py](cross_validate_survival.py) exist. They will run a 4-Fold cross validation for the best n (default 8) models for each metric, model and number of training devices.
The cross validation experiments will also be maintained in MLflow. After the scripts are done, the script [./data/cross_validation/download_cross_validation_csvs.py](./data/cross_validation/download_cross_validation_csvs.py) can be used to download the cross validation results. The notebooks [evaluate_cross_validation_regression.ipynb](evaluate_cross_validation_regression.ipynb) and [evaluate_cross_validation_survival.ipynb](evaluate_cross_validation_survival.ipynb) can be used to evaluate these results.

### Cross Validated best augmentation parameters

To find out the best augmentation parameters like the jittering strength or the amout of augmented data to add, there are the two notebooks [find_best_reg_aug_params.ipynb](find_best_reg_aug_params.ipynb) and [find_best_surv_aug_params.ipynb](find_best_surv_aug_params.ipynb). They run the cross validation and evaluate the results directly. These notebooks may both take several (5-12) hours to run all the experiments.
