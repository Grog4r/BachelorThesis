import os

import mlflow
from dotenv import load_dotenv

load_dotenv()


mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URL"))

run_path = "/home/nkuechen/Documents/Thesis"
os.chdir(run_path)
print(os.getcwd())

surv_name = os.environ.get("MLFLOW_SURVIVAL_CV")
surv_id = mlflow.get_experiment_by_name(surv_name)
if surv_id is None:
    print(f"Could not find experiment {surv_name}")
else:
    command = f"mlflow experiments csv -o code/thesis_code/data/cross_validation/surv_cross_validation.csv -x {surv_id.experiment_id}"
    input(command)
    os.system(command)

reg_name = os.environ.get("MLFLOW_REGRESSION_CV")
reg_id = mlflow.get_experiment_by_name(reg_name)
if reg_id is None:
    print(f"Could not find experiment {reg_name}")
else:
    command = f"mlflow experiments csv -o code/thesis_code/data/cross_validation/reg_cross_validation.csv -x {reg_id.experiment_id}"
    input(command)
    os.system(command)
