import argparse
import os

import mlflow
from dotenv import load_dotenv

load_dotenv()


mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URL"))

parser = argparse.ArgumentParser(description="Download experiment logs from mlflow.")
parser.add_argument(
    "--id",
    "-i",
    type=int,
    help="The experiment number to download for.",
    default=os.environ.get("EXPERIMENT_NUMBER"),
)
args = parser.parse_args()


run_path = "/home/nkuechen/Documents/Thesis"
os.chdir(run_path)
print(os.getcwd())

surv_name = f"survival_experiment_{args.id}"
surv_id = mlflow.get_experiment_by_name(surv_name)
if surv_id is None:
    print(f"Could not find experiment {surv_name}")
else:
    command = f"mlflow experiments csv -o code/thesis_code/data/runs/surv_runs_{args.id}.csv -x {surv_id.experiment_id}"
    input(command)
    os.system(command)

reg_name = f"regression_experiment_{args.id}"
reg_id = mlflow.get_experiment_by_name(reg_name)
if reg_id is None:
    print(f"Could not find experiment {reg_name}")
else:
    command = f"mlflow experiments csv -o code/thesis_code/data/runs/reg_runs_{args.id}.csv -x {reg_id.experiment_id}"
    input(command)
    os.system(command)
