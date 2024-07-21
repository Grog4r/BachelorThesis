import os
from concurrent.futures import ThreadPoolExecutor

import generate_survival_dataset as gen_surv
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
EXPERIMENT_BASE_DATASETS_PATH = os.environ.get("EXPERIMENT_BASE_DATASETS_PATH")
EXPERIMENT_SURVIVAL_DATASETS_PATH = os.environ.get("EXPERIMENT_SURVIVAL_DATASETS_PATH")


def preprocess_dataset(dataset_name: str) -> None:
    surv_path = os.path.join(EXPERIMENT_SURVIVAL_DATASETS_PATH, dataset_name)
    if os.path.exists(surv_path):
        print(f"Dataset {dataset_name} already exists. Skipping...")
        return

    base_path = f"{EXPERIMENT_BASE_DATASETS_PATH}/{dataset_name}"
    print(f"Processing dataset to {surv_path}...")
    df = pd.read_parquet(base_path)

    surv_df = gen_surv.base_to_survival_dataset(df)
    surv_df.to_parquet(surv_path)


def main() -> None:
    # Make sure the destination directory exists
    if not os.path.exists(EXPERIMENT_SURVIVAL_DATASETS_PATH):
        os.makedirs(EXPERIMENT_SURVIVAL_DATASETS_PATH)

    print()
    print("########################################")
    print("SURVIVAL DATASETS")
    print("########################################")
    print()

    # Convert all the base datasets to survival datasets using a thread pool
    base_datasets = os.listdir(EXPERIMENT_BASE_DATASETS_PATH)

    with ThreadPoolExecutor() as executor:
        executor.map(preprocess_dataset, base_datasets)


if __name__ == "__main__":
    main()
