#!/bin/bash

# Preprocess the raw datasets into the different augmented base datasets
python base_experiment_datasets.py

# Convert the base datasets into regression datasets
python regression_experiment_datasets.py

# Train the regression models
python regression_experiment_training.py

# Convert the base datasets into survival datasets
python survival_experiment_datasets.py

# Train the survival models
python survival_experiment_training.py
