---
Date Received: 06/11/2023
Last Updated: 19/01/2024
---

# Datalogger Datasets

This folder contains raw and cleaned datasets for the datalogger usecase.

## Description

### Raw Datasets

| Type        | Name             | Description                                                     |
| ----------- | ---------------- | --------------------------------------------------------------- |
| Metadata    | `batt_types`     | 2-liner containing details about batteries used in the loggers. |
| Metadata    | `equipment`      | Information on where the loggers are deployed.                  |
| Metadata    | `devices`        | Information regarding loggers (devices) and their capabilities. |
| Time Series | `measurements`   | Time series of temperature measurements taken by the loggers.   |
| Time Series | `devices_status` | Time series of battery levels for each device (Older models).   |
| Time Series | `sensors_status` | Time series of battery levels for each device (Newer models).   |

### Clean Datasets

See repo README.

## Access

The datasets are protected with OpenSSL encryption. Consult @dbinmohdkhir for the password.

### Steps

1. Decrypt the archive. You will be prompted for a password.

`openssl enc -d -aes-256-cbc -md sha512 -pbkdf2 -iter 1000000 -salt -in data.tar.gz.enc -out data.tar.gz`

2. Extract the contents.

`tar -xzvf datasets.tar.gz`

3. The datasets are in the `data` folder.

> To reproduce the encryption, first compress with `tar -czvf data.tar.gz data/`. Then, run `openssl enc -aes-256-cbc -md sha512 -pbkdf2 -iter 1000000 -salt -in data.tar.gz -out data.tar.gz.enc`. Provide a strong password when prompted. Optionally generate the password with BitWarden.

## Code

Preliminary code is stored in a [GitLab repo](https://gitlab.inovex.de/dbinmohdkhir/ba-code). Refer to @dbinmohdkhir for repo access.

## Changelog

19/01/24: Add cleaned datasets

26/11/24: Add README and raw datasets
