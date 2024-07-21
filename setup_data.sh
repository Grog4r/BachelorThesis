#!/bin/bash
# Decrypts and extracts datasets from archive.

temp_folder="DeKiOpsLoggerData"
dataset_zip="DeKiOpsLoggerData.zip"

# Check if dataset zip exists
if [ -e "$dataset_zip" ]; then
    echo "Unzipping archive..."
    unzip $dataset_zip

    echo "Decrypting archive..."
    openssl enc -d -aes-256-cbc -md sha512 -pbkdf2 -iter 1000000 -salt -in "$temp_folder/data.tar.gz.enc" -out "$temp_folder/data.tar.gz"

    # Check exit status
    if [ $? -ne 0 ]; then
        echo "Error: OpenSSL decryption failed. Bad decrypt."
    else
        echo "Decryption successful. Extracting data..."
        tar -xzvf "$temp_folder/data.tar.gz"
        mv $temp_folder/README.md data/
    fi

    echo "Cleaning up..."
    rm -r $temp_folder
else
    echo "File does not exist: $dataset_zip"
fi

echo "All done!"
