#!/bin/bash

# Script to move CSV files to directories named after their filenames

# Check if any CSV files exist in current directory
if ! ls *.csv >/dev/null 2>&1; then
    echo "No CSV files found in current directory."
    exit 1
fi

# Process each CSV file
for csv_file in *.csv; do
    # Skip if it's not a regular file
    if [[ ! -f "$csv_file" ]]; then
        continue
    fi
    
    # Get filename without extension
    filename=$(basename "$csv_file" .csv)
    
    # Create directory if it doesn't exist
    if [[ ! -d "$filename" ]]; then
        mkdir -p "$filename"
        echo "Created directory: $filename"
    fi
    
    # Move the CSV file to the directory
    mv "$csv_file" "$filename/"
    echo "Moved $csv_file to $filename/"
done

echo "All CSV files have been moved to their respective directories."