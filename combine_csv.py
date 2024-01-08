import os
import pandas as pd
import gzip

# Specify the root folder where your CSV files are located
root_folder = "D:/clusterdata-2011-2/task_events/"

# Specify the chunk size
chunk_size = 30000

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through all subfolders, including subfolders within subfolders
for folder, _, files in os.walk(root_folder):
    # Loop through each file in the current folder
    for file in files:
        file_path = os.path.join(folder, file)

        # Check if the file is a regular CSV or gzipped CSV
        if file.endswith(".csv"):
            # Read the CSV file into a DataFrame in chunks
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                combined_data = combined_data.append(chunk, ignore_index=True)
        elif file.endswith(".gz"):
            # Read the gzipped CSV file into a DataFrame in chunks
            with gzip.open(file_path, 'rt') as f:
                for chunk in pd.read_csv(f, chunksize=chunk_size):
                    combined_data = combined_data.append(chunk, ignore_index=True)
        else:
            continue  # Skip files that are not CSV or gzipped CSV

# Write the combined data to a new CSV file
combined_data.to_csv("D:/clusterdata-2011-2/combined_data.csv", index=False)

print("Combined data saved to 'D:/clusterdata-2011-2/combined_data.csv'")
