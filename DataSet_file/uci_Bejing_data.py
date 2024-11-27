import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsdb import pickle_dump
from dataset_processing_utils import (
    window_truncate,
    add_artificial_mask,
    saving_into_h5,
    saving_in_h5,
    compute_delta,
)



# Set the main parameters for data processing
# Change file_path to the directory containing the CSV file
file_path = "D:\proj\Time series imputation\Airquality_DataSets\sample data set od air quality"  
artificial_missing_rate = 0.1  # Percentage of values to mask as missing
seq_len = 100  # Length of each time series sequence
dataset_name = "Air_time_series_dataset"  # Name for saving the processed dataset
saving_path = "D:\proj\Time series imputation\Airquality_DataSets"  # Folder to save the processed dataset

# Create the directory for saving if it doesn't exist
dataset_saving_dir = os.path.join(saving_path, dataset_name)
os.makedirs(dataset_saving_dir, exist_ok=True)

# Collect and process all files in the dataset directory
df_collector = []
station_name_collector = []
for filename in os.listdir(file_path):
    # Check if it's a file and ends with '.csv' to avoid processing directories or other file types
    if os.path.isfile(os.path.join(file_path, filename)) and filename.endswith('.csv'):
        file_path_full = os.path.join(file_path, filename)
        current_df = pd.read_csv(file_path_full)
        current_df["date_time"] = pd.to_datetime(current_df[["year", "month", "day", "hour"]])
        station_name_collector.append(current_df.loc[0, "station"])

        # Drop unnecessary columns
        current_df = current_df.drop(["year", "month", "day", "hour", "wd", "No", "station"], axis=1)
        df_collector.append(current_df)



# Combine data and set feature names
date_time = df_collector[0]["date_time"]
df_collector = [i.drop("date_time", axis=1) for i in df_collector]
df = pd.concat(df_collector, axis=1)
feature_names = [
    f"{station}_{feature}"
    for station in station_name_collector
    for feature in df_collector[0].columns
]
df.columns = feature_names
df["date_time"] = date_time

# Split the data into training, validation, and test sets by unique months
unique_months = df["date_time"].dt.to_period("M").unique()
test_months = unique_months[:10]
val_months = unique_months[10:20]
train_months = unique_months[20:]

test_set = df[df["date_time"].dt.to_period("M").isin(test_months)]
val_set = df[df["date_time"].dt.to_period("M").isin(val_months)]
train_set = df[df["date_time"].dt.to_period("M").isin(train_months)]

# Standardize the data
scaler = StandardScaler()
train_set_X = scaler.fit_transform(train_set[feature_names])
val_set_X = scaler.transform(val_set[feature_names])
test_set_X = scaler.transform(test_set[feature_names])

# Truncate the data into sequences
train_set_X = window_truncate(train_set_X, seq_len)
val_set_X = window_truncate(val_set_X, seq_len)
test_set_X = window_truncate(test_set_X, seq_len)

# Add artificial missing values and masks
train_set_dict = add_artificial_mask(train_set_X, artificial_missing_rate, "train")
val_set_dict = add_artificial_mask(val_set_X, artificial_missing_rate, "val")
test_set_dict = add_artificial_mask(test_set_X, artificial_missing_rate, "test")

# Save the processed data and scaler
processed_data = {"train": train_set_dict, "val": val_set_dict, "test": test_set_dict}
saving_in_h5(dataset_saving_dir, processed_data, classification_dataset=False)
pickle_dump(scaler, os.path.join(dataset_saving_dir, 'scaler'))

print(f"Dataset created and saved in '{dataset_saving_dir}'")
