"""
FILE: file.py
PURPOSE:
This script loads the Mine-Intel project dataset (original_data.xlsx) 
from a local folder, checks that it has been read correctly, 
and displays the main input and output columns for roof fall rate analysis.

STEPS PERFORMED:
1. Import the pandas library for data handling.
2. Read the Excel file (original_data.xlsx) from the same folder.
3. Print the total number of rows and columns in the dataset.
4. Display all column names present in the Excel file.
5. Select only the 5 major input parameters and the output:
   - CMRR                → Coal Mine Roof Rating
   - PRSUP               → Primary Roof Support
   - depth_of_ cover     → Depth of Cover
   - intersection_diagonal → Intersection Span
   - mining_hight        → Mining Height
   - roof_fall_rate      → Target variable (dependent)
6. Print the first few rows to verify the selected data.
"""

import pandas as pd

data = pd.read_csv("original_data.csv")
data.to_excel("original_data.xlsx",index=False)

print("\nData loaded successfully!!")
print(data.shape)

# print("\nColumns in excel file :")
# print(data.columns.tolist())

selected_columns = [
    "CMRR",                  # Coal Mine Roof Rating
    "PRSUP",                 # Primary Roof Support
    "depth_of_ cover",       # Depth of Cover
    "intersection_diagonal", # Intersection Span
    "mining_hight",          # Mining Height
    "roof_fall_rate"         # Output variable
]

df = data[selected_columns]

# --- Basic Data Cleaning ---
# 1. Check for nulls and handle them
print("\nMissing values before cleaning:\n", df.isnull().sum())
df = df.dropna()  # Drop rows with any missing values
print("\nMissing values after cleaning:\n", df.isnull().sum())


# 2. Quick sanity check for invalid (negative) numeric values
for col in df.select_dtypes(include=['number']).columns:
    invalid_count = (df[col] < 0).sum()
    if invalid_count > 0:
        print(f"{invalid_count} invalid (negative) values detected in {col}")
        df = df[df[col] >= 0]

pd.set_option('display.max_columns', None)

print("\nSelected columns : ")
print(df.head())

