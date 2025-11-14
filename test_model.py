import joblib
import numpy as np
import glob
import os
import pandas as pd

MODEL_PATH = "models/"

# Load ALL .joblib models from models/ folder
model_files = glob.glob(os.path.join(MODEL_PATH, "*.joblib"))

# Remove preprocessing pipeline file from list
model_files = [m for m in model_files if "preprocessing_pipeline" not in m]

print("\nFound Models:")
for m in model_files:
    print(" -", m)

# Load preprocessing pipeline
preprocess = joblib.load(MODEL_PATH + "preprocessing_pipeline.joblib")

# Hard-coded test input sample
sample = {
    "CMRR": 45,
    "PRSUP": 30,
    "depth_of_ cover": 220,
    "intersection_diagonal": 5.2,
    "mining_hight": 2.8
}

sample_values = np.array([list(sample.values())]).astype(float)
sample_prepared = preprocess.transform(sample_values)

# Predict using each model and store results
results = []

for model_file in model_files:
    model = joblib.load(model_file)
    pred = model.predict(sample_prepared)[0]
    
    results.append([
        os.path.basename(model_file),
        round(pred, 4)
    ])

# Display results in a table
df_results = pd.DataFrame(results, columns=["Model", "Predicted Roof Fall Rate"])

print("\n================ MODEL COMPARISON ================")
print(df_results.to_string(index=False))
print("==================================================\n")
