import joblib
import numpy as np
import os
import pandas as pd

MODEL_PATH = "models/"

# All model files
model_files = [
    "Mining_CatBoost_Model.joblib",
    "Mining_LightGBM_Model.joblib",
    "Mining_XGBoost_Model.joblib",
    "Mining_LinearRegression_Model.joblib",
    "Mining_DecisionTree_Model.joblib",
    "Mining_RandomForest_Model.joblib"
]

# Preprocessing files mapping
preprocess_map = {
    "Mining_CatBoost_Model.joblib": "preprocessing_pipeline_catboost.joblib",
    "Mining_LightGBM_Model.joblib": "preprocessing_pipeline_lightGBM.joblib",
    "Mining_XGBoost_Model.joblib": None,    # already pipeline inside
    "Mining_LinearRegression_Model.joblib": "preprocessing_pipeline_basic.joblib",
    "Mining_DecisionTree_Model.joblib": "preprocessing_pipeline_basic.joblib",
    "Mining_RandomForest_Model.joblib": "preprocessing_pipeline_basic.joblib",
}

# -----------------------------
# SAMPLE INPUT for prediction
# -----------------------------
sample = {
    "CMRR": 45,
    "PRSUP": 30,
    "depth_of_ cover": 220,
    "intersection_diagonal": 5.2,
    "mining_hight": 2.8
}

input_array = np.array([[ 
    sample["CMRR"],
    sample["PRSUP"],
    sample["depth_of_ cover"],
    sample["intersection_diagonal"],
    sample["mining_hight"]
]])

# -----------------------------
# RUN PREDICTIONS
# -----------------------------
results = []

for model_file in model_files:

    model = joblib.load(MODEL_PATH + model_file)

    prep_file = preprocess_map[model_file]

    if prep_file:
        preprocess = joblib.load(MODEL_PATH + prep_file)
        input_prepared = preprocess.transform(input_array)
    else:
        input_prepared = input_array

    # predict
    pred = model.predict(input_prepared)[0]

    results.append([model_file, round(pred, 4)])

# Convert to table
df = pd.DataFrame(results, columns=["Model", "Predicted Roof Fall Rate"])

print("\n==================== MODEL COMPARISON ====================\n")
print(df.to_string(index=False))
print("\n===========================================================\n")
