import joblib
import numpy as np
import os
import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

warnings.filterwarnings(
    "ignore",
    message=".*valid feature names.*"
)

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
    "Mining_XGBoost_Model.joblib": None,    # pipeline inside
    "Mining_LinearRegression_Model.joblib": "preprocessing_pipeline_basic.joblib",
    "Mining_DecisionTree_Model.joblib": "preprocessing_pipeline_basic.joblib",
    "Mining_RandomForest_Model.joblib": "preprocessing_pipeline_basic.joblib",
}

# -------------------------------------------------
# FIXED TEST CASES (Risk categories)
# -------------------------------------------------
samples = [
    {
        "name": "Low Risk",
        "CMRR": 60, "PRSUP": 40, "depth_of_ cover": 100,
        "intersection_diagonal": 3.0, "mining_hight": 2.0
    },
    {
        "name": "Medium Risk",
        "CMRR": 40, "PRSUP": 25, "depth_of_ cover": 180,
        "intersection_diagonal": 4.5, "mining_hight": 2.5
    },
    {
        "name": "High Risk",
        "CMRR": 15, "PRSUP": 10, "depth_of_ cover": 320,
        "intersection_diagonal": 6.0, "mining_hight": 3.2
    },
    {
        "name": "Extreme Dangerous",
        "CMRR": 5, "PRSUP": 3, "depth_of_ cover": 420,
        "intersection_diagonal": 8.0, "mining_hight": 4.0
    }
]

# -------------------------------------------------
# ADD 20 RANDOM SAMPLES
# with realistic mining ranges
# -------------------------------------------------
np.random.seed(42)

for i in range(1, 21):

    rand_sample = {
        "name": f"Random Case {i}",
        "CMRR": np.random.randint(5, 70),          # realistic CMRR
        "PRSUP": np.random.randint(3, 45),         # support pressure
        "depth_of_ cover": np.random.randint(80, 450),
        "intersection_diagonal": round(np.random.uniform(3.0, 9.0), 2),
        "mining_hight": round(np.random.uniform(1.8, 4.5), 2)
    }

    samples.append(rand_sample)

# -------------------------------------------------
# RUN PREDICTIONS FOR EACH SAMPLE
# -------------------------------------------------
final_output = []

for sample in samples:

    print(f"\n\n### Testing sample → {sample['name']} ###\n")

    input_array = np.array([[
        sample["CMRR"],
        sample["PRSUP"],
        sample["depth_of_ cover"],
        sample["intersection_diagonal"],
        sample["mining_hight"]
    ]])

    results = []

    for model_file in model_files:

        model = joblib.load(MODEL_PATH + model_file)
        prep_file = preprocess_map[model_file]

        if prep_file:
            preprocess = joblib.load(MODEL_PATH + prep_file)
            input_prepared = preprocess.transform(input_array)
        else:
            input_prepared = input_array

        pred = model.predict(input_prepared)[0]

        results.append([sample["name"], model_file, round(pred, 4)])

    df = pd.DataFrame(results, columns=["Sample", "Model", "Predicted Roof Fall Rate"])
    print(df.to_string(index=False))

    final_output.append(df)

print("\n\n==================== ALL MODEL RESULTS COMPLETED ====================\n")
