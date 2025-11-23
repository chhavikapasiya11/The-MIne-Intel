import joblib
import numpy as np
import pandas as pd
import warnings

#   CLEAN WARNING FILTERS
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*valid feature names.*")


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#   MODEL PATH
MODEL_PATH = "../models/"     # because script is inside /scripts

#   ONLY THE MODELS YOU HAVE
model_files = [
    "Mining_CatBoost_Model.joblib",
    "Mining_LightGBM_Model_Final.joblib",
    "Mining_XGBoost_Model_Final.joblib"
]

#   EXACT PREPROCESS FILES FROM YOUR FOLDER
preprocess_map = {
    "Mining_CatBoost_Model.joblib": "preprocessing_pipeline_catboost_final.joblib",
    "Mining_LightGBM_Model_Final.joblib": "preprocessing_pipeline_lightGBM_Final.joblib",
    "Mining_XGBoost_Model_Final.joblib": None   # already includes pipeline
}

#   FIXED TEST SAMPLES
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

#   ADDING 20 RANDOM CASES
np.random.seed(42)

for i in range(1, 21):

    rand_sample = {
        "name": f"Random Case {i}",
        "CMRR": np.random.randint(5, 70),
        "PRSUP": np.random.randint(3, 45),
        "depth_of_ cover": np.random.randint(80, 450),
        "intersection_diagonal": round(np.random.uniform(3.0, 9.0), 2),
        "mining_hight": round(np.random.uniform(1.8, 4.5), 2)
    }

    samples.append(rand_sample)

#   RUN PREDICTIONS
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

        # Load model
        model_path = MODEL_PATH + model_file
        model = joblib.load(model_path)

        # Load preprocessing (if required)
        prep_file = preprocess_map[model_file]

        if prep_file:
            preprocess = joblib.load(MODEL_PATH + prep_file)
            input_prepared = preprocess.transform(input_array)
        else:
            input_prepared = input_array

        # Prediction
        pred = model.predict(input_prepared)[0]

        results.append([sample["name"], model_file, round(pred, 4)])

    df = pd.DataFrame(results, columns=["Sample", "Model", "Predicted Roof Fall Rate"])
    print(df.to_string(index=False))

    final_output.append(df)

print("\n\n==================== ALL MODEL RESULTS COMPLETED ====================\n")
