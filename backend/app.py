from flask import Flask,request,jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load("../models/Mining_CatBoost_Model.joblib")
preprocess = joblib.load("../models/preprocessing_pipeline_catboost_final.joblib")

@app.route("/")
def home():
    return {"message" : "Mine-Intel API is running"}

# PREDICT API endpoint
@app.route("/predict",methods=["POST"])
def predict_rate():
    data = request.json     # frontend input

    # Fix inconsistent frontend names -> backend names
    rename_map = {
        "depth_of_cover": "depth_of_ cover",
        "mining_height": "mining_hight"
    }

    for front_key, back_key in rename_map.items():
        if front_key in data and back_key not in data:
            data[back_key] = data.pop(front_key)

    x = pd.DataFrame([data])

    # Force the correct column order (required by sklearn)
    expected_order = [
        "CMRR",
        "PRSUP",
        "depth_of_ cover",
        "intersection_diagonal",
        "mining_hight"
    ]
    x = x[expected_order]

    x_transformed = preprocess.transform(x)     # applying scaling/imputation
    pred = model.predict(x_transformed)[0]      #model prediction

    return jsonify({
        "prediction" : float(pred)
    })

if __name__ == "__main__":
    app.run(debug=True)

