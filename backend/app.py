from flask import Flask,request,jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("../models/Mining_CatBoost_Model.joblib")
preprocess = joblib.load("../models/preprocessing_pipeline_catboost.joblib")

@app.route("/")
def home():
    return {"message" : "Mine-Intel API is running"}

# PREDICT API endpoint
@app.route("/predict",methods=["POST"])
def predict_rate():
    data = request.json     # frontend input
    x = pd.DataFrame([data])

    x_transformed = preprocess.transform(x)     # applying scaling/imputation
    pred = model.predict(x_transformed)[0]      #model prediction

    return jsonify({
        "prediction" : float(pred)
    })

if __name__ == "__main__":
    app.run(debug=True)

