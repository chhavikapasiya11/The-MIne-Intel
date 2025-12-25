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
@app.route("/predict", methods=["POST"], strict_slashes=False)
def predict_rate():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "No JSON payload received"}), 400

    # What your pipeline expects (keep this in sync with your pipeline)
    expected_order = [
        "CMRR",
        "PRSUP",
        "depth_of_ cover",        # note: original pipeline has this space
        "intersection_diagonal",
        "mining_hight"            # note: original pipeline typo 'hight'
    ]

    # Helpful debug: list what keys we actually received
    received_keys = list(data.keys())
    app.logger.debug("predict_rate received keys: %s", received_keys)

    # Build a mapping of likely incoming names -> canonical expected names.
    # Add whatever aliases you expect from frontend here.
    alias_map = {
        # common frontend (camelCase / lower)
        "cmrr": "CMRR",
        "prsup": "PRSUP",
        "depthofcover": "depth_of_ cover",
        "intersectiondiagonal": "intersection_diagonal",
        "miningheight": "mining_hight",

        # camelCase exact keys
        "depthOfCover": "depth_of_ cover",
        "intersectionDiagonal": "intersection_diagonal",
        "miningHeight": "mining_hight",

        # snake_case variants
        "depth_of_cover": "depth_of_ cover",
        "mining_hight": "mining_hight",  # accept typo if present

        # uppercase variants (just in case)
        "CMRR": "CMRR",
        "PRSUP": "PRSUP"
    }

    # Create normalized lookup of incoming keys (strip spaces/underscores and lowercase)
    norm_input = {}
    for k, v in data.items():
        kn = k.lower().replace(" ", "").replace("_", "")
        norm_input[kn] = v

    # Build a row dict with expected_order keys
    row = {}
    missing = []
    for tgt in expected_order:
        # try a direct match in incoming data first
        if tgt in data:
            row[tgt] = data[tgt]; continue

        # normalized target name for direct normalized lookup
        tgt_norm = tgt.lower().replace(" ", "").replace("_", "")

        # 1) if normalized incoming has same normalized name (e.g. "cmrr")
        if tgt_norm in norm_input:
            row[tgt] = norm_input[tgt_norm]; continue

        # 2) check alias_map for keys that map to this target
        found = False
        for alias_key, alias_target in alias_map.items():
            if alias_target == tgt:
                # check alias_key in raw data and in normalized input
                if alias_key in data:
                    row[tgt] = data[alias_key]; found = True; break
                if alias_key.lower().replace(" ", "").replace("_", "") in norm_input:
                    row[tgt] = norm_input[alias_key.lower().replace(" ", "").replace("_", "")]
                    found = True; break
        if not found:
            missing.append(tgt)

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing": missing,
            "expected_order": expected_order,
            "received_keys": received_keys
        }), 400

    # Build DataFrame in the exact expected order
    try:
        x = pd.DataFrame([row], columns=expected_order)
    except Exception as e:
        return jsonify({"error": "Failed to build DataFrame", "detail": str(e)}), 500

    try:
        x_transformed = preprocess.transform(x)
        pred = model.predict(x_transformed)[0]
    except Exception as e:
        return jsonify({"error": "Model/pipeline error", "detail": str(e)}), 500

    return jsonify({"prediction": float(pred)})

if __name__ == "__main__":
    app.run(debug=True)

