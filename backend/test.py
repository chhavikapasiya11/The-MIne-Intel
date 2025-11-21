import joblib
pre = joblib.load("../models/preprocessing_pipeline_catboost.joblib")
print(pre.feature_names_in_)