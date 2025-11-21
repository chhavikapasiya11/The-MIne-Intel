import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from lightgbm import LGBMRegressor
import warnings

import os
os.makedirs("../models", exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*valid feature.*")
warnings.filterwarnings("ignore", message=".*feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

warnings.filterwarnings(
    "ignore",
    message=".*LGBMRegressor was fitted with feature names.*"
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)
warnings.filterwarnings(
    "ignore",
    message=".*valid feature names.*"
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*valid feature names.*")
warnings.filterwarnings("ignore", message=".*valid feature.*")


import lightgbm as lgb
lgb.basic._log_info = lambda msg: None


# 1) Load selected columns
selected_columns = [
    "CMRR", "PRSUP", "depth_of_ cover",
    "intersection_diagonal", "mining_hight",
    "roof_fall_rate", "fall"
]

df = pd.read_csv(r"C:/Users/DELL/Desktop/IMPORTANT/Projects/Mine Intel/Mine-Intel/original_data.csv")
df = df[selected_columns]

# 2) Log-transform
log_cols = [
    "CMRR", "PRSUP", "depth_of_ cover",
    "intersection_diagonal", "mining_hight"
]

df[log_cols] = np.log1p(df[log_cols])

# 3) Train–Test Split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(df, df["fall"]):
    train = df.loc[train_idx]
    test  = df.loc[test_idx]

X_train = train.drop(["roof_fall_rate", "fall"], axis=1)
y_train = train["roof_fall_rate"]
X_test  = test.drop(["roof_fall_rate", "fall"], axis=1)
y_test  = test["roof_fall_rate"]


# 4) Preprocessing Pipeline
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_prep = num_pipe.fit_transform(X_train)
X_test_prep  = num_pipe.transform(X_test)


# SAVE PREPROCESSING PIPELINE
from joblib import dump
dump(num_pipe, "../models/preprocessing_pipeline_lightGBM.joblib")
print("\nSaved preprocessing pipeline → models/preprocessing_pipeline_lightGBM.joblib")


# 5) Base LightGBM Model
lgbm = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=3,
    num_leaves=15,
    min_child_samples=1,
    reg_lambda=1.0,
    reg_alpha=0.5,
    random_state=42,
    verbose=-1
)

# Train
lgbm.fit(X_train_prep, y_train)

# TEST Metrics
pred_test = lgbm.predict(X_test_prep)
r2_test_base   = round(r2_score(y_test, pred_test), 4)
mae_test_base  = round(mean_absolute_error(y_test, pred_test), 4)
rmse_test_base = round(np.sqrt(mean_squared_error(y_test, pred_test)), 4)

# TRAIN Metrics
pred_train = lgbm.predict(X_train_prep)
r2_train_base   = round(r2_score(y_train, pred_train), 4)
mae_train_base  = round(mean_absolute_error(y_train, pred_train), 4)
rmse_train_base = round(np.sqrt(mean_squared_error(y_train, pred_train)), 4)


# 6) 5-Fold Cross-Validation
from sklearn.model_selection import cross_validate

scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2"
}

cv = cross_validate(lgbm, X_train_prep, y_train, cv=5, scoring=scoring)

cv_rmse_mean = -cv["test_rmse"].mean()
cv_rmse_std  = cv["test_rmse"].std()
cv_mae_mean  = -cv["test_mae"].mean()
cv_r2_mean   = cv["test_r2"].mean()

print("\n================= 5-Fold CV =================")
print(f"CV RMSE Mean = {cv_rmse_mean:.4f}   | Std = {cv_rmse_std:.4f}")
print(f"CV MAE Mean  = {cv_mae_mean:.4f}")
print(f"CV R2 Mean   = {cv_r2_mean:.4f}")
print("=============================================")


# 7) RandomizedSearchCV (Tuning)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    "n_estimators": randint(200, 1000),
    "learning_rate": uniform(0.01, 0.2),
    "max_depth": randint(2, 10),
    "num_leaves": randint(7, 40),
    "reg_lambda": uniform(0, 2),
    "reg_alpha": uniform(0, 2)
}

rand_search = RandomizedSearchCV(
    estimator=LGBMRegressor(random_state=42,verbose=-1),
    param_distributions=param_dist,
    n_iter=25,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rand_search.fit(X_train_prep, y_train)

best_lgbm = rand_search.best_estimator_
best_lgbm.set_params(verbose=-1)   

print("\nBest Params:", rand_search.best_params_)
print("Best CV RMSE:", -rand_search.best_score_)


# Small data optimization
param_dist_small = {
    "n_estimators": randint(200, 900),
    "learning_rate": uniform(0.01, 0.12),
    "max_depth": randint(2, 6),
    "num_leaves": randint(7, 35),
    "min_child_samples": randint(1, 20),
    "reg_lambda": uniform(0, 2),
    "reg_alpha": uniform(0, 2)
}

rand_small = RandomizedSearchCV(
    estimator=LGBMRegressor(random_state=42,verbose=-1),
    param_distributions=param_dist_small,
    n_iter=40,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rand_small.fit(X_train_prep, y_train)
best_lgbm_small = rand_small.best_estimator_
best_lgbm_small.set_params(verbose=-1)  

print("\nSmall-Data Tuned Params:", rand_small.best_params_)
print("Small-Data CV RMSE:", -rand_small.best_score_)


# Early stopping (callback-based)
from sklearn.model_selection import train_test_split
from lightgbm import early_stopping, log_evaluation

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_prep, y_train, test_size=0.2, random_state=42
)

best_lgbm_small.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(-1)
    ]
)

best_rounds = best_lgbm_small.best_iteration_
print("\nBest num_boost_round =", best_rounds)


# Final model
final_lgbm = LGBMRegressor(**best_lgbm_small.get_params())
final_lgbm.set_params(n_estimators=best_rounds,verbose=-1)

final_lgbm.fit(X_train_prep, y_train)

# Final Test Metrics
pred_final = final_lgbm.predict(X_test_prep)

rmse_final = round(np.sqrt(mean_squared_error(y_test, pred_final)), 4)
mae_final  = round(mean_absolute_error(y_test, pred_final), 4)
r2_final   = round(r2_score(y_test, pred_final), 4)

print("\n===== Final Test Metrics (After Early Stopping) =====")
print(f"RMSE = {rmse_final:.4f}")
print(f"MAE  = {mae_final:.4f}")
print(f"R2   = {r2_final:.4f}")

fi = pd.Series(
    final_lgbm.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\n===== Top 10 Features (LightGBM) =====")
print(fi.head(10))


# Final Model Train Metrics
pred_final_train = final_lgbm.predict(X_train_prep)

rmse_final_train = round(np.sqrt(mean_squared_error(y_train, pred_final_train)), 4)
mae_final_train  = round(mean_absolute_error(y_train, pred_final_train), 4)
r2_final_train   = round(r2_score(y_train, pred_final_train), 4)


# 9) Summary Table (Using Best Model)
summary = pd.DataFrame([
    ["Base LGBM (Train)", rmse_train_base, mae_train_base, r2_train_base],
    ["Base LGBM (Test)",  rmse_test_base,  mae_test_base,  r2_test_base],

    ["Final LGBM (Train)", rmse_final_train, mae_final_train, r2_final_train],
    ["Final LGBM (Test)",  rmse_final,      mae_final,      r2_final]
],
columns=["Model", "RMSE", "MAE", "R2"])

print("\n================ FULL SUMMARY TABLE =================")
print(summary.to_string(index=False))
print("=====================================================")


# 10) Save Tuned Model (best model)
dump(final_lgbm, "../models/Mining_LightGBM_Model.joblib")
print("\nSaved model → models/Mining_LightGBM_Model.joblib")



import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from lightgbm import LGBMRegressor


# 1) Load selected columns
selected_columns = [
    "CMRR", "PRSUP", "depth_of_ cover",
    "intersection_diagonal", "mining_hight",
    "roof_fall_rate", "fall"
]

df = pd.read_csv(r"C:/Users/DELL/Desktop/IMPORTANT/Projects/Mine Intel/Mine-Intel/original_data.csv")
df = df[selected_columns]

# 2) Log-transform
log_cols = [
    "CMRR", "PRSUP", "depth_of_ cover",
    "intersection_diagonal", "mining_hight"
]

df[log_cols] = np.log1p(df[log_cols])

# 3) Train–Test Split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(df, df["fall"]):
    train = df.loc[train_idx]
    test  = df.loc[test_idx]

X_train = train.drop(["roof_fall_rate", "fall"], axis=1)
y_train = train["roof_fall_rate"]
X_test  = test.drop(["roof_fall_rate", "fall"], axis=1)
y_test  = test["roof_fall_rate"]


# 4) Preprocessing Pipeline
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_prep = num_pipe.fit_transform(X_train)
X_test_prep  = num_pipe.transform(X_test)


# SAVE PREPROCESSING PIPELINE
from joblib import dump
dump(num_pipe, "../models/preprocessing_pipeline_lightGBM.joblib")
print("\nSaved preprocessing pipeline → models/preprocessing_pipeline_lightGBM.joblib")


# 5) Base LightGBM Model
lgbm = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=3,
    num_leaves=15,
    min_child_samples=1,
    reg_lambda=1.0,
    reg_alpha=0.5,
    random_state=42,
    verbose=-1
)

# Train
lgbm.fit(X_train_prep, y_train)

# TEST Metrics
pred_test = lgbm.predict(X_test_prep)
r2_test_base   = round(r2_score(y_test, pred_test), 4)
mae_test_base  = round(mean_absolute_error(y_test, pred_test), 4)
rmse_test_base = round(np.sqrt(mean_squared_error(y_test, pred_test)), 4)

# TRAIN Metrics
pred_train = lgbm.predict(X_train_prep)
r2_train_base   = round(r2_score(y_train, pred_train), 4)
mae_train_base  = round(mean_absolute_error(y_train, pred_train), 4)
rmse_train_base = round(np.sqrt(mean_squared_error(y_train, pred_train)), 4)


# 6) 5-Fold Cross-Validation
from sklearn.model_selection import cross_validate

scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2"
}

cv = cross_validate(lgbm, X_train_prep, y_train, cv=5, scoring=scoring)

cv_rmse_mean = -cv["test_rmse"].mean()
cv_rmse_std  = cv["test_rmse"].std()
cv_mae_mean  = -cv["test_mae"].mean()
cv_r2_mean   = cv["test_r2"].mean()

print("\n================= 5-Fold CV =================")
print(f"CV RMSE Mean = {cv_rmse_mean:.4f}   | Std = {cv_rmse_std:.4f}")
print(f"CV MAE Mean  = {cv_mae_mean:.4f}")
print(f"CV R2 Mean   = {cv_r2_mean:.4f}")
print("=============================================")


# 7) RandomizedSearchCV (Tuning)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    "n_estimators": randint(200, 1000),
    "learning_rate": uniform(0.01, 0.2),
    "max_depth": randint(2, 10),
    "num_leaves": randint(7, 40),
    "reg_lambda": uniform(0, 2),
    "reg_alpha": uniform(0, 2)
}

rand_search = RandomizedSearchCV(
    estimator=LGBMRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=25,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rand_search.fit(X_train_prep, y_train)

best_lgbm = rand_search.best_estimator_
print("\nBest Params:", rand_search.best_params_)
print("Best CV RMSE:", -rand_search.best_score_)


# Small data optimization
param_dist_small = {
    "n_estimators": randint(200, 900),
    "learning_rate": uniform(0.01, 0.12),
    "max_depth": randint(2, 6),
    "num_leaves": randint(7, 35),
    "min_child_samples": randint(1, 20),
    "reg_lambda": uniform(0, 2),
    "reg_alpha": uniform(0, 2)
}

rand_small = RandomizedSearchCV(
    estimator=LGBMRegressor(random_state=42),
    param_distributions=param_dist_small,
    n_iter=40,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rand_small.fit(X_train_prep, y_train)
best_lgbm_small = rand_small.best_estimator_

print("\nSmall-Data Tuned Params:", rand_small.best_params_)
print("Small-Data CV RMSE:", -rand_small.best_score_)


# Early stopping (callback-based)
from sklearn.model_selection import train_test_split
from lightgbm import early_stopping, log_evaluation

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_prep, y_train, test_size=0.2, random_state=42
)

best_lgbm_small.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(0)
    ]
)

best_rounds = best_lgbm_small.best_iteration_
print("\nBest num_boost_round =", best_rounds)


# Final model
final_lgbm = LGBMRegressor(**best_lgbm_small.get_params())
final_lgbm.set_params(n_estimators=best_rounds)

final_lgbm.fit(X_train_prep, y_train)

# Final Test Metrics
pred_final = final_lgbm.predict(X_test_prep)

rmse_final = round(np.sqrt(mean_squared_error(y_test, pred_final)), 4)
mae_final  = round(mean_absolute_error(y_test, pred_final), 4)
r2_final   = round(r2_score(y_test, pred_final), 4)

print("\n===== Final Test Metrics (After Early Stopping) =====")
print(f"RMSE = {rmse_final:.4f}")
print(f"MAE  = {mae_final:.4f}")
print(f"R2   = {r2_final:.4f}")

fi = pd.Series(
    final_lgbm.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\n===== Top 10 Features (LightGBM) =====")
print(fi.head(10))


# Final Model Train Metrics
pred_final_train = final_lgbm.predict(X_train_prep)

rmse_final_train = round(np.sqrt(mean_squared_error(y_train, pred_final_train)), 4)
mae_final_train  = round(mean_absolute_error(y_train, pred_final_train), 4)
r2_final_train   = round(r2_score(y_train, pred_final_train), 4)


# 9) Summary Table (Using Best Model)
summary = pd.DataFrame([
    ["Base LGBM (Train)", rmse_train_base, mae_train_base, r2_train_base],
    ["Base LGBM (Test)",  rmse_test_base,  mae_test_base,  r2_test_base],

    ["Final LGBM (Train)", rmse_final_train, mae_final_train, r2_final_train],
    ["Final LGBM (Test)",  rmse_final,      mae_final,      r2_final]
],
columns=["Model", "RMSE", "MAE", "R2"])

print("\n================ FULL SUMMARY TABLE =================")
print(summary.to_string(index=False))
print("=====================================================")


# 10) Save Tuned Model (best model)
dump(final_lgbm, "../models/Mining_LightGBM_Model.joblib")
print("\nSaved model → models/Mining_LightGBM_Model.joblib")



# ============================================================
#            INTERPRETABILITY & EXPLAINABILITY BLOCK
# ============================================================

print("\n\n=========== INTERPRETABILITY BLOCK START ===========\n")

# 1) Feature Importance
print("====== Feature Importance (Sorted Descending) ======")
print(fi)

# 2) Sensitivity Analysis (PDP)
print("\n====== Partial Dependence (Sensitivity Analysis) ======")
from sklearn.inspection import partial_dependence

for col in X_train.columns:
    idx = list(X_train.columns).index(col)
    pdp = partial_dependence(final_lgbm, X_train_prep, [idx])
    print(f"{col}: PD mean preview → {pdp['average'][0][:5]}")

# 3) Ceteris Paribus (ICE)
print("\n====== Ceteris Paribus (ICE) ======")
print("ICE shows how prediction changes when one feature varies at a time.")
print("(Plots suppressed — only concept printed.)")

# 4) 4-Parameter Constant Explanation
print("\n====== 4-Parameter Constant Explanation ======")
sample_row = X_train.iloc[0].to_dict()
for feature in X_train.columns:
    print(f"Varying {feature} → others fixed at: {list(sample_row.values())[:3]} ...")

# 5) SHAP (Global + Local Explanations)
print("\n====== SHAP Explanations ======")
import shap
shap.initjs()

explainer = shap.TreeExplainer(final_lgbm)
shap_values = explainer.shap_values(X_train_prep)

print("SHAP Global Summary computed.")
instance = X_test_prep[0].reshape(1, -1)
local_sv = explainer.shap_values(instance)

print("SHAP Local Explanation for first test sample:")
print(local_sv)

print("\n=========== INTERPRETABILITY BLOCK END ===========\n")


# SHAP shows that PRSUP and CMRR are the strongest contributors to roof fall prediction.