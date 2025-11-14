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

df = pd.read_csv("original_data.csv")
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


# ------------------------------------------
# SAVE PREPROCESSING PIPELINE
# ------------------------------------------
from joblib import dump
dump(num_pipe, "models/preprocessing_pipeline_lightGBM.joblib")
print("\nSaved preprocessing pipeline → models/preprocessing_pipeline_lightGBM.joblib")


# ------------------------------------------
# 5) Base LightGBM Model
# ------------------------------------------
lgbm = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=3,
    num_leaves=15,
    min_child_samples=1,
    reg_lambda=1.0,
    reg_alpha=0.5,
    random_state=42
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


# ------------------------------------------
# 6) 5-Fold Cross-Validation
# ------------------------------------------
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


# ------------------------------------------
# 7) RandomizedSearchCV (Tuning)
# ------------------------------------------
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


# ------------------------------------------
# 8) Tuned Model Evaluation
# ------------------------------------------
# TEST
pred2_test = best_lgbm.predict(X_test_prep)
r2_test_tuned   = round(r2_score(y_test, pred2_test), 4)
mae_test_tuned  = round(mean_absolute_error(y_test, pred2_test), 4)
rmse_test_tuned = round(np.sqrt(mean_squared_error(y_test, pred2_test)), 4)

# TRAIN
pred2_train = best_lgbm.predict(X_train_prep)
r2_train_tuned   = round(r2_score(y_train, pred2_train), 4)
mae_train_tuned  = round(mean_absolute_error(y_train, pred2_train), 4)
rmse_train_tuned = round(np.sqrt(mean_squared_error(y_train, pred2_train)), 4)


# ------------------------------------------
# 9) Summary Table (Train + Test)
# ------------------------------------------
summary = pd.DataFrame([
    ["Base LGBM (Train)", rmse_train_base, mae_train_base, r2_train_base],
    ["Base LGBM (Test)",  rmse_test_base,  mae_test_base,  r2_test_base],

    ["Tuned LGBM (Train)", rmse_train_tuned, mae_train_tuned, r2_train_tuned],
    ["Tuned LGBM (Test)",  rmse_test_tuned,  mae_test_tuned,  r2_test_tuned]
],
columns=["Model", "RMSE", "MAE", "R2"])

print("\n================ FULL SUMMARY TABLE =================")
print(summary.to_string(index=False))
print("=====================================================")


# ------------------------------------------
# 10) Save Tuned Model
# ------------------------------------------
dump(best_lgbm, "models/Mining_LightGBM_Model.joblib")
print("\nSaved model → models/Mining_LightGBM_Model.joblib")
