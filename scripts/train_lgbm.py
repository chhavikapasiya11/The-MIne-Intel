import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.stats import randint, uniform
from joblib import dump
import lightgbm as lgb
import warnings
import shap
import matplotlib.pyplot as plt
import os

# WARNINGS
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*valid feature.*")
warnings.filterwarnings("ignore", message=".*feature names.*")

lgb.basic._log_info = lambda msg: None

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts → project root
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "original_data.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

# 1) Load selected columns
selected_columns = [
    "CMRR", "PRSUP", "depth_of_ cover",
    "intersection_diagonal", "mining_hight",
    "roof_fall_rate", "fall"
]

df = pd.read_csv(DATA_PATH)
df = df[selected_columns]

# 2) Log-transform
log_cols = [
    "CMRR", "PRSUP", "depth_of_ cover",
    "intersection_diagonal", "mining_hight"
]
df[log_cols] = np.log1p(df[log_cols])

# 3) Train/Test split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(df, df["fall"]):
    train = df.loc[train_idx]
    test = df.loc[test_idx]

X_train = train.drop(["roof_fall_rate", "fall"], axis=1)
y_train = train["roof_fall_rate"]
X_test = test.drop(["roof_fall_rate", "fall"], axis=1)
y_test = test["roof_fall_rate"]

# 4) Preprocessing Pipeline
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_prep = num_pipe.fit_transform(X_train)
X_test_prep = num_pipe.transform(X_test)

dump(num_pipe, os.path.join(MODEL_DIR, "preprocessing_pipeline_lightGBM_Final.joblib"))
print("Saved preprocessing pipeline → models/preprocessing_pipeline_lightGBM_Final.joblib")

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

lgbm.fit(X_train_prep, y_train)

# Metrics
pred_train = lgbm.predict(X_train_prep)
pred_test = lgbm.predict(X_test_prep)

r2_train_base = round(r2_score(y_train, pred_train), 4)
mae_train_base = round(mean_absolute_error(y_train, pred_train), 4)
rmse_train_base = round(np.sqrt(mean_squared_error(y_train, pred_train)), 4)

r2_test_base = round(r2_score(y_test, pred_test), 4)
mae_test_base = round(mean_absolute_error(y_test, pred_test), 4)
rmse_test_base = round(np.sqrt(mean_squared_error(y_test, pred_test)), 4)

# 6) 5-Fold Cross Validation
scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2"
}

cv = cross_validate(lgbm, X_train_prep, y_train, cv=5, scoring=scoring)

cv_rmse_mean = -cv["test_rmse"].mean()
cv_rmse_std = cv["test_rmse"].std()
cv_mae_mean = -cv["test_mae"].mean()
cv_r2_mean = cv["test_r2"].mean()

# 7) RandomizedSearchCV (Tuning)
param_dist = {
    "n_estimators": randint(200, 900),
    "learning_rate": uniform(0.01, 0.12),
    "max_depth": randint(2, 6),
    "num_leaves": randint(7, 35),
    "min_child_samples": randint(1, 20),
    "reg_lambda": uniform(0, 2),
    "reg_alpha": uniform(0, 2)
}

rand_small = RandomizedSearchCV(
    estimator=LGBMRegressor(random_state=42, verbose=-1),
    param_distributions=param_dist,
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

# 8) Early Stopping
X_tr, X_val, y_tr, y_val = train_test_split(X_train_prep, y_train, test_size=0.2, random_state=42)

best_lgbm_small.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(-1)]
)

best_rounds = best_lgbm_small.best_iteration_

# 9) Final model
final_lgbm = LGBMRegressor(**best_lgbm_small.get_params())
final_lgbm.set_params(n_estimators=best_rounds, verbose=-1)
final_lgbm.fit(X_train_prep, y_train)

pred_final = final_lgbm.predict(X_test_prep)

rmse_final = round(np.sqrt(mean_squared_error(y_test, pred_final)), 4)
mae_final = round(mean_absolute_error(y_test, pred_final), 4)
r2_final = round(r2_score(y_test, pred_final), 4)

fi = pd.Series(
    final_lgbm.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

# 10) Save Final Model
dump(final_lgbm, os.path.join(MODEL_DIR, "Mining_LightGBM_Model_Final.joblib"))
print("Saved model → models/Mining_LightGBM_Model_Final.joblib")





print("\n=========== INTERPRETABILITY BLOCK START ===========\n")

# 0) Utility to suppress SHAP internal printing
import sys, contextlib, os

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# 1) FEATURE IMPORTANCE
print("===== Feature Importance (Descending) =====")
fi = pd.Series(
    final_lgbm.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)
print(fi)
print("\n")


# 2) SHAP GLOBAL SUMMARY PLOT
import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(final_lgbm)
shap_values = explainer.shap_values(X_train_prep)

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_train_prep,
    feature_names=X_train.columns,
    plot_type="dot",
    show=False
)
plt.title("SHAP Summary Plot - LightGBM")
plt.tight_layout()
plt.show()
plt.close()


# 3) SHAP DEPENDENCE PLOTS FOR ALL FEATURES
print("Generating SHAP dependence plots...\n")
for feature in X_train.columns:
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(
        feature,
        shap_values,
        X_train_prep,
        feature_names=X_train.columns,
        show=False
    )
    plt.title(f"SHAP Dependence – {feature}")
    plt.tight_layout()
    plt.show()
    plt.close()


print("=========== INTERPRETABILITY BLOCK END ===========\n")
