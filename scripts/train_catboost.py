import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform
from joblib import dump

import os
os.makedirs("../models", exist_ok=True)
print("Saving at:", os.path.abspath("../models/Mining_CatBoost_Model_Final.joblib"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Goes from scripts -> project root
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "original_data.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

selected_columns = [
    "CMRR", "PRSUP", "depth_of_ cover",
    "intersection_diagonal", "mining_hight",
    "roof_fall_rate", "fall"
]

df = pd.read_csv(DATA_PATH)
df = df[selected_columns]

# LOG TRANSFORM
log_cols = ["CMRR","PRSUP","depth_of_ cover","intersection_diagonal","mining_hight"]
df[log_cols] = np.log1p(df[log_cols])

# df.head()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(df, df["fall"]):
    train = df.loc[train_idx]
    test  = df.loc[test_idx]

X_train = train.drop(["roof_fall_rate", "fall"], axis=1)
y_train = train["roof_fall_rate"]

X_test  = test.drop(["roof_fall_rate", "fall"], axis=1)
y_test  = test["roof_fall_rate"]


# PREPROCESSING PIPELINE
num_pipe = Pipeline([
    ("scaler",  StandardScaler())
])

X_train_prep = num_pipe.fit_transform(X_train)
X_test_prep  = num_pipe.transform(X_test)

# SAVE PREPROCESSING PIPELINE  (ADDED)
dump(num_pipe, os.path.join(MODEL_DIR, "preprocessing_pipeline_catboost_final.joblib"))

print("Saved preprocessing pipeline → models/preprocessing_pipeline_catboost_final.joblib")

# BASE CATBOOST MODEL
from catboost import CatBoostRegressor

cat = CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=4,
    loss_function='RMSE',
    random_state=42,
    verbose=False
)

cat.fit(X_train_prep, y_train)
pred = cat.predict(X_test_prep)
# Base CatBoost Train Metrics
train_pred_base = cat.predict(X_train_prep)

r2_base_train   = round(r2_score(y_train, train_pred_base), 4)
mae_base_train  = round(mean_absolute_error(y_train, train_pred_base), 4)
rmse_base_train = round(np.sqrt(mean_squared_error(y_train, train_pred_base)), 4)


# Base Evaluation
r2_base   = round(r2_score(y_test, pred), 4)
mae_base  = round(mean_absolute_error(y_test, pred), 4)
rmse_base = round(np.sqrt(mean_squared_error(y_test, pred)), 4)

print("\n===== Base CATBOOST Test Metrics =====")
print("R2:",   r2_base)
print("MAE:",  mae_base)
print("RMSE:", rmse_base)


# 5-Fold Cross Validation
scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2"
}

cv = cross_validate(cat, X_train_prep, y_train, cv=5, scoring=scoring)

cv_rmse_mean = round(-cv["test_rmse"].mean(), 4)
cv_rmse_std  = round(cv["test_rmse"].std(), 4)
cv_mae_mean  = round(-cv["test_mae"].mean(), 4)
cv_r2_mean   = round(cv["test_r2"].mean(), 4)

print("\n===== 5-Fold CV =====")
print("CV RMSE Mean =", cv_rmse_mean, " | Std =", cv_rmse_std)
print("CV MAE Mean  =", cv_mae_mean)
print("CV R2 Mean   =", cv_r2_mean)


# Randomized Search CV
param_dist = {
    "iterations": randint(200, 800),
    "learning_rate": uniform(0.01, 0.2),
    "depth": randint(3, 10),
    "l2_leaf_reg": uniform(0.0, 5.0)
}

rand_search = RandomizedSearchCV(
    estimator=CatBoostRegressor(loss_function='RMSE', random_state=42, verbose=False),
    param_distributions=param_dist,
    n_iter=25,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rand_search.fit(X_train_prep, y_train)

best_cat = rand_search.best_estimator_
print("\nBest Params:", rand_search.best_params_)
print("Best CV RMSE:", -rand_search.best_score_)


# Evaluate Tuned Model
pred2 = best_cat.predict(X_test_prep)
# Tuned CatBoost Train Metrics
train_pred_tuned = best_cat.predict(X_train_prep)

r2_tuned_train   = round(r2_score(y_train, train_pred_tuned), 4)
mae_tuned_train  = round(mean_absolute_error(y_train, train_pred_tuned), 4)
rmse_tuned_train = round(np.sqrt(mean_squared_error(y_train, train_pred_tuned)), 4)


r2_tuned   = round(r2_score(y_test, pred2), 4)
mae_tuned  = round(mean_absolute_error(y_test, pred2), 4)
rmse_tuned = round(np.sqrt(mean_squared_error(y_test, pred2)), 4)


# Summary Model
summary = pd.DataFrame([
    ["Base CatBoost (Train)",  rmse_base_train, mae_base_train, r2_base_train],
    ["Base CatBoost (Test)",   rmse_base,       mae_base,       r2_base],

    ["Tuned CatBoost (Train)", rmse_tuned_train, mae_tuned_train, r2_tuned_train],
    ["Tuned CatBoost (Test)",  rmse_tuned,       mae_tuned,       r2_tuned]
], columns=["Model", "RMSE", "MAE", "R2"])

summary = summary.round(4)

print("\n================ FULL SUMMARY TABLE ================\n")
print(summary.to_string(index=False))
print("\n====================================================\n")



# Save Tuned Model
dump(best_cat, os.path.join(MODEL_DIR, "Mining_CatBoost_Model.joblib"))
print("Saved tuned model → models/Mining_CatBoost_Model.joblib")



# ============================================================
# INTERPRETABILITY (Feature Importance + SHAP + Line Graphs)
# ============================================================

print("\n=========== INTERPRETABILITY START ===========\n")

# -----------------------------------------------
# 1. FEATURE IMPORTANCE
# -----------------------------------------------
fi = pd.Series(
    best_cat.get_feature_importance(),
    index=X_train.columns
).sort_values(ascending=False)

print("\nFeature Importance (Descending):")
print(fi)


# -----------------------------------------------
# 2. SHAP EXPLANATIONS
# -----------------------------------------------
try:
    import shap
    import matplotlib.pyplot as plt

    shap.initjs()

    # Build explainer
    explainer = shap.TreeExplainer(best_cat)
    shap_values = explainer.shap_values(X_train_prep)

    # -------------------------------------------
    # SHAP SUMMARY (Global Beeswarm Plot)
    # -------------------------------------------
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_train_prep,
        feature_names=X_train.columns,
        plot_type="dot",
        show=False
    )
    plt.title("SHAP Summary Plot - Roof Fall Rate Model")
    plt.tight_layout()
    plt.show()
    plt.close()


    # -------------------------------------------
    # SHAP DEPENDENCE PLOTS FOR ALL 5 FEATURES
    # -------------------------------------------
    print("\nGenerating SHAP dependence plots for all features...")

    for feature in X_train.columns:
        print(f"Plotting dependence: {feature}")

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

    """
    # -------------------------------------------
    # OPTIONAL: LINE GRAPH PER FEATURE
    # SHAP Value vs Feature Value (sorted)
    # -------------------------------------------
    print("\nGenerating SHAP line graphs (sorted)…")

    for idx, feature in enumerate(X_train.columns):
        x_vals = X_train_prep[:, idx]
        y_vals = shap_values[:, idx]

        # Sort for clean line plotting
        order = x_vals.argsort()
        x_sorted = x_vals[order]
        y_sorted = y_vals[order]

        plt.figure(figsize=(7, 5))
        plt.plot(x_sorted, y_sorted, marker="o", linestyle="-")
        plt.xlabel(feature)
        plt.ylabel(f"SHAP Value ({feature})")
        plt.title(f"Line Plot – SHAP Effect of {feature}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    """
    # -------------------------------------------
    # PRINT SHAP STATISTICS (Useful for Report)
    # -------------------------------------------
    print("\n===== SHAP Statistics for Each Feature =====")
    for idx, feature in enumerate(X_train.columns):
        values = shap_values[:, idx]
        print(
            f"{feature:20s}  |  min: {values.min():.4f}  "
            f"max: {values.max():.4f}  mean: {values.mean():.4f}  std: {values.std():.4f}"
        )

except ImportError:
    print("SHAP not installed. Run: pip install shap")

print("\n=========== INTERPRETABILITY END ===========\n")
