import pandas as pd
import numpy as np

from sklearn.model_selection import (
    StratifiedShuffleSplit, 
    cross_validate, 
    KFold, 
    RandomizedSearchCV
)

from sklearn.metrics import (
    mean_squared_error, r2_score, 
    mean_absolute_error, make_scorer
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone

from xgboost import XGBRegressor
import xgboost as xgb
from scipy.stats import randint, uniform
from joblib import dump
import shap
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts → project root
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "original_data.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load Data
selected_columns = [
    "CMRR", "PRSUP", "depth_of_ cover",
    "intersection_diagonal", "mining_hight",
    "roof_fall_rate", "fall"
]

df = pd.read_csv(DATA_PATH)
mine = df[selected_columns]


# Stratified Split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(mine, mine["fall"]):
    strat_train_set = mine.loc[train_index]
    strat_test_set  = mine.loc[test_index]

X_train = strat_train_set.drop(["roof_fall_rate", "fall"], axis=1)
y_train = strat_train_set["roof_fall_rate"].copy()

X_test  = strat_test_set.drop(["roof_fall_rate", "fall"], axis=1)
y_test  = strat_test_set["roof_fall_rate"].copy()


# Scoring Setup
scoring = {
    "rmse": make_scorer(lambda y, yhat: np.sqrt(mean_squared_error(y, yhat)),
                        greater_is_better=False),
    "r2": "r2",
    "mae": "neg_mean_absolute_error"
}


def cv_table(model, X, y, cv=5, random_state=42):
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    cvres = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)

    df = pd.DataFrame({
        "Fold": np.arange(1, cv+1),
        "RMSE": -cvres["test_rmse"],
        "R2":   cvres["test_r2"],
        "MAE":  -cvres["test_mae"]
    })

    df.loc["Mean"] = ["-", df["RMSE"].mean(), df["R2"].mean(), df["MAE"].mean()]
    df.loc["Std"]  = ["-", df["RMSE"].std(),  df["R2"].std(),  df["MAE"].std()]
    print("\n===== 5-Fold CV Metrics =====")
    print(df)
    return df


def train_test_table(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred_tr = model.predict(X_train)
    pred_te = model.predict(X_test)

    df = pd.DataFrame([
        {
            "Split": "Train",
            "RMSE": np.sqrt(mean_squared_error(y_train, pred_tr)),
            "R2":   r2_score(y_train, pred_tr),
            "MAE":  mean_absolute_error(y_train, pred_tr)
        },
        {
            "Split": "Test",
            "RMSE": np.sqrt(mean_squared_error(y_test, pred_te)),
            "R2":   r2_score(y_test, pred_te),
            "MAE":  mean_absolute_error(y_test, pred_te)
        }
    ])

    print("\n===== Train vs Test Metrics =====")
    print(df)
    return df


# Baseline XGB Pipeline
num_pipe = SimpleImputer(strategy="median")

xgb_pipe = Pipeline([
    ("prep", num_pipe),
    ("model", XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    ))
])

# Baseline CV
cv_table(xgb_pipe, X_train, y_train, cv=5)

# Train vs Test
train_test_table(xgb_pipe, X_train, y_train, X_test, y_test)


# RandomizedSearchCV Tuning
param_dist = {
    "model__n_estimators": randint(300, 1200),
    "model__max_depth": randint(3, 12),
    "model__learning_rate": uniform(0.01, 0.2),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4)
}

rand_search = RandomizedSearchCV(
    estimator=xgb_pipe,
    param_distributions=param_dist,
    n_iter=30,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rand_search.fit(X_train, y_train)

print("\n===== Best Params (RandomSearchCV) =====")
print(rand_search.best_params_)
print("Best CV RMSE:", -rand_search.best_score_)

best_xgb = rand_search.best_estimator_


# Tuned Model Evaluation
preds = best_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)
mae  = mean_absolute_error(y_test, preds)

print("\n===== Tuned Model Test Metrics =====")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")
print(f"MAE  = {mae:.4f}")


# Small-Data Optimized RandomizedSearchCV
param_dist_small = {
    "model__n_estimators": randint(200, 900),
    "model__max_depth": randint(2, 6),
    "model__learning_rate": uniform(0.03, 0.12),
    "model__subsample": uniform(0.6, 0.35),
    "model__colsample_bytree": uniform(0.6, 0.35),
    "model__min_child_weight": randint(1, 10),
    "model__gamma": uniform(0.0, 4.0),
    "model__reg_alpha": uniform(0.0, 1.0),
    "model__reg_lambda": uniform(0.5, 2.0)
}

rnd_small = RandomizedSearchCV(
    estimator=xgb_pipe,
    param_distributions=param_dist_small,
    n_iter=40,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rnd_small.fit(X_train, y_train)

print("\n===== Small-Data Tuned Params =====")
print(rnd_small.best_params_)
print(f"Best 5-Fold CV RMSE: {-rnd_small.best_score_:.4f}")

best_pipe = rnd_small.best_estimator_


# CV with Early Stopping
prep = best_pipe.named_steps["prep"]
prep.fit(X_train, y_train)
X_train_np = prep.transform(X_train)

xgb_est = best_pipe.named_steps["model"]
xgb_est.set_params(n_estimators=60, max_depth=4)

xgb_params = xgb_est.get_xgb_params()
xgb_params["eval_metric"] = "rmse"

dtrain = xgb.DMatrix(X_train_np, label=y_train.values)

cv_results = xgb.cv(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=10000,
    nfold=5,
    early_stopping_rounds=50,
    metrics="rmse",
    seed=42,
    as_pandas=True,
    shuffle=True
)

best_rounds = int(cv_results.shape[0])
print(f"\nBest num_boost_round = {best_rounds}")
print(f"Best CV RMSE = {cv_results['test-rmse-mean'].min():.4f}")

final_pipe = clone(best_pipe)
final_pipe.set_params(model__n_estimators=best_rounds)
final_pipe.fit(X_train, y_train)

# Final Test Metrics
pred_test = final_pipe.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
test_r2   = r2_score(y_test, pred_test)
test_mae  = mean_absolute_error(y_test, pred_test)

print("\n===== Final Test Metrics =====")
print(f"RMSE = {test_rmse:.4f}")
print(f"R²   = {test_r2:.4f}")
print(f"MAE  = {test_mae:.4f}")


# Save Model
dump(final_pipe, os.path.join(MODEL_DIR, "Mining_XGBoost_Model_Final.joblib"))
print("\nSaved final model → models/Mining_XGBoost_Model_Final.joblib")


#                 FEATURE IMPORTANCE
try:
    xgb_core = final_pipe.named_steps["model"]
    fi = pd.Series(xgb_core.feature_importances_,
                   index=X_train.columns).sort_values(ascending=False)

    print("\n===== Top 10 Features =====")
    print(fi.head(10))

except Exception as e:
    print("Feature importance error:", e)


#                 SHAP INTERPRETABILITY
print("\n=========== XGBOOST SHAP INTERPRETABILITY START ===========\n")

# SHAP PREP
explainer = shap.TreeExplainer(final_pipe.named_steps["model"])
X_train_np = prep.transform(X_train)
shap_values = explainer.shap_values(X_train_np)

# ---- SHAP Summary ----
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_train_np,
    feature_names=X_train.columns,
    plot_type="dot",
    show=False
)
plt.title("SHAP Summary Plot - XGBoost")
plt.tight_layout()
plt.show()
plt.close()

# ---- SHAP Dependence ----
print("Generating SHAP dependence plots...\n")
for feature in X_train.columns:
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(
        feature,
        shap_values,
        X_train_np,
        feature_names=X_train.columns,
        show=False
    )
    plt.title(f"SHAP Dependence – {feature}")
    plt.tight_layout()
    plt.show()
    plt.close()


print("\n=========== XGBOOST SHAP INTERPRETABILITY END ===========\n")