import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
from joblib import dump

# Load + Split

selected_columns = [
    "CMRR", "PRSUP", "depth_of_ cover",
    "intersection_diagonal", "mining_hight",
    "roof_fall_rate", "fall"
]

mine_org = pd.read_csv("original_data.csv")
mine = mine_org[selected_columns]

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(mine, mine["fall"]):
    strat_train_set = mine.loc[train_index]
    strat_test_set  = mine.loc[test_index]

X_train = strat_train_set.drop(["roof_fall_rate", "fall"], axis=1)
y_train = strat_train_set["roof_fall_rate"].copy()

X_test  = strat_test_set.drop(["roof_fall_rate", "fall"], axis=1)
y_test  = strat_test_set["roof_fall_rate"].copy()


# Helper scoring
scoring = {
    "rmse": make_scorer(lambda y, yhat: np.sqrt(mean_squared_error(y, yhat)), greater_is_better=False),
    "r2": "r2",
    "mae": "neg_mean_absolute_error"
}


def cv_table(model, X, y, cv=5, random_state=42):
    """Return a per-fold table (+ mean/std row) for RMSE, R2, MAE."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    cvres = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)
    
    df = pd.DataFrame({
        "Fold": np.arange(1, cv+1),
        "RMSE": -cvres["test_rmse"],                        # flip sign
        "R2":   cvres["test_r2"],
        "MAE":  -cvres["test_mae"]                          # flip sign
    })

    mean_row = pd.DataFrame({
        "Fold": ["Mean"],
        "RMSE": [df["RMSE"].mean()],
        "R2":   [df["R2"].mean()],
        "MAE":  [df["MAE"].mean()]
    })
    
    std_row = pd.DataFrame({
        "Fold": ["Std"],
        "RMSE": [df["RMSE"].std()],
        "R2":   [df["R2"].std()],
        "MAE":  [df["MAE"].std()]
    })

    return pd.concat([df, mean_row, std_row], ignore_index=True)


def train_test_table(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred_tr = model.predict(X_train)
    pred_te = model.predict(X_test)

    rows = []
    for split, y, p in [("Train", y_train, pred_tr), ("Test", y_test, pred_te)]:
        rows.append({
            "Split": split,
            "RMSE": np.sqrt(mean_squared_error(y, p)),
            "R2":   r2_score(y, p),
            "MAE":  mean_absolute_error(y, p)
        })
    print("\n===== Train vs Test Metrics =====")
    print(pd.DataFrame(rows))


# Baseline XGBoost Pipeline

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

# ===== 5-Fold CV =====
cv_table(xgb_pipe, X_train, y_train, cv=5)

# ===== Train vs Test =====
train_test_table(xgb_pipe, X_train, y_train, X_test, y_test)

#RandomizedSearchCV

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

print("\nBest Params:", rand_search.best_params_)
print("Best CV RMSE:", -rand_search.best_score_)

best_xgb = rand_search.best_estimator_

# Evaluate tuned model
preds = best_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)
mae  = mean_absolute_error(y_test, preds)

print("\n===== Tuned XGBoost Test Metrics =====")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")
print(f"MAE  = {mae:.4f}")

dump(best_xgb, "Mining_XGBoost_Model.joblib")
print("\nSaved model → Mining_XGBoost_Model.joblib")