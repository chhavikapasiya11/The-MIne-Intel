import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from joblib import dump
from scipy.stats import randint, uniform

# Load
mine_org = pd.read_csv("original_data.csv")
selected_columns = ["CMRR", "PRSUP", "depth_of_ cover",
                    "intersection_diagonal", "mining_hight",
                    "roof_fall_rate", "fall"]

mine = mine_org[selected_columns]

# =======================
#  LOG TRANSFORM ADDED
# =======================
log_cols = ["CMRR", "PRSUP", "depth_of_ cover",
            "intersection_diagonal", "mining_hight"]

mine[log_cols] = np.log1p(mine[log_cols])

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(mine, mine["fall"]):
    strat_train_set = mine.loc[train_index]
    strat_test_set = mine.loc[test_index]

# Preprocessing
mine_train = strat_train_set.drop(["roof_fall_rate", "fall"], axis=1)
mine_labels = strat_train_set["roof_fall_rate"].copy()

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

mine_train_prepared = my_pipeline.fit_transform(mine_train)

x_test = strat_test_set.drop(["roof_fall_rate", "fall"], axis=1)
y_test = strat_test_set["roof_fall_rate"].copy()
x_test_prepared = my_pipeline.transform(x_test)

# =================================
# MODELS + RANDOMIZED SEARCH
# =================================
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Base models (before tuning)
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

# Hyperparameter search spaces
param_grids = {
    "Decision Tree": {
        "max_depth": randint(2, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 20)
    },
    "Random Forest": {
        "n_estimators": randint(100, 500),
        "max_depth": randint(2, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 20)
    }
}

train_results = []
test_results = []

tuned_models = {}

for name, model in models.items():
    print(f"\n===== {name} =====")

    # -------------------------
    # RandomizedSearchCV only for DT + RF
    # -------------------------
    if name in param_grids:
        print("• Running RandomizedSearchCV...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grids[name],
            n_iter=20,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=42
        )
        search.fit(mine_train_prepared, mine_labels)
        model = search.best_estimator_
        print("Best Params:", search.best_params_)
    else:
        model.fit(mine_train_prepared, mine_labels)

    tuned_models[name] = model

    # Predict
    pred_train = model.predict(mine_train_prepared)
    pred_test = model.predict(x_test_prepared)

    # Train metrics
    train_results.append([
        name,
        np.sqrt(mean_squared_error(mine_labels, pred_train)),
        mean_absolute_error(mine_labels, pred_train),
        r2_score(mine_labels, pred_train)
    ])

    # Test metrics
    test_results.append([
        name,
        np.sqrt(mean_squared_error(y_test, pred_test)),
        mean_absolute_error(y_test, pred_test),
        r2_score(y_test, pred_test)
    ])

# SUMMARY TABLES
train_df = pd.DataFrame(train_results,
                        columns=["Model", "RMSE", "MAE", "R2"]).round(4)

test_df = pd.DataFrame(test_results,
                       columns=["Model", "RMSE", "MAE", "R2"]).round(4)

print("\n================ TRAIN METRICS ================")
print(train_df.to_string(index=False))
print("===============================================")

print("\n================ TEST METRICS =================")
print(test_df.to_string(index=False))
print("===============================================")

# Save models + preprocessing
dump(my_pipeline, "models/preprocessing_pipeline_basic.joblib")

for name, model in tuned_models.items():
    filename = f"models/Mining_{name.replace(' ', '')}_Model.joblib"
    dump(model, filename)
    print(f"Saved → {filename}")
