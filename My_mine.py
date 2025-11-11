import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import cross_validate
from joblib import dump


mine_org=pd.read_csv("original_data.csv")
selected_columns = ["CMRR", "PRSUP", "depth_of_ cover","intersection_diagonal", "mining_hight", "roof_fall_rate","fall"]
mine= mine_org[selected_columns]

mine.head()
mine.info()
mine.describe()

# for plotting histogram
mine.hist(bins=50 ,figsize=(20,15))

# Stratified split
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2 ,random_state=42)

for train_index ,test_index in split.split (mine,mine['fall']):
    strat_train_set=mine.loc[train_index]
    strat_test_set=mine.loc[test_index]

mine = strat_train_set
strat_test_set.info()


# correlation-matrix ananlysis
corr_matrix=mine.corr()
corr_matrix['roof_fall_rate'].sort_values(ascending=False)

attributes=["roof_fall_rate","PRSUP","CMRR","mining_hight","intersection_diagonal","depth_of_ cover"]
scatter_matrix(mine[attributes],figsize=(12,8))


# Preprocessing
mine = strat_train_set.drop(["roof_fall_rate", "fall"],axis=1)
mine_labels = strat_train_set["roof_fall_rate"].copy()

imputer=SimpleImputer(strategy="median")
imputer.fit (mine)
X=imputer.transform(mine)
mine_tr=pd.DataFrame(X,columns=mine.columns)


# scikit learn design and creating pipeline
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scalaer',StandardScaler()),
])

mine_num_tr=my_pipeline.fit_transform(mine_tr)
mine_num_tr.shape

x_test = strat_test_set.drop(["roof_fall_rate", "fall"], axis=1)
y_test = strat_test_set["roof_fall_rate"].copy()
x_test_prepared = my_pipeline.transform(x_test)


# Model training and its evaluation
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

metrics_columns = ["Model", "Train_RMSE", "Test_RMSE", "Train_MAE", "Test_MAE", "Train_R2", "Test_R2", "CV_RMSE_Mean", "CV_RMSE_Std"]

results = []

for name, model in models.items():
    print(f"\n===== {name} =====")

    # train
    model.fit(mine_num_tr, mine_labels)

    # predict
    pred_train = model.predict(mine_num_tr)
    pred_test  = model.predict(x_test_prepared)

    # train metrics
    rmse_train = np.sqrt(mean_squared_error(mine_labels, pred_train))
    mae_train  = mean_absolute_error(mine_labels, pred_train)
    r2_train   = r2_score(mine_labels, pred_train)

    # test metrics
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    mae_test  = mean_absolute_error(y_test, pred_test)
    r2_test   = r2_score(y_test, pred_test)

    print("Train → RMSE:", rmse_train, " MAE:", mae_train, " R²:", r2_train)
    print("Test  → RMSE:", rmse_test,  " MAE:", mae_test,  " R²:", r2_test)

    # 5-Fold CV
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }
    cv = cross_validate(model, mine_num_tr, mine_labels, cv=5, scoring=scoring)

    cv_rmse_mean = -cv["test_rmse"].mean()
    cv_rmse_std  = cv["test_rmse"].std()

    print("\n5-Fold CV Results:")
    print("RMSE → Mean:", cv_rmse_mean, " Std:", cv_rmse_std)
    print("MAE  → Mean:", -cv["test_mae"].mean(),  " Std:", cv["test_mae"].std())
    print("R²   → Mean:",  cv["test_r2"].mean(),   " Std:", cv["test_r2"].std())

    results.append([
        name,
        rmse_train,
        rmse_test,
        mae_train,
        mae_test,
        r2_train,
        r2_test,
        cv_rmse_mean,
        cv_rmse_std
    ])

# ---- RESULTS TABLE ----
results_df = pd.DataFrame(results, columns=metrics_columns)

# round to 4 decimals
results_df = results_df.round(4)

print("\n\n================ SUMMARY TABLE ================")
print(results_df.to_string(index=False))
print("===============================================")





















# model = LinearRegression()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()

# model.fit(mine_num_tr,mine_labels)


# some_data = mine.iloc[:5]
# some_labels = mine_labels.iloc[:5]
# prepared_data = my_pipeline.transform(some_data)
# model.predict(prepared_data)


# # ## Evaluation
# from sklearn.metrics import mean_squared_error
# mine_predictions = model.predict(mine_num_tr)
# mse = mean_squared_error(mine_labels,mine_predictions)
# rmse = np.sqrt(mse)
# print("RMSE:", rmse)

# # Cross -validation
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(model,mine_num_tr,mine_labels,scoring ="neg_mean_squared_error",cv=5)
# rmse_scores = np.sqrt(-scores)

# print("Cross-val RMSE:", rmse_scores)
# print("Mean:", rmse_scores.mean(), "Std:", rmse_scores.std())

# def print_scores(scores):
#     print("Scores:",scores)
#     print("Mean ",scores.mean())
#     print("Std ",scores.std())


# print_scores(rmse_scores)

# from joblib import dump,load
# dump(model,'Mining.joblib')


# x_test = strat_test_set.drop(["roof_fall_rate", "fall"],axis=1)
# y_test = strat_test_set["roof_fall_rate"].copy()
# x_test_prepared = my_pipeline.transform(x_test)
# final_predictions = model.predict(x_test_prepared)
# final_mse = mean_squared_error(y_test,final_predictions)
# final_rmse = np.sqrt(final_mse)
# print(final_predictions,list(y_test))


# print("Final Test RMSE:", final_rmse)  keep only table for printing...no separately..also make 2 tables for each model..test and train..