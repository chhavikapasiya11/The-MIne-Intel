import pandas as pd
import numpy as np


# In[3]:


mine_org=pd.read_csv("original_data.csv")
selected_columns = ["CMRR", "PRSUP", "depth_of_ cover","intersection_diagonal", "mining_hight", "roof_fall_rate","fall"]
mine= mine_org[selected_columns]


mine.describe()

# for plotting histogram
import matplotlib.pyplot as plt
mine.hist(bins=50 ,figsize=(20,15))


from sklearn.model_selection import train_test_split
train_set ,test_set = train_test_split(mine,test_size=0.2,random_state=42)


# In[10]:


print(f"Rows in train set:{len (train_set)}\nRows in test set:{len(test_set)}\n")


# In[11]:


from sklearn.model_selection import  StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2 ,random_state=42)

for train_index ,test_index in split.split (mine,mine['fall']):
    strat_train_set=mine.loc[train_index]
    strat_test_set=mine.loc[test_index]


# In[12]:


mine=strat_train_set


# In[13]:


strat_test_set.info()


# ## correlation-matrix ananlysis

# In[14]:


corr_matrix=mine.corr()


# In[15]:


corr_matrix['roof_fall_rate'].sort_values(ascending=False)


# In[16]:


from  pandas.plotting import scatter_matrix
attributes=["roof_fall_rate","PRSUP","CMRR","mining_hight","intersection_diagonal","depth_of_ cover"]
scatter_matrix(mine[attributes],figsize=(12,8))


# ## Missing value imputation and data processing

# In[17]:


mine = strat_train_set.drop(["roof_fall_rate", "fall"],axis=1)
mine_labels = strat_train_set["roof_fall_rate"].copy()


# In[18]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit (mine)
X=imputer.transform(mine)
mine_tr=pd.DataFrame(X,columns=mine.columns)


# ## Scikit learn design and creating pipeline

# In[19]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scalaer',StandardScaler()),
])


# In[20]:


mine_num_tr=my_pipeline.fit_transform(mine_tr)


# In[21]:


mine_num_tr.shape


# ## Model training and its evaluation

# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(mine_num_tr,mine_labels)


# In[23]:


some_data = mine.iloc[:5]


# In[24]:


some_labels = mine_labels.iloc[:5]


# In[25]:


prepared_data = my_pipeline.transform(some_data)


# In[26]:


model.predict(prepared_data)


# ## Evaluation

# In[27]:


from sklearn.metrics import mean_squared_error


# In[28]:


mine_predictions = model.predict(mine_num_tr)
mse = mean_squared_error(mine_labels,mine_predictions)
rmse = np.sqrt(mse)


# In[29]:


rmse


# ## Cross -validation

# In[30]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,mine_num_tr,mine_labels,scoring ="neg_mean_squared_error",cv=5)
rmse_scores = np.sqrt(-scores)


# In[31]:


rmse_scores


# In[32]:


def print_scores(scores):
    print("Scores:",scores)
    print("Mean ",scores.mean())
    print("Std ",scores.std())


# In[33]:


print_scores(rmse_scores)


# In[34]:


from joblib import dump,load
dump(model,'Mining.joblib')


# In[35]:


x_test = strat_test_set.drop(["roof_fall_rate", "fall"],axis=1)
y_test = strat_test_set["roof_fall_rate"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions,list(y_test))


# In[36]:


final_rmse


# In[37]:


prepared_data[0]


# ## Setup, Metrics & Helper Functions

# In[38]:


from sklearn.model_selection import cross_validate, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from joblib import dump


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
    """Train the model and report Train vs Test metrics (RMSE, R2, MAE)."""
    model.fit(X_train, y_train)
    pred_tr = model.predict(X_train)
    pred_te = model.predict(X_test)
    rows = []
    for split, y, p in [("Train", y_train, pred_tr), ("Test", y_test, pred_te)]:
        rows.append({
            "Split": split,
            "RMSE": np.sqrt(mean_squared_error(y, p)),
            "R2": r2_score(y, p),
            "MAE": mean_absolute_error(y, p)
        })
    return pd.DataFrame(rows)


# ## Data Split & Preprocessing

# In[39]:


X_train = strat_train_set.drop(["roof_fall_rate", "fall"], axis=1)
y_train = strat_train_set["roof_fall_rate"].copy()
X_test  = strat_test_set.drop(["roof_fall_rate", "fall"], axis=1)
y_test  = strat_test_set["roof_fall_rate"].copy()

# Simple numeric pipeline (median imputation only)
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# In[ ]:


get_ipython().system('pip install xgboost')


# ## Baseline XGBoost Pipeline

# In[ ]:


from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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


# ## 5-Fold Cross-Validation (Baseline)

# In[ ]:


cv_results = cv_table(xgb_pipe, X_train, y_train, cv=5)
print("\n=== XGBoost 5-Fold CV Results ===")
display(cv_results)


# ## Train vs Test Metrics (Baseline)

# In[ ]:


train_test_metrics = train_test_table(xgb_pipe, X_train, y_train, X_test, y_test)
print("\n=== Train vs Test Metrics ===")
display(train_test_metrics)


# ## RandomizedSearchCV — Broad Tuning

# In[ ]:


from scipy.stats import randint, uniform

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
print("Best Params (Random Search):", rand_search.best_params_)
print("Best CV RMSE:", -rand_search.best_score_)
best_xgb = rand_search.best_estimator_


# In[ ]:


preds = best_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"\nFinal XGBoost Test Results:")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")
print(f"MAE  = {mae:.4f}")


# In[ ]:


dump(best_xgb, "Mining_XGBoost_Model.joblib")


# ## Small-Data Optimized RandomizedSearchCV

# In[ ]:


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform

# Define a compact, regularized parameter space (good for smaller data)
param_dist_small = {
    "model__n_estimators": randint(200, 900),       # fewer trees to prevent overfitting
    "model__max_depth": randint(2, 6),              # shallow trees generalize better
    "model__learning_rate": uniform(0.03, 0.12),    # slightly higher learning rate
    "model__subsample": uniform(0.6, 0.35),         # 0.6–0.95
    "model__colsample_bytree": uniform(0.6, 0.35),  # 0.6–0.95
    "model__min_child_weight": randint(1, 10),      # conservative splits
    "model__gamma": uniform(0.0, 4.0),              # split loss regularization
    "model__reg_alpha": uniform(0.0, 1.0),          # L1 regularization
    "model__reg_lambda": uniform(0.5, 2.0)          # L2 regularization
}

# RandomizedSearchCV: 5-fold CV, 40 iterations
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

print("\n[RandomizedSearchCV] Best Parameters (Small-Data Tuned):")
print(rnd_small.best_params_)
print(f"Best 5-Fold CV RMSE: {-rnd_small.best_score_:.4f}")

best_pipe = rnd_small.best_estimator_


# ## Create Validation Split & Use Early Stopping, Final Fit with Best Number of Trees

# In[ ]:


import xgboost as xgb
from sklearn.base import clone
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# 1) Prepare numeric data using the same preprocessor from your pipeline
prep = best_pipe.named_steps["prep"]
prep.fit(X_train, y_train)
X_train_np = prep.transform(X_train)

# 2) Extract tuned XGB params from the pipeline, THEN (optionally) override
xgb_est = best_pipe.named_steps["model"]
xgb_est.set_params(n_estimators=60, max_depth=4)  # <-- if you really want to try deeper/shallower
xgb_params = xgb_est.get_xgb_params()
xgb_params["eval_metric"] = "rmse"

# 3) CV with early stopping to find best_num_boost_round
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
print(f"\n xgboost.cv found best_num_boost_round = {best_rounds}")
print(f"Best CV RMSE: {cv_results['test-rmse-mean'].min():.4f} ± {cv_results['test-rmse-std'].iloc[best_rounds-1]:.4f}")

# 4) Final fit with the best number of trees
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

final_pipe = clone(best_pipe)
final_pipe.set_params(model__n_estimators=best_rounds)
final_pipe.fit(X_train, y_train)

# 5) Test metrics
pred_test = final_pipe.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
test_r2   = r2_score(y_test, pred_test)
test_mae  = mean_absolute_error(y_test, pred_test)

print("\n === Final Test Metrics (CV-tuned rounds) ===")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²:   {test_r2:.4f}")
print(f"MAE:  {test_mae:.4f}")

# 6) Save model
from joblib import dump
dump(final_pipe, "Mining_XGBoost_cv_earlystop.joblib")

# 7) Feature importances
try:
    xgb_core = final_pipe.named_steps["model"]
    fi = pd.Series(xgb_core.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\n Top 10 Features:")
    display(fi.head(10))
except Exception as e:
    print(" Could not compute feature importances:", e)


# In[ ]:




