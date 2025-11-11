print("R2:",   r2_score(y_test, pred))
print("MAE:",  mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))