import numpy as np
import pandas as pd
# Loading the test and training datasets from pickle files
X_train = pd.read_pickle("./data/X_train.pkl")
y_train = pd.read_pickle("./data/y_train.pkl")
X_test = pd.read_pickle("./data/X_test.pkl")
y_test = pd.read_pickle("./data/y_test.pkl")

####################################### MODELLING #######################################

##### Linear Regression #####
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
model = LinearRegression()
model.fit(X_train, y_train)

# TRAIN
train_predictions = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
train_mae = mean_absolute_error(y_train, train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)

print(f"Train RMSE: {train_rmse}") # 4913.004916488663
print(f"Train MAE: {train_mae}") # 3613.8646802218113
print(f"Train MSE: {train_mse}") # 24137617.309441775

# TEST
test_predictions = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f"Test RMSE: {test_rmse}") # 4964.8477713325665
print(f"Test MAE: {test_mae}") # 3658.1130662802334
print(f"Test MSE: {test_mse}") # 24649713.39250595



##### Decision Tree #####
from sklearn import tree
dtr = tree.DecisionTreeRegressor()
dtr = dtr.fit(X_train, y_train)

# TRAIN
train_predictions = dtr.predict(X_train)
train_mae = mean_absolute_error(y_train, train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)

print(f"Train RMSE: {train_rmse}") # 270.2036454665148
print(f"Train MAE: {train_mae}") # 26.501269013344427
print(f"Train MSE: {train_mse}") # 73010.010023394

# TEST
test_predictions = dtr.predict(X_test)
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)

print(f"Test RMSE: {test_rmse}") # 4392.043777131446
print(f"Test MAE: {test_mae}") # 2236.5374367097247
print(f"Test MSE: {test_mse}") # 19290048.540239062



##### KNN regression #####
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 3 
knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(X_train, y_train)

# TRAIN
train_predictions_knn = knn_regressor.predict(X_train)
train_rmse_knn = np.sqrt(mean_squared_error(y_train, train_predictions_knn))
print(f"Train RMSE: {train_rmse_knn}") 

# TEST
test_predictions_knn = knn_regressor.predict(X_test)
test_rmse_knn = np.sqrt(mean_squared_error(y_test, test_predictions_knn))
print(f"Test RMSE: {test_rmse_knn}") # 4487.2253509261645

# 4 
knn_regressor = KNeighborsRegressor(n_neighbors=4)
knn_regressor.fit(X_train, y_train)

# TRAIN
train_predictions_knn = knn_regressor.predict(X_train)
train_rmse_knn = np.sqrt(mean_squared_error(y_train, train_predictions_knn))
print(f"Train RMSE: {train_rmse_knn}") 

# TEST
test_predictions_knn = knn_regressor.predict(X_test)
test_rmse_knn = np.sqrt(mean_squared_error(y_test, test_predictions_knn))
print(f"Test RMSE: {test_rmse_knn}") # 4441.753654473995

# 5
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)

# TRAIN
train_predictions_knn = knn_regressor.predict(X_train)
train_rmse_knn = np.sqrt(mean_squared_error(y_train, train_predictions_knn))
train_mae_knn = mean_absolute_error(y_train, train_predictions_knn)
train_mse_knn = mean_squared_error(y_train, train_predictions_knn)
print(f"Train RMSE: {train_rmse_knn}") # 3555.023642533517 
print(f"Train MAE: {train_mae_knn}") # 2162.8121966728277
print(f"Train MSE: {train_mse_knn}") # 12638193.098972274

# TEST 
test_predictions_knn = knn_regressor.predict(X_test)
test_rmse_knn = np.sqrt(mean_squared_error(y_test, test_predictions_knn))
test_mae_knn = mean_absolute_error(y_test, test_predictions_knn)
test_mse_knn = mean_squared_error(y_test, test_predictions_knn)
print(f"Test RMSE: {test_rmse_knn}") # 4425.809446480019
print(f"Test MAE: {test_mae_knn}") # 2760.0960977124523
print(f"Test MSE: {test_mse_knn}") # 19587789.256551772

# 6
knn_regressor = KNeighborsRegressor(n_neighbors=6)
knn_regressor.fit(X_train, y_train)

# TRAIN
train_predictions_knn = knn_regressor.predict(X_train)
train_rmse_knn = np.sqrt(mean_squared_error(y_train, train_predictions_knn))
print(f"Train RMSE: {train_rmse_knn}") # 3706.2758012616196

# TEST
test_predictions_knn = knn_regressor.predict(X_test)
test_rmse_knn = np.sqrt(mean_squared_error(y_test, test_predictions_knn))
print(f"Test RMSE: {test_rmse_knn}") # 4433.148081128365



##### Random Forest Regression #####
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

random_forest = RandomForestRegressor(n_estimators=500, random_state=42)
random_forest.fit(X_train, y_train)

# TRAIN
train_predictions_randomforest = random_forest.predict(X_train)
train_rmse_randomforest = np.sqrt(mean_squared_error(y_train, train_predictions_randomforest))
train_mae_randomforest = mean_absolute_error(y_train, train_predictions_randomforest)
train_mse_randomforest = mean_squared_error(y_train, train_predictions_randomforest)

print(f"Train RMSE {train_rmse_randomforest}") # 1264.6176841768977
print(f"Train MAE: {train_mae_randomforest}") # 701.9589688148769
print(f"Train MSE: {train_mse_randomforest}") # 1599257.8871329399

# TEST
test_predictions_randomforest = random_forest.predict(X_test)
test_rmse_randomforest = np.sqrt(mean_squared_error(y_test, test_predictions_randomforest))
test_mae_randomforest = mean_absolute_error(y_test, test_predictions_randomforest)
test_mse_randomforest = mean_squared_error(y_test, test_predictions_randomforest)

print(f"Test RMSE: {test_rmse_randomforest}") # 3400.640261293944 
print(f"Test MAE: {test_mae_randomforest}") # 1892.8797055297316
print(f"Test MSE: {test_mse_randomforest}") # 11564354.186733345

# RMSE = 3428.0649056250068 with 50 n_estimators
# RMSE = 3403.083633046189 with 1000 n_estimators


# Plot of RF decission tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Selecting a single tree from the forest
single_tree = random_forest.estimators_[0]

# Visualizing the tree
plt.figure(figsize=(20,10))

plot_tree(single_tree, feature_names=X_train.columns, filled=True, max_depth=3, fontsize=8)
plt.show()



##### Gradient Boosting Regression #####
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Initialize the Gradient Boosting regressor
gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.5, max_depth=5, random_state=42)
gbr.fit(X_train, y_train) 
train_y_pred = gbr.predict(X_train) 
# Calculate the evaluation metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_y_pred))  
train_mae = mean_absolute_error(y_train, train_y_pred) 
train_mse = mean_squared_error(y_train, train_y_pred) 

print("Train RMSE:", train_rmse) # 1816.694616811483
print("Train MAE:", train_mae) # 1259.0532114312662
print("Train MSE:", train_mse) # 3300379.330751821  


test_y_pred = gbr.predict(X_test)
# Calculate the evaluation metrics
test_rmse = np.sqrt(mean_squared_error(y_test, test_y_pred)) 
test_mae = mean_absolute_error(y_test, test_y_pred) 
test_mse = mean_squared_error(y_test, test_y_pred) 

print("Test RMSE:", test_rmse) # 3412.457201897026
print("Test MAE:", test_mae) # 2144.671269246954 
print("Test MSE:", test_mse) # 11642789.014203683 

# Display feature importances
feature_importances = pd.DataFrame(gbr.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)