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

# Create a Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Calculate the evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"Root Mean Squared Error: {rmse}") # 4964.8477713325665
print(f"Mean Absolute Error: {mae}") # 3658.1130662802334
print(f"Mean Squared Error: {mse}") # 24649713.39250595



##### Decision Tree #####
from sklearn import tree
dtr = tree.DecisionTreeRegressor()
dtr = dtr.fit(X_train, y_train)
predictions = dtr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"Root Mean Squared Error: {rmse}") # 4392.043777131446
print(f"Mean Absolute Error: {mae}") # 2236.5374367097247
print(f"Mean Squared Error: {mse}") # 19290048.540239062



##### KNN regression #####
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 3
knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(X_train, y_train)
predictions_knn = knn_regressor.predict(X_test)
rmse_knn = np.sqrt(mean_squared_error(y_test, predictions_knn))
print(f"Root Mean Squared Error: {rmse_knn}") # 4487.2253509261645

# 4
knn_regressor = KNeighborsRegressor(n_neighbors=4)
knn_regressor.fit(X_train, y_train)
predictions_knn = knn_regressor.predict(X_test)
rmse_knn = np.sqrt(mean_squared_error(y_test, predictions_knn))
print(f"Root Mean Squared Error: {rmse_knn}") # 4441.753654473995

# 5
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)
predictions_knn = knn_regressor.predict(X_test)

rmse_knn = np.sqrt(mean_squared_error(y_test, predictions_knn))
mae_knn = mean_absolute_error(y_test, predictions_knn)
mse_knn = mean_squared_error(y_test, predictions_knn)

print(f"Root Mean Squared Error: {rmse_knn}") # 4425.809446480019
print(f"Mean Absolute Error: {mae_knn}") # 2760.0960977124523
print(f"Mean Squared Error: {mse_knn}") # 19587789.256551772

# 6
knn_regressor = KNeighborsRegressor(n_neighbors=6)
knn_regressor.fit(X_train, y_train)
predictions_knn = knn_regressor.predict(X_test)
rmse_knn = np.sqrt(mean_squared_error(y_test, predictions_knn))
print(f"Root Mean Squared Error: {rmse_knn}") # 4433.148081128365



##### Random Forest Regression #####
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

random_forest = RandomForestRegressor(n_estimators=500, random_state=42)
random_forest.fit(X_train, y_train)
predictions_random_forest = random_forest.predict(X_test)

rmse_random_forest = np.sqrt(mean_squared_error(y_test, predictions_random_forest))
mae_random_forest = mean_absolute_error(y_test, predictions_random_forest)
mse_random_forest = mean_squared_error(y_test, predictions_random_forest)

print(f"Root Mean Squared Error: {rmse_random_forest}") # 3400.640261293944 med 500 n_estimators
print(f"Mean Absolute Error: {mae_random_forest}") # 1892.8797055297316
print(f"Mean Squared Error: {mse_random_forest}") # 11564354.186733345

# 3428.0649056250068 med 50 n_estimators
# 3403.083633046189 med 1000 n_estimators



##### Random Forest Regression med hypergrid (long convergence time) #####
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters for Random Forest:", grid_search_rf.best_params_)
        # max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200

predictions_grid_search_rf = grid_search_rf.predict(X_test)
rmse_grid_search_rf = np.sqrt(mean_squared_error(y_test, predictions_grid_search_rf))
print(f"Random Forest Regression Root Mean Squared Error: {rmse_grid_search_rf}") # HAR IKKE KØRT DENNE



##### Gradient Boosting Regression #####
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Initialize the Gradient Boosting regressor
gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.5, max_depth=5, random_state=42)
gbr.fit(X_train, y_train) # Train the model
y_pred = gbr.predict(X_test) # Predict on the test set

# Calculate the evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Calculate the root mean squared error (RMSE)
mae = mean_absolute_error(y_test, y_pred) # Calculate the mean absolute error (MAE)
mse = mean_squared_error(y_test, y_pred) # Calculate the mean squared error (MSE)

print("Root Mean Squared Error:", rmse)
# 5625.007709969297 (n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # IKKE KØRT PÅ NY
# 3834.3064543798773 (n_estimators=500, learning_rate=0.3, max_depth=5, random_state=42) # IKKE KØRT PÅ NY
# 3412.153134635619 (n_estimators=1000, learning_rate=0.5, max_depth=5, random_state=42)
print("Mean Absolute Error:", mae) # 2144.671269246954 for n=1000
print("Mean Squared Error:", mse) # 11642789.014203683 for n=1000

# Display feature importances
feature_importances = pd.DataFrame(gbr.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)



##### Gradient Boosting Regression with hypergrid (long convergence time) #####
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Define the reduced parameter grid for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.2],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}
# Grid search for Gradient Boosting
grid_search_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=5, scoring='neg_mean_squared_error')
grid_search_gb.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters for Gradient Boosting:", grid_search_gb.best_params_) 
        # learning_rate': 0.2, 'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 200

# predictions
predictions_grid_search_gb = grid_search_gb.predict(X_test)
rmse_grid_search_gb = np.sqrt(mean_squared_error(y_test, predictions_grid_search_gb))
print(f"Gradient Boosting Regression Root Mean Squared Error: {rmse_grid_search_gb}") # HAR IKKE KØRT DENNE


