############### Import 
import pandas as pd
vehicles = pd.read_csv('Data/vehicles_FeatureEngineered.csv')

############### Split #######################
# Stratified sampling / split
from sklearn.model_selection import train_test_split
# Assuming 'vehicles_nona' is your DataFrame and 'price' is the column you want to stratify by
# Set random seed for reproducibility
np.random.seed(123)
# Binning prices into categories
bins = pd.cut(vehicles['price'], bins=50, labels=False)  # Adjust the number of bins as needed
vehicles['price_bin'] = bins
# Splitting based on the price bins
train_vehicles, test_vehicles = train_test_split(vehicles, test_size=0.3, stratify=vehicles['price_bin'])
# Dropping 'price_bin' from both train and test sets
test_vehicles.drop(columns=['price_bin'], inplace=True)
train_vehicles.drop(columns=['price_bin'], inplace=True)
# Now, let's check the data types of the columns in the training set
print(test_vehicles.dtypes)

# 'Price' is the target variable and all other columns are features
X_train = train_vehicles.drop('price', axis=1)
y_train = train_vehicles['price']

X_test = test_vehicles.drop('price', axis=1)
y_test = test_vehicles['price']


################ Linear Regression ###################
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

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

print(f"Root Mean Squared Error: {rmse}") # 5366.268210233288
print(f"Mean Absolute Error: {mae}") # 3842.294117647059
print(f"Mean Squared Error: {mse}") # 28796834.504160374


################ Support Vector Machines (Regression) ###################
from sklearn import svm
svm = svm.SVR()
svm.fit(X_train, y_train)
predictions_svm = svm.predict(X_test)

rmse_svm = np.sqrt(mean_squared_error(y_test, predictions_svm))
mae_svm = mean_absolute_error(y_test, predictions_svm)
mse_svm = mean_squared_error(y_test, predictions_svm)

print(f"Root Mean Squared Error: {rmse_svm}") # 12086.086598954298
print(f"Mean Absolute Error: {mae_svm}") # 8899.947097294988
print(f"Mean Squared Error: {mse_svm}") # 146073489.27742267

################ Decision Tree ###################
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"Root Mean Squared Error: {rmse}") # 4711.732945825287
print(f"Mean Absolute Error: {mae}") # 2352.5058552119926
print(f"Mean Squared Error: {mse}") # 22200427.352775436

################ Ridge regression ###################
from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)
reg.fit(X_train, y_train)
predictions_rid = reg.predict(X_test)

rmse_rid = np.sqrt(mean_squared_error(y_test, predictions_rid))
mae_rid = mean_absolute_error(y_test, predictions_rid)
mse_rid = mean_squared_error(y_test, predictions_rid)

print(f"Root Mean Squared Error: {rmse_rid}") # 5366.365037459648
print(f"Mean Absolute Error: {mae_rid}") # 3842.056265424365
print(f"Mean Squared Error: {mse_rid}") # 28797873.71526929

################ Bayesian Ridge regression ###################
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

bay = linear_model.BayesianRidge()
bay.fit(X_train, y_train)
predictions_bay = bay.predict(X_test)

rmse_bay = np.sqrt(mean_squared_error(y_test, predictions_bay))
mae_bay = mean_absolute_error(y_test, predictions_bay)
mse_bay = mean_squared_error(y_test, predictions_bay)

print(f"Root Mean Squared Error: {rmse_bay}") # 5366.62250700759
print(f"Mean Absolute Error: {mae_bay}") # 3842.2151609642274
print(f"Mean Squared Error: {mse_bay}") # 28800637.13272043


################ Lasso regression ###################
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

las = linear_model.Lasso(alpha=0.1)
las.fit(X_train, y_train)
predictions_las = las.predict(X_test)

rmse_las = np.sqrt(mean_squared_error(y_test, predictions_las))
mae_las = mean_absolute_error(y_test, predictions_las)
mse_las = mean_squared_error(y_test, predictions_las)

print(f"Root Mean Squared Error: {rmse_las}") # 5366.601618098152
print(f"Mean Absolute Error: {mae_las}") # 3842.113763171372
print(f"Mean Squared Error: {mse_las}") # 28800412.927373704


################ KNN regression ###################
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 3
knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(X_train, y_train)
predictions_knn = knn_regressor.predict(X_test)
rmse_knn = np.sqrt(mean_squared_error(y_test, predictions_knn))
print(f"Root Mean Squared Error: {rmse_knn}") # 5004.641317204217

# 4
knn_regressor = KNeighborsRegressor(n_neighbors=4)
knn_regressor.fit(X_train, y_train)
predictions_knn = knn_regressor.predict(X_test)
rmse_knn = np.sqrt(mean_squared_error(y_test, predictions_knn))
print(f"Root Mean Squared Error: {rmse_knn}") # 4940.608706349057

# 5
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)
predictions_knn = knn_regressor.predict(X_test)

rmse_knn = np.sqrt(mean_squared_error(y_test, predictions_knn))
mae_knn = mean_absolute_error(y_test, predictions_knn)
mse_knn = mean_squared_error(y_test, predictions_knn)

print(f"Root Mean Squared Error: {rmse_knn}") # 4928.648101840205
print(f"Mean Absolute Error: {mae_knn}") # 2962.8740643171523
print(f"Mean Squared Error: {mse_knn}") # 24291572.111773055

# 6
knn_regressor = KNeighborsRegressor(n_neighbors=6)
knn_regressor.fit(X_train, y_train)
predictions_knn = knn_regressor.predict(X_test)
rmse_knn = np.sqrt(mean_squared_error(y_test, predictions_knn))
print(f"Root Mean Squared Error: {rmse_knn}") # 4935.306805828042

################ ElasticNet Regression ###################
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)
predictions_elastic_net = elastic_net.predict(X_test)

rmse_elastic_net = np.sqrt(mean_squared_error(y_test, predictions_elastic_net))
mae_elastic_net = mean_absolute_error(y_test, predictions_elastic_net)
mse_elastic_net = mean_squared_error(y_test, predictions_elastic_net)

print(f"Root Mean Squared Error: {rmse_elastic_net}") # 6586.208684130186
print(f"Mean Absolute Error: {mae_elastic_net}") # 4746.45460425704
print(f"Mean Squared Error: {mse_elastic_net}") # 43378144.83091188


################ Random Forest Regression ###################
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

random_forest = RandomForestRegressor(n_estimators=500, random_state=42)
random_forest.fit(X_train, y_train)
predictions_random_forest = random_forest.predict(X_test)

rmse_random_forest = np.sqrt(mean_squared_error(y_test, predictions_random_forest))
mae_random_forest = mean_absolute_error(y_test, predictions_random_forest)
mse_random_forest = mean_squared_error(y_test, predictions_random_forest)

print(f"Root Mean Squared Error: {rmse_random_forest}") # 3673.1164056550765 med 200 n_estimators, # 3668.267011017643 med 500 n_estimators
print(f"Mean Absolute Error: {mae_random_forest}") # 1980.435267052644
print(f"Mean Squared Error: {mse_random_forest}") # 13456182.864120314

################ Random Forest Regression med hypergrid ################### # tager 1 time at køre
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
print(f"Random Forest Regression Root Mean Squared Error: {rmse_grid_search_rf}") # 3673.1164056550765


############### Gradient Boosting Regression ###############
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
# 5625.007709969297 (n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# 3834.3064543798773 (n_estimators=500, learning_rate=0.3, max_depth=5, random_state=42)
# 3755.186035015425 (n_estimators=1000, learning_rate=0.5, max_depth=5, random_state=42)
print("Mean Absolute Error:", mae) # 2305.8761729442467 for n=1000
print("Mean Squared Error:", mse) # 14101422.157574872 for n=1000



# Display feature importances
feature_importances = pd.DataFrame(gbr.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)


################ Gradient Boosting Regression with hypergrid ################### # tager lang tid
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
print(f"Gradient Boosting Regression Root Mean Squared Error: {rmse_grid_search_gb}") # 4104.193407930705


###### PLOT WITH RESULTS ######

import matplotlib.pyplot as plt

# Sort the DataFrame by RMSE in descending order
results_df_sorted = results_df.sort_values(by='RMSE', ascending=False)

# Set the style of the plot
plt.style.use('seaborn-darkgrid')

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.barh(results_df_sorted['Model'], results_df_sorted['RMSE'], color='skyblue')
plt.xlabel('Root Mean Squared Error (RMSE)')
plt.ylabel('Model')
plt.title('Comparison of RMSE for Different Models')
plt.tight_layout()

# Show the plot
plt.show()