############### Import 
import pandas as pd
vehicles = pd.read_csv('Data/vehicles_fe_models.csv')

############### Split #######################
# Stratified sampling / split
from sklearn.model_selection import train_test_split
import numpy as np
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


############### Gradient Boosting Regression ###############

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Initialize the Gradient Boosting regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
gbr.fit(X_train, y_train)

# Predict on the test set
y_pred = gbr.predict(X_test)

# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse) # 5625.007709969297

# Optional: Display feature importances
feature_importances = pd.DataFrame(gbr.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)

################ Linear Regression ###################
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Create a Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Calculate the mean squared error
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error: {rmse}") # 5367.323677284773

################ Support Vector Machines (Regression) ###################
from sklearn import svm
svm = svm.SVR()
svm.fit(X_train, y_train)
predictions_svm = svm.predict(X_test)
rmse_svm = np.sqrt(mean_squared_error(y_test, predictions_svm))
print(f"Root Mean Squared Error: {rmse_svm}") # 12086.086598954298

################ Decision Tree ###################
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error: {rmse}") # 4688.806060242256


################ Ridge regression ###################
from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)
reg.fit(X_train, y_train)
predictions_rid = reg.predict(X_test)
rmse_rid = np.sqrt(mean_squared_error(y_test, predictions_rid))
print(f"Root Mean Squared Error: {rmse_rid}") # 5366.365037459648


################ Bayesian Ridge regression ###################
from sklearn import linear_model
bay = linear_model.BayesianRidge()
bay.fit(X_train, y_train)
predictions_bay = bay.predict(X_test)
rmse_bay = np.sqrt(mean_squared_error(y_test, predictions_bay))
print(f"Root Mean Squared Error: {rmse_bay}") # 5366.62250700759


################ Lasso regression ###################
from sklearn import linear_model
las = linear_model.Lasso(alpha=0.1)
las.fit(X_train, y_train)
predictions_las = las.predict(X_test)
rmse_las = np.sqrt(mean_squared_error(y_test, predictions_las))
print(f"Root Mean Squared Error: {rmse_las}") # 5366.601618098152


################ KNN regression ###################
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

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
print(f"Root Mean Squared Error: {rmse_knn}") # 4928.648101840205

# 6
knn_regressor = KNeighborsRegressor(n_neighbors=6)
knn_regressor.fit(X_train, y_train)
predictions_knn = knn_regressor.predict(X_test)
rmse_knn = np.sqrt(mean_squared_error(y_test, predictions_knn))
print(f"Root Mean Squared Error: {rmse_knn}") # 4935.306805828042

################ ElasticNet Regression ###################
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)
predictions_elastic_net = elastic_net.predict(X_test)
rmse_elastic_net = np.sqrt(mean_squared_error(y_test, predictions_elastic_net))
print(f"ElasticNet Regression Root Mean Squared Error: {rmse_elastic_net}") # 6586.208684130186

################ Random Forest Regression ###################
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
random_forest = RandomForestRegressor(n_estimators=5000, random_state=42)
random_forest.fit(X_train, y_train)
predictions_random_forest = random_forest.predict(X_test)
rmse_random_forest = np.sqrt(mean_squared_error(y_test, predictions_random_forest))
print(f"Random Forest Regression Root Mean Squared Error: {rmse_random_forest}") 
# 3673.1164056550765 med 200 n_estimators
# 3668.267011017643 med 500 n_estimators

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

################ Gradient Boosting Regression ################### # tager lang tid
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
plt.show()# hvad med nu