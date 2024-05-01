############### Import 
import pandas as pd
vehicles = pd.read_csv('Data/vehicles_fe_models.csv')

############### Split #######################3
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
regr = svm.SVR()
regr.fit(X_train, y_train)
predictions = regr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error: {rmse}") # 12086.086598954298

################ Decision Tree ###################
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error: {rmse}") # 4688.806060242256

################ Graditio Tree ###################



############### RandomForest ############### 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=15)
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error: {rmse}") 
# n=10: 6080.714677263028
# n=15: 5603.594256935747