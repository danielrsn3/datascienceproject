####################################### MODELLING #######################################
import pandas as pd
pd.set_option('display.max_columns', None) # Allow to see all columns when viewing data
vehicles = pd.read_csv('Data/vehicles_clean.csv') # Uploading the data
vehicles.dtypes # Display the data types of each column
from data_preprocessing import apply_data_types
apply_data_types(vehicles)
vehicles.dtypes # Display the data types of each column







####### Simple linear regression #######
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
# Selecting numerical features and target variable (only works with numerical (int))
numerical_features = ['year', 'odometer']
X = vehicles[numerical_features]  # Features
y = vehicles['price']  # Target variable
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initializing the linear regression model
model = LinearRegression()
# Training the model
model.fit(X_train, y_train)
# Making predictions on the test set
y_pred = model.predict(X_test)
# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
# Calculating root mean squared error
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)
    # Comparing this to the price range from 5015 to 57341, an RMSE of 9562 means that, on average
    # this model's predictions are off by approximately $9562
# Assuming 'model' is already trained and available
# Input values for 'year' and 'odometer'
input_year = int(input("Enter the year of the vehicle: "))
input_odometer = float(input("Enter the odometer reading of the vehicle: "))
# Making prediction for the input values
predicted_price = model.predict([[input_year, input_odometer]])
print("Predicted Price:", predicted_price[0])




####### TEST ANOTHER MODEL #######
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Assuming 'vehicles' is your DataFrame
# Load your data and drop any rows with missing values
# vehicles = pd.read_csv('your_data.csv').dropna()

# Selecting predictors and target variable
X = vehicles.drop(columns=['price'])
y = vehicles['price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline for categorical variables
categorical_cols = X.select_dtypes(include=['category']).columns
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols)
    ])

# Append regression to preprocessing pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Train the model
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
