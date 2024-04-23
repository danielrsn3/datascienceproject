####################################### MODELLING #######################################
import pandas as pd # Import pandas
pd.set_option('display.max_columns', None) # Allow to see all columns when viewing data
vehicles_clean = pd.read_csv('Data/vehicles_clean.csv') # Uploading the data

print(vehicles_clean.columns) # View all cloums in the dataframe
vehicles_clean.info # Summary of the DataFrame
vehicles_clean.dtypes # Display the data types of each column


####### Simple linear regression #######
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
# Selecting numerical features and target variable (only works with numerical (int))
numerical_features = ['year', 'odometer']
X = vehicles_clean[numerical_features]  # Features
y = vehicles_clean['price']  # Target variable
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
