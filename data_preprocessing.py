import pandas as pd # Import pandas
pd.set_option('display.max_columns', None) # Allow to see all columns when viewing data
vehicles = pd.read_csv('Data/vehicles_no_url.csv') # Uploading the data
print(vehicles.columns) # View all cloums in the dataframe
vehicles.info # Summary of the DataFrame
print(vehicles['year']) # View specific variables
vehicles.dtypes # Display the data types of each column


####################################### PREPROCESSING #######################################

####### Remove excessive variables #######
# Drop the 'county' variable from the DataFrame since it still has missings for some reason (find ud af hvorfor)
vehicles.drop(columns=['county'], inplace=True)
# Removing the VIN column as it has no predictive power
vehicles.drop(columns=['VIN'], inplace=True)

####### Correcting data types #######


####### Handle missing values #######
vehicles.isna().any() # In what coloums are there missings?
print(vehicles.isnull().sum()) # Amount of missing in each coloum
# Impute missing values for numeric columns with mean
numeric_columns = vehicles.select_dtypes(include='number').columns
vehicles[numeric_columns] = vehicles[numeric_columns].fillna(vehicles[numeric_columns].mean())
# Impute missing values for categorical columns with mode
categorical_columns = vehicles.select_dtypes(exclude='number').columns
vehicles[categorical_columns] = vehicles[categorical_columns].fillna(vehicles[categorical_columns].mode().iloc[0])
# Removing duplicates
vehicles[vehicles.duplicated()]
vehicles.drop_duplicates(inplace=True)
# View missing values again
print(vehicles.isnull().sum()) # 0 Missing now!


####### Handle Outliers for numerical variables #######
# Remove outliers in the 'price' column using IQR method
Q1 = vehicles['price'].quantile(0.25)
Q3 = vehicles['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
vehicles = vehicles[(vehicles['price'] >= lower_bound) & (vehicles['price'] <= upper_bound)]
vehicles = vehicles[vehicles['price'] > 5000] # Removing rows where price is below 5000
price_range = (vehicles['price'].min(), vehicles['price'].max()) # Calculate 'price' range
print("Price range (min, max):", price_range) # Display 'price range'
# Remove outliers in the 'year' column using IQR method
Q1 = vehicles['year'].quantile(0.25)
Q3 = vehicles['year'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
vehicles = vehicles[(vehicles['year'] >= lower_bound) & (vehicles['year'] <= upper_bound)]
# Remove outliers in the 'odometer' column using IQR method
Q1 = vehicles['odometer'].quantile(0.25)
Q3 = vehicles['odometer'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
vehicles = vehicles[(vehicles['odometer'] >= lower_bound) & (vehicles['odometer'] <= upper_bound)]
# Remove outliers in the 'lat' column using IQR method
Q1 = vehicles['lat'].quantile(0.25)
Q3 = vehicles['lat'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
vehicles = vehicles[(vehicles['lat'] >= lower_bound) & (vehicles['lat'] <= upper_bound)]
# Remove outliers in the 'long' column using IQR method
Q1 = vehicles['long'].quantile(0.25)
Q3 = vehicles['long'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
vehicles = vehicles[(vehicles['long'] >= lower_bound) & (vehicles['long'] <= upper_bound)]

####### Scale numerical variables #######
#from sklearn.preprocessing import StandardScaler, MaxAbsScaler
#scaler = MaxAbsScaler()
#vehicles[["price", "year"]] = scaler.fit_transform(vehicles[["price", "year"]])




####################################### MODELLING #######################################

####### Simple linear regression #######
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
# Selecting numerical features and target variable
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


####### TEST ANOTHER MODEL #######
