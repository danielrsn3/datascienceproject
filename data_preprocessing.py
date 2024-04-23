import pandas as pd # Import pandas
pd.set_option('display.max_columns', None) # Allow to see all columns when viewing data
vehicles = pd.read_csv('Data/vehicles_no_url.csv') # Uploading the data
print(vehicles.columns) # View all cloums in the dataframe
vehicles.info # Summary of the DataFrame
print(vehicles['year']) # View specific variables
vehicles.dtypes # Display the data types of each column

####################################### PREPROCESSING #######################################

####### Remove excessive variables #######
# Removing the 'id' column
vehicles.drop(columns=['id'], inplace=True)
# Drop the 'county' variable from the DataFrame since it still has missings for some reason (find ud af hvorfor)
vehicles.drop(columns=['county'], inplace=True)
# Removing the VIN column as it has no predictive power
vehicles.drop(columns=['VIN'], inplace=True)
# Removing the posting_date column
vehicles.drop(columns=['posting_date'], inplace=True)
# Removing latitude and longtitude columns
vehicles.drop(columns=['lat'], inplace=True)
vehicles.drop(columns=['long'], inplace=True)
# Removing size column because of many missings
vehicles.drop(columns=['size'], inplace=True)

####### Modifying observations #######
price_range = (vehicles['price'].min(), vehicles['price'].max()) # Calculate 'price' range
print("Price range (min, max):", price_range) # Display 'price range'
vehicles = vehicles[vehicles['price'] < 1000] # Removing rows where price is below 5000
vehicles[~vehicles['region'].str.contains('/')] # Removing all observations where region variable has a '/'


####### Correcting data types #######
vehicles['year'] = vehicles['year'].astype('Int64')
vehicles['year'] = vehicles['year'].astype('object')

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
# Remove outliers in the 'odometer' column using IQR method
Q1 = vehicles['odometer'].quantile(0.25)
Q3 = vehicles['odometer'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
vehicles = vehicles[(vehicles['odometer'] >= lower_bound) & (vehicles['odometer'] <= upper_bound)]

####### Scale numerical variables #######
#from sklearn.preprocessing import StandardScaler, MaxAbsScaler
#scaler = MaxAbsScaler()
#vehicles[["price", "year"]] = scaler.fit_transform(vehicles[["price", "year"]])

####### Remove near zero variance #######


#### Saving as a new csv file ####
vehicles.to_csv('Data/vehicles_clean.csv', index=False)
