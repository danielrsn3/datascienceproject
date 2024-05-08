###### GENERAL OPTIONS ######
import pandas as pd # Import pandas
pd.set_option('display.max_columns', None) # Allow to see all columns when viewing data
pd.set_option('display.float_format', lambda x: '%.3f' % x) # Prevent scientific numbers

####### UPLOADING DATA #######
vehicles = pd.read_csv('Data/vehicles_no_url.csv') # Uploading the data 

####### Viewing the data #######
print(vehicles.columns) # View all cloums in the dataframe
vehicles.info() # Summary of the DataFrame
vehicles.dtypes # Display the data types of each column


####################################### PREPROCESSING #######################################
####### Remove excessive variables #######
vehicles.drop(columns=['id'], inplace=True) # ID it not required
vehicles.drop(columns=['county'], inplace=True) # 100 % missing / coloum left in by mistake
vehicles.drop(columns=['VIN'], inplace=True) # Useless
vehicles.drop(columns=['posting_date'], inplace=True) # We just need the year of the car, not the posting date
vehicles.drop(columns=['lat'], inplace=True) # Excessive because of state coloum
vehicles.drop(columns=['long'], inplace=True) # Excessive because of state coloum
vehicles.drop(columns=['region'], inplace=True) # Excessive because of state coloum
vehicles.drop(columns=['size'], inplace=True) # Too Many missings / Craigslist sellers are not using this / 72 % missing
vehicles.drop(columns=['model'], inplace=True) # Poor data quality
vehicles.drop(columns=['title_status'], inplace=True) # Poor data quality

# Removing the wording of "cylinders" in all the rows from the column cylinder, we just need the number of cylinders.
vehicles['cylinders']
vehicles['cylinders'] = vehicles['cylinders'].str.replace(' cylinders', '').str.replace('cylinders', '') 

######## Renaming variables #######
vehicles.rename(columns={'fuel': 'fuel_type'}, inplace=True)
vehicles.rename(columns={'type': 'car_type'}, inplace=True)

##### Modifying observations #####
vehicles = vehicles[vehicles['condition'] != 'salvage'] # Remove rows containing 'salvage' in the 'condition' variable
vehicles = vehicles[~vehicles['manufacturer'].isin(['harley-davidson', 'kawasaki'])] # removing motorcycle brands

##### Setting max and min ranges for our data #####
# Removing rows where price is below 1000
vehicles = vehicles[vehicles['price'] > 1000] # To exclude damaged cars and listings with no intention of selling to that price
vehicles = vehicles[vehicles['price'] < 57300] # To exclude observations where the price is above 57300 (our upper whisker from the boxplot)
print((vehicles['price'].min(), vehicles['price'].max())) # View price range
vehicles = vehicles[vehicles['years_old'] <= 30] # To exclude cars older than 1980 
vehicles['years_old'] = vehicles['years_old'].astype('category') # Treat years_old as category

# Assuming 'vehicles' is your DataFrame containing the 'odometer' variable
vehicles['odometer'] = vehicles['odometer'].astype('float64') # Treat years_old as float64
vehicles = vehicles[vehicles['odometer'] < 300000]


###### Plot the percentage of missing values in each column ######
import matplotlib.pyplot as plt
import seaborn as sns
# Calculate the percentage of missing values for each column
missing_data = vehicles.isnull().mean() * 100
# Create a bar plot to visualize the percentage of missing data by column
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_data.values, y=missing_data.index, palette='viridis')
plt.title('Percentage of Missing Data by Column')
plt.xlabel('Percentage of Missing Values')
plt.ylabel('Columns')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
# Droping NA's
vehicles_nona = vehicles.dropna() # dropping all remaining missings

# Remove unused categories
categorical_columns = ['manufacturer', 'condition', 'cylinders', 'fuel_type', 'transmission',
                       'drive', 'car_type', 'paint_color', 'state', 'years_old', 'odometer_range']
# Iterate through each categorical column and remove unused categories
for col in categorical_columns:
    vehicles_nona[col] = vehicles_nona[col].cat.remove_unused_categories()

####### Saving as a new csv file #######
vehicles.to_csv('Data/vehicles_clean.csv', index=False)

#### FUNCTION TO BE CALLED TO CORRECT DATA TYPES ####
# Convert each column to correct data type (creating function)
# This function will be ran in 'feature engineering'
def apply_data_types(df):
    df['manufacturer'] = df['manufacturer'].astype('category')
    df['condition'] = df['condition'].astype('category')
    df['cylinders'] = df['cylinders'].astype('category')
    df['fuel_type'] = df['fuel_type'].astype('category')
    df['transmission'] = df['transmission'].astype('category')
    df['drive'] = df['drive'].astype('category')
    df['car_type'] = df['car_type'].astype('category')
    df['paint_color'] = df['paint_color'].astype('category')
    df['state'] = df['state'].astype('category')
    df['year'] = df['year'].astype('Int64') # To remove decimal points / Truncates
    df['year'] = df['year'].astype('category')
    df['odometer'] = df['odometer'].astype('category')
    return df