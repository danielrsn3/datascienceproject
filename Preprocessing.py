####################################### PREPROCESSING #######################################
# GENERAL OPTIONS
import pandas as pd # Import pandas
pd.set_option('display.max_columns', None) # Allow to see all columns when viewing data
pd.set_option('display.float_format', lambda x: '%.3f' % x) # Prevent scientific numbers

# UPLOADING DATA
vehicles = pd.read_csv('Data/vehicles_Raw.csv')

# Viewing the data
print(vehicles.columns) # View all cloums in the dataframe
vehicles.info() # Summary of the DataFrame
vehicles.dtypes # Display the data types of each column

# Remove excessive variables
vehicles.drop(columns=['id'], inplace=True) # ID it not required
vehicles.drop(columns=['county'], inplace=True) # 100 % missing / coloum left in by mistake
vehicles.drop(columns=['VIN'], inplace=True) # Irrelevant
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

# Renaming variables
vehicles.rename(columns={'fuel': 'fuel_type'}, inplace=True)
vehicles.rename(columns={'type': 'car_type'}, inplace=True)

# Modifying observations
vehicles = vehicles[vehicles['condition'] != 'salvage'] # Remove rows containing 'salvage' in the 'condition' variable
vehicles = vehicles[~vehicles['manufacturer'].isin(['harley-davidson', 'kawasaki'])] # removing motorcycle brands

# Creating the years_old variable and removing the original variable 'year'
vehicles['year'] = vehicles['year'].astype('Int64') # Treat year as numeric to be able to substract
vehicles['years_old'] = 2021 - vehicles['year']
vehicles.drop(columns=['year'], inplace=True) # deleting the original year column


# Setting max and min ranges for our data
    # Removing rows where price is below 1000
vehicles = vehicles[vehicles['price'] > 1000] # To exclude damaged cars and listings with no intention of selling to that price
vehicles = vehicles[vehicles['price'] < 57300] # To exclude observations where the price is above 57300 (our upper whisker from the boxplot)
print((vehicles['price'].min(), vehicles['price'].max())) # View price range
vehicles = vehicles[vehicles['years_old'] <= 30] # To exclude cars older than 1980 
vehicles['years_old'] = vehicles['years_old'].astype('category') # Treat years_old as category

    # Removing rows where odometer is above 300000
vehicles['odometer'] = vehicles['odometer'].astype('float64') # Treat years_old as category
vehicles = vehicles[vehicles['odometer'] < 300000]

# Creating new variables
    # Creating Ranges for 'odometer'
    # Define the ranges for the bins
bins = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000, float('inf')]
    # Define labels for the bins
labels = ['0-10000', '10000-20000', '20000-30000', '30000-40000', '40000-50000', '50000-60000', '60000-70000', '70000-80000', '80000-90000', '90000-100000', '100000-110000', '110000-120000', '120000-130000', '130000-140000', '140000-150000', '150000-160000', '160000-170000', '170000-180000', '180000-190000', '190000-200000', '200000+']
    # Create a new column 'odometer_range' with the binned values
vehicles['odometer_range'] = pd.cut(vehicles['odometer'], bins=bins, labels=labels, right=False)
    # Drop the old 'odometer' variable
vehicles.drop(columns=['odometer'], inplace=True)
    # Display the count of values in each bin
print(vehicles['odometer_range'].value_counts())
vehicles['odometer_range'] = vehicles['odometer_range'].astype('category')
vehicles.dtypes

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

# Drop NA
vehicles = vehicles.dropna() # dropping all remaining missings

# Saving as a new csv file
vehicles.to_csv('Data/vehicles_Preprocessed.csv', index=False)