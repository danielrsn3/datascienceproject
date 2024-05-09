####################################### PREPROCESSING #######################################

# GENERAL OPTIONS
import pandas as pd
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

# Missing values
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

vehicles.info() # 426880 entries before remvoing missings

# Remove missing values
vehicles = vehicles.dropna() # dropping all remaining missings

vehicles.info() # 117169 entries before remvoing missings

# Saving as a new csv file
vehicles.to_csv('Data/vehicles_Preprocessed.csv', index=False)