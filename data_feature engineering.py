####################################### Feature Engineering #######################################
import pandas as pd
vehicles = pd.read_csv('Data/vehicles_clean.csv') # Uploading the data
vehicles.dtypes # Display the data types of each column
from data_preprocessing import apply_data_types
apply_data_types(vehicles)
vehicles.dtypes # Display the data types of each column

##### Modifying observations #####
vehicles = vehicles[vehicles['condition'] != 'salvage'] # Remove rows containing 'salvage' in the 'condition' variable
vehicles = vehicles[~vehicles['manufacturer'].isin(['harley-davidson', 'kawasaki'])] # removing motorcycle brands


# Creating the years_old variable
vehicles['year'] = vehicles['year'].astype('Int64') # Treat year as numeric to be able to substract
vehicles['years_old'] = 2021 - vehicles['year']
vehicles.drop(columns=['year'], inplace=True) # deleting the original year column

##### Setting max and min ranges for our data #####
# Removing rows where price is below 1000
vehicles = vehicles[vehicles['price'] > 1000] # To exclude damaged cars and listings with no intention of selling to that price
vehicles = vehicles[vehicles['price'] < 57300] # To exclude observations where the price is above 57300 (our upper whisker from the boxplot)
print((vehicles['price'].min(), vehicles['price'].max())) # View price range
vehicles = vehicles[vehicles['years_old'] <= 30] # To exclude cars older than 1980 
vehicles['years_old'] = vehicles['years_old'].astype('category') # Treat years_old as category

# Assuming 'vehicles' is your DataFrame containing the 'odometer' variable
vehicles['odometer'] = vehicles['odometer'].astype('float64') # Treat years_old as category
vehicles = vehicles[vehicles['odometer'] < 300000]


##### Creating new variables #####
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

vehicles_nona = vehicles.dropna() # dropping all remaining missings

# Stratified sampling / split
from sklearn.model_selection import train_test_split
# Assuming 'vehicles' is your DataFrame and 'price' is the column you want to stratify by
# You need to replace 'vehicles' and 'price' with the actual column names in your DataFrame
# Set random seed for reproducibility
import numpy as np
np.random.seed(123)
# Binning prices into categories
bins = pd.cut(vehicles_nona['price'], bins=50, labels=False)  # Adjust the number of bins as needed
vehicles_nona['price_bin'] = bins
# Splitting based on the price bins
train_vehicles, test_vehicles = train_test_split(vehicles_nona, test_size=0.3, stratify=vehicles_nona['price_bin'])
# slet price_bin fra test og train
test_vehicles = test_vehicles.drop(columns=['price_bin'], inplace=True)
train_vehicles = train_vehicles.drop(columns=['price_bin'], inplace=True)





####### Lumping #######
print(vehicles['manufacturer'].value_counts())

# Example of how to perform lumping based on a threshold
# Define a Function to Lump Categories
def lump_categories(series, threshold=100): # set threshold
    counts = series.value_counts()
    return series.apply(lambda x: x if counts[x] > threshold else 'Other')
# Using the threshold-based lumping
vehicles['manufacturer'] = lump_categories(vehicles['manufacturer'], threshold=3)

# Example of how to perform lumping based on a manual specification
def manual_lump(series):
    mapping = {
        'fiat': 'other', # insert all categories of the variable and manually specify new category
        'mini': 'other',
        'mitsubishi': 'other',
        'volvo': 'other'
    }
    return series.map(mapping)
vehicles['manufacturer'] = manual_lump(vehicles['manufacturer'])

# ONE HOT ENCODING







####### Remove near zero variance #######
from sklearn.feature_selection import VarianceThreshold
# Assuming vehicles is your dataset
selector = VarianceThreshold()
# Fit the selector to your data and transform it
vehicles_selected = selector.fit_transform(vehicles_nona)
# vehicles_selected will contain only the features with non-zero variance

vehicles.to_csv('Data/vehicles_nona.csv', index=False)