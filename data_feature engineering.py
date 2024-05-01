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
vehicles.dtypes


###### Plot the percentage of missing values in each column ######
import matplotlib.pyplot as plt
#import seaborn as sns
# Calculate the percentage of missing values for each column
#missing_data = vehicles.isnull().mean() * 100
# Create a bar plot to visualize the percentage of missing data by column
#plt.figure(figsize=(10, 6))
#sns.barplot(x=missing_data.values, y=missing_data.index, palette='viridis')
#plt.title('Percentage of Missing Data by Column')
#plt.xlabel('Percentage of Missing Values')
#plt.ylabel('Columns')
#plt.grid(True, linestyle='--', alpha=0.6)
#plt.show()

# Drop NA
vehicles_nona = vehicles.dropna() # dropping all remaining missings

# Remove unused categories
categorical_columns = ['manufacturer', 'condition', 'cylinders', 'fuel_type', 'transmission',
                       'drive', 'car_type', 'paint_color', 'state', 'years_old', 'odometer_range']
# Iterate through each categorical column and remove unused categories
for col in categorical_columns:
    vehicles_nona[col] = vehicles_nona[col].cat.remove_unused_categories()

# Stratified sampling / split
from sklearn.model_selection import train_test_split
import numpy as np
# Assuming 'vehicles_nona' is your DataFrame and 'price' is the column you want to stratify by
# Set random seed for reproducibility
np.random.seed(123)
# Binning prices into categories
bins = pd.cut(vehicles_nona['price'], bins=50, labels=False)  # Adjust the number of bins as needed
vehicles_nona['price_bin'] = bins
# Splitting based on the price bins
train_vehicles, test_vehicles = train_test_split(vehicles_nona, test_size=0.3, stratify=vehicles_nona['price_bin'])
# Dropping 'price_bin' from both train and test sets
test_vehicles.drop(columns=['price_bin'], inplace=True)
train_vehicles.drop(columns=['price_bin'], inplace=True)
# Now, let's check the data types of the columns in the training set
print(train_vehicles.dtypes)



####### Lumping ####### (other)
train_vehicles.dtypes

# Manufacturer lumping
mf = train_vehicles['manufacturer'].value_counts()
train_vehicles['manufacturer'] = train_vehicles['manufacturer'].apply(lambda s: s if str(s) in mf[:20] else 'others')
train_vehicles['manufacturer'].value_counts()
train_vehicles['manufacturer'] = train_vehicles['manufacturer'].astype('category')

# Condition lumping (SHOULDNT BE DONE)
train_vehicles['condition'].value_counts()

# Cylinders lumping (SHOULDNT BE DONE)
train_vehicles['cylinders'].value_counts()

# Fuel type lumping (SHOULDNT BE DONE)
train_vehicles['fuel_type'].value_counts()

# Transmission lumping (SHOULDNT BE DONE)
train_vehicles['transmission'].value_counts()

# Drive lumping (SHOULDNT BE DONE)
train_vehicles['drive'].value_counts()

# Car type lumping (already a other)
train_vehicles['car_type'].value_counts()

# Paint color lumping
paint_color = train_vehicles['paint_color'].value_counts()
train_vehicles['paint_color'] = train_vehicles['paint_color'].apply(lambda s: s if str(s) in paint_color[:9] else 'others')
train_vehicles['paint_color'].value_counts()
train_vehicles['paint_color'] = train_vehicles['paint_color'].astype('category')

# State lumping (SHOULDNT BE DONE)
train_vehicles['state'].value_counts()

# Years old lumping (SHOULDNT BE DONE)
train_vehicles['years_old'].value_counts()


###### ONE HOT ENCODING ######
train_vehicles.dtypes
# Select coloumns
categorical_columns = ['manufacturer', 'condition', 'cylinders', 'fuel_type', 'transmission', 'drive', 'car_type', 'paint_color', 'state', 'years_old', 'odometer_range']
# Apply one-hot encoding
train_vehicles = pd.get_dummies(train_vehicles, columns=categorical_columns, dtype=int)
# Optionally, drop the original categorical columns if you want only the encoded data
# data_encoded = data_encoded.drop(categorical_columns, axis=1)
print(train_vehicles.head())
train_vehicles.dtypes


###### Removing zero and near zero variance #####
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)  # Adjust threshold as needed
train_vehicles_selected = selector.fit_transform(train_vehicles)
# vehicles_selected will contain only the features with variance above the threshold
# Creating df
train_vehicles = pd.DataFrame(train_vehicles_selected, columns=train_vehicles.columns[selector.get_support()])


####### Saving as a new csv file #######
vehicles.to_csv('Data/train_vehicles_models.csv', index=False)
