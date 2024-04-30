####################################### Feature Engineering #######################################
import pandas as pd
vehicles = pd.read_csv('Data/vehicles_clean.csv') # Uploading the data
vehicles.dtypes # Display the data types of each column
from data_preprocessing import apply_data_types
apply_data_types(vehicles)
vehicles.dtypes # Display the data types of each column


# Stratified sampling / split
import pandas as pd
from sklearn.model_selection import train_test_split
# Assuming 'vehicles' is your DataFrame and 'price' is the column you want to stratify by
# You need to replace 'vehicles' and 'price' with the actual column names in your DataFrame
# Set random seed for reproducibility
import numpy as np
np.random.seed(123)
# Binning prices into categories
bins = pd.cut(vehicles['price'], bins=5, labels=False)  # Adjust the number of bins as needed
vehicles['price_bin'] = bins
# Splitting based on the price bins
train_strat, test_strat = train_test_split(vehicles, test_size=0.3, stratify=vehicles['price_bin'])


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









####### Modifying observations #######
# Remove rows containing 'salvage' in the 'condition' variable
vehicles = vehicles[vehicles['condition'] != 'salvage']

# Removing rows where price is below 1000
vehicles = vehicles[vehicles['price'] > 1000] # To exclude damaged cars and listings with no intention of selling to that price
print((vehicles['price'].min(), vehicles['price'].max())) # View price range



# Creating Ranges for 'odometer' ( LAV OM TIL mileage)
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

# df['odometer_range'] = df['odometer_range'].astype('category')




# Remove motercycles
#harley d   
#kawasaki
