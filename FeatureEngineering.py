# Importing libraries
import pandas as pd
import numpy as np

# Importing the data
vehicles = pd.read_csv('Data/vehicles_Preprocessed.csv') # Uploading the data

# Viewing data types
vehicles.dtypes

####################################### Feature Engineering #######################################

####### Ektra preprossing steps caused by data Exploration ##########

# Extra preprocessing steps caused by data Exploration
    # Creating the years_old variable and removing the original variable 'year'
vehicles['year'] = vehicles['year'].astype('Int64') # Treat year as numeric to be able to substract
vehicles['years_old'] = 2021 - vehicles['year']
vehicles.drop(columns=['year'], inplace=True) # deleting the original year column

    # Removing rows where price is above the upper whisker
    # Caculate the exact point of upper whisker
Q1 = np.percentile(vehicles['price'], 25)
Q3 = np.percentile(vehicles['price'], 75)
IQR = Q3 - Q1                                     # Calculate IQR
upper_whisker = Q3 + 1.5 * IQR                    # Calculate upper whisker
print("Upper Whisker:", upper_whisker)            # Upper whisker: 48325
vehicles = vehicles[vehicles['price'] < 48325]    # To exclude observations where the price is above 48325 (our upper whisker from the boxplot)
    # Removing rows where price is below 1000
vehicles = vehicles[vehicles['price'] > 1000] # To exclude damaged cars and listings with no intention of selling to that price
print((vehicles['price'].min(), vehicles['price'].max())) # View price range

    # Exclude cars with more than 20 years of age
vehicles = vehicles[vehicles['years_old'] <= 20]
vehicles['years_old'] = vehicles['years_old'].astype('object')
    
    # Removing rows where odometer is above 300000
vehicles['odometer'] = vehicles['odometer'].astype('float64') # Treat years_old as float64
vehicles = vehicles[vehicles['odometer'] < 300000]
    
    # Modifying observations
vehicles = vehicles[vehicles['condition'] != 'salvage'] # Remove rows containing 'salvage' in the 'condition' variable
vehicles = vehicles[~vehicles['manufacturer'].isin(['harley-davidson', 'kawasaki'])] # removing motorcycle brands
    
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

# Convert each column to correct data type
vehicles.dtypes # Checking data types
vehicles['manufacturer'] = vehicles['manufacturer'].astype('category')
vehicles['condition'] = vehicles['condition'].astype('category')
vehicles['cylinders'] = vehicles['cylinders'].astype('category')
vehicles['fuel_type'] = vehicles['fuel_type'].astype('category')
vehicles['transmission'] = vehicles['transmission'].astype('category')
vehicles['drive'] = vehicles['drive'].astype('category')
vehicles['car_type'] = vehicles['car_type'].astype('category')
vehicles['paint_color'] = vehicles['paint_color'].astype('category')
vehicles['state'] = vehicles['state'].astype('category')
vehicles['years_old'] = vehicles['years_old'].astype('category')
vehicles['odometer_range'] = vehicles['odometer_range'].astype('category')
# Checking data types agian
vehicles.dtypes # Now all features are converted into categories except our target variable 'price'.

# Lumping
# Manufacturer lumping (creating a 'others' level inside the variable manufacturer)
mf = vehicles['manufacturer'].value_counts()
vehicles['manufacturer'] = vehicles['manufacturer'].apply(lambda s: s if str(s) in mf[:20] else 'others')
vehicles['manufacturer'].value_counts()
vehicles['manufacturer'] = vehicles['manufacturer'].astype('category')
# Paint_color lumping (creating a 'others' level inside the variable paint_color)
paint_color = vehicles['paint_color'].value_counts()
vehicles['paint_color'] = vehicles['paint_color'].apply(lambda s: s if str(s) in paint_color[:9] else 'others')
vehicles['paint_color'].value_counts()
vehicles['paint_color'] = vehicles['paint_color'].astype('category')

# ONE HOT ENCODING
vehicles.dtypes
# Apply one-hot encoding for our features
categorical_columns = ['manufacturer', 'condition', 'cylinders', 'fuel_type', 'transmission', 'drive', 'car_type', 'paint_color', 'state', 'years_old', 'odometer_range']
vehicles = pd.get_dummies(vehicles, columns=categorical_columns, dtype=int)
print(vehicles.head())
vehicles.dtypes

# Removing zero and near zero variance
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)  # Setting threshold at 0.01
vehicles_selected = selector.fit_transform(vehicles)
# vehicles_selected will contain only the features with variance above the threshold
# Imcorporating into a new dataframe:
vehicles = pd.DataFrame(vehicles_selected, columns=vehicles.columns[selector.get_support()])

######################################## Split ########################################

# Stratified sampling / split
from sklearn.model_selection import train_test_split
# Set random seed for reproducibility
import numpy as np
np.random.seed(123)
vehicles.info
# Binning prices into categories 
bins = pd.cut(vehicles['price'], bins=50, labels=False) 
vehicles['price_bin'] = bins
# Splitting based on the price bins
train_vehicles, test_vehicles = train_test_split(vehicles, test_size=0.3, stratify=vehicles['price'])
# Dropping 'price_bin' from both train and test sets
test_vehicles.drop(columns=['price_bin'], inplace=True)
train_vehicles.drop(columns=['price_bin'], inplace=True)
# Now, let's check the data types of the columns in the training set
print(test_vehicles.dtypes)

# 'Price' is the target variable and all other columns are features
X_train = train_vehicles.drop('price', axis=1)
y_train = train_vehicles['price']

X_test = test_vehicles.drop('price', axis=1)
y_test = test_vehicles['price']

# Saving the datasets to pickle files
X_train.to_pickle("./data/X_train.pkl")
y_train.to_pickle("./data/y_train.pkl")
X_test.to_pickle("./data/X_test.pkl")
y_test.to_pickle("./data/y_test.pkl")