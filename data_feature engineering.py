####################################### Feature Engineering #######################################
# Importing pandas library
import pandas as pd

# Importing the data
vehicles = pd.read_csv('Data/vehicles_Preprocessed.csv') # Uploading the data

# Viewing data types
vehicles.dtypes

# Convert each column to correct data type
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


# Saving as a new csv file
vehicles.to_csv('Data/vehicles_FeatureEngineered.csv', index=False)