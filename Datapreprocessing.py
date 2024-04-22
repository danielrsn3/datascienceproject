# Uploading the data
import pandas as pd
vehicles = pd.read_csv('Data/vehicles_no_url.csv')
vehicles
vehicles.info()
# Showing whether a columns has missings and not
vehicles.isna().any()
# Example of showing how many missings in one column
vehicles[vehicles['year'].isna()].shape[0]
# Getting summary statistics of the numeric variables
vehicles.describe()