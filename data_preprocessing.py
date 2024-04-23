####### UPLOADING DATA #######
import pandas as pd # Import pandas
pd.set_option('display.max_columns', None) # Allow to see all columns when viewing data
vehicles = pd.read_csv('Data/vehicles_no_url.csv') # Uploading the data

####### GENERAL COMMANDS TO VIEW STUFF #######
print(vehicles.columns) # View all cloums in the dataframe
vehicles.info # Summary of the DataFrame
print(vehicles['year']) # View specific variables
vehicles.dtypes # Display the data types of each column

####################################### PREPROCESSING #######################################
####### Remove excessive variables #######
# Removing the columns
vehicles.drop(columns=['id'], inplace=True) # Useless
vehicles.drop(columns=['county'], inplace=True) # No predictive power
vehicles.drop(columns=['VIN'], inplace=True) # Useless
vehicles.drop(columns=['posting_date'], inplace=True) # We just need the year of the car, not the posting date
vehicles.drop(columns=['lat'], inplace=True) # Useless
vehicles.drop(columns=['long'], inplace=True) # Useless
vehicles.drop(columns=['size'], inplace=True) # Many missings
vehicles.drop(columns=['region'], inplace=True) # Execcive beacuse of state coloum
vehicles.drop(columns=['model'], inplace=True) # Poor data quality

####### Modifying observations #######


# Remove rows containing 'salvage' in the 'condition' variable
vehicles[vehicles['condition'] != 'salvage']

# Removing rows where price is below 1000
vehicles = vehicles[vehicles['price'] > 1000] # To exclude damaged cars and listings with no intention of selling to that price
print((vehicles['price'].min(), vehicles['price'].max())) # View price range

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

# Removing "cylinders" in all the rows from the column cylinder
vehicles['cylinders']
vehicles['cylinders'] = vehicles['cylinders'].str.replace(' cylinders', '').str.replace('cylinders', '') 



####### Saving as a new csv file #######
vehicles.to_csv('Data/vehicles_clean.csv', index=False)





####### Correcting data types #######
# Convert each column to correct data type (creating function)
def apply_data_types(df):
    df['manufacturer'] = df['manufacturer'].astype('category')
    df['condition'] = df['condition'].astype('category')
    df['cylinders'] = df['cylinders'].astype('category')
    df['fuel'] = df['fuel'].astype('category')
    df['title_status'] = df['title_status'].astype('category')
    df['transmission'] = df['transmission'].astype('category')
    df['drive'] = df['drive'].astype('category')
    df['type'] = df['type'].astype('category')
    df['paint_color'] = df['paint_color'].astype('category')
    df['state'] = df['state'].astype('category')
    df['year'] = df['year'].astype('Int64') # To remove decimal points / Truncates
    df['year'] = df['year'].astype('category')
    return df
