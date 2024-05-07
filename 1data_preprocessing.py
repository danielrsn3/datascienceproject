####### UPLOADING DATA #######
import pandas as pd # Import pandas
pd.set_option('display.max_columns', None) # Allow to see all columns when viewing data
pd.set_option('display.float_format', lambda x: '%.3f' % x) # Prevent scientific numbers
vehicles = pd.read_csv('Data/vehicles_no_url.csv') # Uploading the data

####### GENERAL COMMANDS TO VIEW STUFF #######
print(vehicles.columns) # View all cloums in the dataframe
vehicles.info() # Summary of the DataFrame
vehicles.dtypes # Display the data types of each column


####################################### PREPROCESSING #######################################
####### Remove excessive variables #######
# Removing the columns
vehicles.drop(columns=['id'], inplace=True) # Useless
vehicles.drop(columns=['county'], inplace=True) # 100 % missing / Useless coloum left in by mistake
vehicles.drop(columns=['VIN'], inplace=True) # Useless
vehicles.drop(columns=['posting_date'], inplace=True) # We just need the year of the car, not the posting date
vehicles.drop(columns=['lat'], inplace=True) # Excessive because of state coloum
vehicles.drop(columns=['long'], inplace=True) # Excessive because of state coloum
vehicles.drop(columns=['region'], inplace=True) # Excessive because of state coloum
vehicles.drop(columns=['size'], inplace=True) # Too Many missings / Craigslist sellers are not using this / 72 % missing
vehicles.drop(columns=['model'], inplace=True) # Poor data quality
vehicles.drop(columns=['title_status'], inplace=True) # Poor data quality

# Removing "cylinders" in all the rows from the column cylinder
vehicles['cylinders']
vehicles['cylinders'] = vehicles['cylinders'].str.replace(' cylinders', '').str.replace('cylinders', '') 

######## Renaming variables #######
# fuel
vehicles.rename(columns={'fuel': 'fuel_type'}, inplace=True)
# type
vehicles.rename(columns={'type': 'car_type'}, inplace=True)


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

####### Saving as a new csv file #######
vehicles.to_csv('Data/vehicles_clean.csv', index=False)