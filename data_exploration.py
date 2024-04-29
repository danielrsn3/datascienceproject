# To prevent scientific numbers
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Importing the relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
vehicles = pd.read_csv('Data/vehicles_clean.csv') # Uploading the data
vehicles.dtypes # Display the data types of each column
from data_preprocessing import apply_data_types
apply_data_types(vehicles)
vehicles.dtypes # Display the data types of each column

vehicles['year'] = vehicles['year'].astype('Int64') # Treat year as numeric to get summary statistics
vehicles.describe(include='all') # Get summary statistics of all variables

# Histogram of the 'price' column with prices up to $100,000
plt.figure(figsize=(12, 6))
plt.hist(vehicles['price'][vehicles['price'] <= 100000], bins=50, color='blue', edgecolor='black')
plt.title('Histogram of Vehicle Prices (up to $100,000)')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Scatter plot of price vs. year for vehicles priced up to $100,000
plt.figure(figsize=(12, 8))
plt.scatter(vehicles['year'], vehicles['price'], alpha=0.5, edgecolors='w', color='green')
plt.title('Scatter Plot of Vehicle Price vs. Year (Prices up to $100,000)')
plt.xlabel('Year')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Filter the data to ensure both 'year' and 'price' have valid entries and price is up to $100,000
filtered_data = vehicles[(vehicles['price'] <= 100000) & (vehicles['year'].notna())]
# Hexbin plot with filtered data
plt.figure(figsize=(12, 8))
plt.hexbin(filtered_data['year'], filtered_data['price'], gridsize=50, cmap='Greens', bins='log')
plt.colorbar(label='Log10(N)')
plt.title('Hexbin Plot of Vehicle Price vs. Year (Prices up to $100,000)')
plt.xlabel('Year')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Creating a more detailed boxplot for the 'price' column with a zoomed-in view
plt.figure(figsize=(10, 6))
plt.boxplot(vehicles['price'].dropna(), vert=False, flierprops=dict(marker='o', color='red', alpha=0.5))
plt.xlabel('Price')
plt.title('Boxplot of Vehicle Prices with Outlier Detection')
plt.grid(True)
plt.xlim(0, 100000)  # Limiting the x-axis to enhance detail around the typical price range
plt.show()

# Grouping data by manufacturer and calculating the average price per manufacturer
manufacturer_price_avg = vehicles.groupby('manufacturer')['price'].mean().sort_values(ascending=False)
# Creating a bar chart of average prices per manufacturer
plt.figure(figsize=(14, 8))
manufacturer_price_avg.plot(kind='bar', color='teal')
plt.title('Average Vehicle Price by Manufacturer')
plt.xlabel('Manufacturer')
plt.ylabel('Average Price')
plt.xticks(rotation=90)  # Rotating the manufacturer names for better visibility
plt.grid(axis='y')
plt.show()

# Filtering the data for entries where the manufacturer is 'mercedes-benz'
mercedes_benz_data = vehicles[vehicles['manufacturer'] == 'mercedes-benz']
# Displaying the filtered data
mercedes_benz_data.sort_values(ascending=False, by='price')
    # Outliers
