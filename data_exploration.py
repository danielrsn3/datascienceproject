# Importing the relevant libraries
import pandas as pd
import matplotlib.pyplot as plt

# To prevent scientific numbers
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load the data
vehicles = pd.read_csv('Data/vehicles_clean.csv') # Uploading the data generated in the preprocessing step
vehicles.dtypes # Display the data types of each column

###### Summary statistics ######
vehicles.describe() # for numerical variable
vehicles.select_dtypes(include=['object']).describe() # for categorical variables

###### Visualizations ######
# Histogram of the 'price' column with prices up to $100,000
plt.figure(figsize=(12, 6))
plt.hist(vehicles['price'][vehicles['price'] <= 100000], bins=50, color='blue', edgecolor='black')
plt.title('Histogram of Vehicle Prices (up to $100,000)')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# HEXBIN
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
plt.boxplot(vehicles['price'], vert=False, flierprops=dict(marker='o', color='red', alpha=0.5))
plt.xlabel('Price')
plt.title('Boxplot of Vehicle Prices')
plt.grid(True)
plt.xlim(0, 100000)  # Limiting the x-axis to enhance detail around the typical price range
plt.show()
    # Comments:
    # Observations above 57300 should be removed.



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
    # Comments:
    # Remove motorcycles 
    # Multiple manufacturers with very high average price


filtered_data = vehicles[(vehicles['price'] <= 100000) & (vehicles['year'].notna())]
# Correlation between categorical features
from scipy.stats import chi2_contingency
import numpy as np

def cramers_v(x, y):
    """Calculate Cramer's V statistic for two categorical series."""
    contingency_table = pd.crosstab(x, y) # Creating a contingency table
    chi2, _, _, _ = chi2_contingency(contingency_table) # Perform the Chi-Squared Test
    n = contingency_table.sum().sum()
    phi2 = chi2 / n # Calculate Phi Squared 
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)) # Correct Phi Squared for Bias
    rcorr = r - ((r-1)**2)/(n-1) # Calculate Corrected Row Totals
    kcorr = k - ((k-1)**2)/(n-1) # Calculate Corrected Column Totals
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) # Compute Cramer's V

# Identify categorical columns
categorical_columns = vehicles.select_dtypes(include=['object']).columns

# Compute Cramer's V matrix
cramers_v_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)

# Optimizing the computation of Cramer's V matrix
for i, col1 in enumerate(categorical_columns):
    for col2 in categorical_columns[i:]:  # Start from current index to avoid recomputation
        if col1 == col2:
            # The correlation of a variable with itself is 1
            cramers_v_value = 1.0
        else:
            cramers_v_value = cramers_v(vehicles[col1], vehicles[col2])
        
        cramers_v_matrix.loc[col1, col2] = cramers_v_value
        cramers_v_matrix.loc[col2, col1] = cramers_v_value  # Symmetric value

cramers_v_matrix

##### HEATMAP ######
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure the data is in float format for plotting
cramers_v_matrix = cramers_v_matrix.astype(float)

# Create a heatmap for Cramer's V matrix
plt.figure(figsize=(10, 8))  # Adjust the size of the plot as necessary
heatmap = sns.heatmap(
    cramers_v_matrix,        # The Cramer's V matrix data
    annot=True,              # Enable annotations inside the quadrants
    fmt=".2f",               # Format numbers to two decimal places
    cmap="viridis",          # Color map for different values in the heatmap
    cbar=True,               # Enable the color bar on the side
    annot_kws={'size': 12, 'color': 'black'}  # Set annotation font size and color for visibility
)
plt.title('Heatmap of Cramer\'s V Statistics Between Categorical Features')  # Title of the plot
plt.show()  # Display the plot