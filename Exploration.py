####################################### EXPLORATION #######################################

# Importing the relevant libraries
import pandas as pd
import matplotlib.pyplot as plt

# To prevent scientific numbers
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load the data
vehicles = pd.read_csv('Data/vehicles_Preprocessed.csv') # Uploading the data generated in the preprocessing step
vehicles.dtypes # Display the data types of each column

# Summary statistics
vehicles.describe() # for numerical variables
vehicles.select_dtypes(include=['object']).describe() # for categorical variables

# Filtering the data to ensure visualization clarity
filtered_data = vehicles[(vehicles['price'] <= 100000)] # excluding observations where price is above $100,000

# Histogram of the 'price' column 
plt.figure(figsize=(12, 6)) # specifying the dimensions of the figure
plt.hist(filtered_data['price'], bins=100, color='blue', edgecolor='black') # 100 bins in blue color with black edges
plt.title('Histogram of Vehicle Prices') # specifying the title of the plot
plt.xlabel('Price') # labelling the x-axis
plt.ylabel('Frequency') # labelling the y-axis
plt.grid(True) # enabling a grid to ease interpretation
plt.show() # displaying the plot

# Hexbin plot of 'price' vs. 'year'
plt.figure(figsize=(12, 8)) # specifying the dimensions of the figure
plt.hexbin(filtered_data['year'], filtered_data['price'], gridsize=50, cmap='Greens', bins='log') # gridsize specifies the number of hexagons, cmap specifies the colormap, bins='log' applies log scale to the count of points in each bin/hex
plt.colorbar(label='Log10(N)') # adding a color bar to the plot named 'Log10(N)' to indicate that the color bar represents the log base 10 og the count of data points in each bin/hex
plt.title('Hexbin Plot of Vehicle Price vs. Year') # specifying the title of the plot
plt.xlabel('Year') # labelling the x-axis
plt.ylabel('Price') # labelling the y-axis
plt.grid(True) # enabling a grid to ease interpretation
plt.show() # displaying the plot

# Boxplot of Vehicle Price (outlier detection)
        # Limiting the x-axis instead of using the filtered dataset to ensure all data is being considered when calculating quartiles and medians
plt.figure(figsize=(10, 6)) # specifying the dimensions of the figure
plt.boxplot(vehicles['price'], vert=False, flierprops=dict(marker='o', color='red', alpha=0.5)) # generating a horizontal boxplot and specifying the appearance of the outliers
plt.xlabel('Price') # labelling the x-axis
plt.title('Boxplot of Vehicle Prices') # specifying the title of the plot
plt.grid(True) # enabling a grid
plt.xlim(0, 100000) # Limiting the x-axis to enhance detail around the typical price range  
plt.show() # displaying the plot

vehicles.sort_values(by='price', ascending=False)

# Grouping data by manufacturer and calculating the average price per manufacturer
manufacturer_price_avg = vehicles.groupby('manufacturer')['price'].mean().sort_values(ascending=False)
# Bar chart of average 'price' per 'manufacturer'
plt.figure(figsize=(14, 8)) # specifying the dimensions of the figure
manufacturer_price_avg.plot(kind='bar', color='teal') # kind specifies bar chat
plt.title('Average Vehicle Price by Manufacturer') # specifying the title of the plot
plt.xlabel('Manufacturer') # labelling the x-axis
plt.ylabel('Average Price') # labelling the y-axis
plt.xticks(rotation=90)  # Rotating the manufacturer names for better visibility
plt.grid(axis='y') # enabling grid lines along the y-axis
plt.show() # displaying the plot
    # Comments:
    # Remove motorcycle manufacturers
    # Multiple manufacturers with very high average price, these highly prices vechiles will be removed later.

# Correlation between categorical features
from scipy.stats import chi2_contingency # used to perform the Chi-squared test for independence on a contingency table
import numpy as np
def cramers_v(x, y): # defining the Cramer's V calculation function
    """Calculate Cramer's V statistic for two categorical series."""
    contingency_table = pd.crosstab(x, y) # creating a contingency table using pandas
    chi2, _, _, _ = chi2_contingency(contingency_table) # perform the chi-squared test
    n = contingency_table.sum().sum() # calculating the total number of observations
    phi2 = chi2 / n # calculate phi squared 
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)) # correct phi squared for bias
    rcorr = r - ((r-1)**2)/(n-1) # correcting the number of rows for bias.
    kcorr = k - ((k-1)**2)/(n-1) # correcting the number of columns for bias.
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) # Compute Cramer's V
# Identify categorical columns
categorical_columns = vehicles.select_dtypes(include=['object']).columns 
# Compute Cramer's V matrix
cramers_v_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns) # initializing an empty dataframe to store the Cramer's V values
# Iterate over each pair of categorical columns to compute Cramer's V
# Optimizing by starting the inner loop from the current index of the outer loop to avoid redundant computations
for i, col1 in enumerate(categorical_columns):
    for col2 in categorical_columns[i:]:  # Start from current index to avoid recomputation
        if col1 == col2: 
            cramers_v_value = 1.0 # The correlation of a variable with itself is 1
        else:
            cramers_v_value = cramers_v(vehicles[col1], vehicles[col2]) # if two different variables, calculate Cramer's V
        
        cramers_v_matrix.loc[col1, col2] = cramers_v_value # storing the computed value in the matrix
        cramers_v_matrix.loc[col2, col1] = cramers_v_value  # Symmetric value

cramers_v_matrix


# HEATMAP
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
plt.title('Heatmap of Cramer\'s V Statistics Between Categorical Features') # specifying the title of the plot
plt.show() # displaying the plot