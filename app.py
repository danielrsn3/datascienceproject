####################################### APP CREATION (Deployment) #######################################
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from datetime import datetime

# Load and prepare data
vehicles = pd.read_csv('Data/vehicles_FeatureEngineered.csv')

# Stratified sampling based on price
np.random.seed(123)
bins = pd.cut(vehicles['price'], bins=50, labels=False)
vehicles['price_bin'] = bins
train_vehicles, test_vehicles = train_test_split(vehicles, test_size=0.3, stratify=vehicles['price_bin'])
train_vehicles.drop(columns=['price_bin'], inplace=True)
test_vehicles.drop(columns=['price_bin'], inplace=True)

# Separate features and target
X_train = train_vehicles.drop('price', axis=1)
y_train = train_vehicles['price']
X_test = test_vehicles.drop('price', axis=1)
y_test = test_vehicles['price']

# Train a Decision Tree Regressor
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error: {rmse}")

# Flask web application
app = Flask(__name__)

# Helper function for creating dropdown options
def options_html(options_list):
    return ''.join(f'<option value="{option}">{str(option).capitalize()}</option>' for option in options_list)

@app.route('/')
def home():
    # Define dropdown options
    manufacturers = ['bmw', 'ford', 'chevrolet', 'honda', 'toyota', 'others']
    conditions = ['excellent', 'good', 'fair', 'like new']
    fuel_types = ['gas', 'diesel', 'hybrid', 'other']
    transmissions = ['automatic', 'manual', 'other']
    drives = ['4wd', 'fwd', 'rwd']
    colors = ['black', 'white', 'red', 'blue', 'grey', 'green']
    years = list(range(2000, 2023))
    mileages = ['0-10,000', '10,001-50,000', '50,001-100,000', '100,001-150,000', '150,001-200,000', '200,001+']

    # Form for user input
    return f'''
        <form action="/evaluate">
            Manufacturer: <select name="manufacturer">{options_html(manufacturers)}</select><br>
            Condition: <select name="condition">{options_html(conditions)}</select><br>
            Fuel Type: <select name="fuel_type">{options_html(fuel_types)}</select><br>
            Transmission: <select name="transmission">{options_html(transmissions)}</select><br>
            Drive: <select name="drive">{options_html(drives)}</select><br>
            Color: <select name="color">{options_html(colors)}</select><br>
            Year: <select name="year">{options_html(years)}</select><br>
            Mileage: <select name="mileage">{options_html(mileages)}</select><br>
            Price: <input type="text" name="price"><br>
            <input type="submit" value="Submit">
        </form>
    '''

@app.route('/evaluate')
def evaluate():
    # Retrieve user inputs
    manufacturer = request.args.get('manufacturer')
    condition = request.args.get('condition')
    fuel_type = request.args.get('fuel_type')
    transmission = request.args.get('transmission')
    drive = request.args.get('drive')
    color = request.args.get('color')
    year = int(request.args.get('year'))
    mileage_str = request.args.get('mileage')
    price = float(request.args.get('price'))

    # Calculate years_old from the provided year
    reference_year = datetime.now().year
    years_old = reference_year - year

    # Create a default feature set
    model_input = {col: 0 for col in X_train.columns}
    model_input[f'manufacturer_{manufacturer}'] = 1
    model_input[f'condition_{condition}'] = 1
    model_input[f'fuel_type_{fuel_type}'] = 1
    model_input[f'transmission_{transmission}'] = 1
    model_input[f'drive_{drive}'] = 1
    model_input[f'paint_color_{color}'] = 1

    # Check if years_old_X exists in the model
    years_old_key = f'years_old_{years_old}'
    if years_old_key in model_input:
        model_input[years_old_key] = 1
    else:
        return jsonify({'error': f'Invalid car age: {years_old} not in range'})

    # Convert mileage to a feature
    mileage_feature = f'odometer_range_{mileage_str.replace(",", "").replace("-", "").replace("+", "plus")}'
    if mileage_feature in model_input:
        model_input[mileage_feature] = 1

    # Convert dictionary to array format for model prediction
    input_features = np.array([list(model_input.values())])

    # Predict the fair price
    predicted_price = clf.predict(input_features)[0]
    difference = predicted_price - price
    assessment = "Fair" if abs(difference) < 1000 else "Unfair"

    return jsonify({
        'Your Price': price,
        'Predicted Fair Price': predicted_price,
        'Assessment': assessment
    })

if __name__ == '__main__':
    app.run(debug=True)
