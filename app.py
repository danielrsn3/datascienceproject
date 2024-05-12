####################################### APP CREATION (Deployment) #######################################
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from datetime import datetime
pd.set_option('display.max_rows', None)  # Set to None to display all rows
pd.set_option('display.max_columns', None)  # Set to None to display all columns
pd.set_option('display.max_colwidth', None)  # Set to None to display full width of column content
pd.set_option('display.width', None)  # Set to None to make sure the display width fits the terminal or notebook cell

# Loading the test and training datasets from pickle files
X_train = pd.read_pickle("./data/X_train.pkl")
y_train = pd.read_pickle("./data/y_train.pkl")
X_test = pd.read_pickle("./data/X_test.pkl")
y_test = pd.read_pickle("./data/y_test.pkl")
X_train.dtypes
# Train the Random Forest Regressor model again (due to large files, we could not save the previously trained model)
random_forest = RandomForestRegressor(n_estimators=30, random_state=42)
random_forest = random_forest.fit(X_train, y_train)
predictions_random_forest = random_forest.predict(X_test)

# Flask web application
app = Flask(__name__)

def options_html(options_list):
    return ''.join(f'<option value="{option}">{str(option).capitalize()}</option>' for option in options_list)

@app.route('/')
def home():
    # Dropdown options for various car attributes
    manufacturers = ['bmw', 'buick', 'cadillac', 'chevrolet', 'chrysler', 'dodge', 'ford', 'gmc', 'honda', 'hyundai', 'infiniti', 'jeep', 'kia', 'lexus', 'mercedes-benz', 'nissan', 'ram', 'subaru', 'toyota', 'volkswagen', 'others']
    conditions = ['excellent', 'fair', 'good', 'like new']
    cylinders = ['4', '6', '8']
    fuel_types = ['gas', 'diesel', 'hybrid', 'other']
    transmissions = ['automatic', 'manual', 'other']
    drives = ['4wd', 'fwd', 'rwd']
    car_types = ['SUV', 'convertible', 'coupe', 'hatchback', 'mini-van', 'pickup', 'sedan', 'truck', 'van', 'wagon', 'other']
    colors = ['black', 'blue', 'brown', 'custom', 'green', 'grey', 'others', 'red', 'silver', 'white']
    states = ['al', 'az', 'ca', 'co', 'ct', 'fl', 'ga', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'ma', 'mi', 'mn', 'mo', 'nc', 'nj', 'ny', 'oh', 'ok', 'or', 'pa', 'sc', 'tn', 'tx', 'va', 'vt', 'wa', 'wi']
    years = list(range(2001, 2022))  # User selects car manufacturing year between 2001 and 2021
    mileages = ['0-10000', '10000-20000', '20000-30000', '30000-40000', '40000-50000', '50000-60000',
            '60000-70000', '70000-80000', '80000-90000', '90000-100000', '100000-110000', '110000-120000',
            '120000-130000', '130000-140000', '140000-150000', '150000-160000', '160000-170000', '170000-180000',
            '180000-190000', '190000-200000', '200000+']


    return f'''
        <form action="/evaluate">
            Manufacturer: <select name="manufacturer">{options_html(manufacturers)}</select><br>
            Condition: <select name="condition">{options_html(conditions)}</select><br>
            Cylinders: <select name="cylinders">{options_html(cylinders)}</select><br>
            Fuel Type: <select name="fuel_type">{options_html(fuel_types)}</select><br>
            Transmission: <select name="transmission">{options_html(transmissions)}</select><br>
            Drive: <select name="drive">{options_html(drives)}</select><br>
            Car Type: <select name="car_type">{options_html(car_types)}</select><br>
            Color: <select name="color">{options_html(colors)}</select><br>
            State for sale: <select name="state">{options_html(states)}</select><br>
            Car Year: <select name="car_year">{options_html(years)}</select><br>
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
    cylinders = request.args.get('cylinders')
    fuel_type = request.args.get('fuel_type')
    transmission = request.args.get('transmission')
    drive = request.args.get('drive')
    car_type = request.args.get('car_type')
    color = request.args.get('color')
    state = request.args.get('state')
    car_year = int(request.args.get('car_year'))
    mileage_str = request.args.get('mileage')
    price = float(request.args.get('price'))

    # Calculate the age of the car based on the selected year and reference year 2021
    reference_year = 2021
    years_old = reference_year - car_year

    # Setup model input dictionary, initializing all to 0
    model_input = {col: 0 for col in X_train.columns}
    model_input[f'manufacturer_{manufacturer}'] = 1
    model_input[f'condition_{condition}'] = 1
    model_input[f'cylinders_{cylinders}'] = 1
    model_input[f'fuel_type_{fuel_type}'] = 1
    model_input[f'transmission_{transmission}'] = 1
    model_input[f'drive_{drive}'] = 1
    model_input[f'car_type_{car_type}'] = 1
    model_input[f'paint_color_{color}'] = 1
    model_input[f'state_{state}'] = 1
    model_input[f'years_old_{years_old}'] = 1 # use calculated years_old

    # Convert mileage string to feature format
    if mileage_str.endswith('+'):
        mileage_feature = f'odometer_range_{mileage_str.replace("+", "plus")}'  # Handle the open-ended range
    else:
        mileage_feature = f'odometer_range_{mileage_str}'

    # Ensure the mileage feature exists in the model input
    if mileage_feature in model_input:
        model_input[mileage_feature] = 1
    else:
        return jsonify({'error': f'Mileage range {mileage_str} not recognized'})  # Handle invalid mileage range


    # Prepare features for model prediction
    input_features = np.array([list(model_input.values())])

    # Prepare features for model prediction
    predicted_price = random_forest.predict(input_features)[0]
    difference = predicted_price - price
    assessment = "Fair" if abs(difference) < 1000 else "Unfair"

    return jsonify({
        'Your Price': price,
        'Predicted Fair Price': predicted_price,
        'Assessment': assessment
    })

if __name__ == '__main__':
    app.run(debug=True)