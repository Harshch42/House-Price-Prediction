import numpy as np
import pickle
import json
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pickled model
with open('Banglore_house_price_prediction.pkl', 'rb') as model_file:
    regressor1 = pickle.load(model_file)

# Load the feature names from columns.json
with open('columns.json', 'r') as columns_file:
    columns = json.load(columns_file)
    feature_names = columns['data_columns']

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

# Update your prediction function to handle 242 features
def predict_price(sqft, bath, bhk, location):
    x = np.zeros(241)  # Create a vector with zeros for all features
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # Set the location feature to 1 if it's in the feature_names list
    loc_index = feature_names.index(location) if location in feature_names else -1
    if loc_index >= 3:
        x[loc_index] = 1

    return regressor1.predict([x])[0]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    sqft = float(data['sqft'])
    bath = int(data['bath'])
    bhk = int(data['bhk'])
    location = data['location']

    predicted_price = predict_price(sqft, bath, bhk, location)

    return render_template('index.html', predicted_price=predicted_price, feature_names=feature_names, data=data)


if __name__ == '__main__':
    app.run(debug=True)
