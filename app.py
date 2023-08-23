from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import math

app = Flask(__name__)

# Load the trained pipeline
loaded_pipeline = joblib.load('trained_pipeline.pkl')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            'Company': request.form['company'],
            'TypeName': request.form['typename'],
            'Ram': int(request.form['ram']),
            'Weight': float(request.form['weight']),
            'Touchscreen': int(request.form['touchscreen']),
            'Ips': int(request.form['ips']),
            'ppi': 321,
            'Cpu brand': request.form['cpubrand'],
            'HDD': int(request.form['hdd']),
            'SSD': int(request.form['ssd']),
            'Gpu brand': request.form['gpubrand'],
            'os': request.form['os']
        }

        # Convert user input to a DataFrame
        user_df = pd.DataFrame([user_input])

        # Preprocess user input using the same encoder as before
        encoded_user_input = loaded_pipeline.named_steps['step1'].transform(user_df)

        # Make predictions using the loaded pipeline
        predicted_price = loaded_pipeline.named_steps['step2'].predict(encoded_user_input)

        return render_template('result.html', predicted_price=math.floor(np.exp(predicted_price[0])))

if __name__ == '__main__':
    app.run(debug=True)
