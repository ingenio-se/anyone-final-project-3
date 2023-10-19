from flask import Flask, render_template, request
import pandas as pd
import joblib
from xgboost import XGBClassifier

app = Flask(__name__)

# Load the XGBoost model from a binary file
loaded_model = XGBClassifier()
model_filename = 'models/xgb_model.bin'
loaded_model.load_model(model_filename)


@app.route('/')
def index():
    """
    Render the index.html template when the root URL is accessed.
    """
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def predict():
    """
    Handle the POST request to make a prediction based on form data.
    """
    # Convert form data to a dictionary
    data = request.form.to_dict()

    # Prepare the data and make a prediction with the model
    result, value = prepare_data(data)

    return render_template('result.html', result=result, value=value)

def prepare_data(data):
    """
    Prepare the input data for prediction and return the result and probability.
    Args:
        data (dict): A dictionary containing input data from the form.

    Returns:
        str: The prediction result (0 or 1).
        float: The probability of the positive class in percentage.
    """
    # Verify if 'lift' and 'pusha' fields are missing and set them to 0
    data.setdefault('lift', 0)
    data.setdefault('pusha', 0)

    # Create a DataFrame from the input data
    df = pd.DataFrame([data])

    # Calculate BMI index
    bmi = float(df['weight']) / float(df['height']) ** 2
    df['bmi'] = bmi

    # Columns for adltot6 feature
    adltot6_columns = ['bath', 'dress', 'eat', 'bed', 'walk', 'toilet']

    # Calculate the adltot6 feature based on available columns
    adltot6 = sum(1 for col in adltot6_columns if col in df.columns)
    df['adltot6'] = adltot6

    # Calculate the doctor1y feature
    df['doctor1y'] = 1 if int(df['doctim1y']) > 0 else 0

    # Columns for iadlfour feature
    iadlfour_columns = ['money', 'meds', 'shop', 'meals']

    # Calculate the iadlfour feature based on available columns
    iadlfour = sum(1 for col in iadlfour_columns if col in df.columns)
    df['iadlfour'] = iadlfour

    # Select the final columns to be used for prediction
    df_predict = df[["adltot6", "bmi", "cholst", "decsib", "dentim1y", "doctim1y", "doctor1y", "fallnum", "hltc",
                     "iadlfour", "lift", "momage", "oopden1y", "oopdoc1y", "oopmd1y", "pusha", "sight", "weight"]]

    # Convert all columns to the appropriate data types
    for column in df_predict:
        if column == 'bmi':
            df_predict[column] = df_predict[column].astype(float)
        else:
            df_predict[column] = df_predict[column].astype(int)

    # Use the loaded model to make a prediction
    prediction = loaded_model.predict(df_predict)
    print(prediction)
    
    predict_proba = loaded_model.predict_proba(df_predict)
    print(predict_proba)

    # Convert prediction to a string and calculate the probability in percentage
    result = str(prediction[0])
    proba = predict_proba[0]
    value = round(proba[1] * 100, 1)

    return result, value

if __name__ == '__main__':
    # Run the Flask app on all available network interfaces
    app.run(host="0.0.0.0", debug=True,port=5001)
