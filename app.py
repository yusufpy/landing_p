from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

def load_model():
    with open('stress_level_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
model = data

label_encoder = LabelEncoder()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    age = float(data['age'])
    heart_rate = float(data['heart_rate'])
    gender = data['gender']
    sleep_hours = float(data['sleep_hours'])

    # Convert gender to numerical value
    gender_encoded = label_encoder.fit_transform([gender])[0]

    # Make prediction
    prediction = model.predict([[age, heart_rate, gender_encoded, sleep_hours]])

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
