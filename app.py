from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Load model, encoder, and symptoms
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('symptoms.pkl', 'rb') as f:
    symptoms = pickle.load(f)

def predict_disease(symptoms_text):
    try:
        # Lowercase and clean input
        symptoms_list = [s.strip() for s in symptoms_text.lower().split(',') if s.strip()]
        input_data = [1 if symptom in symptoms_list else 0 for symptom in symptoms]
        input_df = pd.DataFrame([input_data], columns=symptoms)

        # Make predictions
        predicted_class = model.predict(input_df)[0]

        # Try to inverse transform, and refit if needed
        try:
            predicted_disease = label_encoder.inverse_transform([predicted_class])[0]
        except ValueError as e:
             if "y contains previously unseen labels" in str(e):
                #Refit Label Encoder
                print("Detected unseen label. Refitting LabelEncoder...")
                label_encoder.fit(np.append(label_encoder.classes_, predicted_class))
                predicted_disease = label_encoder.inverse_transform([predicted_class])[0]

        probabilities = {}
        if predicted_disease != "Other":
            prediction_prob = model.predict_proba(input_df)[0] * 100
            for i, prob in enumerate(prediction_prob):
                disease_name = label_encoder.inverse_transform([i])[0]
                if disease_name != "Other":  # Exclude "Other" from probabilities
                    probabilities[disease_name] = prob

        if predicted_disease == "Other":
            return "Rare Disease (Not in Main Dataset)", {}

        return predicted_disease, probabilities

    except Exception as e:
      print("error" ,e)
      return "Error","{}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms_text = data['symptoms']
    predicted_disease, probabilities = predict_disease(symptoms_text)

    return jsonify({
        'prediction': predicted_disease,
        'probabilities': probabilities
    })

if __name__ == '__main__':
    app.run(debug=True)
