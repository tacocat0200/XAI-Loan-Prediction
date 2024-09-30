from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import shap
import lime
import lime.lime_tabular
import json
import os

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('loan_approval_model.pkl')

# Define feature names and load training data statistics for preprocessing
feature_names = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term',
                 'Credit_History', 'Property_Area_Semi-Urban', 'Property_Area_Rural',
                 'Education_Level_Graduate', 'Education_Level_Not Graduate',
                 'Employment_Status_Employed', 'Employment_Status_Self-employed']

# Load mean and std for scaling (replace with your actual values)
mean = {
    'Applicant_Income': 5000,
    'Coapplicant_Income': 1500,
    'Loan_Amount': 150,
    'Loan_Amount_Term': 360
}
std = {
    'Applicant_Income': 2000,
    'Coapplicant_Income': 500,
    'Loan_Amount': 50,
    'Loan_Amount_Term': 60
}

# Initialize SHAP explainer
explainer_shap = shap.Explainer(model, feature_names=feature_names)

# Initialize LIME explainer
# For LIME, you need the training data; assuming you have X_train loaded
X_train = pd.read_csv('loan_data.csv')  # Replace with actual training data
X_train_processed = preprocess_input(X_train)
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_processed.values,
    feature_names=feature_names,
    class_names=['Not Approved', 'Approved'],
    mode='classification'
)

def preprocess_input(input_data):
    # Example preprocessing steps:
    
    # Convert categorical variables using the same encoding as training
    categorical_features = ['Property_Area', 'Education_Level', 'Employment_Status']
    input_data = pd.get_dummies(input_data, columns=categorical_features)
    
    # Ensure all expected dummy columns are present
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0
    
    # Reorder columns to match training data
    input_data = input_data[feature_names]
    
    # Fill missing values if any
    input_data.fillna(0, inplace=True)
    
    # Scale numerical features
    numerical_features = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term']
    for feature in numerical_features:
        input_data[feature] = (input_data[feature] - mean[feature]) / std[feature]
    
    return input_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        data = request.form.to_dict()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess input
        input_processed = preprocess_input(input_df)
        
        # Make prediction
        prediction = model.predict(input_processed)[0]
        prediction_proba = model.predict_proba(input_processed)[0][1]  # Probability of approval
        
        # Generate SHAP explanation
        shap_values = explainer_shap(input_processed)
        shap_explanation = shap_values.values.tolist()
        
        # Generate LIME explanation
        lime_exp = explainer_lime.explain_instance(
            input_processed.iloc[0].values,
            model.predict_proba
        )
        lime_explanation = lime_exp.as_list()
        
        # Prepare response
        response = {
            'prediction': 'Approved' if prediction == 1 else 'Not Approved',
            'probability': round(prediction_proba * 100, 2),
            'shap_explanation': shap_explanation,
            'lime_explanation': lime_explanation
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
