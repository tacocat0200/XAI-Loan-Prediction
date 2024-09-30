import lime
import lime.lime_tabular
import pandas as pd
import joblib
import json
import os

def load_model(model_path='loan_approval_model.pkl'):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        raise FileNotFoundError(f"Model file {model_path} not found.")

def initialize_lime_explainer(training_data, feature_names, categorical_features, class_names=['Not Approved', 'Approved']):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data.values,
        feature_names=feature_names,
        categorical_features=categorical_features,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True
    )
    return explainer

def preprocess_input(input_data, preprocessor):
    """
    Preprocess the input data using the same preprocessing pipeline as the model.
    """
    processed_data = preprocessor.transform(input_data)
    return processed_data

def generate_lime_explanation(explainer, model, input_data, num_features=5):
    """
    Generate a LIME explanation for a single prediction.
    
    Parameters:
    - explainer: The initialized LIME explainer.
    - model: The trained machine learning model.
    - input_data: A single data instance (pandas DataFrame or Series).
    - num_features: Number of features to include in the explanation.
    
    Returns:
    - explanation: A list of tuples with feature names and their contribution to the prediction.
    """
    # Convert input_data to numpy array if it's a DataFrame
    if isinstance(input_data, pd.DataFrame):
        input_instance = input_data.iloc[0].values
    elif isinstance(input_data, pd.Series):
        input_instance = input_data.values
    else:
        input_instance = input_data
    
    # Generate explanation
    explanation = explainer.explain_instance(
        data_row=input_instance,
        predict_fn=model.predict_proba,
        num_features=num_features
    )
    
    # Format the explanation as a list of tuples
    explanation_list = explanation.as_list()
    
    return explanation_list

def format_lime_explanation(explanation_list):
    """
    Format the LIME explanation list into a dictionary for easier handling.
    
    Parameters:
    - explanation_list: List of tuples containing feature contributions.
    
    Returns:
    - explanation_dict: Dictionary with feature names as keys and contributions as values.
    """
    explanation_dict = {feature: contribution for feature, contribution in explanation_list}
    return explanation_dict

def save_explanation(explanation, file_path):
    """
    Save the explanation to a JSON file.
    
    Parameters:
    - explanation: The explanation to save.
    - file_path: Path to the JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(explanation, f, indent=4)

def main():
    # Load the pre-trained model
    model = load_model('loan_approval_model.pkl')
    print("Model loaded successfully.")
    
    # Extract preprocessor from the model pipeline
    preprocessor = model.named_steps['preprocessor']
    
    # Load training data for LIME explainer
    training_data = pd.read_csv('loan_data.csv')[['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term',
                                                'Credit_History', 'Property_Area', 'Education_Level', 'Employment_Status']]
    
    # Define feature names and categorical feature indices
    feature_names = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term',
                     'Credit_History', 'Property_Area', 'Education_Level', 'Employment_Status']
    # Assuming zero-based indices for categorical features
    categorical_features = [5, 6, 7]
    
    # Initialize LIME explainer
    explainer = initialize_lime_explainer(training_data, feature_names, categorical_features)
    print("LIME explainer initialized.")
    
    # Example input data for explanation
    example_input = pd.DataFrame({
        'Applicant_Income': [5000],
        'Coapplicant_Income': [0],
        'Loan_Amount': [200],
        'Loan_Amount_Term': [360],
        'Credit_History': [1],
        'Property_Area': ['Urban'],
        'Education_Level': ['Graduate'],
        'Employment_Status': ['Employed']
    })
    
    # Preprocess the input data
    input_processed = preprocess_input(example_input, preprocessor)
    print("Input data preprocessed.")
    
    # Generate LIME explanation
    lime_explanation = generate_lime_explanation(explainer, model, example_input, num_features=5)
    print("LIME explanation generated.")
    
    # Format the explanation
    formatted_explanation = format_lime_explanation(lime_explanation)
    print("LIME explanation formatted.")
    
    # Save the explanation
    save_explanation(formatted_explanation, 'lime_explanation.json')
    print("LIME explanation saved to lime_explanation.json.")
    
    # Print the explanation
    print("\nLIME Explanation:")
    for feature, contribution in formatted_explanation.items():
        print(f"{feature}: {contribution}")

if __name__ == '__main__':
    main()
