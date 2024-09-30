import shap
import pandas as pd
import pickle

# Load the trained loan approval model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Prepare the data for SHAP analysis
def prepare_data(input_data):
    # Assuming input_data is a DataFrame or needs to be converted to one
    return pd.DataFrame(input_data)

# Calculate SHAP values for the predictions
def calculate_shap_values(model, input_data):
    # Prepare the input data
    X = prepare_data(input_data)
    
    # Create a SHAP explainer based on the model
    explainer = shap.Explainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(X)
    return shap_values

# Visualize SHAP values
def visualize_shap_values(shap_values):
    # Summary plot
    shap.summary_plot(shap_values, plot_type="bar")

    # Dependence plot for a specific feature
    # Here 'income' can be replaced with any feature of interest
    shap.dependence_plot("income", shap_values.values, feature_names=["loan_amount", "income", "credit_score", "employment_status"])

# Main function to execute SHAP analysis
if __name__ == "__main__":
    model_path = 'loan_approval_model.pkl'  # Path to the trained model
    input_data = {
        "loan_amount": [5000],
        "income": [60000],
        "credit_score": [700],
        "employment_status": ["employed"]
    }
    
    model = load_model(model_path)
    shap_values = calculate_shap_values(model, input_data)
    visualize_shap_values(shap_values)
