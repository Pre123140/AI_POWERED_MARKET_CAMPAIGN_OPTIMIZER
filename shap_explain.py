import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt

def load_model(model_filepath):
    """Load the trained model from file."""
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Model not found: {model_filepath}")
    return joblib.load(model_filepath)

def load_data(filepath):
    """Load dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    return pd.read_csv(filepath)

def shap_explain(model_filepath, data_filepath, output_dir):
    """Perform SHAP analysis and save results."""
    # Load model and dataset
    model = load_model(model_filepath)
    data = load_data(data_filepath)

    # Ensure the correct target column is removed
    X = data.drop(columns=['deposit'], errors='ignore')

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "shap_summary_plot.png")

    # Generate SHAP summary plot
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(output_filepath)  # Save SHAP plot
    plt.close()

    print(f"✅ SHAP analysis completed. Summary plot saved at {output_filepath}")

def main():
    model_filepath = "models/campaign_model.pkl"
    data_filepath = "data/bank_selected_features.csv"
    output_dir = "reports/figures"

    shap_explain(model_filepath, data_filepath, output_dir)

if __name__ == "__main__":
    main()
