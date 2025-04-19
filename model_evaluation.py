import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    return pd.read_csv(filepath)

def load_model(filepath):
    """Load trained model from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    return joblib.load(filepath)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print performance metrics."""
    y_pred = model.predict(X_test)

    # Print classification report & confusion matrix
    print("\n🔍 **Model Evaluation Metrics**")
    print(classification_report(y_test, y_pred))
    print("\n🟩 **Confusion Matrix:**")
    print(confusion_matrix(y_test, y_pred))

def main():
    input_filepath = "data/bank_selected_features.csv"
    model_filepath = "models/campaign_model.pkl"

    # Load data
    df = load_data(input_filepath)
    
    # Split features & target
    X = df.drop(columns=['deposit'])
    y = df['deposit']

    # Train-test split (80% train, 20% test)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load trained model
    model = load_model(model_filepath)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
