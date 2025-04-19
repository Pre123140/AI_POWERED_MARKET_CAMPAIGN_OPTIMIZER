import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(filepath):
    """Load dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    return pd.read_csv(filepath)

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """Save trained model."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✅ Model training completed. Model saved at {filepath}")

def main():
    input_filepath = "data/bank_selected_features.csv"
    model_filepath = "models/campaign_model.pkl"

    # Load data
    df = load_data(input_filepath)
    
    # Split features & target
    X = df.drop(columns=['deposit'])
    y = df['deposit']

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🔹 Model Accuracy: {accuracy:.4f}")

    # Save the trained model
    save_model(model, model_filepath)

if __name__ == "__main__":
    main()

