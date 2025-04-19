import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data(filepath):
    """Load dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """Handle missing values and clean the dataset."""
    df.dropna(inplace=True)  # Remove rows with missing values
    return df

def encode_categorical_features(df, categorical_cols):
    """Encode categorical features using Label Encoding."""
    encoder_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoder_dict[col] = le  # Store encoders for later use
    return df, encoder_dict

def scale_features(df, numerical_cols):
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

def preprocess_data(input_filepath, output_filepath):
    """Full data preprocessing pipeline: clean, encode, and scale features."""
    df = load_data(input_filepath)
    df = clean_data(df)
    
    # Identify categorical and numerical columns
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    
    df, encoders = encode_categorical_features(df, categorical_cols)
    df, scaler = scale_features(df, numerical_cols)

    # Save processed dataset
    df.to_csv(output_filepath, index=False)
    
    print(f"✅ Data preprocessing completed. Processed data saved at {output_filepath}")

    return df, encoders, scaler

if __name__ == "__main__":
    input_filepath = "data/bank.csv"
    output_filepath = "data/bank_processed.csv"

    preprocess_data(input_filepath, output_filepath)
