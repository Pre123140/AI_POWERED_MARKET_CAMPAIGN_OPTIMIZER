import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import os

def load_data(filepath):
    """Load preprocessed dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed dataset not found at {filepath}")
    return pd.read_csv(filepath)

def select_features(df):
    """Perform feature selection using Mutual Information."""
    
    # Separate features and target
    X = df.drop(columns=['deposit'])
    y = df['deposit']
    
    # Compute Mutual Information Scores
    mi_scores = mutual_info_classif(X, y, discrete_features='auto')
    
    # Store MI scores in a DataFrame
    mi_scores_df = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    
    # Select the top 5 most important features
    selected_features = mi_scores_df.nlargest(5).index.tolist()
    
    return selected_features, X, y, mi_scores_df

def save_selected_features(X, y, selected_features, output_filepath):
    """Save the dataset with selected features."""
    X_selected = X[selected_features]
    X_selected['deposit'] = y  # Re-add target variable

    X_selected.to_csv(output_filepath, index=False)
    print(f"✅ Feature selection completed. Processed data saved at {output_filepath}")

    return X_selected

def main():
    input_filepath = "data/bank_processed.csv"
    output_filepath = "data/bank_selected_features.csv"

    df = load_data(input_filepath)
    selected_features, X, y, mi_scores_df = select_features(df)

    # Display selected features
    print("\n🔹 Selected Features:")
    print(selected_features)

    save_selected_features(X, y, selected_features, output_filepath)

if __name__ == "__main__":
    main()
