import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

# Define paths
data_path = "data/bank_selected_features.csv"  # Processed dataset
model_path = "models/campaign_model.pkl"  # Trained model
output_dir = "reports/figures"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load processed data
df = pd.read_csv(data_path)

# Load trained model
model = joblib.load(model_path)

# Extract features (excluding target variable 'deposit')
X = df.drop(columns=["deposit"])
y = df["deposit"]

# Ensure feature order matches training data
model_features = model.feature_names_in_  # Extract features model was trained on
X = X[model_features]  # Reorder dataset to match

# Standardize data (if model was trained with scaling)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 🔍 Feature Importance Analysis
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# 📊 Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index, palette="viridis")
plt.title("Feature Importance in Predicting Deposits")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.savefig(f"{output_dir}/feature_importance.png")
plt.show()

print("\n🔍 **Marketing Campaign Insights:**")
print("📌 Top features influencing customer conversions:")
print(feature_importance.head(5))

# 💡 SHAP Analysis for Deeper Insights
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled, check_additivity=False)  # FIXED: Pass check_additivity in shap_values()

# 📊 SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_scaled, show=False)  # `show=False` to avoid multiple plots
plt.savefig(f"{output_dir}/shap_summary_campaign.png")
plt.close()

print("✅ **Campaign Optimization Completed!** Insights saved at reports/figures.")
