import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# Load the processed dataset
data_path = "data/bank_processed.csv"
df = pd.read_csv(data_path)

# Ensure reports/figures directory exists
os.makedirs("reports/figures", exist_ok=True)

# Segment customers based on balance and previous campaign interactions
def segment_customers(df):
    df["customer_segment"] = "General"
    df.loc[df["balance"] >= df["balance"].median(), "customer_segment"] = "High Balance"
    df.loc[df["previous"] > 0, "customer_segment"] = "Returning Customer"
    return df

df = segment_customers(df)

# Assign campaigns strategically
def assign_campaigns(df):
    df["campaign_group"] = "C"  # Default to Campaign C
    df.loc[df["customer_segment"] == "High Balance", "campaign_group"] = "A"
    df.loc[df["customer_segment"] == "General", "campaign_group"] = "B"
    return df

df = assign_campaigns(df)

# Compute conversion rates
conversion_rates = df.groupby("campaign_group")["deposit"].value_counts(normalize=True).unstack()
print("\n🔍 **A/B/C Testing Results**")
print("📊 Conversion Rates:")
print(conversion_rates)

# Perform Chi-Square Test
contingency_table = df.pivot_table(index="campaign_group", columns="deposit", aggfunc="size", fill_value=0)
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\n📈 Chi-Square Test: Chi2 = {chi2:.4f}, p-value = {p:.4f}")
if p < 0.05:
    print("✅ Statistically significant difference found between campaigns!")
else:
    print("⚠️ No significant difference between campaigns.")

# Plot conversion rates
plt.figure(figsize=(8, 5))
sns.barplot(x=conversion_rates.index, y=conversion_rates["yes"], palette="viridis")
plt.xlabel("Campaign Group")
plt.ylabel("Deposit Conversion Rate")
plt.title("A/B/C Testing Results - Conversion Rates")
plt.savefig("reports/figures/ab_test_results.png")
plt.show()

print("📊 A/B/C testing completed! Results saved at 'reports/figures/ab_test_results.png'.")
