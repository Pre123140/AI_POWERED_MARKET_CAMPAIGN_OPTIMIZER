import os
import subprocess

# Ensure necessary directories exist
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Define script execution order
pipeline_steps = [
    "python src/data_preprocessing.py",
    "python src/feature_engineering.py",
    "python src/model_training.py",
    "python src/model_evaluation.py",
    "python src/campaign_optimizer.py",
    "python src/ab_testing.py"
]

# Execute scripts sequentially
for step in pipeline_steps:
    print(f"\n🚀 Running: {step}")
    result = subprocess.run(step, shell=True, capture_output=True, text=True)

    # Print output & errors (if any)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ Error in {step}: {result.stderr}")

print("\n✅ **Automated Pipeline Execution Completed Successfully!** 🚀")
