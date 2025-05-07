# AI-Powered Marketing Campaign Optimizer

This project applies machine learning, explainability, and statistical testing to optimize marketing campaigns. Using historical banking campaign data, we predict deposit conversions, analyze top drivers with SHAP, and run simulated A/B/C tests for campaign strategy selection.

---

## Project Objective

To develop an end-to-end AI solution for optimizing marketing strategies by:
- Predicting customer responses to marketing campaigns.
- Interpreting top conversion drivers with SHAP.
- Running statistical tests to compare campaign versions (A/B/C).

---

## Features

- Predictive model for deposit conversion
- Mutual Information-based feature selection
- SHAP values for model transparency
- Automated pipeline for preprocessing, training, and testing
- A/B/C testing simulation using Chi-Square statistics
- Visualization suite for strategic decision-making

---

## Conceptual Study
Want to understand the full methodology and thinking behind this project?
[Read the Full Conceptual Study →](https://github.com/Pre123140/AI_Marketing_Campaign_Optimizer/blob/main/AI_Marketing_Campaign_Optimizer.pdf)

Includes:
- Supervised learning workflow
- SHAP theory and explainability rationale
- Statistical testing (Chi-Square method)
- Campaign optimization principles
- Strategic recommendations

---

## Tech Stack

- pandas – Data loading and transformation
- numpy – Numeric operations
- scikit-learn – ML modeling and feature selection
- SHAP – Explainability
- seaborn, matplotlib – Data visualizations
- joblib – Model saving and loading
- scipy – Statistical A/B testing
- streamlit (optional) – For future UI integration

---

## Folder Structure
```
AI_Marketing_Campaign_Optimizer/
├── data/
│   ├── bank.csv
│   ├── bank_processed.csv
│   ├── bank_selected_features.csv
│
├── models/
│   ├── campaign_model.pkl
│   ├── campaign_model_pipeline.pkl
│
├── reports/
│   └── figures/
│       ├── ab_test_results.png
│       ├── feature_importance.png
│       ├── shap_summary_campaign.png
│       └── shap_summary_plot.png
│
├── src/
│   ├── ab_testing.py
│   ├── automated_pipeline.py
│   ├── campaign_optimizer.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── model_tuning.py
│   └── shap_explain.py
│
└── README.md
```

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/Pre123140/AI_Marketing_Campaign_Optimizer.git
cd AI_Marketing_Campaign_Optimizer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline
```bash
python src/automated_pipeline.py
```

### 4. (Optional) Run Individual Scripts
```bash
python src/data_preprocessing.py
python src/model_training.py
python src/shap_explain.py
```

---

## Visual Outputs

- Feature Importance Plot
- SHAP Summary Plot
- A/B/C Conversion Testing Charts
- Customer Segment Visuals
- Confusion Matrix & Classification Report

---

## Deliverables

- Processed and cleaned marketing data
- Feature selection using mutual_info_classif
- Random Forest classifier
- SHAP-based insight generation
- Statistical testing outputs (Chi-Square)
- Visual summaries of all results

---

## Next Steps (Future Enhancements)

- Integrate dashboard for real-time campaign monitoring
- Apply uplift modeling for causal impact evaluation
- Extend to multi-channel campaigns (email, call, in-app)
- Add CRM or MarTech integration
- Implement RL-based personalization

---

## License

This project is open for educational use only. For commercial deployment, contact the author.

---

##  Contact
If you'd like to learn more or collaborate on projects or other initiatives, feel free to connect on [LinkedIn](https://www.linkedin.com/in/prerna-burande-99678a1bb/) or check out my [portfolio site](https://youtheleader.com/).

