# AI-Powered Marketing Campaign Optimizer

This project applies machine learning, explainability, and statistical testing to optimize marketing campaigns. Using historical banking campaign data, we predict deposit conversions, analyze top drivers with SHAP, and run simulated A/B/C tests for campaign strategy selection.

---

## 🚀 Project Highlights

- Predict customer response to a term deposit campaign using ML
- Identify high-impact features through Mutual Information & SHAP
- Segment customers and simulate A/B/C campaign testing
- Generate interactive plots and visual explanations
- Run entire pipeline via a single automation script

---

## 🧠 Use Case

Businesses often run multiple marketing campaigns without knowing which strategy resonates best with which segment. This project solves that by using AI to:

- Forecast conversion likelihood
- Interpret why customers said “yes”
- Statistically validate best-performing strategies

---

## 📁 Folder Structure

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
│   ├── __pycache__/
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
├── README.md
└── requirements.txt








---

## 📊 Visual Outputs

✅ Feature Importance Bar Plot  
✅ SHAP Summary Plot (model interpretability)  
✅ A/B/C Conversion Rate Bar Plot  
✅ SHAP Insights for Campaign Impact  
✅ Customer Segment Plot  
✅ Campaign Testing Summary Charts  
✅ Confusion Matrix & Classification Report  

---

## 📦 Key Deliverables

✅ Cleaned and encoded datasets  
✅ Feature selection using Mutual Information  
✅ Random Forest model for conversion prediction  
✅ SHAP explanations for model transparency  
✅ Automated pipeline to orchestrate all steps  
✅ A/B/C campaign testing with statistical validation  
✅ Insightful visual reports for decision-makers

---

## 🧰 Tools and Libraries Used

- **Pandas** – Data loading and manipulation  
- **NumPy** – Numerical operations  
- **Scikit-learn** – ML models, feature selection, and evaluation  
- **SHAP** – Explainable AI (SHAP values for model interpretation)  
- **Matplotlib & Seaborn** – Visualizations  
- **Joblib** – Model serialization  
- **Scipy** – Chi-Square statistical testing  
- **Streamlit (optional)** – For dashboard UI extension

---

## 📌 Conceptual Study

For a deeper dive into the algorithms, concepts, and strategic thinking behind this project, refer to the accompanying [**Conceptual Study PDF**](./conceptual_study.pdf).

It covers:
- Supervised learning principles
- SHAP theory for explainability
- A/B testing foundations (Chi-Square test)
- Business relevance of AI-driven marketing
- Strategic reflections and next steps

---

## 📈 Performance Summary

- **Model Used:** Random Forest Classifier  
- **Evaluation Metrics:** Accuracy, Confusion Matrix, Classification Report  
- **Explainability:** SHAP summary plot and feature breakdown  
- **Testing:** Statistically validated A/B/C testing using Chi-Square

---

## ✅ Automation Pipeline

This project includes an `automated_pipeline.py` script that runs the entire end-to-end process:
- Data Preprocessing  
- Feature Engineering  
- Model Training & Evaluation  
- SHAP-based Explanation  
- A/B Testing for Campaign Validation

This ensures reproducibility, modularity, and scalability.

---

## 📌 Next Steps (Optional Extensions)

- Integrate real-time customer scoring with CRM tools  
- Deploy dashboard for campaign managers  
- Apply uplift modeling for causal impact estimation  
- Add personalization layer using reinforcement learning

---

**Author:** *Prerna Burande*  
**License:** For educational and portfolio use only.  
Commercial use or adaptation without permission is prohibited.


