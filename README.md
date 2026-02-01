# Predictive Analytics for CKD Detection

## Overview

This project presents an end-to-end Predictive Analytics framework for Chronic Kidney Disease (CKD) detection using clinical and laboratory data. It covers robust data handling, exploratory data analysis, clinically informed feature engineering, and advanced machine learning modeling. Multiple algorithms were evaluated using rigorous validation strategies, with ensemble models demonstrating superior performance. The framework emphasizes clinical interpretability through statistical analysis and SHAP-based explanations, enabling transparent decision support. An interactive visualization and deployment pipeline built with Streamlit supports real-time inference and risk stratification. Overall, the system is designed to be accurate, explainable, and suitable for real-world clinical decision-support applications.

---

## Key Features

- End-to-end CKD predictive analytics pipeline  
- Advanced missing data handling and outlier treatment  
- Statistical validation using hypothesis testing  
- Biomarker analysis and clinical interpretation  
- Interactive visualizations using Plotly  
- Model-ready data processing workflow  

---

## Environment Setup

### Step 1: Create Conda Environment

```bash
conda create -n CKD_Env python=3.8 
```

### Step 2: Activate Environment

```bash
conda activate CKD_Env
```

### Step 3: Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── CKD_analysis.ipynb                # Data preprocessing, EDA, and statistical analysis
├── model_evaluation_validation.ipynb # Modeling, Evalaution and Prediction
├── Future_scope.ipynb                # Hybrid Unsuperived Learning with RUL analysis
├── app.py                            # Streamlit application
├── brief.pdf                         # Solution approach and methodology
├── requirements.txt                  # Project dependencies
└── README.md
```

---

## Methodology

The proposed methodology follows a structured and clinically driven machine learning pipeline for Chronic Kidney Disease (CKD) prediction. Initially, raw clinical and laboratory data were explored to understand feature distributions, missingness patterns, and data quality issues. A hybrid imputation strategy was applied, combining statistical and multivariate techniques to handle missing values while preserving clinical relationships. Outliers were identified using robust statistical methods and clinically defined thresholds, followed by winsorization to maintain pathological relevance.

Comprehensive exploratory data analysis and statistical hypothesis testing were conducted to identify significant biomarkers associated with CKD. Clinically meaningful features were then engineered, including ratio-based indicators, interaction terms, abnormality flags, and aggregated severity scores. The processed dataset was used to train multiple classification models, with hyperparameters optimized via nested cross-validation to prevent data leakage.

Model performance was evaluated using classification, probabilistic, and clinical metrics. Interpretability was ensured through feature importance analysis and SHAP-based explanations. Finally, the optimized model was deployed using a consistent and reproducible inference pipeline.

---

## Usage

- Refer to **CKD_analysis.ipynb** for complete data handling, exploratory data analysis, visualization, and statistical validation.
- Review **model_evaluation_validation.ipynb** for model training, performance evaluation, and prediction workflows.
- Consult **brief.pdf** for the conceptual explanation, methodology, and overall solution design.
- Explore the deployed application here:  
  **Website:** https://predictive-analytics-for-ckd-detection-je9vzk6jyeqgow54w5cwdh.streamlit.app

---

## Applications

- Clinical decision support  
- Early-stage CKD screening  
- Medical data analytics  
- Healthcare predictive modeling  

---

## Author

**Sampan Sanjay Naik**

---
