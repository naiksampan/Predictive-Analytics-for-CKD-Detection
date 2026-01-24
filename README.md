# Predictive Analytics for CKD Detection

## Overview

**Predictive-Analytics-for-CKD-Detection** is a comprehensive machine learning and data analytics project focused on the **early detection of Chronic Kidney Disease (CKD)** using clinical and laboratory data. The project integrates advanced **data preprocessing, exploratory data analysis (EDA), statistical validation, feature engineering, and predictive modeling** to support **clinical decision-making** and improve diagnostic accuracy.

The primary goal is to build a robust and interpretable analytics pipeline that identifies key biomarkers, handles real-world data challenges such as missing values and outliers, and evaluates predictive performance using clinically relevant metrics.

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
├── CKD_analysis.ipynb   # Data preprocessing, EDA, and statistical analysis
├── brief.pdf            # Solution approach and methodology
├── requirements.txt     # Project dependencies
└── README.md
```

---

## Methodology

The analytical approach includes systematic **data preprocessing**, **missing value imputation**, and **clinically guided outlier treatment**. Exploratory data analysis (EDA) and **statistical hypothesis testing (t-test and Mann–Whitney U test)** are performed to identify significant biomarkers. Comorbidity analysis evaluates the impact of diabetes and hypertension on CKD progression. Feature relevance is clinically validated using **reference range comparisons and interactive visualizations**. The complete analytical workflow is detailed in **brief.pdf**, while hands-on implementation is provided in **CKD_analysis.ipynb**.

---

## Usage

- Refer to **CKD_analysis.ipynb** for full data handling, visualization, and statistical analysis.
- Review **brief.pdf** for conceptual explanation and solution design.

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
