# 🏦 LoanSense — Intelligent Loan Approval Predictor

> An end-to-end Machine Learning project that predicts loan approval
> decisions using 19 applicant features — with **Precision** as the
> key optimization metric to minimize costly false approvals.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Best Model](https://img.shields.io/badge/Best%20Model-Naive%20Bayes-green)
![Accuracy](https://img.shields.io/badge/Accuracy-82%25-brightgreen)
![Precision](https://img.shields.io/badge/Precision-80.35%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-1000%20Records-orange)
![Models Tested](https://img.shields.io/badge/Models%20Tested-3-purple)

---

## 📌 Problem Statement

Banks process thousands of loan applications daily. A false approval
(approving a bad loan) is far more costly than a false rejection.

This project automates loan approval decisions using ML — prioritizing
**Precision** over raw accuracy to minimize financial risk from
false approvals.

---

## 📊 Dataset Overview

| Property | Detail |
|----------|--------|
| Total Records | 1,000 applicants |
| Input Features | 19 features |
| Target Variable | Loan_Approved (Yes / No) |
| Class Distribution | 68.7% Rejected · 31.37% Approved |
| Missing Values | 50 records — handled via imputation |

### 🔑 Key Features

| Feature | Insight |
|---------|---------|
| Credit_Score | Avg: 676 · Strongest predictor of approval |
| DTI_Ratio | Avg: 0.35 · High DTI = lower approval chance |
| Applicant_Income | Avg: $10,852 |
| Loan_Amount | Avg: $20,522 requested |
| Employment_Status | Contract workers had highest approval (37%) |
| Education_Level | Graduates approved more (31.1% vs 25.9%) |

---

## 🔍 Key EDA Findings

- 📉 **Imbalanced dataset** — only 31.37% approvals (298 out of 950)
- 💳 **Credit Score ≥ 650** was a strong threshold for loan approval
- 💼 **Contract employees** had the highest approval rate at 37.09%
- 🎓 **Graduates** were approved 5.2% more than non-graduates
- 📊 **DTI Ratio & Credit Score** were the top correlated features
  with loan approval (confirmed via correlation heatmap)

---

## 🔄 Complete ML Pipeline
```
Raw CSV (1,000 records, 20 columns)
        ↓
Missing Value Imputation
  → Categorical: Most Frequent Strategy
  → Numerical: Mean Strategy
        ↓
Exploratory Data Analysis (EDA)
  → Class distribution · Income analysis
  → Outlier detection (Boxplots)
  → Credit Score vs Approval histogram
        ↓
Feature Encoding
  → Label Encoding (Education_Level, Loan_Approved)
  → One-Hot Encoding (Employment, Marital, Purpose,
    Area, Gender, Employer — drop='first')
        ↓
Feature Engineering
  → DTI_Ratio² (squared for non-linear patterns)
  → Credit_Score² (squared for non-linear patterns)
        ↓
Train/Test Split (80/20 · random_state=42)
        ↓
Feature Scaling (StandardScaler)
        ↓
Model Comparison (3 algorithms)
        ↓
Best Model: Naive Bayes ✅
```

---

## 🤖 Model Comparison

| Model | Key Metric (Precision) | Notes |
|-------|----------------------|-------|
| Logistic Regression | Tested | Baseline model |
| KNN (k=7) | Tested | Weak on high-dim sparse data |
| **Naive Bayes** ✅ | **80.35%** | **Best Precision — selected** |

> 💡 KNN was expected to underperform — with a large number of
> encoded columns, data becomes sparse, reducing KNN's effectiveness.
> This was noted in the code comments before testing.

---

## 📈 Final Model Performance (Naive Bayes)

| Metric | Score |
|--------|-------|
| **Precision** | **80.35%** |
| **Accuracy** | **86.5%** |
| Train/Test Split | 80% / 20% |
| Test Set Size | 200 records |

> ⚡ **Why Precision?** A False Positive (approving a bad loan) 
> is more costly than a False Negative (rejecting a good one).
> Precision directly measures how many predicted approvals
> were actually correct.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| NumPy | Numerical computing & feature engineering |
| Pandas | Data loading, cleaning & EDA |
| Scikit-learn | Imputation, encoding, scaling, modelling |
| Matplotlib | Pie charts, histograms, subplots |
| Seaborn | Boxplots, heatmaps, bar charts |
| JupyterLab | Development environment |

---

LoanSense/
│
├── 📓 LoanSense.ipynb              # Main Jupyter notebook
├── 📂 data/
│   └── loan_approval_data.csv      # Raw dataset
├── requirements.txt                # Dependencies
└── README.md                       # Documentation

## 👨‍💻 Author

**Vaibhav** — ECE Undergrad @ NIT Raipur
🎯 Aspiring AI/ML Engineer | Competitive Programmer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/vaibhav1o1/)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:techvaibhav27@gmail.com)
