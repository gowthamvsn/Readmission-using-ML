# Hospital Readmission Prediction using Machine Learning

This project predicts whether a diabetic patient will be readmitted to the hospital
after discharge using machine learning models.

## Problem Statement
Hospital readmissions are costly and often preventable.  
The goal of this project is to identify patients at high risk of readmission
so hospitals can improve discharge planning and follow-up care.

## Dataset
- Diabetes Readmission Dataset (UCI / Kaggle)
- ~100,000 patient encounters
- Target: Any readmission (Yes / No)
Dataset available from UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

## Approach
- Data cleaning and preprocessing
- Feature engineering from clinical records
- Training and comparing multiple ML models
- Model evaluation using ROC-AUC and Recall

## Models Used
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine

## Results
XGBoost achieved the best overall performance with strong recall
for identifying patients who were readmitted.

## Files
- `model_experiments.py` – Main training and evaluation pipeline
- `interactive_analysis.py` – Interactive and exploratory analysis utilities
- `Complete Final Code.pdf` – Final project report

## How to Run
1. Clone the repository  
2. Install required libraries  
3. Run the Jupyter notebook or Python scripts

## Contributors
- Gowtham
- Hye Eunkg
