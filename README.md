# Parkinson’s Disease Progression Analysis: A Biomedical Machine Learning Validation Study

## Overview

This project explores whether biomedical voice biomarkers can predict Parkinson’s disease symptom severity using the UCI Parkinson’s Telemonitoring dataset.

Rather than simply optimizing model performance, this analysis focuses on methodological rigor in biomedical machine learning, specifically identifying and eliminating data leakage sources that can artificially inflate predictive performance.

## Research Question

Can voice-derived biomarkers reliably predict Parkinson’s disease symptom severity, and how does evaluation methodology affect apparent model performance?

## Dataset

Source: UCI Parkinson’s Telemonitoring Dataset

- 5,875 observations
- 42 Parkinson’s disease patients
- repeated longitudinal measurements
- biomedical voice features including jitter, shimmer, RPDE, DFA, PPE, and HNR

Target variable:
- total_UPDRS (Unified Parkinson’s Disease Rating Scale)

## Methodology

Initial naive models showed near-perfect performance due to data leakage.
Leakage sources identified included correlated clinical variables, repeated patient measurements, and temporal structure.

Revised results:
- demographics + voice biomarkers: MAE 4.84, R² 0.595
- voice biomarkers only: MAE 7.57, R² 0.200
- patient-level grouped validation: MAE 12.06, R² -1.34

## Key Finding

Apparent biomedical ML performance can collapse under rigorous validation, highlighting the importance of leakage-aware evaluation.

## Technologies

Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter

## Future Improvements

- Group cross-validation
- SHAP interpretability
- XGBoost benchmarking
- additional Parkinson’s datasets
- longitudinal progression modeling

## Disclaimer

Educational biomedical ML validation case study only. Not a clinical diagnostic tool.
