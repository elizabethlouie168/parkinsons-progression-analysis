# Parkinson's Disease Progression Analysis: An Explainable Machine Learning Framework

## Overview

This project investigates whether longitudinal Parkinson's disease data can be used to predict future disease progression and identify the factors associated with rapid neurological decline.

Unlike traditional Parkinson's classification studies, this project focuses on progression forecasting, risk stratification, and explainable biomedical machine learning.

## Research Question

Can clinical and biomarker-derived features be used to predict which Parkinson's disease patients will experience faster progression over time?

## Scientific Motivation

Parkinson's disease affects more than 10 million people worldwide. Disease progression varies substantially between patients, making treatment planning difficult.

Accurate progression prediction could support earlier intervention, improved clinical trial design, and more personalized treatment strategies.

## Dataset

Current dataset:
- UCI Parkinson's Telemonitoring Dataset
- 5,875 observations
- 42 Parkinson's patients
- longitudinal voice measurements
- UPDRS severity scores

Future expansion:
- Parkinson's Progression Markers Initiative (PPMI)
- additional longitudinal cohorts

## Methodological Focus

This project emphasizes rigorous validation and leakage detection in biomedical machine learning.

Potential sources of inflated performance include:
- repeated measurements from the same patient
- correlated clinical variables
- temporal leakage
- improper train/test separation

## Current Findings

Initial analyses demonstrated that apparent model performance can collapse after leakage-aware validation.

This highlights the importance of reproducible and clinically realistic evaluation strategies in biomedical AI.

## Planned ISEF-Level Extensions

### 1. Progression Modeling
Predict future disease worsening rather than current disease severity.

### 2. Progression Risk Score (PPRS)
Develop a Parkinson's Progression Risk Score to stratify patients into slow, moderate, and fast progressors.

### 3. Explainable AI
Use SHAP analysis and feature attribution methods to identify biologically meaningful predictors.

### 4. Model Benchmarking
Compare:
- Linear Regression
- Random Forest
- XGBoost
- Survival Models

### 5. Longitudinal Analysis
Investigate temporal trajectories of disease progression across patient subgroups.

## Planned Outputs

- Reproducible Python pipeline
- Progression prediction framework
- Feature importance analysis
- Publication-style figures
- Scientific poster
- Conference manuscript

## Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, Jupyter

## Disclaimer

Educational biomedical machine learning research project. Not intended for clinical diagnosis or medical decision-making.