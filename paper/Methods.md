# Methods

## Study Design

This study was designed as a retrospective biomedical machine learning validation study. The central goal was to evaluate whether Parkinson disease severity could be predicted from voice-derived biomarkers and to determine how model performance changes under progressively stricter validation strategies.

Rather than treating model accuracy as the only endpoint, the study focuses on methodological reliability. In particular, the analysis tests whether apparently strong predictive performance remains stable after accounting for repeated patient measurements, correlated clinical variables, and patient-level leakage.

## Dataset

The primary dataset is the UCI Parkinson Telemonitoring dataset. The dataset contains longitudinal voice recordings and clinical severity measures from patients with Parkinson disease. Each patient contributed repeated observations over time, creating a structure in which random train-test splitting may place measurements from the same patient in both training and testing sets.

The primary outcome variable is total UPDRS score. Secondary analyses may examine motor UPDRS separately if available.

## Feature Groups

Features were organized into the following groups:

1. Demographic and clinical variables
2. Voice-derived biomedical features
3. Combined clinical and voice biomarker feature sets

Voice features include measurements related to jitter, shimmer, harmonic-to-noise ratio, recurrence period density entropy, detrended fluctuation analysis, and pitch period entropy.

## Validation Strategy

Three validation strategies were used.

### Naive Random Split

A standard random train-test split was used as an initial baseline. This approach estimates apparent model performance but may overstate generalization because observations from the same patient can appear in both training and testing sets.

### Cross-Validation

K-fold cross-validation was used to estimate performance stability across different partitions of the dataset.

### Patient-Level Grouped Validation

Grouped validation was used to separate patients rather than individual rows. Under this approach, all observations from a given patient are assigned either to the training set or the testing set, but not both. This more closely approximates the clinical task of predicting severity for unseen patients.

## Machine Learning Models

The analysis compares interpretable and nonlinear models:

- Linear regression
- Ridge regression
- Elastic Net regression
- Random Forest regression
- Gradient boosting or XGBoost regression

Models are evaluated using mean absolute error, root mean squared error, and coefficient of determination.

## Leakage Analysis

Potential leakage sources were evaluated by comparing model performance across validation strategies. Large performance declines between random splitting and grouped validation were interpreted as evidence that random splitting may inflate apparent performance.

Additional leakage checks include:

- Removing strongly correlated target-adjacent variables
- Comparing patient-level and row-level splits
- Evaluating whether model performance is driven by patient identity rather than disease signal
- Testing feature groups separately

## Explainability Analysis

Feature importance analysis will be used to interpret model behavior. For tree-based models, SHAP values will be used to estimate the contribution of each feature to predicted UPDRS score. The goal is to determine whether high-performing models rely on biologically plausible voice biomarkers or on artifacts of the dataset structure.

## Statistical Analysis

Model performance will be reported as mean and standard deviation across validation folds. Statistical comparisons between validation strategies may use paired tests or bootstrap confidence intervals. Results will be interpreted cautiously because the dataset has a limited number of unique patients.

## External Validation Plan

To strengthen generalizability, the pipeline is designed to be extended to additional Parkinson disease datasets, including larger longitudinal cohorts such as PPMI or other approved public research datasets. External datasets will not be included unless obtained through proper data access procedures.