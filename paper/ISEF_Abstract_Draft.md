# Abstract

## Background
Machine learning models frequently report strong performance in Parkinson's disease prediction; however, methodological weaknesses such as data leakage may substantially inflate apparent accuracy. The impact of leakage-aware validation on Parkinson's disease severity prediction remains insufficiently studied.

## Objective
To evaluate the reliability of voice-derived biomarkers for Parkinson's disease severity prediction and quantify the effect of validation methodology on model performance.

## Methods
Using the UCI Parkinson's Telemonitoring dataset, multiple regression models were trained to predict total Unified Parkinson's Disease Rating Scale (UPDRS) scores from voice biomarkers. Successively stricter validation protocols were implemented to identify and mitigate potential sources of data leakage, including repeated patient measurements and correlated clinical variables.

## Results
Models demonstrating strong apparent performance under naive validation experienced substantial declines following leakage-aware evaluation. These findings suggest that a significant portion of reported predictive performance may be attributable to methodological artifacts rather than generalizable disease-related signals.

## Conclusions
Rigorous validation procedures are critical for clinically meaningful biomedical machine-learning applications. Leakage-aware evaluation should be considered essential for future Parkinson's disease progression modeling studies.