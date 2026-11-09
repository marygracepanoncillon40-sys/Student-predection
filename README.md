# Student Performance Prediction System

A classroom prototype that estimates whether a secondary-school student is at risk of failing final Mathematics.

## Problem
Academic risk is often noticed only after poor results. This prototype explores early identification using available student indicators (study time, absences, past failures, G1, G2, etc.).

## Dataset
UCI Machine Learning Repository, *Student Performance* (ID 320), Mathematics file (`student-mat.csv`):
https://archive.ics.uci.edu/dataset/320/student+performance

## Prediction target
`G3 < 10` = **AT RISK**; `G3 >= 10` = **NOT AT RISK**. G3 is the target only and is never used as an input feature.

## Technologies
Python, pandas, scikit-learn (Random Forest), joblib, Streamlit, GitHub.

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py        # downloads data, trains, saves student_model.pkl
streamlit run app.py         # opens http://localhost:8501
```
Use "Download sample input CSV" in the app, then upload it to test predictions.

## Limitations
- Trained on one Portuguese secondary-school dataset; may not generalize to other institutions.
- G1 and G2 (earlier period grades) are strong predictors, so accuracy is high but the model is less useful for *early* warning.
- Small dataset (~395 records); evaluate precision/recall, not accuracy alone.
- Predictions are not diagnoses and must not be the sole basis for real student decisions.
