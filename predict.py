import joblib
import pandas as pd

model = joblib.load("model/churn_model.pkl")

def predict_churn(input_data):
    df = pd.DataFrame([input_data])
    churn_prob = model.predict_proba(df)[0][1]
    return churn_prob
