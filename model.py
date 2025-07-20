import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load data
df = pd.read_csv("data/customer_data.csv")
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

X = df.drop(columns=["Churn"])
y = df["Churn"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

clf.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/churn_model.pkl")
print("✅ Model saved to model/churn_model.pkl")
