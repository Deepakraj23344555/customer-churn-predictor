import streamlit as st
import pandas as pd
import plotly.express as px
from predict import predict_churn
from retention import generate_retention_email

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("🔁 Customer Churn Predictor + Retention Tool")

st.markdown("""
Upload a CSV with customer data, get churn predictions, retention emails, and visual insights.
""")

uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Sample data preview")
    st.dataframe(df.head())

    churn_probs = []
    retention_msgs = []

    for idx, row in df.iterrows():
        try:
            prob = predict_churn(row.to_dict())
            churn_probs.append(prob)
            if prob > 0.5:
                msg = generate_retention_email(row.to_dict())
            else:
                msg = "No retention action needed."
            retention_msgs.append(msg)
        except Exception as e:
            churn_probs.append(None)
            retention_msgs.append(f"Error: {e}")

    df["Churn Probability"] = churn_probs
    df["Retention Email"] = retention_msgs

    st.subheader("Prediction Results")
    st.dataframe(df)

    st.download_button(
        "Download results as CSV",
        df.to_csv(index=False),
        file_name="churn_predictions.csv"
    )

    # Dashboard visualization
    st.subheader("Churn Probability Distribution")
    fig = px.histogram(df, x="Churn Probability", nbins=20, title="Churn Probability Histogram")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn Probability by Contract Type")
    if "Contract" in df.columns:
        fig2 = px.box(df, x="Contract", y="Churn Probability", title="Churn by Contract")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Upload data with 'Contract' column for more visual insights.")
else:
    st.info("Upload a CSV file to get started.")
