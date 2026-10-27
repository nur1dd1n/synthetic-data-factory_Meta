import pandas as pd
import streamlit as st

from app.services.profiler import profile_dataframe

st.title("Synthetic Data Factory")
st.write("Upload a CSV dataset to analyze its structure.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        report = profile_dataframe(df)

        st.subheader("Dataset overview")
        st.metric("Rows", report["rows"])
        st.metric("Columns", report["columns_count"])
        st.metric("Duplicate rows", report["duplicate_rows"])

        st.subheader("Data preview")
        st.dataframe(df.head(20))

        st.subheader("Column statistics")
        st.dataframe(pd.DataFrame(report["columns"]))

    except Exception as error:
        st.error(f"Could not analyze dataset: {error}")