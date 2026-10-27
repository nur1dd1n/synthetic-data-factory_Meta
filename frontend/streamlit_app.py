from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st

from app.services.profiler import profile_dataframe
from app.services.synthesizer import generate_synthetic_data
from app.services.ai_assistant import explain_dataset


st.set_page_config(
    page_title="Synthetic Data Factory",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Synthetic Data Factory")
st.write(
    "Upload a CSV dataset to analyze its structure, "
    "generate synthetic data, and get AI insights."
)

uploaded_file = st.file_uploader(
    "Upload CSV dataset",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("The uploaded CSV contains no data.")
            st.stop()

        report = profile_dataframe(df)

        # Dataset overview
        st.subheader("📊 Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", report["rows"])
        col2.metric("Columns", report["columns_count"])
        col3.metric("Duplicate rows", report["duplicate_rows"])

        # Data preview
        st.subheader("👀 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # Column statistics
        st.subheader("📋 Column Statistics")
        st.dataframe(
            pd.DataFrame(report["columns"]),
            use_container_width=True,
        )

        # Synthetic data generation
        st.divider()
        st.subheader("🧬 Synthetic Data Generation")

        num_rows = st.number_input(
            "Number of synthetic rows",
            min_value=1,
            max_value=100000,
            value=max(1, len(df)),
            step=100,
        )

        if st.button("Generate Synthetic Data"):
            try:
                with st.spinner(
                    "Training synthesizer and generating data..."
                ):
                    synthetic_df = generate_synthetic_data(
                        df,
                        int(num_rows),
                    )

                st.session_state["synthetic_df"] = synthetic_df

                st.success("Synthetic dataset generated!")

            except Exception as error:
                st.error(f"Generation failed: {error}")

        # Show generated data and download button
        synthetic_df = st.session_state.get("synthetic_df")

        if synthetic_df is not None:
            st.subheader("🔬 Synthetic Data Preview")

            st.dataframe(
                synthetic_df.head(20),
                use_container_width=True,
            )

            st.download_button(
                label="⬇️ Download Synthetic CSV",
                data=synthetic_df.to_csv(index=False).encode("utf-8"),
                file_name="synthetic_data.csv",
                mime="text/csv",
            )

        # Meta AI analysis
        st.divider()
        st.subheader("🤖 Meta AI Analysis")

        if st.button("Analyze Dataset with Meta AI"):
            try:
                with st.spinner("Meta AI is analyzing the dataset..."):
                    explanation = explain_dataset(report)

                st.markdown(explanation)

            except Exception as error:
                st.error(f"Meta AI analysis failed: {error}")

    except Exception as error:
        st.error(f"Could not analyze dataset: {error}")

else:
    st.info("Upload a CSV file to get started.")