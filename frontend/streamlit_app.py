import hashlib
import json

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st

from app.services.profiler import profile_dataframe
from app.services.ai_assistant import explain_dataset
from app.services.demo_data import make_demo_data
from app.services.trust_report import run_generation
from frontend.trust_report_ui import render_trust_report


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

use_demo = st.checkbox("Try invented transaction demo data")
uploaded_file = st.file_uploader(
    "Upload CSV dataset",
    type=["csv"],
    disabled=use_demo,
)

if uploaded_file is not None or use_demo:
    try:
        if use_demo:
            df = make_demo_data()
            source_id = "invented-transactions-v1"
            st.info("300 entirely invented transactions. Try target 'fraud' and classification for the demo.")
        else:
            if len(uploaded_file.getvalue()) > 10 * 1024 * 1024:
                st.error("This local MVP accepts CSV files up to 10 MB.")
                st.stop()
            source_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            df = pd.read_csv(uploaded_file)

        if st.session_state.get("source_id") != source_id:
            st.session_state["source_id"] = source_id
            for key in ("generation_result", "run_config", "excluded_columns", "target_column", "comparison_column"):
                st.session_state.pop(key, None)

        if df.empty:
            st.warning("The uploaded CSV contains no data.")
            st.stop()
        if len(df) > 100000 or len(df.columns) > 50:
            st.error("For this local MVP, use at most 100,000 rows and 50 columns.")
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
        excluded = st.multiselect(
            "Exclude columns from generation and evaluation", list(df.columns), key="excluded_columns",
            help="For example, remove direct identifiers such as email or account ID.",
        )
        generation_source = df.drop(columns=excluded)
        if generation_source.shape[1] == 0:
            st.session_state.pop("generation_result", None)
            st.warning("Keep at least one column.")
            st.stop()

        num_rows = st.number_input(
            "Number of synthetic rows",
            min_value=1,
            max_value=100000,
            value=min(100000, max(1, len(df))),
            step=100,
        )
        target_option = st.selectbox(
            "Optional ML target column", options=[None] + list(generation_source.columns),
            format_func=lambda value: "None" if value is None else str(value), key="target_column",
            help="Reserve 25% of distinct labeled real rows before training the generator. Both models use that same test set.",
        )
        task = st.selectbox("ML task", ["auto", "classification", "regression"], disabled=target_option is None)
        config = json.dumps([source_id, int(num_rows), excluded, target_option, task])
        if st.session_state.get("run_config") != config:
            st.session_state.pop("generation_result", None)
            st.session_state.pop("comparison_column", None)
            st.session_state["run_config"] = config

        if st.button("Generate Synthetic Data"):
            st.session_state.pop("generation_result", None)
            try:
                with st.spinner("Training synthesizer and evaluating data..."):
                    synthetic_df, reference, trust_report = run_generation(
                        generation_source, int(num_rows), target_option, task,
                    )
                    trust_report["excluded_columns"] = excluded
                st.session_state["generation_result"] = (synthetic_df, reference, trust_report)
                st.success("Synthetic dataset generated!")
            except Exception as error:
                st.error(f"Generation failed: {error}")

        result = st.session_state.get("generation_result")
        if result is not None:
            synthetic_df, reference, trust_report = result
            st.subheader("🔬 Synthetic Data Preview")
            st.dataframe(synthetic_df.head(20), use_container_width=True)
            st.download_button(
                label="⬇️ Download Synthetic CSV",
                data=synthetic_df.to_csv(index=False).encode("utf-8"),
                file_name="synthetic_data.csv", mime="text/csv",
            )
            render_trust_report(trust_report, reference, synthetic_df)

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
    st.session_state.pop("generation_result", None)
    st.info("Upload a CSV file to get started, or select the invented-data demo.")
