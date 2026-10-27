import hashlib

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st

from app.services.profiler import profile_dataframe
from app.services.synthesizer import generate_synthetic_data
from app.services.ai_assistant import explain_dataset
from app.services.ml_utility import evaluate_ml_utility
from app.services.privacy import evaluate_privacy_risk
from app.services.quality import evaluate_statistical_quality


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
        source_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if st.session_state.get("source_id") != source_id:
            st.session_state["source_id"] = source_id
            st.session_state.pop("synthetic_df", None)
            st.session_state.pop("quality_report", None)
            st.session_state.pop("privacy_report", None)
            st.session_state.pop("ml_utility_report", None)

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

        target_option = st.selectbox(
            "Optional ML target column",
            options=[None] + list(df.columns),
            format_func=lambda value: "None" if value is None else str(value),
            help=(
                "Select a prediction target to compare a model trained on real "
                "data with a model trained on synthetic data."
            ),
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

                    quality_report = evaluate_statistical_quality(
                        df,
                        synthetic_df,
                    )
                    privacy_report = evaluate_privacy_risk(
                        df,
                        synthetic_df,
                    )
                    ml_utility_report = None
                    if target_option is not None:
                        ml_utility_report = evaluate_ml_utility(
                            df,
                            synthetic_df,
                            target_option,
                        )

                st.session_state["synthetic_df"] = synthetic_df
                st.session_state["quality_report"] = quality_report
                st.session_state["privacy_report"] = privacy_report
                st.session_state["ml_utility_report"] = ml_utility_report

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

            quality_report = st.session_state.get("quality_report")
            privacy_report = st.session_state.get("privacy_report")

            if quality_report and privacy_report:
                st.divider()
                st.subheader("🛡️ Automated Trust Report")

                ml_utility_report = st.session_state.get("ml_utility_report")
                score_col, correlation_col, utility_col, risk_col = st.columns(4)
                score_col.metric(
                    "Statistical similarity",
                    f'{quality_report["overall_score"]:.1f}/100',
                )
                correlation_score = quality_report["correlation_score"]
                correlation_col.metric(
                    "Correlation similarity",
                    (
                        f"{correlation_score:.1f}/100"
                        if correlation_score is not None
                        else "N/A"
                    ),
                )
                utility_col.metric(
                    "ML utility",
                    (
                        f'{ml_utility_report["utility_score"]:.1f}/100'
                        if ml_utility_report
                        else "Not evaluated"
                    ),
                )
                risk_col.metric(
                    "Privacy risk",
                    f'{privacy_report["risk_score"]:.1f}/100',
                )

                st.caption(
                    "Higher similarity is better; lower privacy risk is better. "
                    "Privacy risk is a heuristic screening score, not a formal "
                    "privacy guarantee."
                )

                if privacy_report["risk_level"] == "high":
                    st.error(
                        "High privacy risk: the synthetic data may reproduce "
                        "records that are too close to the source data."
                    )
                elif privacy_report["risk_level"] == "medium":
                    st.warning(
                        "Medium privacy risk: review close matches and direct "
                        "identifiers before sharing this dataset."
                    )
                else:
                    st.success(
                        "Low heuristic privacy risk. Additional privacy testing "
                        "is still required for sensitive production data."
                    )

                identifiers = privacy_report["potential_identifier_columns"]
                if identifiers:
                    st.warning(
                        "Potential identifier columns: " + ", ".join(identifiers)
                    )

                detail_col, privacy_col = st.columns(2)
                with detail_col:
                    st.markdown("**Per-column similarity**")
                    st.dataframe(
                        pd.DataFrame(quality_report["per_column"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                with privacy_col:
                    st.markdown("**Privacy diagnostics**")
                    st.metric(
                        "Exact source-row matches",
                        f'{privacy_report["exact_match_rate"]:.2f}%',
                    )
                    st.metric(
                        "5th percentile nearest distance",
                        f'{privacy_report["nearest_distance_p05"]:.4f}',
                    )
                    st.caption(privacy_report["warning"])

                if ml_utility_report:
                    st.markdown("**ML utility benchmark**")
                    ml_col1, ml_col2, ml_col3 = st.columns(3)
                    ml_col1.metric(
                        "Task",
                        ml_utility_report["task"].title(),
                    )
                    ml_col2.metric(
                        "Real-trained score",
                        f'{ml_utility_report["baseline_score"]:.3f}',
                    )
                    ml_col3.metric(
                        "Synthetic-trained score",
                        f'{ml_utility_report["synthetic_score"]:.3f}',
                    )
                    st.caption(
                        f'Metric: {ml_utility_report["metric"]}. '
                        + ml_utility_report["warning"]
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
