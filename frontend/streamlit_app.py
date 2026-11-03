import hashlib
import json

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st

from app.services.profiler import profile_dataframe
from app.services.ai_assistant import explain_dataset
from app.services.demo_data import make_demo_data
from app.services.trust_report import run_generation, run_improvement_loop
from frontend.trust_report_ui import render_trust_report
from app.services.generation_planner import create_generation_plan
from app.services.generation_planner import create_next_recommendation

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

        st.session_state["current_generation_profile"] = profile_dataframe(generation_source)
        st.session_state["current_ml_task"] = task

        st.divider()
        st.subheader("🤖 AI Synthetic Data Copilot")

        if "ai_recommendation_history" not in st.session_state:
            st.session_state["ai_recommendation_history"] = []

        if "ai_recommendation_iteration" not in st.session_state:
            st.session_state["ai_recommendation_iteration"] = 0

        if "ai_recommendation_limit" not in st.session_state:
            st.session_state["ai_recommendation_limit"] = 5





        def get_next_ai_recommendation():
            """Ask Meta AI for the next recommendation."""
            try:
                result = create_next_recommendation(
                    user_request=st.session_state.get(
                        "copilot_request_input", ""
                    ),
                    profile=st.session_state["current_generation_profile"],
                    history=st.session_state["ai_recommendation_history"],
                    target_column=st.session_state.get("target_column"),
                    task=st.session_state.get("current_ml_task", "auto"),
                )

                if result["done"]:
                    st.session_state["ai_recommendations_finished"] = True
                    st.session_state.pop("current_ai_recommendation", None)
                else:
                    st.session_state["current_ai_recommendation"] = (
                        result["recommendation"]
                    )

            except Exception as error:
                st.session_state["ai_recommendation_error"] = str(error)


        def save_ai_decision(recommendation, decision):
            """Save the user's Accept or Reject decision."""

            st.session_state["ai_recommendation_history"].append(
                {
                    "title": recommendation["title"],
                    "problem": recommendation["problem"],
                    "action": recommendation["action"],
                    "decision": decision,
                }
            )

            st.session_state["ai_recommendation_iteration"] += 1

                # Remove the current recommendation.
            st.session_state.pop("current_ai_recommendation", None)

            get_next_ai_recommendation()



        def apply_ai_recommendation(goal):
            st.session_state["copilot_request_input"] = goal
            st.session_state.pop("copilot_plan", None)
            st.session_state.pop("copilot_plan_config", None)

        user_request = st.text_area(
        "Describe what synthetic data you want",
        key="copilot_request_input",
        placeholder=(
            "Example: Generate synthetic transactions while "
            "preserving fraud patterns and reducing privacy risks."
        ),
        help="Describe your desired result in natural language.",
        )

        config = json.dumps([source_id, int(num_rows), excluded, target_option, task])


        history = st.session_state["ai_recommendation_history"]

        if history:
            st.markdown("### 🔄 AI Improvement History")

            for item in history:
                if item["decision"] == "accepted":
                    st.success(f"Accepted: {item['title']}")
                else:
                    st.info(f"Rejected: {item['title']}")

            st.caption(
                f"Iteration: {st.session_state['ai_recommendation_iteration']}"
                f"/{st.session_state['ai_recommendation_limit']}"
            )

        if st.session_state.get("ai_recommendation_error"):
            st.error(
                "Could not get the next AI recommendation: "
                    + st.session_state["ai_recommendation_error"]
            )
        if st.session_state.get("ai_recommendation_error"):
            st.success(
                "Meta AI has finshed the recommendation proccess!"
            )

        # Keep the AI plan valid only for the current dataset and settings.
        copilot_config = json.dumps(
            [
                source_id,
                int(num_rows),
                excluded,
                target_option,
                task,
                user_request,
            ],
            ensure_ascii=False,
        )

        if st.session_state.get("copilot_plan_config") != copilot_config:
            st.session_state.pop("copilot_plan", None)
            st.session_state.pop("copilot_plan_config", None)

        if st.button("✨ Create AI Generation Plan"):
            if not user_request.strip():
                st.warning("Please describe what synthetic data you need.")
            else:
                try:
                    with st.spinner("Meta AI is preparing your generation plan..."):
                        plan = create_generation_plan(
                            user_request=user_request,
                            profile=profile_dataframe(generation_source),
                            target_column=target_option,
                            task=task,
                        )

                    st.session_state["copilot_plan"] = plan
                    st.session_state["copilot_plan_config"] = copilot_config

                    recommendations = plan.get("recommendations", [])
                    if recommendations:
                        st.session_state["current_ai_recommendation"] = recommendations[0]

                    st.success("Meta AI generation plan is ready!")

                except Exception as error:
                    st.error(f"Could not create AI plan: {error}")

        # Display the generated plan.
        plan = st.session_state.get("copilot_plan")

        if plan is not None and st.session_state.get("copilot_plan_config") == copilot_config:
            st.markdown("### 📋 Meta AI Generation Plan")

            st.markdown("**Goal**")
            st.write(plan["goal"])

            st.markdown("**Priorities**")
            st.write(", ".join(plan["priorities"]))

            st.markdown("**Checks**")
            for check in plan["checks"]:
                st.write(f"• {check}")

            st.markdown("**Why this plan?**")
            for reason in plan["rationale"]:
                st.write(f"• {reason}")

            if plan["warnings"]:
                st.warning("Please review the following limitations:")
                for warning in plan["warnings"]:
                    st.write(f"• {warning}")

            if plan.get("recommendations"):
                st.json(plan["recommendations"])
                st.markdown("### 💡 AI Recommendations")

                for index, recommendation in enumerate(plan["recommendations"]):
                    with st.container(border=True):
                        st.markdown(f"{recommendation['title']}")
                        st.write(recommendation["problem"])
                        st.info(recommendation["action"])

                        suggested_goal = recommendation.get("suggested_goal")

                        if suggested_goal:
                            if st.button(
                                "✅ Apply Recommendation",
                                key=f"applyrecommendation{index}",
                                on_click=apply_ai_recommendation,
                                args=(suggested_goal,),
                            ):
                                st.session_state["copilot_request_input"] = suggested_goal
                                st.session_state.pop("copilot_plan", None)
                                st.session_state.pop("copilot_plan_config", None)
                                st.rerun()

                        suggested_target = recommendation.get("suggested_target")
                        if suggested_target:
                            st.caption(
                                f"Suggested target column: {suggested_target}. "
                                "Select it manually above before creating a new plan."
                            )

            st.info("Review the plan before approving the first generation.")

            current_recommendation = st.session_state.get(
                    "current_ai_recommendation"
                )

            iteration = st.session_state["ai_recommendation_iteration"]
            limit = st.session_state["ai_recommendation_limit"]

            if current_recommendation and iteration < limit:

                st.markdown("### 💡 Current AI Recommendation")

                with st.container(border=True):

                    st.markdown(
                        f"#### {current_recommendation['title']}"
                    )

                    st.write("**Problem:**")
                    st.write(current_recommendation["problem"])

                    st.write("**AI recommendation:**")
                    st.info(current_recommendation["action"])

                    col_accept, col_reject = st.columns(2)

                    with col_accept:
                        st.button(
                            "✅ Accept",
                            key=f"accept_recommendation_{iteration}",
                            on_click=save_ai_decision,
                            args=(current_recommendation, "accepted"),
                        )

                    with col_reject:
                        st.button(
                            "❌ Reject",
                            key=f"reject_recommendation_{iteration}",
                            on_click=save_ai_decision,
                            args=(current_recommendation, "rejected"),
                        )

            if st.button("✅ Approve Plan & Generate", type="primary"):
                st.session_state.pop("generation_result", None)

                try:
                    with st.spinner("Generating synthetic data and running AI improvements..."):
                        synthetic_df, reference, trust_report = run_improvement_loop(
                            source=generation_source,
                            num_rows=int(num_rows),
                            user_request=user_request,
                            profile=st.session_state["current_generation_profile"],
                            target=target_option,
                            task=task,
                            max_iterations=5,
                        )

                        trust_report["excluded_columns"] = excluded
                        trust_report["ai_plan"] = plan

                    st.session_state["generation_result"] = (
                        synthetic_df,
                        reference,
                        trust_report,
                    )
                    st.success("AI-approved synthetic dataset generated!")

                except Exception as error:
                    st.error(f"AI generation failed: {error}")


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
            st.caption(f"All {len(synthetic_df):,} generated rows are available below. Scroll to view them.")
            st.dataframe(synthetic_df, use_container_width=True, height=500)
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
