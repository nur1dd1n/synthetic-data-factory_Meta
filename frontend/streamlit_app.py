import hashlib
import json
import os

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st

from app.services.ai_assistant import explain_dataset
from app.services.demo_data import make_demo_data
from app.services.generation_planner import (
    create_generation_plan,
    create_next_recommendation,
)
from app.services.profiler import profile_dataframe
from app.services.trust_report import run_generation, run_improvement_loop
from frontend.trust_report_ui import render_trust_report


MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 50
AI_STATE_KEYS = (
    "copilot_plan",
    "copilot_plan_config",
    "current_ai_recommendation",
    "ai_recommendation_history",
    "ai_recommendation_iteration",
    "ai_recommendations_finished",
    "ai_recommendation_error",
    "ai_explanation",
)
SOURCE_STATE_KEYS = (
    "generation_result",
    "run_config",
    "excluded_columns",
    "target_column",
    "task_type",
    "comparison_column",
    "copilot_request_input",
) + AI_STATE_KEYS


def clear_session_keys(keys):
    for key in keys:
        st.session_state.pop(key, None)


def meta_ai_is_configured():
    api_key = os.getenv("MODEL_API_KEY", "").strip()
    return bool(
        api_key
        and "replace_with" not in api_key
        and "your_numeric_id" not in api_key
        and "your_secret" not in api_key
    )


def ai_error_message(error):
    message = str(error)
    if "401" in message or "invalid_api_key" in message:
        return (
            "Meta AI rejected MODEL_API_KEY. Create a fresh Model API key, "
            "replace the value in .env, and restart Streamlit."
        )
    if "429" in message:
        return "Meta AI rate limit reached. Wait briefly and try again."
    return message

st.set_page_config(
    page_title="Synthetic Data Factory",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Synthetic Data Factory")
st.write(
    "Generate synthetic tabular data and measure statistical similarity, "
    "ML utility, and record-copying signals before export."
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
            st.info(
                "300 entirely invented transactions. The fraud target and "
                "classification task are preselected for a fast demo."
            )
        else:
            if len(uploaded_file.getvalue()) > MAX_INPUT_BYTES:
                st.error("This local MVP accepts CSV files up to 10 MB.")
                st.stop()
            source_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            df = pd.read_csv(uploaded_file)

        if st.session_state.get("source_id") != source_id:
            clear_session_keys(SOURCE_STATE_KEYS)
            st.session_state["source_id"] = source_id
            st.session_state["target_column"] = "fraud" if use_demo else None
            st.session_state["task_type"] = (
                "classification" if use_demo else "auto"
            )
            st.session_state["copilot_request_input"] = (
                "Preserve fraud patterns and useful feature relationships "
                "while reducing record-copying signals."
                if use_demo else ""
            )

        if df.empty:
            st.warning("The uploaded CSV contains no data.")
            st.stop()
        if len(df) > MAX_ROWS or len(df.columns) > MAX_COLUMNS:
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
            hide_index=True,
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
            max_value=MAX_ROWS,
            value=min(MAX_ROWS, max(1, len(df))),
            step=100,
        )
        target_option = st.selectbox(
            "Optional ML target column", options=[None] + list(generation_source.columns),
            format_func=lambda value: "None" if value is None else str(value), key="target_column",
            help="Reserve 25% of distinct labeled real rows before training the generator. Both models use that same test set.",
        )
        selected_task = st.selectbox(
            "ML task",
            ["auto", "classification", "regression"],
            key="task_type",
            disabled=target_option is None,
        )
        task = selected_task if target_option is not None else "auto"

        generation_profile = profile_dataframe(generation_source)
        ai_enabled = meta_ai_is_configured()

        st.divider()
        st.subheader("🤖 AI Synthetic Data Copilot")
        if ai_enabled:
            st.caption(
                "Meta AI receives aggregate schema statistics and your goal, "
                "not uploaded source rows."
            )
        else:
            st.info(
                "Meta AI is optional. Add MODEL_API_KEY to .env to enable the "
                "Copilot; local generation and diagnostics remain available."
            )
        def get_next_ai_recommendation():
            """Ask Meta AI for the next recommendation."""
            st.session_state.pop("ai_recommendation_error", None)
            result = create_next_recommendation(
                user_request=st.session_state.get("copilot_request_input", ""),
                profile=generation_profile,
                history=st.session_state["ai_recommendation_history"],
                target_column=target_option,
                task=task,
            )
            if result["done"]:
                st.session_state["ai_recommendations_finished"] = True
                st.session_state.pop("current_ai_recommendation", None)
            else:
                st.session_state["current_ai_recommendation"] = result[
                    "recommendation"
                ]


        def save_ai_decision(recommendation, decision):
            """Save feedback before requesting the next recommendation."""
            st.session_state["ai_recommendation_history"].append(
                {
                    "title": recommendation["title"],
                    "problem": recommendation["problem"],
                    "action": recommendation["action"],
                    "decision": decision,
                }
            )
            st.session_state["ai_recommendation_iteration"] += 1
            st.session_state.pop("current_ai_recommendation", None)

        def apply_ai_recommendation(recommendation):
            goal = recommendation.get("suggested_goal")
            target = recommendation.get("suggested_target")
            suggested_task = recommendation.get("suggested_task")
            if goal:
                st.session_state["copilot_request_input"] = goal
            if target in generation_source.columns:
                st.session_state["target_column"] = target
            if suggested_task in {"auto", "classification", "regression"}:
                st.session_state["task_type"] = suggested_task

        user_request = st.text_area(
            "Describe what synthetic data you want",
            key="copilot_request_input",
            placeholder=(
                "Example: Preserve fraud patterns while reducing "
                "record-copying signals."
            ),
            help="Describe your desired result in natural language.",
        )

        config = json.dumps(
            [source_id, int(num_rows), excluded, target_option, task],
            ensure_ascii=False,
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
            clear_session_keys(AI_STATE_KEYS)
            st.session_state["copilot_plan_config"] = copilot_config

        st.session_state.setdefault("ai_recommendation_history", [])
        st.session_state.setdefault("ai_recommendation_iteration", 0)

        if st.button("✨ Create AI Generation Plan", disabled=not ai_enabled):
            if not user_request.strip():
                st.warning("Please describe what synthetic data you need.")
            else:
                try:
                    with st.spinner("Meta AI is preparing your generation plan..."):
                        plan = create_generation_plan(
                            user_request=user_request,
                            profile=generation_profile,
                            target_column=target_option,
                            task=task,
                        )

                    st.session_state["copilot_plan"] = plan
                    st.session_state["copilot_plan_config"] = copilot_config
                    st.session_state["ai_recommendation_history"] = []
                    st.session_state["ai_recommendation_iteration"] = 0
                    st.session_state.pop("ai_recommendation_error", None)
                    st.session_state.pop("ai_recommendations_finished", None)
                    recommendations = plan.get("recommendations", [])
                    if recommendations:
                        st.session_state["current_ai_recommendation"] = recommendations[0]
                    else:
                        st.session_state.pop("current_ai_recommendation", None)

                    st.success("Meta AI generation plan is ready!")

                except Exception as error:
                    st.error(
                        "Could not create AI plan: " + ai_error_message(error)
                    )

        # Display the generated plan.
        plan = st.session_state.get("copilot_plan")

        if plan is not None and st.session_state.get("copilot_plan_config") == copilot_config:
            st.markdown("### 📋 Meta AI Generation Plan")

            st.markdown("**Goal**")
            st.write(plan["goal"])

            st.markdown("**Priorities**")
            st.write(", ".join(plan["priorities"]) or "No priorities returned")

            checks_column, rationale_column = st.columns(2)
            with checks_column:
                st.markdown("**Checks**")
                for check in plan["checks"]:
                    st.write(f"• {check}")
            with rationale_column:
                st.markdown("**Why this plan?**")
                for reason in plan["rationale"]:
                    st.write(f"• {reason}")

            if plan["warnings"]:
                st.warning("Please review the following limitations:")
                for warning in plan["warnings"]:
                    st.write(f"• {warning}")

            if plan.get("recommendations"):
                st.markdown("### 💡 AI Recommendations")

                for index, recommendation in enumerate(plan["recommendations"]):
                    with st.container(border=True):
                        st.markdown(f"**{recommendation['title']}**")
                        st.write(recommendation["problem"])
                        st.info(recommendation["action"])
                        has_suggestion = (
                            bool(recommendation.get("suggested_goal"))
                            or recommendation.get("suggested_target") is not None
                            or recommendation.get("suggested_task") != task
                        )
                        if has_suggestion:
                            st.button(
                                "Apply suggested settings",
                                key=f"apply_recommendation_{index}",
                                on_click=apply_ai_recommendation,
                                args=(recommendation,),
                            )

            history = st.session_state["ai_recommendation_history"]
            if history:
                with st.expander("Recommendation review history"):
                    for item in history:
                        decision = (
                            "Useful"
                            if item["decision"] == "accepted"
                            else "Skipped"
                        )
                        st.write(f"• {decision}: {item['title']}")

            current_recommendation = st.session_state.get(
                "current_ai_recommendation"
            )

            iteration = st.session_state["ai_recommendation_iteration"]
            if current_recommendation and iteration < 5:
                st.markdown("### Recommendation review")
                st.caption(
                    "Feedback guides the next AI suggestion. Use ‘Apply suggested "
                    "settings’ above to change the actual inputs."
                )

                with st.container(border=True):
                    st.markdown(f"**{current_recommendation['title']}**")
                    st.write(current_recommendation["problem"])
                    st.info(current_recommendation["action"])
                    col_accept, col_reject = st.columns(2)
                    useful = col_accept.button(
                        "👍 Useful",
                        key=f"accept_recommendation_{iteration}",
                    )
                    skipped = col_reject.button(
                        "Skip",
                        key=f"reject_recommendation_{iteration}",
                    )

                if useful or skipped:
                    save_ai_decision(
                        current_recommendation,
                        "accepted" if useful else "rejected",
                    )
                    try:
                        with st.spinner(
                            "Meta AI is preparing the next suggestion..."
                        ):
                            get_next_ai_recommendation()
                    except Exception as error:
                        st.session_state["ai_recommendation_error"] = (
                            ai_error_message(error)
                        )
                    st.rerun()

            if st.session_state.get("ai_recommendation_error"):
                st.error(
                    "Could not get the next AI recommendation: "
                    + st.session_state["ai_recommendation_error"]
                )
            if st.session_state.get("ai_recommendations_finished"):
                st.success("Meta AI has finished the recommendation process.")

            st.info(
                "Approve the plan to generate a baseline and keep only AI-tuned "
                "settings that improve measured diagnostics without regressions."
            )
            if st.button("✅ Approve Plan & Generate", type="primary"):
                st.session_state.pop("generation_result", None)
                st.session_state.pop("ai_explanation", None)

                try:
                    with st.spinner(
                        "Generating data and evaluating AI-proposed settings..."
                    ):
                        synthetic_df, reference, trust_report = run_improvement_loop(
                            source=generation_source,
                            num_rows=int(num_rows),
                            user_request=user_request,
                            profile=generation_profile,
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
                    loop_history = trust_report["improvement_loop"]["history"]
                    if any(
                        item["status"] == "error" for item in loop_history
                    ):
                        st.warning(
                            "The baseline dataset was generated, but AI tuning "
                            "stopped early. See the iteration history below."
                        )
                    else:
                        st.success("Synthetic dataset generated and evaluated!")

                except Exception as error:
                    st.error(f"AI generation failed: {error}")


        if st.session_state.get("run_config") != config:
            st.session_state.pop("generation_result", None)
            st.session_state.pop("comparison_column", None)
            st.session_state.pop("ai_explanation", None)
            st.session_state["run_config"] = config

        if st.button("Generate Synthetic Data"):
            st.session_state.pop("generation_result", None)
            st.session_state.pop("ai_explanation", None)
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
        st.caption(
            "After generation, Meta AI can explain the measured aggregate "
            "report. It does not receive uploaded source rows."
        )

        if st.button("Analyze Results with Meta AI", disabled=not ai_enabled):
            try:
                with st.spinner("Meta AI is analyzing the measured results..."):
                    st.session_state["ai_explanation"] = explain_dataset(
                        report,
                        result[2] if result is not None else None,
                    )
            except Exception as error:
                st.error("Meta AI analysis failed: " + ai_error_message(error))

        if st.session_state.get("ai_explanation"):
            st.markdown(st.session_state["ai_explanation"])

    except Exception as error:
        st.error(f"Could not analyze dataset: {error}")

else:
    clear_session_keys(("generation_result", "ai_explanation"))
    st.info("Upload a CSV file to get started, or select the invented-data demo.")
