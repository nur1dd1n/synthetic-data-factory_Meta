"""Trust report presentation kept separate from upload and Meta AI integration."""
import numpy as np
import pandas as pd
import streamlit as st

from app.services.comparison import tokens
from app.services.trust_report import report_json, report_markdown


def distribution_table(original, synthetic):
    if (pd.api.types.is_numeric_dtype(original) and pd.api.types.is_numeric_dtype(synthetic)
            and not pd.api.types.is_bool_dtype(original)):
        a, b = original.dropna().to_numpy(dtype=float), synthetic.dropna().to_numpy(dtype=float)
        if not len(a) and not len(b):
            return pd.DataFrame()
        edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=20)
        return pd.DataFrame({"bin_center": (edges[:-1] + edges[1:]) / 2,
                             "Original": np.histogram(a, edges)[0] / max(len(a), 1),
                             "Synthetic": np.histogram(b, edges)[0] / max(len(b), 1)}).set_index("bin_center")
    a, b = tokens(original).value_counts(normalize=True), tokens(synthetic).value_counts(normalize=True)
    labels = a.add(b, fill_value=0).nlargest(15).index
    result = pd.DataFrame({"Original": a.reindex(labels, fill_value=0),
                           "Synthetic": b.reindex(labels, fill_value=0)})
    remainder = [1 - result[c].sum() for c in result]
    if max(remainder) > 1e-9:
        result.loc["[remaining categories]"] = remainder
    return result


def render_trust_report(report, original, synthetic):
    st.divider()
    st.subheader("Automated Trust Report")
    st.caption(f"This run: {report['training_rows']} training rows, {report['test_rows']} reserved test rows, "
               f"{report['generated_rows']} synthetic rows. Target: {report['target'] or 'none'}.")
    quality, privacy, utility = (report[k] for k in ("quality", "privacy", "ml_utility"))
    columns = st.columns(3)
    for column, label, value in zip(columns,
            ["Statistical similarity", "ML utility", "Copying-risk heuristic"],
            [quality["overall_score"] if quality else None,
             utility["utility_score"] if utility else None, privacy["risk_score"] if privacy else None]):
        column.metric(label, f"{value:.1f}/100" if value is not None else "Not evaluated / undefined")
    st.caption("Higher similarity and utility are better. The copying heuristic is uncalibrated: "
               "a low value does not demonstrate overall privacy.")
    for stage, error in report["errors"].items():
        st.warning(f"{stage} could not be evaluated: {error}. The generated CSV is still available.")
    if utility:
        st.write(f"{utility['metric']}: real-trained {utility['baseline_score']:.4f}; "
                 f"synthetic-trained {utility['synthetic_score']:.4f}.")
        st.caption(utility["warning"])
    distribution, correlation, details = st.tabs(["Distributions", "Correlations", "Diagnostics"])
    with distribution:
        column = st.selectbox("Compare column", list(original.columns), key="comparison_column")
        table = distribution_table(original[column], synthetic[column])
        if not table.empty:
            st.bar_chart(table)
        st.caption("Shares use common bins for numbers, or top 15 categories plus the remainder. "
                   "Numeric missing values are excluded from the histogram.")
        st.write({"Original missing (%)": round(original[column].isna().mean() * 100, 2),
                  "Synthetic missing (%)": round(synthetic[column].isna().mean() * 100, 2)})
    with correlation:
        numeric = [c for c in original.select_dtypes(include="number") if pd.api.types.is_numeric_dtype(synthetic[c])]
        if len(numeric) >= 2:
            if len(numeric) > 20:
                st.caption("Showing the first 20 numeric columns.")
            numeric = numeric[:20]
            a, b = st.columns(2)
            for panel, label, frame in ((a, "Original training data", original), (b, "Synthetic data", synthetic)):
                with panel:
                    st.write(label)
                    st.dataframe(frame[numeric].corr().style.format("{:.2f}", na_rep="N/A"))
            st.caption("Pearson correlation matrices. N/A means the relationship could not be estimated.")
        else:
            st.info("At least two numeric columns are needed.")
    with details:
        if quality:
            st.dataframe(pd.DataFrame(quality["per_column"]), hide_index=True)
            st.caption(quality["methodology"])
        if privacy:
            st.write(f"Exact matches: {privacy['exact_match_rate']:.2f}%. "
                     f"Nearest-distance p05: {privacy['nearest_distance_p05']:.4f}.")
            st.caption(f"Searched all {privacy['reference_rows']} original training rows for "
                       f"{privacy['distance_sample_size']} sampled synthetic rows.")
            if privacy["potential_identifier_columns"]:
                st.warning("Possible identifiers: " + ", ".join(privacy["potential_identifier_columns"]))
                st.write("Non-null identifier value overlap (%)", privacy["identifier_value_overlap_percent"])
            st.caption(privacy["warning"])
    st.markdown("**Recommended next steps**")
    for recommendation in report["recommendations"]:
        st.write("• " + recommendation)
    st.caption("Report exports contain aggregate results and column names, without source records. Review before sharing.")
    a, b = st.columns(2)
    a.download_button("Download report JSON", report_json(report), "trust_report.json", "application/json")
    b.download_button("Download report Markdown", report_markdown(report), "trust_report.md", "text/markdown")
