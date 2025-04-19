
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency

def create_segment(df, col):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    median = df[col].median()
    return df[col].apply(lambda x: "Low" if x < median else "High")

def prettify_label(col):
    return col.replace("_", " ").replace(":", "").title()

def generate_natural_summary(source, target, means_or_props, p, is_percentage=False):
    high = means_or_props.get("High", 0)
    low = means_or_props.get("Low", 0)
    direction = "more likely to" if high > low else "less likely to"
    value_type = "percentage" if is_percentage else "score"
    return f"People who rate **{prettify_label(source)}** high are **{direction}** show higher **{prettify_label(target)}** ({value_type}) (p = {p:.3f})."

def column_root(colname):
    return colname.lower().replace("_selected", "").replace("_score", "").strip()

def is_same_variable(a, b):
    return column_root(a) == column_root(b)

def run_segment_analysis(df, segment_col, col_types):
    insights = []
    df = df.copy()
    df["Segment"] = create_segment(df, segment_col)
    numeric_targets = col_types["likert"] + col_types["rating"] + col_types["matrix"] + col_types["semantic"] + col_types["ranking"]

    for col in numeric_targets:
        if is_same_variable(col, segment_col):
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        g1 = df[df["Segment"] == "Low"][col].dropna()
        g2 = df[df["Segment"] == "High"][col].dropna()
        if len(g1) > 5 and len(g2) > 5:
            stat, p = ttest_ind(g1, g2, equal_var=False)
            if p < 0.05:
                means = df.groupby("Segment")[col].mean().round(2).to_dict()
                summary = generate_natural_summary(segment_col, col, means, p)
                insights.append((summary, means, g1, g2, col, "boxplot"))

    for prefix in set("_".join(col.split("_")[:3]) for col in col_types["checkbox"]):
        group = [col for col in col_types["checkbox"] if col.startswith(prefix)]
        for col in group:
            if is_same_variable(col, segment_col):
                continue
            df[col + "_selected"] = df[col].notna().astype(int)
            ctab = pd.crosstab(df["Segment"], df[col + "_selected"])
            if ctab.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(ctab)
                if p < 0.05:
                    prop = df.groupby("Segment")[col + "_selected"].mean().round(2).mul(100).to_dict()
                    summary = generate_natural_summary(segment_col, col, prop, p, is_percentage=True)
                    insights.append((summary, prop, ctab, None, col, "bar"))

    for prefix in set("_".join(col.split("_")[:2]) for col in col_types["radio"]):
        group = [col for col in col_types["radio"] if col.startswith(prefix)]
        for col in group:
            if is_same_variable(col, segment_col):
                continue
            df[col + "_selected"] = df[col].notna().astype(int)
            ctab = pd.crosstab(df["Segment"], df[col + "_selected"])
            if ctab.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(ctab)
                if p < 0.05:
                    prop = df.groupby("Segment")[col + "_selected"].mean().round(2).mul(100).to_dict()
                    summary = generate_natural_summary(segment_col, col, prop, p, is_percentage=True)
                    insights.append((summary, prop, ctab, None, col, "bar"))

    return insights

def show_phase2(df, col_types):
    st.header("🧠 Smart Segment Explorer")
    segment_sources = col_types["likert"] + col_types["rating"] + col_types["semantic"] + col_types["ranking"]

    for source in segment_sources:
        insights = run_segment_analysis(df, source, col_types)
        for summary, values, data1, data2, col, chart_type in insights:
            with st.expander("✅ " + summary):
                if chart_type == "boxplot":
                    st.write(f"**Group Means:** High = {values.get('High', 'N/A')} | Low = {values.get('Low', 'N/A')}")
                    fig, ax = plt.subplots()
                    ax.boxplot([data1, data2], labels=["Low", "High"])
                    ax.set_title(prettify_label(col))
                    st.pyplot(fig)
                elif chart_type == "bar":
                    fig, ax = plt.subplots()
                    ax.bar(values.keys(), values.values())
                    for i, v in enumerate(values.values()):
                        ax.text(i, v + 1, f"{v:.1f}%", ha='center')
                    ax.set_ylabel("Selected (%)")
                    ax.set_title(prettify_label(col))
                    st.pyplot(fig)
