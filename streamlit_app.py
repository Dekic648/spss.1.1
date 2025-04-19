
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def detect_column_types(df):
    col_types = {
        "segments": [],
        "likert": [],
        "rating": [],
        "matrix": [],
        "radio": [],
        "checkbox": [],
        "ranking": [],
        "semantic": [],
        "open_ended": []
    }
    for col in df.columns:
        cl = col.lower()
        if "segment_" in cl:
            col_types["segments"].append(col)
        elif "likert_" in cl:
            col_types["likert"].append(col)
        elif "rating_" in cl or "nps_" in cl:
            col_types["rating"].append(col)
        elif "matrix_" in cl:
            col_types["matrix"].append(col)
        elif "rb_" in cl:
            col_types["radio"].append(col)
        elif "checkbox_" in cl:
            col_types["checkbox"].append(col)
        elif "rank_" in cl:
            col_types["ranking"].append(col)
        elif "sd_" in cl:
            col_types["semantic"].append(col)
        elif "open_ended_" in cl or "comment_" in cl or "feedback_" in cl:
            col_types["open_ended"].append(col)
    return col_types

def plot_avg(df, cols, title, segment):
    numeric_df = df[cols].apply(pd.to_numeric, errors='coerce')
    if segment:
        grouped = df[[segment]].join(numeric_df).groupby(segment).mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        grouped.plot(kind='bar', ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        st.pyplot(fig)
    else:
        avg = numeric_df.mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        avg.plot(kind='bar', ax=ax)
        ax.bar_label(ax.containers[0], fmt='%.2f')
        st.pyplot(fig)

def plot_checkbox_group(df, group_cols, segment):
    group_label = "_".join(group_cols[0].split("_")[:3])
    st.subheader(f"📦 {group_label} (Checkbox Group)")
    if segment:
        grouped = df.groupby(segment)[group_cols].apply(lambda x: x.notna().sum())
        total = df.groupby(segment).size()
        percent = grouped.div(total, axis=0) * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        percent.plot(kind='bar', ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')
        st.pyplot(fig)
    else:
        total = df[group_cols].notna().sum()
        percent = total / len(df) * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        percent.plot(kind='bar', ax=ax)
        ax.bar_label(ax.containers[0], fmt='%.1f%%')
        st.pyplot(fig)

def plot_radio_group(df, group_cols, segment):
    group_label = "_".join(group_cols[0].split("_")[:2])
    st.subheader(f"🔘 {group_label} (Radio Button Group)")
    if segment:
        grouped = df.groupby(segment)[group_cols].apply(lambda x: x.notna().sum())
        total = df.groupby(segment).size()
        percent = grouped.div(total, axis=0) * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        percent.plot(kind='bar', ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')
        st.pyplot(fig)
    else:
        total = df[group_cols].notna().sum()
        percent = total / len(df) * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        percent.plot(kind='bar', ax=ax)
        ax.bar_label(ax.containers[0], fmt='%.1f%%')
        st.pyplot(fig)

def plot_ranking(df, cols):
    st.subheader("🥇 Ranking (lower = better)")
    numeric_df = df[cols].apply(pd.to_numeric, errors='coerce')
    avg_ranks = numeric_df.mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    avg_ranks.plot(kind='barh', ax=ax)
    ax.bar_label(ax.containers[0], fmt='%.2f')
    ax.set_xlabel("Average Rank")
    st.pyplot(fig)

def plot_semantic_diff(df, cols):
    st.subheader("🔄 Semantic Differential")
    numeric_df = df[cols].apply(pd.to_numeric, errors='coerce')
    avg = numeric_df.mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    avg.plot(kind='bar', ax=ax)
    ax.axhline(0, color='gray', linewidth=1)
    ax.bar_label(ax.containers[0], fmt='%.2f')
    st.pyplot(fig)

def preview_open_ended(df, cols):
    st.subheader("💬 Open-Ended Responses")
    for col in cols:
        st.markdown(f"**{col}**")
        responses = df[col].dropna().astype(str)
        for val in responses.head(5):
            st.markdown(f"- {val}")
        st.markdown("---")

def show_phase1(df, col_types):
    st.header("📥 Upload & Overview")
    segment = st.selectbox("Group by segment (optional)", ["None"] + col_types["segments"])
    seg = segment if segment != "None" else None

    if col_types["likert"]:
        st.subheader("📊 Likert Scale")
        plot_avg(df, col_types["likert"], "Likert", seg)

    if col_types["rating"]:
        st.subheader("📊 Rating Scale")
        plot_avg(df, col_types["rating"], "Rating", seg)

    if col_types["matrix"]:
        st.subheader("📊 Matrix Questions")
        plot_avg(df, col_types["matrix"], "Matrix", seg)

    if col_types["checkbox"]:
        prefixes = set("_".join(c.split("_")[:3]) for c in col_types["checkbox"])
        for prefix in prefixes:
            group = [col for col in col_types["checkbox"] if col.startswith(prefix)]
            plot_checkbox_group(df, group, seg)

    if col_types["radio"]:
        prefixes = set("_".join(c.split("_")[:2]) for c in col_types["radio"])
        for prefix in prefixes:
            group = [col for col in col_types["radio"] if col.startswith(prefix)]
            plot_radio_group(df, group, seg)

    if col_types["ranking"]:
        plot_ranking(df, col_types["ranking"])

    if col_types["semantic"]:
        plot_semantic_diff(df, col_types["semantic"])

    if col_types["open_ended"]:
        preview_open_ended(df, col_types["open_ended"])

    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head(20))
