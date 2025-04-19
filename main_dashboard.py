
import streamlit as st
import pandas as pd
from streamlit_app import detect_column_types, show_phase1
from phase2_segment_explorer import show_phase2

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

st.set_page_config(page_title="Survey Dashboard", layout="wide")
st.title("📊 Unified Survey Analysis Dashboard")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    col_types = detect_column_types(df)

    tab1, tab2 = st.tabs(["📥 Upload & Overview", "🧠 Smart Segment Explorer"])

    with tab1:
        show_phase1(df, col_types)

    with tab2:
        show_phase2(df, col_types)
