# Author: Hardik Singh
# Last updated: October 2025
# Purpose: load data, run automatic EDA, detect anomalies, generate AI insights

import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.ensemble import IsolationForest
from scipy import stats
import tempfile
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------- Generate Profiling Report --------
def generate_report(df):
    profile = ProfileReport(
        df,
        title="AI Agent Automated EDA Report",
        explorative=True
    )
    return profile

# -------- Anomaly Detection --------
def detect_anomalies(df, numeric_columns=["Quantity", "Sales"]):
    anomalies = {}
    iso = IsolationForest(contamination=0.05, random_state=42)

    for col in numeric_columns:
        if col in df.columns:
            preds = iso.fit_predict(df[[col]])
            df[f"{col}_anomaly"] = preds
            anomalies[col] = df[df[f"{col}_anomaly"] == -1]

    for col in numeric_columns:
        if col in df.columns:
            df[f"{col}_zscore"] = stats.zscore(df[col].astype(float))
            anomalies[f"{col}_zscore"] = df[abs(df[f"{col}_zscore"]) > 3]

    return anomalies


# -------- Safe device detection --------
def get_safe_device_map():
    """Decide automatically between GPU (8‑bit/FP16) and CPU."""
    if torch.cuda.is_available():
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            total_vram = gpu_props.total_memory / (1024 ** 3)
            free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            print(f"Detected GPU: {gpu_props.name} | "
                  f"Total VRAM {total_vram:.1f} GB | Free {free_vram:.1f} GB")
            if free_vram >= 5:
                return {"device_map": "auto",
                        "torch_dtype": torch.float16,
                        "low_cpu_mem_usage": True}
            else:
                return {"device_map": "auto", "load_in_8bit": True}
        except Exception as e:
            print(f"GPU check failed ({e}) – using CPU fallback.")
    return {"device_map": "cpu", "torch_dtype": torch.float32}


# -------- AI Insights Generation (Phi‑2) --------
def generate_ai_insights(df, anomalies):
    """Generate consultant‑style insights with Phi‑2."""
    model_id = "microsoft/phi-2"
    options = get_safe_device_map()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **options)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Prepare analytical context
    summary = df.describe(include="all").round(2).to_string()
    corr = df.corr(numeric_only=True).round(2).to_string()
    anomaly_cols = [k for k, v in anomalies.items() if not v.empty]
    anomaly_text = ", ".join(anomaly_cols) if anomaly_cols else "none detected"

    prompt = f"""
You are a data and business analytics consultant.

Dataset summary:
{summary}

Correlations:
{corr}

Columns with anomalies: {anomaly_text}

Write 5 concise, high‑value recommendations covering:
1. Data cleaning and structuring
2. New analytical approaches or KPIs
3. Creative dashboards or visualisations
4. Automation of recurring reports
5. Governance or collaboration practices
"""

    result = generator(
        prompt,
        max_new_tokens=350,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )[0]["generated_text"]

    return result.strip()


# -------- Streamlit App --------
st.set_page_config(page_title="🤖 AI Data Analyst Agent", layout="wide")
st.title("🤖 AI‑Powered Data Analytics Agent")

uploaded_file = st.file_uploader("📂 Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"✅ File uploaded – {df.shape[0]} rows, {df.shape[1]} columns")

    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10))

    st.subheader("📊 Automated EDA Report")
    profile = generate_report(df)
    st_profile_report(profile)

    st.subheader("⚠️ Anomaly Detection Results")
    anomalies = detect_anomalies(df)
    for key, subset in anomalies.items():
        if not subset.empty:
            st.warning(f"{len(subset)} anomalies detected in {key}")
            st.dataframe(subset[["OrderID", "Customer", "Region", "Product", "Quantity", "Sales"]].head(5))
        else:
            st.info(f"No major anomalies detected in {key}")

    st.subheader("🧠 AI‑Generated Insights")
    insights = generate_ai_insights(df, anomalies)
    st.write(insights)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(insights.encode("utf-8"))
        tmp_path = tmp.name
    with open(tmp_path, "rb") as file:
        st.download_button("💾 Download Insights", data=file, file_name="ai_insights.txt")