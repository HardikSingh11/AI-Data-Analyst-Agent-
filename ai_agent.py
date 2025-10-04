# Author: Hardik Singh
# Last updated: October 2025
# Purpose: load data, run automatic EDA, detect anomalies, generate AI insights

import pandas as pd
import sys
from ydata_profiling import ProfileReport
from sklearn.ensemble import IsolationForest
from scipy import stats
from transformers import pipeline

# -------- Load Dataset --------
def load_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")
    return df

# -------- Generate Profiling Report --------
def generate_report(df, output_file="eda_report.html"):
    profile = ProfileReport(
        df,
        title="AI Agent Automated EDA Report",
        explorative=True
    )
    profile.to_file(output_file)
    print(f"\n✅ Report generated successfully: {output_file}")

# -------- Anomaly Detection --------
def detect_anomalies(df, numeric_columns=["Quantity", "Sales"]):
    print("\n🔎 Running anomaly detection...")

    anomalies = {}

    # 1. Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    for col in numeric_columns:
        if col in df.columns:
            preds = iso.fit_predict(df[[col]])
            df[f"{col}_anomaly"] = preds
            anomalies[col] = df[df[f"{col}_anomaly"] == -1]

    # 2. Z-score statistical method
    for col in numeric_columns:
        if col in df.columns:
            df[f"{col}_zscore"] = stats.zscore(df[col].astype(float))
            anomalies[f"{col}_zscore"] = df[abs(df[f"{col}_zscore"]) > 3]

    # Print anomalies summary
    for key, subset in anomalies.items():
        if not subset.empty:
            print(f"\n⚠️ Anomalies detected in {key}: {len(subset)} rows")
            print(subset[["OrderID", "Customer", "Region", "Product", "Quantity", "Sales"]].head(5))
        else:
            print(f"\n✅ No major anomalies detected in {key}")

    return anomalies

# -------- Generate AI Insights --------
def generate_ai_insights(df, anomalies):
    print("\n🧠 Generating AI Insights... (using FLAN-T5)")

    summarizer = pipeline("text2text-generation", model="google/flan-t5-base")

    # Build context for the LLM
    summary = f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n"
    summary += f"Columns available: {', '.join(df.columns)}\n"

    if anomalies:
        summary += "\nAnomaly highlights:\n"
        for key, subset in anomalies.items():
            if not subset.empty:
                summary += f"- {len(subset)} anomalies in {key}\n"

    prompt = (
        f"You are a data analyst. Based on this dataset summary and anomalies:\n"
        f"{summary}\n"
        f"Write a clear report with insights on what the data shows, "
        f"why anomalies might exist, and what analysis approaches to try next."
    )

    insights = summarizer(prompt, max_length=256, do_sample=True)[0]['generated_text']

    print("\n📢 AI-Generated Insights:\n")
    print(insights)
    return insights

# -------- Append AI Insights into HTML --------
def append_ai_to_html(insights, html_file="eda_report.html"):
    try:
        with open(html_file, "a", encoding="utf-8") as f:
            f.write("<hr>")
            f.write("<h2>🤖 AI Generated Insights</h2>")
            f.write(f"<p>{insights}</p>")
        print("💡 AI Insights added into HTML report.")
    except Exception as e:
        print("Error appending insights to HTML:", e)

    # Save insights to file
    with open("ai_insights.txt", "w", encoding="utf-8") as f:
        f.write("📢 AI Generated Insights\n")
        f.write("="*40 + "\n\n")
        f.write(insights)

    print("\n💾 Insights saved to ai_insights.txt")

# -------- Main Script --------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai_agent.py <data_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"\n📂 Loading dataset: {file_path}")
    df = load_data(file_path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Generate report
    generate_report(df)

    # Detect anomalies
    anomalies = detect_anomalies(df)

# Generate AI insights
insights = generate_ai_insights(df, anomalies)

# Append insights into the HTML report
append_ai_to_html(insights)