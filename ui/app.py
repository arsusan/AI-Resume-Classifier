import streamlit as st
import requests
import json
import base64
import pandas as pd
from datetime import datetime
import plotly.express as px

# --- Page Setup ---
st.set_page_config(page_title="AI Resume Classifier", layout="centered")
st.title("📄 AI Resume Classifier")

st.markdown("""
Upload your resume (PDF or DOCX) and get an instant prediction of your most likely job role.
""")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; }
    .stButton button {
        background-color: #2c3e50;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    .stSpinner { color: #2c3e50; }
    .stMarkdown { font-size: 1.1em; }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .confidence-fill {
        background-color: #2c3e50;
        height: 20px;
        border-radius: 5px;
        text-align: right;
        padding-right: 8px;
        color: white;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# --- File Upload ---
file = st.file_uploader("📎 Upload your resume", type=["pdf", "docx"])

# --- Classification Logic ---
if file:
    files = {"file": (file.name, file.getvalue(), file.type)}

    with st.spinner("🔍 Classifying your resume..."):
        try:
            response = requests.post("http://127.0.0.1:8000/classify", files=files)
            response.raise_for_status()
            result = response.json()

            label = result["label"]
            confidence = round(result["confidence"], 3)
            top_matches = result["top_matches"]

            # --- Display Results ---
            st.success("✅ Classification Complete")

            st.subheader("🎯 Predicted Role")
            st.markdown(f"<h3 style='color:#2c3e50'>{label}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Confidence:</strong> {confidence}</p>", unsafe_allow_html=True)

            st.subheader("🏆 Top 3 Similar Roles")
            for role, score in top_matches.items():
                pct = int(score * 100)
                st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{pct}%;">{role}: {round(score, 3)}</div>
                    </div>
                """, unsafe_allow_html=True)

            # --- Download Button ---
            result_json = json.dumps(result, indent=2)
            b64 = base64.b64encode(result_json.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="classification_result.json">📥 Download Result</a>'
            st.markdown(href, unsafe_allow_html=True)

            # --- Feedback Loop ---
            st.subheader("✏️ Feedback: Correct the Role")
            role_list = list(top_matches.keys()) + [label]  # Add predicted + top matches
            corrected_label = st.selectbox("Select correct role (if misclassified)", sorted(set(role_list)))

            if st.button("Submit Correction"):
                feedback_entry = {
                    "filename": file.name,
                    "predicted_label": label,
                    "corrected_label": corrected_label,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }
                feedback_df = pd.DataFrame([feedback_entry])
                feedback_df.to_csv("corrections_log.csv", mode="a", header=False, index=False)
                st.success("✅ Correction submitted!")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Error: {e}")
        except Exception as e:
            st.error(f"⚠️ Unexpected Error: {e}")

# --- Performance Dashboard ---
st.markdown("---")
st.subheader("📊 Model Performance Dashboard")

try:
    df = pd.read_csv("corrections_log.csv", names=["filename", "predicted_label", "corrected_label", "confidence", "timestamp"])
    df["correct"] = df["predicted_label"] == df["corrected_label"]

    # Accuracy over time
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    accuracy_by_date = df.groupby("date")["correct"].mean().reset_index()
    fig1 = px.line(accuracy_by_date, x="date", y="correct", title="Accuracy Over Time")
    st.plotly_chart(fig1)

    # Most corrected roles
    corrections = df[df["correct"] == False]
    correction_counts = corrections["predicted_label"].value_counts().reset_index()
    correction_counts.columns = ["predicted_label", "count"]  # Rename columns explicitly
    fig2 = px.bar(correction_counts, x="predicted_label", y="count", title="Most Corrected Roles")
    st.plotly_chart(fig2)

    # Confidence histogram
    fig3 = px.histogram(df, x="confidence", nbins=20, title="Confidence Score Distribution")
    st.plotly_chart(fig3)

except FileNotFoundError:
    st.info("ℹ️ No feedback data available yet. Submit corrections to populate the dashboard.")
