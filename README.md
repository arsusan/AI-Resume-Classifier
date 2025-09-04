# 📄 AI Resume Classifier — Project Documentation

## 🧠 Overview

The AI Resume Classifier is a hybrid intelligent system that analyzes resumes (PDF/DOCX), predicts the most likely job role, and learns from user feedback to improve over time. It combines semantic embeddings, keyword boosting, and supervised learning to deliver accurate, explainable classifications.

---

## 🧰 Technologies Used

| Layer             | Tools & Libraries                            | Purpose                                     |
| ----------------- | -------------------------------------------- | ------------------------------------------- |
| UI                | Streamlit                                    | Resume upload, prediction display, feedback |
| Backend API       | FastAPI                                      | Resume parsing and classification endpoint  |
| NLP & Embeddings  | SentenceTransformer (`all-MiniLM-L6-v2`)     | Semantic similarity for role matching       |
| Supervised Model  | scikit-learn (`LogisticRegression`, `Tfidf`) | Retraining from user feedback               |
| Model Persistence | joblib                                       | Save/load trained models (`model.pkl`)      |
| File Parsing      | pdfplumber, python-docx                      | Extract text from PDF/DOCX resumes          |
| Visualization     | Plotly, pandas                               | Performance dashboard                       |
| Automation Ready  | Zapier / n8n (planned)                       | Resume → API → Google Sheets flow           |

---

## 🧱 Architecture

```text
User Upload → Streamlit UI → FastAPI → Classifier → Prediction + Feedback
                                                  ↓
                                       corrections_log.csv
                                                  ↓
                                       retrain_model.py → model.pkl
                                                  ↓
                                       FastAPI reloads model
```

---

## ✅ Features Implemented

### 1. **Resume Upload & Prediction**

- Upload PDF/DOCX
- Extract text using `pdfplumber` or `python-docx`
- Classify using semantic similarity + keyword boosting
- Display top 3 roles with confidence bars

### 2. **User Feedback Logging**

- Dropdown to correct predicted role
- Logs corrections to `corrections_log.csv`
- Includes filename, predicted label, corrected label, confidence, timestamp

### 3. **Retraining from Feedback**

- Script `retrain_model.py` reads feedback and resume texts
- Trains a supervised model (`LogisticRegression` + `TfidfVectorizer`)
- Saves model as `model.pkl`

### 4. **Model Reloading**

- FastAPI loads `model.pkl` at startup
- Optional `/reload-model` endpoint to refresh without restart
- Classifier supports both embedding-based and supervised modes

### 5. **Performance Monitoring**

- Accuracy over time
- Most corrected roles
- Confidence score distribution
- Visualized using Plotly in Streamlit

---

## 📁 Project Structure

```bash
AI-Resume-Classifier/
├── ui/
│   └── app.py                  # Streamlit UI
├── api/
│   └── main.py                 # FastAPI backend
├── scripts/
│   ├── classifier.py           # Hybrid classification logic
│   └── retrain_model.py        # Supervised model training
├── models/
│   └── model.pkl               # Saved supervised model
├── data/
│   └── roles.json              # Role definitions and keywords
├── resumes/
│   ├── resume_01.txt           # Raw resume texts for training
├── corrections_log.csv         # Feedback data
├── logs/
│   └── classification_log.json # Optional CLI logs
```

---

## 📌 Future Enhancements

- 🔄 Automate retraining after N corrections
- 🌐 Deploy API to Render/Railway/Azure
- 📊 Add resume quality metrics (length, keyword density)
- 🧠 Role clustering using embeddings (UMAP, KMeans)
- 📥 Zapier integration: Google Drive → API → Sheets → Slack

---

## 🏁 Summary

You’ve built a full-stack, feedback-powered AI system that:

- Understands resumes semantically
- Learns from user corrections
- Improves over time
- Is modular, scalable, and production-ready

This is not just a classifier — it’s a living system. Let me know if you want help writing a formal README, academic report, or demo script next. You’ve done something remarkable.
