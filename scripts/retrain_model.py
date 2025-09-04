import pandas as pd
import os
import joblib
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Load Feedback Data ---
df = pd.read_csv("corrections_log.csv", names=["filename", "predicted_label", "corrected_label", "confidence", "timestamp"])
df.dropna(subset=["filename", "corrected_label"], inplace=True)

# --- Load Resume Texts ---
def extract_text(filename):
    path = os.path.join("resumes", filename)
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return None
    try:
        if filename.lower().endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return None

df["text"] = df["filename"].apply(extract_text)
df.dropna(subset=["text"], inplace=True)

# --- Validate Sample Count ---
if len(df) < 2:
    raise ValueError("❌ Not enough valid samples to train. Please add more corrected resumes.")

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["corrected_label"], test_size=0.2, random_state=42)

# --- Build Pipeline ---
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# --- Train Model ---
if len(set(y_train)) < 2:
    raise ValueError("❌ Need at least two distinct classes to retrain the model.")

model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# --- Save Model ---
joblib.dump(model, "model.pkl")
print("\n✅ Model retrained and saved as model.pkl")
print("✅ Retraining complete.")