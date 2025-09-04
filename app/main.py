from fastapi import FastAPI, File, UploadFile, HTTPException
import io
import pdfplumber
from docx import Document
from scripts.classifier import classify_resume
import joblib
import os

app = FastAPI()

# --- Load Model ---
MODEL_PATH = "model.pkl"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded from model.pkl")
    else:
        model = None
        print("⚠️ model.pkl not found. Using embedding-based fallback.")

load_model()

# --- File Extraction ---
def extract_file(contents: bytes, ext: str) -> str:
    """Extract text from uploaded PDF or DOCX file"""
    try:
        if ext == "pdf":
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                text = "\n".join([p.extract_text() or "" for p in pdf.pages])
        elif ext == "docx":
            doc = Document(io.BytesIO(contents))
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")

    if not text or len(text.strip()) < 100:
        raise HTTPException(status_code=422, detail="Resume text is too short or unreadable.")
    
    return text.strip()

# --- Classification Endpoint ---
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["pdf", "docx"]:
        raise HTTPException(status_code=415, detail="Only PDF and DOCX files are supported.")

    contents = await file.read()
    resume_text = extract_file(contents, ext)

    # 🧠 Use enhanced classifier
    result = classify_resume(resume_text, top_n=3, model=model)

    return {
        "label": result["label"],
        "confidence": float(result["confidence"]),
        "top_matches": {k: float(v) for k, v in result["top_matches"].items()},
        "similarities": {k: float(v) for k, v in result["similarities"].items()},
        "matched_keywords": result["matched_keywords"],
        "categories": result["categories"]
    }

# --- Optional: Reload Model Endpoint ---
@app.get("/reload-model")
def reload_model():
    try:
        load_model()
        return {"status": "✅ Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
