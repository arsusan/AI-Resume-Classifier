import os
import json
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load embedding model once ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load role definitions from JSON ---
ROLE_PATH = "data/roles.json"
with open(ROLE_PATH, "r", encoding="utf-8") as f:
    categories = json.load(f)

# --- Precompute category embeddings ---
cat_embeddings = {
    role: embed_model.encode(data["description"])
    for role, data in categories.items()
}

def match_keywords(text, keywords):
    """Return list of matched keywords from resume text"""
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]

def classify_resume(text, top_n=3, model=None):
    """
    Classify resume text using either:
    - Supervised model (if provided)
    - Embedding-based similarity with keyword boosting
    """
    if model:
        # --- Supervised classification ---
        label = model.predict([text])[0]
        probs = model.predict_proba([text])[0]
        role_index = list(model.classes_).index(label)
        confidence = round(probs[role_index], 4)

        # Top N matches
        top_indices = np.argsort(probs)[::-1][:top_n]
        top_matches = {model.classes_[i]: round(probs[i], 4) for i in top_indices}

        # Keyword matching
        keyword_hits = {
            role: match_keywords(text, categories[role]["keywords"])
            for role in categories
        }

        return {
            "label": label,
            "confidence": confidence,
            "top_matches": top_matches,
            "similarities": top_matches,
            "matched_keywords": keyword_hits,
            "categories": {r: categories[r].get("category", "Uncategorized") for r in categories}
        }

    else:
        # --- Embedding-based classification ---
        resume_vec = embed_model.encode(text)
        sims = {}
        keyword_hits = {}

        for role, data in categories.items():
            emb_sim = cosine_similarity([resume_vec], [cat_embeddings[role]])[0][0]
            matched = match_keywords(text, data["keywords"])
            keyword_boost = len(matched) * 0.01
            sims[role] = round(emb_sim + keyword_boost, 4)
            keyword_hits[role] = matched

        sorted_roles = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        best = sorted_roles[0][0]
        top_matches = dict(sorted_roles[:top_n])

        return {
            "label": best,
            "confidence": top_matches[best],
            "top_matches": top_matches,
            "similarities": sims,
            "matched_keywords": keyword_hits,
            "categories": {r: categories[r].get("category", "Uncategorized") for r in categories}
        }

# --- Optional CLI entry point ---
if __name__ == "__main__":
    import sys
    import joblib

    if len(sys.argv) < 2:
        print("Usage: python scripts/classifier.py <resume_txt_file> [model.pkl]")
        sys.exit(1)

    resume_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) == 3 else None

    if not os.path.exists(resume_path):
        print(f"[!] File not found: {resume_path}")
        sys.exit(1)

    with open(resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    model = joblib.load(model_path) if model_path else None
    result = classify_resume(resume_text, model=model)

    print(f"[✔] Predicted Role: {result['label']}")
    print("🏆 Top Matches:")
    for role, score in result["top_matches"].items():
        print(f"  {role}: {score:.4f}")

    print("\n🔍 Matched Keywords:")
    for role, keywords in result["matched_keywords"].items():
        if keywords:
            print(f"  {role}: {', '.join(keywords)}")

    # --- Log result ---
    log = {
        "timestamp": datetime.now().isoformat(),
        "file": os.path.basename(resume_path),
        "result": result
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/classification_log.json", "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log) + "\n")
