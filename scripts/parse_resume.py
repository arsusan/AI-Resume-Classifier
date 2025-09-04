import os
import pdfplumber
from docx import Document
from PyPDF2 import PdfReader

# OCR libraries
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
from pdf2image import convert_from_path
from PIL import Image

# -----------------------
# PDF extract functions
# -----------------------

def extract_pdf_plumber(path):
    """Try extracting text with pdfplumber"""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_pdf_pypdf2(path):
    """Try extracting text with PyPDF2"""
    text = ""
    reader = PdfReader(path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def extract_pdf_ocr(path):
    """Fallback OCR for scanned PDFs"""
    text = ""
    images = convert_from_path(path)
    for img in images:
        text += pytesseract.image_to_string(img)
    return text.strip()

def extract_pdf(path):
    """Try multiple methods for PDF parsing"""
    try:
        text = extract_pdf_plumber(path)
        if text:
            print("[✔] Extracted text using pdfplumber")
            return text
    except Exception as e:
        print("[!] pdfplumber failed:", e)

    try:
        text = extract_pdf_pypdf2(path)
        if text:
            print("[✔] Extracted text using PyPDF2")
            return text
    except Exception as e:
        print("[!] PyPDF2 failed:", e)

    print("[*] Falling back to OCR...")
    return extract_pdf_ocr(path)

# -----------------------
# DOCX extract
# -----------------------

def extract_docx(path):
    """Extract text from DOCX resumes"""
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()]).strip()

# -----------------------
# Main save function
# -----------------------

def save_text(input_path, output_path):
    """Extract text from PDF or DOCX and save to .txt"""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".pdf":
        text = extract_pdf(input_path)
    elif ext == ".docx":
        text = extract_docx(input_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[✔] Saved extracted text to {output_path}")

# -----------------------
# Test run
# -----------------------

if __name__ == "__main__":
    input_file = "data/sample_resume.pdf"   # change file name here
    output_file = "data/sample_resume.txt"
    save_text(input_file, output_file)
