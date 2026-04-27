"""
==========================================================
  STAGE 1: DATA PRE-PROCESSOR
  Input  : ZIP file containing PDF resumes
  Output : Clean Excel file (Resume_ID, Resume_Text, Email,
           Phone, Exp_Years, Skills_List)
==========================================================
"""

# !pip install -q pymupdf openpyxl pandas

import os
import re
import zipfile
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================
OUTPUT_EXCEL = "processed_resumes.xlsx"
EXTRACT_DIR = "extracted_pdfs"

# Skills dictionary (extend as needed)
SKILLS_VOCAB = [
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust",
    "sql", "mysql", "postgresql", "mongodb", "redis", "oracle",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "data analysis", "data science", "statistics", "power bi", "tableau",
    "aws", "azure", "gcp", "docker", "kubernetes", "linux", "git",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "spring", "rest api", "graphql", "microservices", "ci/cd",
    "html", "css", "tailwind", "sass",
    "excel", "communication", "leadership", "project management", "agile", "scrum",
]

# ==========================================================
# 1. ILLEGAL CHARACTER SANITIZATION
# ==========================================================
# openpyxl rejects characters in these ranges (XML 1.0 illegal chars)
ILLEGAL_CHARS_RE = re.compile(
    r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]'
)

def sanitize_text(text: str) -> str:
    """Remove illegal characters & normalize unicode for Excel safety."""
    if text is None:
        return ""
    text = str(text)
    # Normalize unicode (NFKC handles ligatures, weird spaces, etc.)
    text = unicodedata.normalize("NFKC", text)
    # Remove illegal control characters
    text = ILLEGAL_CHARS_RE.sub(" ", text)
    # Collapse excessive whitespace but preserve line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ==========================================================
# 2. PDF TEXT EXTRACTION
# ==========================================================
def extract_pdf_text(pdf_path: str) -> str:
    """Extract raw text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        print(f"⚠️  Failed to read {pdf_path}: {e}")
        return ""

# ==========================================================
# 3. FIELD EXTRACTION (Name, Email, Phone, Experience, Skills)
# ==========================================================
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}"
)
EXP_PATTERNS = [
    re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience", re.I),
    re.compile(r"experience\s*[:\-]?\s*(\d{1,2})\+?\s*(?:years?|yrs?)", re.I),
    re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?)", re.I),
]

def extract_email(text: str) -> str:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else ""

def extract_phone(text: str) -> str:
    for m in PHONE_RE.finditer(text):
        candidate = re.sub(r"\D", "", m.group(0))
        if 8 <= len(candidate) <= 15:
            return m.group(0).strip()
    return ""

def extract_experience_years(text: str) -> float:
    years = []
    for pat in EXP_PATTERNS:
        for m in pat.finditer(text):
            try:
                y = int(m.group(1))
                if 0 <= y <= 50:
                    years.append(y)
            except (ValueError, IndexError):
                continue
    return float(max(years)) if years else 0.0

def extract_name(text: str, fallback: str) -> str:
    """Heuristic: first non-empty line that looks like a name."""
    for line in text.splitlines()[:10]:
        line = line.strip()
        if not line or len(line) > 60:
            continue
        if EMAIL_RE.search(line) or any(ch.isdigit() for ch in line):
            continue
        words = line.split()
        if 2 <= len(words) <= 5 and all(w[0].isupper() for w in words if w):
            return line
    return fallback

def extract_skills(text: str) -> List[str]:
    text_low = text.lower()
    found = []
    for skill in SKILLS_VOCAB:
        # Word-boundary match for short skills, substring for multi-word
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_low):
            found.append(skill)
    return sorted(set(found))

# ==========================================================
# 4. ZIP HANDLING
# ==========================================================
def unzip_resumes(zip_path: str, extract_dir: str) -> List[str]:
    """Extract ZIP and return list of PDF paths."""
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    pdf_paths = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, f))
    return sorted(pdf_paths)

# ==========================================================
# 5. MAIN PIPELINE
# ==========================================================
def process_pdfs_to_excel(zip_path: str,
                           output_excel: str = OUTPUT_EXCEL,
                           extract_dir: str = EXTRACT_DIR) -> str:
    """
    Full pipeline: ZIP -> PDFs -> Excel.
    Returns the path of the generated Excel file.
    """
    print(f"📂 Unzipping: {zip_path}")
    pdf_paths = unzip_resumes(zip_path, extract_dir)
    print(f"   Found {len(pdf_paths)} PDF files.")

    if not pdf_paths:
        raise ValueError("No PDF files found inside the ZIP archive.")

    rows = []
    for idx, pdf_path in enumerate(pdf_paths, start=1):
        filename = Path(pdf_path).stem
        raw_text = extract_pdf_text(pdf_path)
        clean = sanitize_text(raw_text)

        if not clean:
            print(f"   ⚠️  Skipped empty PDF: {filename}")
            continue

        rows.append({
            "Resume_ID": f"R{idx:04d}",
            "File_Name": sanitize_text(filename),
            "Candidate_Name": sanitize_text(extract_name(clean, filename)),
            "Email": sanitize_text(extract_email(clean)),
            "Phone": sanitize_text(extract_phone(clean)),
            "Exp_Years": extract_experience_years(clean),
            "Skills_List": ", ".join(extract_skills(clean)),
            "Resume_Text": clean,  # ORIGINAL text — no augmentation
        })
        print(f"   ✅ [{idx}/{len(pdf_paths)}] {filename}")

    df = pd.DataFrame(rows)
    df.to_excel(output_excel, index=False, engine="openpyxl")
    print(f"\n💾 Saved: {output_excel}  ({len(df)} resumes)")
    return output_excel