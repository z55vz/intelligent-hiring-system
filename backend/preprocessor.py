"""
==========================================================
  STAGE 1: DATA PRE-PROCESSOR (Cloud-Native)
  Input  : ZIP file (BytesIO or path) containing PDF resumes
  Output : pandas.DataFrame with columns:
           [Resume_ID, Resume_Text, Email, Phone,
            Exp_Years, Skills_List, Source_File]
==========================================================
"""

import io
import re
import zipfile
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional

import pandas as pd
import fitz  # PyMuPDF

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("pdf_processor")

# ---------- Constants ----------
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")
EXP_PATTERNS = [
    re.compile(r"(\d+)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience", re.I),
    re.compile(r"experience[:\s]+(\d+)\s*\+?\s*(?:years?|yrs?)", re.I),
    re.compile(r"(\d+)\s*\+\s*(?:years?|yrs?)", re.I),
]

# Common technical skills dictionary (extend as needed)
SKILLS_VOCAB = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "git", "linux", "bash", "ci/cd", "agile", "scrum",
    "tableau", "power bi", "excel", "spark", "hadoop", "kafka",
    "data analysis", "data science", "statistics", "etl",
}


# ============================================================
#  CORE EXTRACTION FUNCTIONS
# ============================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract raw text from PDF bytes using PyMuPDF."""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
        return ""


def extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text)
    return m.group(0).lower() if m else None


def extract_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text)
    if not m:
        return None
    phone = re.sub(r"[\s\-\(\)]", "", m.group(0))
    return phone if 8 <= len(re.sub(r"\D", "", phone)) <= 15 else None


def extract_experience_years(text: str) -> int:
    """Return the maximum years of experience mentioned, else 0."""
    years = []
    for pat in EXP_PATTERNS:
        years.extend(int(x) for x in pat.findall(text) if x.isdigit())
    return max(years) if years else 0


def extract_skills(text: str) -> List[str]:
    """Match skills from vocabulary against resume text."""
    text_lower = text.lower()
    found = {skill for skill in SKILLS_VOCAB if skill in text_lower}
    return sorted(found)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove control chars."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    return text.strip()


# ============================================================
#  MAIN PIPELINE
# ============================================================

def process_single_pdf(filename: str, pdf_bytes: bytes, idx: int) -> Optional[Dict]:
    """Process one PDF and return a structured row."""
    raw_text = extract_text_from_pdf(pdf_bytes)
    if not raw_text or len(raw_text.strip()) < 50:
        logger.warning(f"Skipping {filename}: insufficient text")
        return None

    cleaned = clean_text(raw_text)
    return {
        "Resume_ID": f"R{idx:04d}",
        "Source_File": filename,
        "Resume_Text": cleaned,
        "Email": extract_email(raw_text) or "",
        "Phone": extract_phone(raw_text) or "",
        "Exp_Years": extract_experience_years(raw_text),
        "Skills_List": ", ".join(extract_skills(raw_text)),
    }


def process_zip(zip_input: Union[str, Path, bytes, io.BytesIO]) -> pd.DataFrame:
    """
    Main entry point.
    Accepts a path, raw bytes, or BytesIO (from Streamlit's file_uploader).
    Returns a clean DataFrame.
    """
    # Normalize input to a ZipFile object
    if isinstance(zip_input, (str, Path)):
        zf_source = str(zip_input)
    elif isinstance(zip_input, bytes):
        zf_source = io.BytesIO(zip_input)
    elif isinstance(zip_input, io.BytesIO):
        zf_source = zip_input
    else:
        # Streamlit UploadedFile has .read()
        zf_source = io.BytesIO(zip_input.read())

    rows = []
    with zipfile.ZipFile(zf_source, "r") as zf:
        pdf_names = [n for n in zf.namelist()
                     if n.lower().endswith(".pdf") and not n.startswith("__MACOSX")]
        logger.info(f"Found {len(pdf_names)} PDF(s) in archive")

        for idx, name in enumerate(pdf_names, start=1):
            try:
                with zf.open(name) as f:
                    pdf_bytes = f.read()
                row = process_single_pdf(Path(name).name, pdf_bytes, idx)
                if row:
                    rows.append(row)
            except Exception as e:
                logger.error(f"Failed to process {name}: {e}")

    if not rows:
        raise ValueError("No valid resumes were extracted from the ZIP file.")

    df = pd.DataFrame(rows)
    logger.info(f"Successfully processed {len(df)} resume(s)")
    return df


# ============================================================
#  CLI / Standalone test
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <path_to_zip>")
        sys.exit(1)

    df = process_zip(sys.argv[1])
    out_path = "resumes_extracted.xlsx"
    df.to_excel(out_path, index=False)
    print(f"✅ Saved {len(df)} rows to {out_path}")
    print(df.head())
