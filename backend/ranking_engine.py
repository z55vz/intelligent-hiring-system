"""
==========================================================
  STAGE 2: RANKING ENGINE (Cloud-Native)
  Input  : DataFrame from pdf_processor + UI inputs
  Output : Dict with top candidates, others, metrics, charts
==========================================================
"""

import io
import os
import json
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Headless backend (required for cloud)
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, accuracy_score
)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ranking_engine")

# ---------- Config ----------
MODEL_NAME = "BAAI/bge-small-en-v1.5"
WEIGHTS = {"semantic": 0.55, "experience": 0.20, "skills": 0.25}
CHARTS_DIR = Path("static/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
#  MODEL LOADER (cached singleton)
# ============================================================
_MODEL_CACHE = {}

def get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    if "model" not in _MODEL_CACHE:
        logger.info(f"Loading model: {MODEL_NAME}")
        _MODEL_CACHE["model"] = SentenceTransformer(MODEL_NAME)
    return _MODEL_CACHE["model"]


# ============================================================
#  SCORING FUNCTIONS
# ============================================================

def compute_semantic_scores(jd: str, resumes: List[str]) -> np.ndarray:
    """Cosine similarity between JD and resume embeddings."""
    model = get_model()
    jd_emb = model.encode([jd], normalize_embeddings=True)
    res_emb = model.encode(resumes, normalize_embeddings=True,
                           batch_size=16, show_progress_bar=False)
    return cosine_similarity(jd_emb, res_emb)[0]


def compute_experience_score(years: int, min_exp: int) -> float:
    """Normalized experience score [0, 1]."""
    if years < min_exp:
        return 0.0
    return min(1.0, (years - min_exp + 1) / 10.0)


def compute_skills_score(
    candidate_skills: str,
    required: List[str],
    mandatory: List[str]
) -> Tuple[float, bool]:
    """
    Returns (score, passes_mandatory).
    Mandatory skills (prefixed with '!' in UI) MUST be present.
    """
    cand_set = {s.strip().lower() for s in candidate_skills.split(",") if s.strip()}
    req_set = {s.lower() for s in required}
    mand_set = {s.lower() for s in mandatory}

    # Hard filter: mandatory skills
    if mand_set and not mand_set.issubset(cand_set):
        return 0.0, False

    if not req_set:
        return 1.0, True

    matched = len(req_set & cand_set)
    return matched / len(req_set), True


def parse_skills_input(skills: List[str]) -> Tuple[List[str], List[str]]:
    """Split skills into (all_required, mandatory_only). '!' prefix = mandatory."""
    required, mandatory = [], []
    for s in skills:
        s = s.strip()
        if s.startswith("!"):
            clean = s[1:].strip()
            required.append(clean)
            mandatory.append(clean)
        else:
            required.append(s)
    return required, mandatory


# ============================================================
#  CHART GENERATION (Base64 for cloud)
# ============================================================

def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for inline display."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def chart_score_distribution(scores: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(scores, bins=20, kde=True, color="#3B82F6", ax=ax)
    ax.set_title("Final Score Distribution")
    ax.set_xlabel("Score"); ax.set_ylabel("Count")
    return fig_to_base64(fig)


def chart_experience_distribution(years: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(years, bins=15, color="#10B981", ax=ax)
    ax.set_title("Years of Experience Distribution")
    ax.set_xlabel("Years"); ax.set_ylabel("Candidates")
    return fig_to_base64(fig)


def chart_confusion_matrix(y_true, y_pred, labels) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    return fig_to_base64(fig)


def chart_per_class_metrics(y_true, y_pred, labels) -> str:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    x = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, p, w, label="Precision", color="#3B82F6")
    ax.bar(x, r, w, label="Recall", color="#10B981")
    ax.bar(x + w, f, w, label="F1", color="#F59E0B")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Per-Class Metrics"); ax.legend(); ax.set_ylim(0, 1)
    return fig_to_base64(fig)


# ============================================================
#  THRESHOLDS & LABELS (auto-derived from quantiles)
# ============================================================

def derive_thresholds(scores: np.ndarray) -> Dict[str, float]:
    return {
        "low": float(np.quantile(scores, 0.40)),
        "high": float(np.quantile(scores, 0.75)),
    }


def label_score(score: float, thresholds: Dict[str, float]) -> str:
    if score >= thresholds["high"]:
        return "High"
    if score >= thresholds["low"]:
        return "Medium"
    return "Low"


# ============================================================
#  REASON BUILDER
# ============================================================

def build_reason(row: pd.Series, jd_skills: List[str]) -> str:
    parts = []
    if row["Semantic_Score"] >= 0.75:
        parts.append("strong semantic alignment with the job description")
    elif row["Semantic_Score"] >= 0.55:
        parts.append("moderate semantic alignment")

    if row["Exp_Years"] > 0:
        parts.append(f"{row['Exp_Years']} years of relevant experience")

    cand_skills = {s.strip().lower() for s in row["Skills_List"].split(",")}
    matched = [s for s in jd_skills if s.lower() in cand_skills]
    if matched:
        parts.append(f"matches skills: {', '.join(matched[:5])}")

    return "Selected because of: " + "; ".join(parts) if parts else "Baseline match"


# ============================================================
#  MAIN ENGINE
# ============================================================

def ranking_engine(
    df: pd.DataFrame,
    job_description: str,
    top_n: int = 5,
    min_exp: int = 0,
    skills: Optional[List[str]] = None,
) -> Dict:
    """
    Main ranking pipeline.

    Args:
        df: DataFrame from pdf_processor.process_zip()
        job_description: Free-text JD
        top_n: Number of top candidates to return
        min_exp: Minimum years of experience filter
        skills: List of skills (prefix with '!' for mandatory)

    Returns:
        JSON-serializable dict for Streamlit frontend.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    skills = skills or []
    required_skills, mandatory_skills = parse_skills_input(skills)

    df = df.copy().reset_index(drop=True)
    logger.info(f"Ranking {len(df)} candidates...")

    # 1) Semantic scores
    df["Semantic_Score"] = compute_semantic_scores(
        job_description, df["Resume_Text"].tolist()
    )

    # 2) Experience scores
    df["Exp_Score"] = df["Exp_Years"].apply(
        lambda y: compute_experience_score(int(y), min_exp)
    )

    # 3) Skills scores + mandatory filter
    skill_results = df["Skills_List"].apply(
        lambda s: compute_skills_score(s, required_skills, mandatory_skills)
    )
    df["Skill_Score"] = [r[0] for r in skill_results]
    df["Passes_Mandatory"] = [r[1] for r in skill_results]

    # 4) Final weighted score
    df["Final_Score"] = (
        WEIGHTS["semantic"] * df["Semantic_Score"]
        + WEIGHTS["experience"] * df["Exp_Score"]
        + WEIGHTS["skills"] * df["Skill_Score"]
    ).round(4)

    # 5) Eligibility
    df["Eligible"] = (
        (df["Exp_Years"] >= min_exp) & (df["Passes_Mandatory"])
    )

    # 6) Labels & thresholds
    thresholds = derive_thresholds(df["Final_Score"].values)
    df["Label"] = df["Final_Score"].apply(lambda s: label_score(s, thresholds))

    # 7) Reasons
    df["Reason"] = df.apply(lambda r: build_reason(r, required_skills), axis=1)

    # 8) Sort + split
    eligible_df = df[df["Eligible"]].sort_values("Final_Score", ascending=False)
    others_df = df[~df["Eligible"]].sort_values("Final_Score", ascending=False)

    top = eligible_df.head(top_n)
    others = pd.concat([eligible_df.iloc[top_n:], others_df])

    # 9) Metrics (using Label as pseudo-truth vs threshold predictions)
    y_true = df["Label"].tolist()
    y_pred = df["Final_Score"].apply(lambda s: label_score(s, thresholds)).tolist()
    labels_order = ["High", "Medium", "Low"]

    accuracy = float(accuracy_score(y_true, y_pred))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_order, zero_division=0, average="macro"
    )

    # 10) Charts
    charts = {
        "score_distribution": chart_score_distribution(df["Final_Score"].values),
        "experience_distribution": chart_experience_distribution(df["Exp_Years"]),
        "confusion_matrix": chart_confusion_matrix(y_true, y_pred, labels_order),
        "per_class_metrics": chart_per_class_metrics(y_true, y_pred, labels_order),
    }

    # 11) Build response
    def df_to_records(d: pd.DataFrame) -> List[Dict]:
        cols = ["Resume_ID", "Source_File", "Email", "Phone",
                "Final_Score", "Semantic_Score", "Exp_Years",
                "Skills_List", "Label", "Reason"]
        return d[cols].to_dict(orient="records")

    return {
        "summary": {
            "total": int(len(df)),
            "eligible": int(df["Eligible"].sum()),
            "returned_top": int(len(top)),
        },
        "top_candidates": df_to_records(top),
        "others": df_to_records(others),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(float(p), 4),
            "macro_recall": round(float(r), 4),
            "macro_f1": round(float(f), 4),
        },
        "thresholds": thresholds,
        "charts": charts,  # base64 PNG strings
    }


# ============================================================
#  CLI test
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python ranking_engine.py <excel_path> <job_description>")
        sys.exit(1)

    df = pd.read_excel(sys.argv[1])
    result = ranking_engine(
        df=df,
        job_description=sys.argv[2],
        top_n=5,
        min_exp=2,
        skills=["python", "machine learning", "!sql"],
    )
    # Print without the heavy base64 charts
    light = {k: v for k, v in result.items() if k != "charts"}
    print(json.dumps(light, indent=2, ensure_ascii=False))
