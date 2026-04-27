"""
==========================================================
  INTELLIGENT HIRING SYSTEM - Streamlit Cloud Edition
  Modern Glassmorphism UI + Cloud-Native Architecture
==========================================================
"""
import streamlit as st
import pandas as pd
import os
import json
import hashlib
from pathlib import Path
from backend.pdf_processor import process_pdfs_to_excel
from backend.ranking_engine import ranking_engine

# ===== Page Config =====
st.set_page_config(
    page_title="Intelligent Hiring System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Cloud-Native Paths =====
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static" / "charts"
CACHE_DIR = BASE_DIR / "cache" / "embeddings"
UPLOAD_DIR = BASE_DIR / "uploads"
for d in [STATIC_DIR, CACHE_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===== Custom CSS - Glassmorphism V0-style =====
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
        background-size: 200% 200%;
        animation: gradientShift 15s ease infinite;
    }
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Hero header */
    .hero {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818CF8, #C084FC, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero p { color: #CBD5E1; font-size: 1.1rem; margin-top: 0.5rem; }
    
    /* Glassmorphism candidate cards */
    .candidate-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .candidate-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366F1, #A855F7, #EC4899);
    }
    .candidate-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99,102,241,0.5);
        box-shadow: 0 20px 40px rgba(99,102,241,0.2);
    }
    .candidate-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 0.5rem;
    }
    .candidate-score {
        display: inline-block;
        background: linear-gradient(135deg, #6366F1, #A855F7);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    .candidate-meta {
        color: #94A3B8;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    .skill-tag {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        color: #C7D2FE;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
        font-size: 0.75rem;
        margin: 0.2rem 0.2rem 0 0;
    }
    
    /* Metrics cards */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1, #A855F7);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 14px rgba(99,102,241,0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.6);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.3);
        border: 2px dashed rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ===== Hero Header =====
st.markdown("""
<div class="hero">
    <h1>🎯 Intelligent Hiring System</h1>
    <p>AI-Powered Resume Ranking with Semantic Understanding</p>
</div>
""", unsafe_allow_html=True)

# ===== Session State =====
if "excel_path" not in st.session_state:
    st.session_state.excel_path = None
if "results" not in st.session_state:
    st.session_state.results = None

# ===== STAGE 1: Upload =====
st.markdown("### 📤 Stage 1 — Upload Resume Bundle")
uploaded_zip = st.file_uploader(
    "Drop a ZIP file containing PDF resumes",
    type=["zip"],
    help="Maximum 200 MB. PDFs will be parsed automatically.",
)

if uploaded_zip and st.button("🚀 Process Resumes", use_container_width=True):
    zip_path = UPLOAD_DIR / uploaded_zip.name
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())
    
    with st.spinner("Extracting & parsing PDFs..."):
        excel_path = process_pdfs_to_excel(str(zip_path), output_dir=str(UPLOAD_DIR))
        st.session_state.excel_path = excel_path
    st.success(f"✅ Processed successfully — Excel ready at `{excel_path}`")

# ===== STAGE 2: Job Criteria =====
if st.session_state.excel_path:
    st.markdown("---")
    st.markdown("### 🎛️ Stage 2 — Define Job Criteria")
    
    with st.sidebar:
        st.markdown("## ⚙️ Ranking Settings")
        job_description = st.text_area(
            "Job Description",
            value="AI Engineer with Machine Learning and NLP experience",
            height=120,
        )
        top_n = st.slider("Top N Candidates", 1, 20, 5)
        min_exp = st.number_input("Minimum Experience (years)", 0, 30, 2)
        skills_input = st.text_input(
            "Required Skills (comma-separated, prefix with ! for mandatory)",
            value="!python, machine learning, sql",
        )
        skills = [s.strip() for s in skills_input.split(",") if s.strip()]
        
        run_btn = st.button("🔍 Rank Candidates", use_container_width=True)
    
    # ===== Cached Embeddings =====
    @st.cache_data(show_spinner=False)
    def cached_ranking(excel_path: str, jd: str, top_n: int, min_exp: int, skills_tuple: tuple):
        """Cache by JD + filters hash to avoid recomputing embeddings."""
        ui_inputs = {
            "job_description": jd,
            "top_n": top_n,
            "min_exp": min_exp,
            "skills": list(skills_tuple),
        }
        return ranking_engine(
            excel_path,
            ui_inputs,
            charts_dir=str(STATIC_DIR),
            cache_dir=str(CACHE_DIR),
        )
    
    if run_btn:
        with st.spinner("🧠 Computing semantic embeddings..."):
            results = cached_ranking(
                st.session_state.excel_path,
                job_description,
                top_n,
                min_exp,
                tuple(skills),
            )
            st.session_state.results = results

# ===== STAGE 3: Results =====
if st.session_state.results:
    results = st.session_state.results
    st.markdown("---")
    st.markdown("### 🏆 Stage 3 — Ranked Candidates")
    
    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Resumes", results["metrics"]["n_samples"])
    c2.metric("Eligible", len(results["top_candidates"]) + len(results.get("rest", [])))
    c3.metric("Top Picks", len(results["top_candidates"]))
    c4.metric("Accuracy", f"{results['metrics'].get('accuracy', 0)*100:.1f}%")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🥇 Top Candidates", "📊 Analytics", "📋 Full Pool"])
    
    with tab1:
        for i, c in enumerate(results["top_candidates"], 1):
            skills_html = "".join([f'<span class="skill-tag">{s}</span>' for s in c.get("skills", [])[:6]])
            st.markdown(f"""
            <div class="candidate-card">
                <div style="display:flex; justify-content:space-between; align-items:start;">
                    <div>
                        <div class="candidate-name">#{i} — {c.get('name', 'N/A')}</div>
                        <div class="candidate-meta">
                            📧 {c.get('email', 'N/A')} &nbsp;•&nbsp; 
                            📱 {c.get('phone', 'N/A')} &nbsp;•&nbsp; 
                            💼 {c.get('exp_years', 0)} yrs
                        </div>
                        <div style="margin-top:0.75rem;">{skills_html}</div>
                        <div style="margin-top:0.75rem; color:#CBD5E1; font-size:0.9rem; font-style:italic;">
                            "{c.get('reason', '')}"
                        </div>
                    </div>
                    <span class="candidate-score">{c.get('score', 0):.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        cols = st.columns(2)
        charts = results.get("charts", {})
        for i, (name, path) in enumerate(charts.items()):
            full_path = BASE_DIR / path
            if full_path.exists():
                with cols[i % 2]:
                    st.image(str(full_path), caption=name.replace("_", " ").title(), use_container_width=True)
    
    with tab3:
        rest = results.get("rest", [])
        if rest:
            df = pd.DataFrame(rest)
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.info("No additional candidates in the pool.")

# ===== Footer =====
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748B; padding:1rem;'>"
    "Built with 🧠 BGE-Small Embeddings • Deployed on Streamlit Cloud"
    "</div>",
    unsafe_allow_html=True,
)