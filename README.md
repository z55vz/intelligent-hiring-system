# 🎯 Intelligent Hiring System (IHS)
### [cite_start]"Beyond Keywords: Understanding the Context of Talent" [cite: 104]

The **Intelligent Hiring System** is an AI-powered platform designed to help HR teams identify the best candidates through **Deep Semantic Analysis**. [cite_start]Instead of simply looking for exact words, our system understands the meaning behind the experience and skills listed in a resume. [cite: 106, 118]

---

## 🌟 Why this system?
[cite_start]Traditional hiring systems (ATS) often miss great candidates because they use different terminology. [cite: 104] [cite_start]Our system solves this by using **Semantic Search**, allowing it to recognize that a "Machine Learning Expert" and an "AI Specialist" are conceptually the same. 

---

## [cite_start]🛠️ How it Works (The Pipeline) [cite: 119]
1. [cite_start]**Data Ingestion:** Upload a ZIP file of PDF resumes. [cite: 120]
2. [cite_start]**Preprocessing:** The system extracts text, contact info (Email/Phone), and experience years automatically. [cite: 121]
3. [cite_start]**Semantic Embedding:** Texts are converted into high-dimensional vectors using the **BAAI/bge-small-en-v1.5** model (also supports **Instructor-XL** and **E5-Large**). [cite: 109, 115]
4. [cite_start]**Hybrid Scoring:** Candidates are ranked based on a weighted formula: **65% Semantic Match**, **20% Experience**, and **15% Skill Match**. [cite: 125]
5. [cite_start]**Interactive Dashboard:** View the top candidates, their AI-generated explanations, and visual performance charts. [cite: 127, 128, 131]

---

## [cite_start]🏗️ Project Architecture [cite: 108]
The project is built with a **Modular Design** to ensure speed and scalability:
* **`app.py`**: The main Streamlit interface (The Frontend).
* **`backend/preprocessor.py`**: Handles PDF parsing and data cleaning.
* **`backend/ranking_engine.py`**: The AI "brain" that calculates scores and rankings.
* **`.embedding_cache/`**: Speeds up the system by saving processed data. [cite: 123]

---

## [cite_start]📊 Performance & Evaluation [cite: 129]
Our system includes a built-in evaluation module that monitors:
* [cite_start]**Accuracy & F1-Score:** To ensure ranking quality. 
* **Confusion Matrix:** To visualize how well the AI distinguishes between "Strong" and "Reject" candidates. 
* **Top-N Accuracy:** Measuring if the best candidates are truly landing in the top spots.

---

## ⚙️ Installation
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/intelligent-hiring.git`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

---

## 👥 The Team
* [cite_start]**Abdulrahman Ageeli** (ID: 445004733) [cite: 4]
* **Al-Waleed Al-Suwaiheri** (ID: 445003109) [cite: 4]
* [cite_start]**Sultan Al-Otaibi** (ID: 444002637) [cite: 4]

[cite_start]*Designed for the AI System Design Course - Umm Al-Qura University (2026).* [cite: 5]
