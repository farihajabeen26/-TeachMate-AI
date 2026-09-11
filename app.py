import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# --- PATH & ENV SETUP ---
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env")

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TeachMate AI | Smart Teaching Suite",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 700; color: #4F46E5; margin-bottom: 0px; }
    .sub-title { font-size: 1rem; color: #6B7280; margin-bottom: 25px; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Safe API key retrieval
api_key = None
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

try:
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        model_name = st.secrets.get("GROQ_MODEL", model_name)
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ Groq API Key missing! Set it in your `.env` file or Streamlit Cloud Secrets.")
    st.stop()

client = Groq(api_key=api_key)

# --- RAG ENGINE ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

def extract_pdf_text(uploaded_files):
    text_chunks = []
    for file in uploaded_files:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text = page.get_text()
            if text.strip():
                chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 50]
                text_chunks.extend(chunks)
    return text_chunks

def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def retrieve_context(query, index, chunks, top_k=2):
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return "\n\n".join(results)

# NEW VERSION WITH TENACITY RETRY LOGIC:
@retry(wait=wait_exponential(min=3, max=15), stop=stop_after_attempt(4))
def run_agent(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content
    

# --- SIDEBAR UI ---

with st.sidebar:
    st.markdown("## 🎓 TeachMate AI")
    st.caption("Your Syllabus-Aware AI Teaching Assistant")
    st.divider()

    st.subheader("1. Document Hub")
    uploaded_files = st.file_uploader(
        "Upload Syllabi / Textbooks",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload curriculum documents to align generations with your syllabus."
    )

    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} PDF(s) Attached")

    st.divider()
    st.caption(f"**Engine:** Groq (`{model_name}`)")

# --- MAIN DASHBOARD UI ---
st.markdown('<p class="main-title">🎓 TeachMate AI Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Generate multi-agent study packs, quizzes, and lesson plans aligned to your syllabus.</p>', unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("2. Lesson Configuration")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        grade_level = st.selectbox("Grade Level", ["Class 12", "Class 11", "Class 10", "Class 9", "Class 8", "Class 7", "Class 6"])
    with col2:
        subject = st.text_input("Subject", value="BIOLOGY")
    with col3:
        topic_input = st.text_input("Target Topic / Instruction", value="Prepare tomorrow's lesson for Class 9 Biology on Photosynthesis.")

    generate_btn = st.button("🚀 Generate Complete Study Pack", type="primary")

# --- EXECUTION & DISPLAY ---
if generate_btn:
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF curriculum document in the left sidebar.")
        st.stop()

    combined_prompt = f"Grade: {grade_level}, Subject: {subject}, Request: {topic_input}"

    with st.status("⚙️ Processing Curriculum & Executing Agents...", expanded=True) as status:
        st.write("📄 Indexing PDF text into FAISS vector database...")
        chunks = extract_pdf_text(uploaded_files)
        index, embeddings = create_faiss_index(chunks)
        retrieved_context = retrieve_context(combined_prompt, index, chunks)
        time.sleep(1)

        st.write("🔍 **Stage 3:** Analyzing syllabus alignment...")
        syllabus_prompt = f"Extract key subtopics and learning outcomes for: {combined_prompt}\nContext:\n{retrieved_context}"
        syllabus_out = run_agent("You are an expert curriculum evaluator.", syllabus_prompt)
        time.sleep(2)

        st.write("📝 **Stage 5:** Drafting structured lesson plan...")
        lesson_prompt = f"Draft an interactive lesson plan based on:\n{syllabus_out}\nContext:\n{retrieved_context}"
        lesson_out = run_agent("You are a Master Educator.", lesson_prompt)
        time.sleep(2)

        st.write("❓ **Stage 6:** Generating quizzes and assessments...")
        assessment_prompt = f"Generate 3 MCQs, 2 short-answer questions, and interactive class activities based on:\n{lesson_out}"
        assessment_out = run_agent("You are an Assessment Specialist.", assessment_prompt)
        time.sleep(2)

        st.write("✨ **Stage 8:** Finalizing Study Pack formatting...")
        final_pack_prompt = f"Format the following cleanly into Markdown with headers:\n\n### LESSON PLAN\n{lesson_out}\n\n### QUIZ & ACTIVITIES\n{assessment_out}"
        final_output = run_agent("You are a Senior Academic Reviewer.", final_pack_prompt)

        status.update(label="✅ Study Pack Generation Complete!", state="complete", expanded=False)

    # Output presented in structured tabs
    st.subheader("📚 Generated Study Pack")
    tab1, tab2, tab3 = st.tabs(["📖 Lesson Plan", "🎯 Quiz & Activities", "📥 Export Options"])

    with tab1:
        st.markdown(lesson_out)

    with tab2:
        st.markdown(assessment_out)

    with tab3:
        st.success("Your complete study pack is ready for download.")
        st.download_button(
            label="💾 Download Full Study Pack (.md)",
            data=final_output,
            file_name=f"TeachMate_{subject}_{grade_level}.md",
            mime="text/markdown",
            use_container_width=True
        )