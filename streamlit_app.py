"""
Resume Screener — Streamlit UI (v2: dynamic ranking + styled UI)
pip install streamlit instructor groq pydantic python-dotenv pypdf python-docx pandas
Run: python3 -m streamlit run streamlit_app.py
"""
import os, json, re, time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import instructor
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document

load_dotenv()
MODEL = "openai/gpt-oss-120b"

st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
    padding: 1.75rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
}
.main-header h1 { color: white; margin: 0; font-size: 1.9rem; }
.main-header p { color: #E0E7FF; margin: 0.25rem 0 0 0; font-size: 0.95rem; }

.candidate-card {
    background: #ffffff;
    border: 1px solid #E5E7EB;
    border-left: 6px solid #6366F1;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.candidate-card.low { border-left-color: #EF4444; }
.candidate-card.high { border-left-color: #10B981; }

.score-badge {
    display: inline-block;
    padding: 0.15rem 0.7rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    color: white;
}
.score-high { background: #10B981; }
.score-mid { background: #F59E0B; }
.score-low { background: #EF4444; }

.skill-pill {
    display: inline-block;
    background: #EEF2FF;
    color: #4338CA;
    padding: 0.1rem 0.55rem;
    border-radius: 999px;
    font-size: 0.78rem;
    margin: 0.1rem;
}
.skill-pill.missing { background: #FEF2F2; color: #B91C1C; }

.stButton>button[kind="primary"] {
    background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
    border: none;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class JobD(BaseModel):
    role: str
    required_skills: list[str]
    preferred_skills: list[str]
    minimum_experience: float | None
    education_requirements: list[str]
    responsibilities: list[str]


class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []


class MatchResult(BaseModel):
    candidate_name: str | None
    score: float
    matching_skills: list[str]
    missing_skills: list[str]
    experience_requirement_met: bool
    verdict: str


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def get_client(api_key: str):
    return instructor.from_groq(Groq(api_key=api_key))


_WAIT_SECONDS_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def extract(client, model_cls, prompt: str, max_attempts: int = 6):
    """Call the LLM with automatic backoff on 429 (rate limit) errors.
    Groq's free tier has a low tokens-per-minute cap, so retrying with the
    server-suggested wait time (or exponential backoff as a fallback) is
    what actually recovers from this, rather than instructor's built-in
    max_retries (which only retries schema-validation failures)."""
    last_err = None
    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(
                model=MODEL,
                response_model=model_cls,
                max_retries=2,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            msg = str(e)
            last_err = e
            is_rate_limit = "429" in msg or "rate_limit" in msg.lower()
            if not is_rate_limit or attempt == max_attempts - 1:
                raise
            match = _WAIT_SECONDS_RE.search(msg)
            wait = float(match.group(1)) + 0.5 if match else (2 ** attempt)
            time.sleep(wait)
    raise last_err


def read_resume(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = BytesIO(uploaded_file.getvalue())
    if name.endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(data).pages)
    if name.endswith(".docx"):
        doc = Document(data)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return ""


def screen_one(client, uploaded_file, job: JobD) -> dict:
    text = read_resume(uploaded_file)
    if not text.strip():
        return {"error": f"{uploaded_file.name}: no extractable text (likely a scanned/image PDF)"}
    try:
        resume = extract(client, Resume, f"Extract structured info from this resume:\n\n{text}")
        result = extract(
            client,
            MatchResult,
            f"Compare this resume:\n{resume.model_dump_json()}\n\n"
            f"Against this job:\n{job.model_dump_json()}\n\n"
            "Score 0-100 and give matching/missing skills, whether experience is met, and a short verdict.",
        )
        data = result.model_dump()
        if not data.get("candidate_name"):
            data["candidate_name"] = resume.name or uploaded_file.name
        return data
    except Exception as e:
        return {"error": f"{uploaded_file.name}: {e}"}


def score_class(score: float) -> str:
    if score >= 70:
        return "high"
    if score >= 40:
        return "mid"
    return "low"


def render_card(r: dict, variant: str):
    cls = score_class(r["score"])
    badge_cls = f"score-{cls}"
    card_cls = "high" if variant == "top" else "low"
    matching = "".join(f"<span class='skill-pill'>{s}</span>" for s in r["matching_skills"][:8]) or "—"
    missing = "".join(f"<span class='skill-pill missing'>{s}</span>" for s in r["missing_skills"][:8]) or "—"
    exp_icon = "✅ Met" if r["experience_requirement_met"] else "❌ Not met"
    st.markdown(f"""
    <div class="candidate-card {card_cls}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong style="font-size:1.05rem;">{r['candidate_name']}</strong>
            <span class="score-badge {badge_cls}">{r['score']:.0f}%</span>
        </div>
        <p style="margin:0.4rem 0; color:#374151; font-size:0.9rem;">{r['verdict']}</p>
        <p style="margin:0.2rem 0; font-size:0.85rem;"><strong>Experience:</strong> {exp_icon}</p>
        <p style="margin:0.2rem 0; font-size:0.85rem;"><strong>Matching:</strong><br>{matching}</p>
        <p style="margin:0.2rem 0; font-size:0.85rem;"><strong>Missing:</strong><br>{missing}</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>📄 Resume Screener</h1>
    <p>Paste a job description, upload resumes, get ranked candidates — powered by AI.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("⚙️ Setup")
    api_key = st.text_input("Groq API key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    max_workers = st.slider("Parallel requests", 1, 10, 2)
    st.caption("⚠️ Groq's free tier is capped at 8,000 tokens/minute. "
               "Keep this low (1–2) to avoid rate-limit errors — the app "
               "will auto-retry with backoff, but fewer parallel calls means fewer retries.")
    st.divider()
    st.caption("Built with Groq + Instructor + Streamlit")

col_jd, col_up = st.columns([1, 1])
with col_jd:
    job_description = st.text_area("📋 Job description", height=220, placeholder="Paste the JD here...")
with col_up:
    uploaded_files = st.file_uploader(
        "📁 Upload resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")

run = st.button(
    "🚀 Screen candidates", type="primary",
    disabled=not (api_key and job_description and uploaded_files),
)

if run:
    client = get_client(api_key)
    with st.spinner("Parsing job description..."):
        job = extract(client, JobD, f"Extract structured job info:\n\n{job_description}")

    progress = st.progress(0.0, text="Screening resumes...")
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(screen_one, client, f, job): f for f in uploaded_files}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            progress.progress(done / len(futures), text=f"Screened {done}/{len(futures)}")
            (errors if "error" in r else results).append(r)

    progress.empty()
    st.session_state["results"] = sorted(results, key=lambda r: r["score"], reverse=True)
    st.session_state["errors"] = errors
    st.session_state["job"] = job.model_dump()
    st.session_state["total_uploaded"] = len(uploaded_files)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if "results" in st.session_state:
    job = st.session_state["job"]
    results = st.session_state["results"]
    errors = st.session_state.get("errors", [])
    total = st.session_state.get("total_uploaded", len(results))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Uploaded", total)
    m2.metric("Scored", len(results))
    m3.metric("Failed", len(errors))
    m4.metric("Avg score", f"{(sum(r['score'] for r in results) / len(results)):.0f}%" if results else "—")

    with st.expander("📋 Parsed job requirements"):
        st.json(job)

    if errors:
        st.warning(f"⚠️ {len(errors)} file(s) skipped: " + "; ".join(e["error"] for e in errors))

    if results:
        # dynamic top/bottom count — scales with how many candidates you actually have
        n = len(results)
        if n <= 2:
            # slider needs min != max; with 1-2 results there's no meaningful
            # choice anyway, so just show everything as "top" and skip "lowest"
            top_n = n
        else:
            default_top = max(1, min(3, n // 2)) if n < 6 else 3
            top_n = st.slider("How many top candidates to highlight?", 1, n - 1, min(default_top, n - 1))
        show_lowest = n > top_n  # only makes sense if there's a distinct remainder

        st.subheader("Ranked candidates")
        df = pd.DataFrame([
            {
                "Candidate": r["candidate_name"],
                "Score": r["score"],
                "Experience Met": "✅" if r["experience_requirement_met"] else "❌",
                "Matching Skills": ", ".join(r["matching_skills"]),
                "Missing Skills": ", ".join(r["missing_skills"]),
            }
            for r in results
        ])
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d%%")},
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 🏆 Top {top_n}")
            for r in results[:top_n]:
                render_card(r, "top")
        with col2:
            if show_lowest:
                bottom_n = min(top_n, n - top_n)
                st.markdown(f"#### ⚠️ Lowest {bottom_n}")
                for r in results[-bottom_n:]:
                    render_card(r, "bottom")
            else:
                st.info("Not enough candidates yet for a separate 'lowest' group.")

        st.divider()
        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Download JSON", data=json.dumps(results, indent=2),
            file_name="screening_results.json", mime="application/json", use_container_width=True,
        )
        d2.download_button(
            "⬇️ Download CSV", data=df.to_csv(index=False),
            file_name="screening_results.csv", mime="text/csv", use_container_width=True,
        )
    else:
        st.info("No resumes scored successfully.")