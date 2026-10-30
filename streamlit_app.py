"""Launch from the repository: streamlit run streamlit_app.py"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from langchain_core.documents import Document
from pydantic import ValidationError

from cv_stress_test.documents import MAX_BYTES, read_document
from cv_stress_test.service import assess_cv

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="CVTester · Know your evidence", page_icon="📄", layout="wide")


def invalidate():
    st.session_state.pop("result", None)


def load_example():
    st.session_state["cv_text"] = (ROOT / "examples/cv.md").read_text()
    st.session_state["job_text"] = (ROOT / "examples/job.md").read_text()
    st.session_state["input_method"] = "Paste text"
    st.session_state["mode"] = "Offline demo"
    invalidate()


with st.sidebar:
    st.title("CVTester")
    st.caption("A rehearsal before you apply.")
    mode = st.radio("Assessment mode", ["Offline demo", "AI assessment"], key="mode", on_change=invalidate)
    demo = mode == "Offline demo"
    api_key = ""
    if demo:
        st.info("Free, offline keyword matching. Try the workflow; this is not an AI review.")
    else:
        api_key = st.text_input("OpenAI API key", type="password", key="api_key", on_change=invalidate,
                                help="Optional if OPENAI_API_KEY is already set in your terminal.")
        st.caption("AI mode sends CV text and the job description to OpenAI and uses paid API credits. The key is not saved to disk.")
    st.divider()
    st.button("Load sample CV & job", on_click=load_example, use_container_width=True)
    st.caption("PDF, TXT or Markdown · up to 10 MB. Reports stay in this session unless you download them.")

st.caption("YOUR EXPERIENCE. YOUR EVIDENCE.")
st.title("Put your CV to the test.")
st.write("See where your experience supports a role, where claims need detail, and what an interviewer might ask.")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("1. Your CV")
    method = st.radio("CV input", ["Paste text", "Upload file"], horizontal=True,
                      key="input_method", on_change=invalidate)
    uploaded = None
    cv_text = ""
    if method == "Upload file":
        uploaded = st.file_uploader("Choose your CV", type=["pdf", "txt", "md"], on_change=invalidate)
        st.caption("Use a text-based PDF. Scanned PDFs need OCR first.")
    else:
        cv_text = st.text_area("CV text", height=300, key="cv_text", on_change=invalidate,
                               placeholder="Paste your experience, projects and skills…")
with right:
    st.subheader("2. Target role")
    job = st.text_area("Job description", height=350, key="job_text", on_change=invalidate,
                       placeholder="Paste the job description or a list of role requirements…")
    if demo:
        st.caption("Paste plain text, numbered requirements, or bullet lists. Demo mode checks up to 12 sentences or lines.")

if st.button("Stress-test my CV", type="primary", use_container_width=True):
    invalidate()
    try:
        with st.spinner("Reading requirements, checking evidence, and preparing feedback…"):
            if method == "Upload file":
                if uploaded is None:
                    raise ValueError("Upload a CV first, or switch to Paste text.")
                if uploaded.size > MAX_BYTES:
                    raise ValueError("Maximum file size is 10 MB.")
                with TemporaryDirectory(prefix="cvtester-") as folder:
                    path = Path(folder) / Path(uploaded.name).name
                    path.write_bytes(uploaded.getvalue())
                    documents = read_document(path)
            else:
                if not cv_text.strip():
                    raise ValueError("Paste your CV or load the sample to get started.")
                documents = [Document(page_content=cv_text, metadata={"source": "Pasted CV", "page": 1})]
            st.session_state["result"] = assess_cv(documents, job, demo=demo, api_key=api_key)
    except ValidationError:
        st.error("The model returned an invalid report. Please try again.")
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Could not complete the review. Check the document format, connection, API key, model access and quota.")

if "result" in st.session_state:
    result = st.session_state["result"]
    st.divider()
    st.subheader("Your evidence review")
    if result["mode"] == "demo":
        st.warning("Demo result — keyword matching only. Use AI assessment for substantive coaching.")
    st.caption("Evidence not found does not mean you lack a skill. Review interpretations and suggested edits before using them.")
    findings = result["assessment"]["findings"]
    labels = {"supported": "Direct evidence", "weak_evidence": "Needs detail", "not_found": "Evidence not found"}
    for column, (status, label) in zip(st.columns(3), labels.items()):
        column.metric(label, sum(f["status"] == status for f in findings))
    feedback, report = st.tabs(["Requirement by requirement", "Full report"])
    with feedback:
        for finding in sorted(findings, key=lambda f: f["requirement_index"]):
            index = finding["requirement_index"]
            requirement = result["requirements"][index]
            with st.expander(f"{labels[finding['status']]} · {requirement['text']}", expanded=True):
                st.text(finding["explanation"])
                st.caption("From the job description")
                st.text(requirement["job_quote"])
                evidence = {item["chunk_id"]: item for item in result["evidence"][index]}
                for citation in finding["citations"]:
                    item = evidence[citation["chunk_id"]]
                    st.caption(f"CV evidence · {item['source']} · page {item['page']}")
                    st.text(citation["quote"])
                st.markdown("**Be ready to answer**")
                st.text(finding["interview_question"])
                st.markdown("**Make this stronger**")
                st.text(finding["improvement"])
    with report:
        st.text(result["report"])
    download, evidence_download = st.columns(2)
    download.download_button("Download report", result["report"], "cv-review.md", "text/markdown", use_container_width=True)
    evidence_download.download_button("Download evidence (JSON)", json.dumps(result, indent=2, ensure_ascii=False),
                                      "cv-evidence.json", "application/json", use_container_width=True)
