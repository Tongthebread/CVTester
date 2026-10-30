import copy
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

from cv_stress_test.demo import DemoEmbeddings
from cv_stress_test.documents import chunk_documents, read_document
from cv_stress_test.models import Assessment, Requirements
from cv_stress_test.workflow import build_graph, validate_assessment

ROOT = Path(__file__).resolve().parents[1]


def test_demo_graph_has_evidence_and_missing_skill():
    chunks = chunk_documents(read_document(ROOT / "examples/cv.md"))
    job = (ROOT / "examples/job.md").read_text()
    result = build_graph(chunks, DemoEmbeddings(), demo=True).invoke({"job": job})
    assert len(result["assessment"]["findings"]) == 5
    assert result["assessment"]["findings"][3]["status"] == "not_found"
    assert "DEMO" in result["report"]
    assert "cv-1" in result["report"]


@pytest.fixture
def grounded():
    requirements = [{"text": "Python", "job_quote": "Python"}]
    evidence = {0: [{"chunk_id": "cv-1", "text": "Built a Python API."}]}
    assessment = {"findings": [{"requirement_index": 0, "status": "supported",
                  "explanation": "Built an API", "citations": [{"chunk_id": "cv-1", "quote": "Python API"}],
                  "interview_question": "How?", "improvement": "Add verified outcomes"}]}
    return assessment, requirements, evidence


@pytest.mark.parametrize("mutation", ["invented_quote", "wrong_chunk", "no_citation", "duplicate", "missing", "wrong_index"])
def test_invalid_model_evidence_is_rejected(grounded, mutation):
    assessment, requirements, evidence = copy.deepcopy(grounded)
    finding = assessment["findings"][0]
    if mutation == "invented_quote":
        finding["citations"][0]["quote"] = "Led a team of 20"
    elif mutation == "wrong_chunk":
        finding["citations"][0]["chunk_id"] = "another-cv"
    elif mutation == "no_citation":
        finding["citations"] = []
    elif mutation == "duplicate":
        assessment["findings"].append(finding)
    elif mutation == "missing":
        assessment["findings"] = []
    else:
        finding["requirement_index"] = 99
    with pytest.raises(ValueError):
        validate_assessment(assessment, requirements, evidence)


def test_valid_evidence(grounded):
    assert validate_assessment(*grounded)[0].status == "supported"


def test_document_validation_and_chunk_provenance(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("  ")
    with pytest.raises(ValueError, match="empty"):
        read_document(empty)
    chunks = chunk_documents([Document(page_content="A" * 1601, metadata={"source": "cv", "page": 2})])
    assert len(chunks) == 3
    assert chunks[1].metadata["offset"] == 800
    assert chunks[1].metadata["page"] == 2
    assert chunks[0].page_content[800:] == chunks[1].page_content[:200]


def test_live_branch_with_stub_model():
    class StubModel:
        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, messages):
            assert "untrusted" in messages[0].content
            if self.schema is Requirements:
                return Requirements(items=[{"text": "Python", "job_quote": "Python"}])
            return Assessment(findings=[{"requirement_index": 0, "status": "supported",
                "explanation": "Specific implementation evidence", "citations": [{"chunk_id": "cv-1", "quote": "Python API"}],
                "interview_question": "What design choices did you make?", "improvement": "Add [verified outcome]"}])

    chunks = chunk_documents([Document(page_content="Built a Python API.", metadata={"source": "cv", "page": 1})])
    result = build_graph(chunks, DemoEmbeddings(), StubModel()).invoke({"job": "Python"})
    assert "Direct evidence" in result["report"]
    assert "DEMO" not in result["report"]


def test_cli_writes_outputs_and_preserves_existing_report(tmp_path):
    output = tmp_path / "report.md"
    command = [sys.executable, "-m", "cv_stress_test.cli", "--demo", "--cv", str(ROOT / "examples/cv.md"),
               "--job", str(ROOT / "examples/job.md"), "--out", str(output)]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert output.with_suffix(".json").exists()
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert second.returncode != 0
    assert "already exists" in second.stderr
