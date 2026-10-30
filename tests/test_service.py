import pytest
from langchain_core.documents import Document

from cv_stress_test.service import assess_cv


@pytest.mark.parametrize("job, message", [(" ", "Add a target"), ("x" * 100001, "exceeds")])
def test_invalid_job_is_rejected(job, message):
    with pytest.raises(ValueError, match=message):
        assess_cv([Document(page_content="Python")], job, demo=True)


def test_missing_key_is_actionable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        assess_cv([Document(page_content="Python")], "Python required")


@pytest.mark.parametrize("job", ["Python required", "1. Python required\n2) SQL experience",
                                  "• Python required\n* SQL experience",
                                  "Python required. SQL experience is useful."])
def test_demo_accepts_pasted_job_descriptions(job):
    result = assess_cv([Document(page_content="Python", metadata={"source": "CV", "page": 1})], job, demo=True)
    assert result["assessment"]["findings"]
    assert all(item["job_quote"] in job for item in result["requirements"])
