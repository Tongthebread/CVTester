import pytest

from cv_stress_test.demo import demo_requirements


def test_mixed_format_preserves_quotes_and_deduplicates():
    job = "# Developer\nRequirements:\n- Python\n• SQL\n1. Docker\n(2) Git\nPython\nBuild APIs. Test services."
    result = demo_requirements(job)
    assert [item.text for item in result.items] == ["Python", "SQL", "Docker", "Git", "Build APIs.", "Test services."]
    assert all(item.job_quote in job for item in result.items)


def test_limit_and_chinese_sentences():
    assert len(demo_requirements("\n".join(f"Skill {i}" for i in range(20))).items) == 12
    assert len(demo_requirements("熟悉Python。熟悉SQL。").items) == 2


@pytest.mark.parametrize("text", ["# Title\nRequirements:\n- \n•", "...", ""])
def test_no_usable_requirements(text):
    with pytest.raises(ValueError, match="Add job requirements"):
        demo_requirements(text)
