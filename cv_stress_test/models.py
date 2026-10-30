from typing import Literal

from pydantic import BaseModel, Field


class Requirement(BaseModel):
    text: str = Field(min_length=1)
    job_quote: str = Field(min_length=1, description="Exact quote from the job description")


class Requirements(BaseModel):
    items: list[Requirement] = Field(min_length=1, max_length=12)


class Citation(BaseModel):
    chunk_id: str
    quote: str = Field(min_length=1)


class Finding(BaseModel):
    requirement_index: int
    status: Literal["supported", "weak_evidence", "not_found"]
    explanation: str
    citations: list[Citation]
    interview_question: str
    improvement: str


class Assessment(BaseModel):
    findings: list[Finding]
