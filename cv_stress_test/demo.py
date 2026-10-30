"""Deterministic offline smoke test; does not simulate semantic understanding."""
import hashlib
import re

from langchain_core.embeddings import Embeddings

from .models import Assessment, Requirements


def demo_requirements(job):
    """Split ordinary text and lists while preserving exact source quotes."""
    items = []
    seen = set()
    for line in job.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.endswith((":", "：")):
            continue
        line = re.sub(r"^(?:[-*+•–—]|\d+[.)]|\(\d+\))(?:\s+|$)", "", line).strip()
        # shortcut: sentence/line splitting, not semantic requirement extraction.
        for text in re.split(r"(?<=[.!?])\s+|(?<=[。！？])", line):
            text = text.strip()
            if not any(char.isalnum() for char in text) or text in seen:
                continue
            seen.add(text)
            items.append({"text": text, "job_quote": text})
            if len(items) == 12:
                return Requirements(items=items)
    if not items:
        raise ValueError("Add job requirements as sentences or a list; headings alone are not enough.")
    return Requirements(items=items)


def tokens(text):
    return set(re.findall(r"[\w+#]+", text.lower()))


class DemoEmbeddings(Embeddings):
    def embed_query(self, text):
        vector = [0.0] * 256
        for token in tokens(text):
            vector[int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 256] += 1
        return vector

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def demo_assessment(state):
    findings = []
    for index, requirement in enumerate(state["requirements"]):
        words = tokens(requirement["text"])
        match = next(((item, line) for item in state["evidence"][index]
                      for line in item["text"].splitlines() if line.strip() and words & tokens(line)), None)
        findings.append({
            "requirement_index": index,
            "status": "weak_evidence" if match else "not_found",
            "explanation": "Demo found a word overlap; a person must check relevance." if match else
                           "Demo found no word overlap. Semantic matches may have been missed.",
            "citations": [{"chunk_id": match[0]["chunk_id"], "quote": match[1]}] if match else [],
            "interview_question": f"What did you personally deliver related to {requirement['text']}?",
            "improvement": "If accurate, add [specific action], [scope], and [verified outcome].",
        })
    return Assessment(findings=findings)
