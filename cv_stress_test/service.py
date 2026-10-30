"""Shared assessment entry point for the CLI and local UI."""
import os

from .documents import MAX_CHARS, chunk_documents
from .workflow import build_graph


def assess_cv(documents, job, *, demo=False, api_key=None):
    if not job.strip():
        raise ValueError("Add a target job description.")
    if len(job) > MAX_CHARS:
        raise ValueError("Job description exceeds 100,000 characters.")
    if sum(len(doc.page_content) for doc in documents) > MAX_CHARS:
        raise ValueError("CV exceeds 100,000 characters.")
    chunks = chunk_documents(documents)
    if demo:
        from .demo import DemoEmbeddings

        embeddings, model = DemoEmbeddings(), None
    else:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Add an OpenAI API key or select Offline demo.")
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(api_key=key, model=os.getenv("CV_EMBEDDING_MODEL", "text-embedding-3-small"))
        model = ChatOpenAI(api_key=key, model=os.getenv("CV_MODEL", "gpt-4.1-mini"), timeout=60, max_retries=2)
    return {"mode": "demo" if demo else "live", **build_graph(chunks, embeddings, model, demo).invoke({"job": job})}
