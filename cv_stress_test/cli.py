import argparse
import json
import os
from pathlib import Path

from .documents import chunk_documents, read_document
from .workflow import build_graph


def main():
    parser = argparse.ArgumentParser(description="Stress-test your CV against a target job description")
    parser.add_argument("--cv", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/report.md"))
    parser.add_argument("--demo", action="store_true", help="Offline keyword smoke test; no API key required")
    args = parser.parse_args()
    json_path = args.out.with_suffix(".json")
    if args.out.suffix.lower() != ".md":
        parser.error("--out must end in .md")
    if args.out.exists() or json_path.exists():
        parser.error("Output already exists; choose a new --out path to preserve earlier reports")
    try:
        chunks = chunk_documents(read_document(args.cv))
        job = "\n\n".join(doc.page_content for doc in read_document(args.job))
        if args.demo:
            from .demo import DemoEmbeddings

            embeddings, model = DemoEmbeddings(), None
            if not any(line.strip().startswith("- ") for line in job.splitlines()):
                raise ValueError("Demo job descriptions need requirements on '- ' bullet lines")
        else:
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("Set OPENAI_API_KEY or run with --demo")
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(model=os.getenv("CV_EMBEDDING_MODEL", "text-embedding-3-small"))
            model = ChatOpenAI(model=os.getenv("CV_MODEL", "gpt-4.1-mini"), timeout=60, max_retries=2)
        result = build_graph(chunks, embeddings, model, args.demo).invoke({"job": job})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result["report"], encoding="utf-8")
        json_path.write_text(json.dumps({"mode": "demo" if args.demo else "live", **result},
                                       indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        # Avoid printing provider errors, which can include submitted CV text or credentials.
        if isinstance(exc, (ValueError, FileNotFoundError)):
            parser.exit(1, f"Cannot complete assessment: {exc}\n")
        parser.exit(1, f"Assessment failed ({type(exc).__name__}). Check inputs, connectivity, model access and quota.\n")
    print(f"Report: {args.out.resolve()}\nEvidence: {json_path.resolve()}")


if __name__ == "__main__":
    main()
