import argparse
import json
from pathlib import Path

from .documents import read_document
from .service import assess_cv


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
        documents = read_document(args.cv)
        job = "\n\n".join(doc.page_content for doc in read_document(args.job))
        result = assess_cv(documents, job, demo=args.demo)
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
