"""python -m lesson_plan_ai.v2.cli extract|index|ingest|status"""

import argparse
import json
from pathlib import Path

from .ingestion import extract_pdf, load_questions, write_jsonl
from .models import Curriculum
from .settings import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["extract", "index", "ingest", "status"])
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/raw/0607_Scheme_of_Work_(for_examination_from_2025).pdf"),
    )
    parser.add_argument("--questions", type=Path, default=Path("data/processed/questions.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ocr", action="store_true", help="Use installed OCRmyPDF/Tesseract on scans"
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Index accepted records only, acknowledging report exclusions",
    )
    args = parser.parse_args()
    settings = Settings()
    output = args.output or Path(settings.corpus_directory)
    if args.command in ("extract", "ingest"):
        report = extract_pdf(args.pdf, output, ocr=args.ocr)
        questions, question_report = load_questions(args.questions)
        write_jsonl(output / "questions.jsonl", questions)
        (output / "question-report.json").write_text(json.dumps(question_report, indent=2))
        print(
            json.dumps(
                {
                    "curriculum_records": report["records"],
                    "question_records": len(questions),
                    "reports": str(output),
                }
            )
        )
    if args.command in ("index", "ingest", "status"):
        from .corpus import Corpus, connect
        from .embeddings import Embedder

        client = connect(settings)
        try:
            corpus = Corpus(
                client,
                settings.collection,
                Embedder(settings),
                batch_size=settings.ingestion_batch_size,
            )
            if args.command == "status":
                print(json.dumps(corpus.metadata(), indent=2))
                return
            report = json.loads((output / "extraction-report.json").read_text())
            qreport = json.loads((output / "question-report.json").read_text())
            if (
                report.get("ocr_used")
                or report["unresolved_associations"]
                or report["rejected_records"]
                or qreport["rejected_records"]
            ) and not args.allow_unresolved:
                raise SystemExit(
                    "Review extraction/question reports first. To explicitly exclude "
                    "unresolved/rejected records, pass --allow-unresolved."
                )
            curriculum = [
                Curriculum.model_validate_json(line)
                for line in (output / "curriculum.jsonl").read_text().splitlines()
                if line
            ]
            questions, qreport = load_questions(output / "questions.jsonl")
            if qreport["rejected_records"]:
                raise SystemExit("Invalid prepared question records")
            print(json.dumps(corpus.ingest(curriculum, questions), indent=2))
        finally:
            client.close()


if __name__ == "__main__":
    main()
