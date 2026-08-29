import json
import sys
from pathlib import Path

# Allow direct execution from the repository without requiring an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lesson_plan_ai.ingestion.json_loader import load_json_records
from lesson_plan_ai.ingestion.question_processor import process_questions

raw_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/questions")
output = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed/questions.jsonl")
records, parse_errors = load_json_records(raw_root)
questions, report = process_questions(records)
report["parse_errors"] = len(parse_errors)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(question.model_dump_json() for question in questions), encoding="utf-8")
(output.with_suffix(".quality.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
