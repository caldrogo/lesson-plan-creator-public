import re
from collections import Counter
from pathlib import Path
from typing import Any

from lesson_plan_ai.models.evidence import Question

_ID_FIELDS = ("question_id", "question_number", "id", "uid", "uuid")
_TEXT_FIELDS = ("question_text", "question", "text", "prompt")


def _first(record: dict[str, Any], fields: tuple[str, ...]) -> Any:
    return next((record[field] for field in fields if record.get(field) not in (None, "")), None)


def _marks(record: dict[str, Any]) -> int | None:
    value = _first(record, ("marks", "marks_available", "mark_count"))
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def normalise_question(record: dict[str, Any]) -> Question:
    question_id = _first(record, ("question_id", "id", "uid", "uuid"))
    if question_id is None:
        question_number = _first(record, ("question_number",))
        source = Path(str(record.get("source_filename", "unknown"))).stem
        question_id = f"{source}:{question_number}" if question_number is not None else None
    question_text = _first(record, _TEXT_FIELDS)
    if question_id is None or question_text is None:
        raise ValueError("question record requires an id and question text")
    known = {
        "question_id": str(question_id), "question_text": str(question_text),
        "topic": record.get("topic"), "subtopic": record.get("subtopic"),
        "exam_board": record.get("exam_board") or record.get("board"),
        "year": record.get("year"), "paper": record.get("paper"),
        "marks": _marks(record), "question_type": record.get("question_type"),
        "mark_scheme": record.get("mark_scheme"),
        "source_filename": str(record.get("source_filename", "unknown")),
        "raw_metadata": record,
    }
    return Question.model_validate(known)


def process_questions(records: list[dict[str, Any]]) -> tuple[list[Question], dict[str, Any]]:
    questions: list[Question] = []
    invalid = 0
    for record in records:
        try:
            questions.append(normalise_question(record))
        except (ValueError, TypeError):
            invalid += 1
    seen: set[str] = set()
    unique: list[Question] = []
    duplicates = 0
    for question in questions:
        if question.question_id in seen:
            duplicates += 1
            continue
        seen.add(question.question_id)
        unique.append(question)
    report = {
        "input_records": len(records), "valid_records": len(questions),
        "invalid_records": invalid, "duplicate_records": duplicates,
        "unique_topics": len({q.topic for q in unique if q.topic}),
        "unique_subtopics": len({q.subtopic for q in unique if q.subtopic}),
        "missing_fields": dict(Counter(
            field for q in unique for field in ("topic", "subtopic", "year", "marks")
            if getattr(q, field) is None
        )),
    }
    return unique, report
