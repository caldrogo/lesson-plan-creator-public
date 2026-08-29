import json

from lesson_plan_ai.ingestion.json_loader import load_json_records
from lesson_plan_ai.ingestion.question_processor import process_questions
from lesson_plan_ai.models.evidence import CurriculumRecord, Question
from lesson_plan_ai.retrieval.curriculum_search import search_curriculum
from lesson_plan_ai.retrieval.filters import filter_questions
from lesson_plan_ai.retrieval.search import search_questions


def test_malformed_json_is_reported(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps({"id": "q1", "text": "Explain fields"}))
    (tmp_path / "bad.json").write_text("{")
    records, errors = load_json_records(tmp_path)
    questions, report = process_questions(records)
    assert len(questions) == 1
    assert len(errors) == 1
    assert report["valid_records"] == 1


def test_filtering_and_provenance():
    question = Question(question_id="q1", question_text="Explain fields", topic="Fields", year=2024, source_filename="paper.json")
    result = search_questions("fields", [question], {"topic": "Fields"})
    assert result.resources[0].provenance["source_filename"] == "paper.json"
    assert filter_questions([question], {"year": 2023}) == []


def test_curriculum_search_preserves_location():
    record = CurriculumRecord(source_document="scheme.docx", source_location="table 1", text="Magnetic fields")
    results = search_curriculum("magnetic fields", [record])
    assert results[0].provenance["source_location"] == "table 1"
