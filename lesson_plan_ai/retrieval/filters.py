from typing import Any

from lesson_plan_ai.models.evidence import Question


def filter_questions(questions: list[Question], filters: dict[str, Any]) -> list[Question]:
    def matches(question: Question) -> bool:
        for field, expected in filters.items():
            actual = getattr(question, field, None)
            if expected is None or expected == "":
                continue
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True
    return [question for question in questions if matches(question)]
