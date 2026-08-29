import pytest

from lesson_plan_ai.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from lesson_plan_ai.models.lesson_plan import (
    Activity,
    Assessment,
    Differentiation,
    LessonObjective,
    LessonPlan,
)


def make_plan(minutes: int = 60):
    return LessonPlan(
        title="Test", subject="Physics", year_group="Year 10", duration_minutes=minutes,
        learning_objectives=[LessonObjective(objective="Explain induction")],
        activities=[Activity(name="Work", minutes=minutes, teacher_actions="Teach", student_actions="Learn")],
        differentiation=Differentiation(),
        assessment=Assessment(method="Exit ticket", expected_evidence="An explanation"),
    )


def test_plan_requires_exact_timings():
    with pytest.raises(ValueError):
        LessonPlan(
            title="Test", subject="Physics", year_group="Year 10", duration_minutes=60,
            learning_objectives=[LessonObjective(objective="Explain induction")],
            activities=[Activity(name="Work", minutes=59, teacher_actions="Teach", student_actions="Learn")],
            differentiation=Differentiation(),
            assessment=Assessment(method="Exit ticket", expected_evidence="An explanation"),
        )


def test_retrieval_metrics():
    retrieved, relevant = ["a", "b", "c"], {"b", "c"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, relevant, 3) > 0
