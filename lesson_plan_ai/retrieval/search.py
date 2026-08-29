from collections.abc import Callable

from lesson_plan_ai.models.evidence import Question, RetrievalResult, RetrievedResource
from lesson_plan_ai.retrieval.filters import filter_questions


def lexical_candidates(query: str, questions: list[Question], top_k: int = 20) -> list[Question]:
    terms = set(query.lower().split())
    scored = []
    for question in questions:
        haystack = " ".join(filter(None, [question.question_text, question.topic, question.subtopic])).lower()
        score = sum(term in haystack for term in terms)
        scored.append((score, question))
    return [question for score, question in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0][:top_k]


def search_questions(query: str, questions: list[Question], filters: dict | None = None,
                     top_k: int = 20, reranker: Callable[[str, list[Question]], list[Question]] | None = None) -> RetrievalResult:
    filtered = filter_questions(questions, filters or {})
    candidates = lexical_candidates(query, filtered, top_k)
    selected = reranker(query, candidates) if reranker else candidates
    resources = [RetrievedResource(
        resource_type="question", resource_id=q.question_id, title=q.question_id,
        excerpt=q.question_text, provenance=q.model_dump(exclude={"raw_metadata"}),
    ) for q in selected]
    return RetrievalResult(resources=resources, query=query, filters=filters or {})
