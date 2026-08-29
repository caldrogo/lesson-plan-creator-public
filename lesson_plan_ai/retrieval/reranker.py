from lesson_plan_ai.models.evidence import Question


def rerank_questions(query: str, candidates: list[Question], top_k: int = 5,
                     model_name: str | None = None) -> list[Question]:
    """Use CrossEncoder when available; otherwise keep deterministic candidate order."""
    if not candidates:
        return []
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = model.predict([(query, item.question_text) for item in candidates])
        return [item for _, item in sorted(zip(scores, candidates), reverse=True)][:top_k]
    except (ImportError, OSError):
        return candidates[:top_k]
