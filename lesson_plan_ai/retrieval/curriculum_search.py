from lesson_plan_ai.models.evidence import CurriculumRecord, RetrievedResource


def search_curriculum(query: str, records: list[CurriculumRecord], top_k: int = 5) -> list[RetrievedResource]:
    terms = set(query.lower().split())
    scored: list[tuple[int, CurriculumRecord]] = []
    for record in records:
        score = sum(term in record.text.lower() for term in terms)
        scored.append((score, record))
    resources = []
    for score, record in sorted(scored, key=lambda item: item[0], reverse=True):
        if score <= 0:
            continue
        resources.append(RetrievedResource(
            resource_type="curriculum",
            resource_id=f"{record.source_document}:{record.source_location}",
            title=record.unit_topic or record.source_location,
            excerpt=record.text,
            score=float(score),
            provenance={
                "source_document": record.source_document,
                "source_location": record.source_location,
            },
        ))
        if len(resources) >= top_k:
            break
    return resources