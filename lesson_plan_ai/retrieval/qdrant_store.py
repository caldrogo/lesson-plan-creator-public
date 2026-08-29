from lesson_plan_ai.models.evidence import Question


class QdrantQuestionStore:
    def __init__(self, url: str, collection: str, embedder):
        self.collection = collection
        self.embedder = embedder
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=url)

    def upsert(self, questions: list[Question]) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams
        vectors = self.embedder.encode([q.question_text for q in questions])
        if not vectors:
            return
        collections = {item.name for item in self.client.get_collections().collections}
        if self.collection not in collections:
            self.client.create_collection(self.collection, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
        self.client.upsert(self.collection, points=[PointStruct(
            id=index, vector=vector, payload=question.model_dump()
        ) for index, (question, vector) in enumerate(zip(questions, vectors))])

    def search(self, query: str, limit: int = 20, query_filter=None) -> list[Question]:
        vector = self.embedder.encode([query])[0]
        hits = self.client.search(self.collection, query_vector=vector, limit=limit, query_filter=query_filter)
        return [Question.model_validate(hit.payload) for hit in hits]
