from collections.abc import Iterable


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        return self._load().encode(list(texts), normalize_embeddings=True).tolist()


def question_text(question) -> str:
    fields = [question.question_text, question.topic, question.subtopic]
    return " | ".join(value for value in fields if value)
