import hashlib
import json

import numpy as np

from .settings import Settings


class Embedder:
    """Local embeddings; teacher Gemini credentials never enter this component."""

    def __init__(self, settings: Settings):
        from sentence_transformers import SentenceTransformer

        self.batch_size = settings.embedding_batch_size
        self.model = SentenceTransformer(
            settings.embedding_model, revision=settings.embedding_revision
        )
        self.dimension = self.model.get_embedding_dimension()
        if self.dimension != settings.embedding_dimension:
            raise ValueError("Embedding dimension differs from deployment configuration")
        self.signature = hashlib.sha256(
            json.dumps(
                {
                    "model": settings.embedding_model,
                    "revision": settings.embedding_revision,
                    "dimension": self.dimension,
                    "preprocessing": "token-windows-192-mean-l2-v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        tokenizer = self.model.tokenizer
        result = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            # Accumulate one text's mean using bounded token-window batches.
            # Preserve the existing 192-token windows and normalized mean semantics.
            starts = range(0, len(ids), 192) if ids else range(1)
            total = None
            count = 0
            for offset in range(0, len(starts), self.batch_size):
                chunks = [
                    tokenizer.decode(ids[i : i + 192], skip_special_tokens=True)
                    for i in starts[offset : offset + self.batch_size]
                ]
                embeddings = self.model.encode(
                    chunks, batch_size=self.batch_size, normalize_embeddings=True
                )
                subtotal = np.asarray(embeddings, dtype=np.float64).sum(axis=0)
                total = subtotal if total is None else total + subtotal
                count += len(chunks)
            vector = total / count
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError("Embedding model produced a zero vector")
            result.append((vector / norm).tolist())
        return result
