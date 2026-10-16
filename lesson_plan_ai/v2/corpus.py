"""Versioned Qdrant corpus. Only the CLI writes; requests are read-only."""

import hashlib
from itertools import islice
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .models import Curriculum, Question

META_ID = str(uuid5(NAMESPACE_URL, "lesson-plan-v2:manifest"))


class CorpusError(RuntimeError):
    pass


def connect(settings):
    return QdrantClient(
        url=settings.qdrant_url,
        timeout=30,
        api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
    )


def text_for(record):
    if isinstance(record, Curriculum):
        return f"{record.source_notes}\n{record.objective_text}\n{record.activity_text}"
    return "\n".join(filter(None, [record.topic, record.subtopic, record.question_text]))


class Corpus:
    def __init__(self, client, collection, embedder, *, batch_size=8):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.client, self.collection, self.embedder = client, collection, embedder

    def metadata(self):
        if not self.client.collection_exists(self.collection):
            raise CorpusError("Corpus is not ingested. Run the ingestion CLI first.")
        config = self.client.get_collection(self.collection).config.params.vectors
        if (
            not isinstance(config, qm.VectorParams)
            or config.size != self.embedder.dimension
            or config.distance != qm.Distance.COSINE
        ):
            raise CorpusError("Incompatible vector dimension/distance; use a new collection.")
        points = self.client.retrieve(self.collection, ids=[META_ID])
        if not points:
            raise CorpusError("Corpus has no completed manifest; finish ingestion first.")
        meta = points[0].payload
        if meta.get("embedding_signature") != self.embedder.signature:
            raise CorpusError(
                "Embedding model/revision/preprocessing mismatch; use a new collection."
            )
        return meta

    def ingest(self, curriculum, questions):
        if not curriculum or not questions:
            raise CorpusError("Both curriculum and questions are required before activation.")
        exists = self.client.collection_exists(self.collection)
        if not exists:
            # Disk backing reduces resident memory; it is not a fixed RAM guarantee.
            self.client.create_collection(
                self.collection,
                vectors_config=qm.VectorParams(
                    size=self.embedder.dimension, distance=qm.Distance.COSINE, on_disk=True
                ),
                on_disk_payload=True,
                hnsw_config=qm.HnswConfigDiff(on_disk=True, max_indexing_threads=1),
                optimizers_config=qm.OptimizersConfigDiff(
                    default_segment_number=1,
                    max_optimization_threads=1,
                ),
            )
            for field in ("version", "kind"):
                self.client.create_payload_index(
                    self.collection, field, field_schema=qm.PayloadSchemaType.KEYWORD
                )
            meta = {"embedding_signature": self.embedder.signature, "version": None}
            self._manifest(meta)

        meta = self.metadata()

        # Apply the same policy to persisted collections, even on an unchanged corpus.
        # Validate compatibility above before changing collection settings.
        if exists:
            self.client.update_collection(
                self.collection,
                vectors_config={"": qm.VectorParamsDiff(on_disk=True)},
                collection_params=qm.CollectionParamsDiff(on_disk_payload=True),
                hnsw_config=qm.HnswConfigDiff(on_disk=True, max_indexing_threads=1),
                optimizers_config=qm.OptimizersConfigDiff(
                    default_segment_number=1,
                    max_optimization_threads=1,
                ),
            )

        # Hash records individually without allocating a corpus-sized JSON string.
        hasher = hashlib.sha256(("activity-windows-120-v1" + self.embedder.signature).encode())
        for r in curriculum:
            hasher.update(r.model_dump_json(serialize_as_any=True).encode())
        for r in questions:
            hasher.update(r.model_dump_json(serialize_as_any=True).encode())
        version = hasher.hexdigest()

        if meta.get("version") == version:
            return {
                "version": version,
                "changed": False,
                "records": len(curriculum) + len(questions),
            }

        for kind, records in [("curriculum", curriculum), ("question", questions)]:
            entries = self._entries(kind, records)
            while batch := list(islice(entries, self.batch_size)):
                vectors = self.embedder.encode([entry[2] for entry in batch])
                points = []
                for (record, index, _), vector in zip(batch, vectors, strict=True):
                    rid = record.record_id if kind == "curriculum" else record.question_id
                    points.append(
                        qm.PointStruct(
                            id=str(uuid5(NAMESPACE_URL, f"{version}:{kind}:{rid}:{index}")),
                            vector=vector if isinstance(vector, list) else vector.tolist(),
                            payload={
                                "kind": kind,
                                "version": version,
                                "record": record.model_dump(),
                            },
                        )
                    )
                self.client.upsert(self.collection, points, wait=True)

        self._manifest(
            {
                "version": version,
                "embedding_signature": self.embedder.signature,
                "curriculum_count": len(curriculum),
                "question_count": len(questions),
            }
        )
        return {"version": version, "changed": True, "records": len(curriculum) + len(questions)}

    @staticmethod
    def _entries(kind, records):
        for record in records:
            if kind == "curriculum":
                yield record, 0, record.source_notes + "\n" + record.objective_text
                words = record.activity_text.split()
                for index, start in enumerate(range(0, len(words), 100), start=1):
                    yield (
                        record,
                        index,
                        record.source_notes + "\n" + " ".join(words[start : start + 120]),
                    )
            else:
                yield record, 0, text_for(record)

    def _manifest(self, payload):
        self.client.upsert(
            self.collection,
            [
                qm.PointStruct(
                    id=META_ID,
                    vector=[1.0] + [0.0] * (self.embedder.dimension - 1),
                    payload={"kind": "manifest", **payload},
                )
            ],
            wait=True,
        )

    def search(self, query, kind, *, limit=16, scope=None):
        meta = self.metadata()
        if not meta.get("version"):
            raise CorpusError("Corpus ingestion is incomplete.")
        must = [
            qm.FieldCondition(key="version", match=qm.MatchValue(value=meta["version"])),
            qm.FieldCondition(key="kind", match=qm.MatchValue(value=kind)),
        ]
        if kind == "curriculum" and scope:
            must.append(
                qm.FieldCondition(
                    key="record.scope", match=qm.MatchAny(any=[scope, "Core/Extended"])
                )
            )
        points = self.client.query_points(
            self.collection,
            query=self.embedder.encode([query])[0],
            query_filter=qm.Filter(must=must),
            limit=limit * 12 if kind == "curriculum" else limit,
            with_payload=True,
        ).points
        cls = Curriculum if kind == "curriculum" else Question
        records, seen = [], set()
        for point in points:
            record = cls.model_validate(point.payload["record"])
            rid = record.record_id if kind == "curriculum" else record.question_id
            if rid not in seen:
                seen.add(rid)
                records.append(record)
            if len(records) == limit:
                break
        return records
