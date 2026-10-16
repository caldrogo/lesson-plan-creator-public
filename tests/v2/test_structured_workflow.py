"""Clearly labelled synthetic evidence and fake model responses; no Gemini calls."""

import asyncio
import os

import pytest
from qdrant_client import QdrantClient

from lesson_plan_ai.v2.corpus import Corpus, CorpusError
from lesson_plan_ai.v2.ingestion import join_rows, load_questions
from lesson_plan_ai.v2.models import (
    Activity,
    Alignment,
    Curriculum,
    Fragment,
    Intent,
    Plan,
    Question,
    Request,
    Review,
    Selection,
)
from lesson_plan_ai.v2.runtime import ADKRuntime, public_error
from lesson_plan_ai.v2.settings import Settings
from lesson_plan_ai.v2.workflow import run, validate


@pytest.fixture
def curriculum():
    return Curriculum(
        record_id="fixture-c1",
        document_id="synthetic-fixture",
        file_hash="fixture",
        title="Synthetic curriculum fixture — not Cambridge source",
        source_filename="fixture",
        references=["E1.17"],
        scope="Extended",
        objective_text="1. Simplify surds",
        objectives=[{"number": "1", "text": "Simplify surds"}],
        activity_text="Demonstrate square factors.",
        source_notes="E1.17 Surds",
        fragments=[],
    )


@pytest.fixture
def question():
    return Question(
        question_id="fixture:q1",
        question_text="Simplify √18.",
        marks=2,
        source_filename="fixture.json",
        raw_metadata={"fixture": True, "null": None},
    )


def plan_for(c, q):
    return Plan(
        title="Surds",
        scope="Extended",
        duration_minutes=60,
        objectives=[
            Alignment(objective="Simplify surds", curriculum_id=c.record_id, objective_number="1")
        ],
        activities=[
            Activity(
                title="Guided practice",
                minutes=60,
                teacher_actions="Model square factors.",
                learner_actions="Simplify and explain.",
                assessment="Check √18.",
                curriculum_ids=[c.record_id],
                question_ids=[q.question_id],
            )
        ],
        differentiation="Use factor pairs as scaffolds.",
        resources=["Board"],
        assumptions=["Learners know square numbers."],
        questions=[q],
    )


class FakeCorpus:
    def __init__(self, c, q):
        self.c, self.q = c, q

    def search(self, query, kind, **kwargs):
        return [self.c] if kind == "curriculum" else [self.q]


class FakeRuntime:
    def __init__(self, c, q, failures=0):
        self.c, self.q, self.failures = c, q, failures
        self.calls, self.closed = [], False

    async def verify(self):
        pass

    async def close(self):
        self.closed = True

    async def call(self, stage, payload):
        self.calls.append(stage)
        if stage == "interpreter":
            return Intent(search_query="surds", interpretation="Simplifying surds")
        if stage == "evidence":
            return Selection(
                curriculum_ids=[self.c.record_id],
                question_ids=[self.q.question_id],
                sufficient=True,
                rationale="Direct match",
            )
        if stage == "designer":
            return plan_for(self.c, self.q)
        passed = self.calls.count("critic") > self.failures
        return Review(passed=passed, issues=[] if passed else ["Add worked modelling."])


def execute(c, q, *, failures=0, scope=None):
    runtime = FakeRuntime(c, q, failures)
    result = asyncio.run(
        run(Request(topic="surds", scope=scope), Settings(), FakeCorpus(c, q), runtime)
    )
    assert runtime.closed
    return result, runtime


@pytest.mark.parametrize(
    "failures,expected,count", [(0, "success", 1), (2, "success", 3), (9, "validation_failed", 3)]
)
def test_revision_bound(curriculum, question, failures, expected, count):
    result, runtime = execute(curriculum, question, failures=failures)
    assert result.outcome == expected
    assert len(result.attempts) == count
    assert runtime.calls.count("designer") == count
    assert result.model == "gemini-3.5-flash-lite"


def test_scope_clarification(curriculum, question):
    curriculum.scope = "Core/Extended"
    curriculum.references = ["C1.17", "E1.17"]  # Synthetic shared-scope case only.
    result, runtime = execute(curriculum, question)
    assert result.outcome == "needs_clarification"
    assert "designer" not in runtime.calls


def test_core_cannot_use_extended(curriculum, question):
    result, _ = execute(curriculum, question, scope="Core")
    assert result.outcome == "validation_failed"


def test_deterministic_validation(curriculum, question):
    plan = plan_for(curriculum, question)
    plan.activities[0].minutes = 59
    plan.questions[0] = question.model_copy(update={"marks": None})
    plan.objectives[0].objective_number = "999"
    issues = validate(plan, Request(topic="surds"), "Extended", [curriculum], [question])
    assert len(issues) == 3


def test_bad_json_revision_limit(curriculum, question):
    class Bad(FakeRuntime):
        async def call(self, stage, payload):
            if stage == "designer":
                self.calls.append(stage)
                raise ValueError("invalid JSON with potentially sensitive text")
            return await super().call(stage, payload)

    runtime = Bad(curriculum, question)
    result = asyncio.run(
        run(Request(topic="surds"), Settings(), FakeCorpus(curriculum, question), runtime)
    )
    assert result.outcome == "validation_failed"
    assert runtime.calls.count("designer") == 3
    assert result.plan is None


def test_session_credentials_and_error_redaction(curriculum, question, monkeypatch):
    before = dict(os.environ)
    a = ADKRuntime("teacher-one-secret", "gemini-3.5-flash-lite", "/tmp")
    b = ADKRuntime("teacher-two-secret", "gemini-3.5-flash-lite", "/tmp")
    assert a.model.client_kwargs["api_key"] != b.model.client_kwargs["api_key"]
    assert dict(os.environ) == before
    runtime = FakeRuntime(curriculum, question)
    result = asyncio.run(
        run(
            Request(topic="surds teacher-one-secret"),
            Settings(),
            FakeCorpus(curriculum, question),
            runtime,
            secret="teacher-one-secret",
        )
    )
    assert "teacher-one-secret" not in result.model_dump_json()
    assert "secret" not in public_error(RuntimeError("teacher-two-secret"))


@pytest.mark.parametrize(
    "code,word",
    [
        (401, "key"),
        (403, "permission"),
        (404, "unavailable"),
        (429, "quota"),
        (503, "temporarily unavailable"),
        (400, "rejected"),
    ],
)
def test_safe_errors(code, word):
    error = RuntimeError("sensitive provider payload")
    error.code = code
    assert word in public_error(error)
    assert "sensitive" not in public_error(error)


def test_timeout(curriculum, question):
    class Slow(FakeRuntime):
        async def verify(self):
            await asyncio.sleep(10)

    runtime = Slow(curriculum, question)
    result = asyncio.run(
        run(
            Request(topic="surds"),
            Settings(timeout_seconds=1),
            FakeCorpus(curriculum, question),
            runtime,
        )
    )
    assert result.outcome == "error"
    assert "timed out" in result.message
    assert runtime.closed


def test_continuations_and_negative_references():
    def row(page, ref="", objective="", activity="continued"):
        return Fragment(
            page=page,
            bbox=[0, 0, 100, 100],
            reference_text=ref,
            objective_text=objective,
            activity_text=activity,
        )

    rows = [
        row(1),
        row(2, "C2.1 and E2.1 Formulae", "Transform formulae"),
        row(3),
        row(3, "E2.8 Proportion\n(note there is no C2.8)", "1. Express proportion"),
        row(3, "E1.17 Surds\n(note: there is no C1.17)", "1. Simplify surds"),
        row(4, objective="2. Rationalise the denominator."),
    ]
    records, unresolved = join_rows(
        rows,
        {
            "document_id": "fixture",
            "file_hash": "fixture",
            "title": "Fixture",
            "source_filename": "fixture",
        },
    )
    assert len(unresolved) == 1
    assert [f.page for f in records[0].fragments] == [2, 3]
    assert records[1].references == ["E2.8"]
    assert records[2].references == ["E1.17"]
    assert [o.number for o in records[2].objectives] == ["1", "2"]


def test_lossless_question_roundtrip(tmp_path, question):
    path = tmp_path / "bank.jsonl"
    path.write_text(question.model_dump_json() + "\n" + question.model_dump_json())
    records, report = load_questions(path)
    assert records == [question]
    assert report["duplicates"] == [question.question_id]
    assert not report["rejected_records"]


def test_malformed_and_conflicting_bank(tmp_path, question):
    path = tmp_path / "bank.json"
    import json

    changed = question.model_copy(update={"question_text": "Changed wording"})
    path.write_text(json.dumps([question.model_dump(), changed.model_dump(), 3]))
    records, report = load_questions(path)
    assert records == [question]
    assert len(report["rejected_records"]) == 2


class FixtureEmbedder:
    dimension = 4
    signature = "synthetic-fixture-embedding-v1"
    calls = 0

    def encode(self, texts):
        self.calls += 1
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def test_real_qdrant_local_idempotence_and_mismatch(curriculum, question):
    client = QdrantClient(":memory:")
    embedder = FixtureEmbedder()
    corpus = Corpus(client, "fixture", embedder)
    assert corpus.ingest([curriculum], [question])["changed"]
    count = embedder.calls
    assert not corpus.ingest([curriculum], [question])["changed"]
    assert embedder.calls == count
    assert corpus.search("surds", "question") == [question]
    assert corpus.search("surds", "curriculum", scope="Core") == []
    question2 = question.model_copy(update={"question_id": "fixture:replacement"})
    corpus.ingest([curriculum], [question2])
    assert corpus.search("surds", "question") == [question2]
    embedder.signature = "incompatible-same-dimension"
    with pytest.raises(CorpusError, match="mismatch"):
        corpus.metadata()
    client.close()


def test_interrupted_ingestion_keeps_old_version(curriculum, question):
    client = QdrantClient(":memory:")
    embedder = FixtureEmbedder()
    corpus = Corpus(client, "fixture", embedder)
    corpus.ingest([curriculum], [question])
    old_version = corpus.metadata()["version"]

    def fail(texts):
        raise RuntimeError("embedding interrupted")

    embedder.encode = fail
    with pytest.raises(RuntimeError):
        corpus.ingest([curriculum], [question.model_copy(update={"marks": 7})])
    assert corpus.metadata()["version"] == old_version
    client.close()


def test_actual_adk_runner_with_fixture_model():
    """Exercise the production Runner, message transport and schema parser without network."""
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class FixtureModel(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            assert any(
                "surds" in (part.text or "")
                for content in llm_request.contents
                for part in content.parts or []
            )
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text='{"search_query":"surds","interpretation":"Simplifying surds"}'
                        )
                    ],
                )
            )

    runtime = ADKRuntime("fixture-key", "gemini-3.5-flash-lite", "/tmp")
    runtime.model = FixtureModel(model="synthetic-fixture")
    intent = asyncio.run(runtime.call("interpreter", {"request": {"topic": "surds"}}))
    assert intent.search_query == "surds"


def test_raster_grid_fallback():
    """Synthetic raster table, not an OCR accuracy test."""
    from types import SimpleNamespace

    from PIL import Image, ImageDraw

    from lesson_plan_ai.v2.ingestion import raster_tables

    image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(image)
    for x in [40, 180, 320, 760]:
        draw.line((x, 60, x, 550), fill="black", width=2)
    for y in [60, 110, 300, 550]:
        draw.line((40, y, 760, y), fill="black", width=2)

    class RasterPage:
        width, height = 800, 600

        def to_image(self, **kwargs):
            return SimpleNamespace(original=image)

        def find_tables(self, settings):
            assert len(settings["explicit_vertical_lines"]) == 4
            assert len(settings["explicit_horizontal_lines"]) == 4
            return ["synthetic-table"]

    assert raster_tables(RasterPage()) == ["synthetic-table"]


def test_all_agent_schemas_are_accepted_by_gemini_sdk():
    from google.genai import _transformers
    from google.genai._api_client import BaseApiClient

    from lesson_plan_ai.v2.runtime import SCHEMAS

    client = BaseApiClient(api_key="synthetic-fixture-key", vertexai=False)
    try:
        for schema in SCHEMAS.values():
            assert _transformers.t_schema(client, schema).type == "OBJECT"
    finally:
        client.close()


def test_designer_attaches_original_questions_through_real_adk(curriculum, question):
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    from lesson_plan_ai.v2.models import Draft

    data = plan_for(curriculum, question).model_dump(exclude={"questions"})
    data["question_ids"] = [question.question_id]
    draft = Draft.model_validate(data)

    class FixtureDesigner(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text=draft.model_dump_json())]
                )
            )

    runtime = ADKRuntime("synthetic-fixture-key", "gemini-3.5-flash-lite", "/tmp")
    runtime.model = FixtureDesigner(model="synthetic-fixture")
    plan = asyncio.run(
        runtime.call("designer", {"questions": [question.model_dump()], "curriculum": []})
    )
    assert plan.questions == [question]
    assert plan.question_ids == [question.question_id]


@pytest.mark.parametrize("invalid_field", ["curriculum_ids", "question_ids"])
@pytest.mark.parametrize("recover", [True, False])
def test_evidence_unknown_ids_are_repaired_or_rejected(
    curriculum, question, invalid_field, recover
):
    class BadEvidence(FakeRuntime):
        async def call(self, stage, payload):
            value = await super().call(stage, payload)
            if stage == "evidence":
                attempt = self.calls.count("evidence")
                if attempt > 1:
                    feedback = payload["evidence_validation_feedback"]
                    assert feedback["allowed_curriculum_ids"] == [curriculum.record_id]
                    assert feedback["allowed_question_ids"] == [question.question_id]
                if not recover or attempt == 1:
                    setattr(value, invalid_field, ["E1.17"])
            return value

    runtime = BadEvidence(curriculum, question)
    result = asyncio.run(
        run(Request(topic="surds"), Settings(), FakeCorpus(curriculum, question), runtime)
    )
    assert result.outcome == ("success" if recover else "validation_failed")
    assert runtime.calls.count("evidence") == (2 if recover else 3)
    assert runtime.calls.count("designer") == (1 if recover else 0)
    if recover:
        assert result.selection.curriculum_ids == [curriculum.record_id]
        assert result.selection.question_ids == [question.question_id]


def test_evidence_schema_failure_retries(curriculum, question):
    class MalformedEvidence(FakeRuntime):
        async def call(self, stage, payload):
            if stage == "evidence" and self.calls.count("evidence") == 0:
                self.calls.append(stage)
                raise ValueError("invalid provider JSON")
            return await super().call(stage, payload)

    runtime = MalformedEvidence(curriculum, question)
    result = asyncio.run(
        run(Request(topic="surds"), Settings(), FakeCorpus(curriculum, question), runtime)
    )
    assert result.outcome == "success"
    assert runtime.calls.count("evidence") == 2


def test_request_local_evidence_enums_are_sdk_compatible(curriculum, question):
    from google.genai import _transformers
    from google.genai._api_client import BaseApiClient
    from pydantic import ValidationError

    from lesson_plan_ai.v2.runtime import evidence_schema

    payload = {"curriculum": [curriculum.model_dump()], "questions": [question.model_dump()]}
    schema = evidence_schema(payload)
    data = Selection(
        curriculum_ids=[curriculum.record_id],
        question_ids=[question.question_id],
        sufficient=True,
        rationale="Fixture",
    ).model_dump()
    assert schema.model_validate(data).curriculum_ids == [curriculum.record_id]
    with pytest.raises(ValidationError):
        schema.model_validate({**data, "curriculum_ids": ["E1.17"]})
    second = evidence_schema({"curriculum": [], "questions": []})
    with pytest.raises(ValidationError):
        second.model_validate(data)
    client = BaseApiClient(api_key="synthetic-fixture-key", vertexai=False)
    try:
        converted = _transformers.t_schema(client, schema)
        # SDK converts singleton Literal to a one-value enum for Gemini.
        assert converted.properties["curriculum_ids"].items.enum == [curriculum.record_id]
        assert converted.properties["question_ids"].items.enum == [question.question_id]
    finally:
        client.close()


def test_large_curriculum_uses_bounded_point_batches(curriculum, question, monkeypatch):
    client = QdrantClient(":memory:")
    embedder = FixtureEmbedder()
    corpus = Corpus(client, "bounded", embedder, batch_size=3)
    curriculum.activity_text = "word " * 2100
    uploads = []
    original = client.upsert

    def upsert(collection, points, **kwargs):
        uploads.append(len(points))
        return original(collection, points, **kwargs)

    monkeypatch.setattr(client, "upsert", upsert)
    corpus.ingest([curriculum], [question])
    assert max(uploads) == 3
    assert client.count("bounded").count == 24  # 22 curriculum + question + manifest
    assert corpus.search("word", "curriculum") == [curriculum]
    client.close()


def test_unchanged_corpus_updates_existing_storage_policy(curriculum, question, monkeypatch):
    client = QdrantClient(":memory:")
    corpus = Corpus(client, "existing", FixtureEmbedder())
    corpus.ingest([curriculum], [question])
    updates = []
    monkeypatch.setattr(client, "update_collection", lambda *a, **kw: updates.append(kw))
    assert not corpus.ingest([curriculum], [question])["changed"]
    assert updates[0]["vectors_config"][""].on_disk is True
    assert updates[0]["collection_params"].on_disk_payload is True
    assert updates[0]["hnsw_config"].max_indexing_threads == 1
    assert updates[0]["optimizers_config"].max_optimization_threads == 1
    client.close()


def test_embedding_windows_are_bounded_and_mean_is_preserved():
    import numpy as np

    from lesson_plan_ai.v2.embeddings import Embedder

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(range(int(text)))

        def decode(self, ids, **kwargs):
            return str(len(ids))

    class Model:
        tokenizer = Tokenizer()
        sizes = []

        def encode(self, chunks, **kwargs):
            self.sizes.append(len(chunks))
            assert kwargs["batch_size"] == 2
            return np.array([[float(c), 1.0] for c in chunks])

    embedder = Embedder.__new__(Embedder)
    embedder.model = Model()
    embedder.batch_size = 2
    result = embedder.encode([str(192 * 5 + 30), "0"])
    expected = np.array([(192 * 5 + 30) / 6, 1.0])
    np.testing.assert_allclose(result[0], expected / np.linalg.norm(expected))
    np.testing.assert_allclose(result[1], [0, 1])
    assert embedder.model.sizes == [2, 2, 2, 1]
    assert embedder.encode([]) == []
