from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Scope = Literal["Core", "Extended"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        schema = super().__get_pydantic_json_schema__(core_schema, handler)
        # Gemini Developer API rejects additionalProperties, including false.
        # Local Pydantic validation still rejects extra keys.
        schema.pop("additionalProperties", None)
        return schema


class Request(Contract):
    topic: str = Field(min_length=1, max_length=2000)
    scope: Scope | None = None
    duration_minutes: int = Field(default=60, ge=10, le=240)
    duration_assumed: bool = True
    learner_level: str = ""
    prior_knowledge: str = ""
    ability: str = ""
    learning_needs: str = ""
    resources: str = ""
    additional_instructions: str = ""


class Objective(Contract):
    number: str | None = None
    text: str


class Fragment(Contract):
    page: int
    printed_label: str | None = None
    bbox: list[float]
    reference_text: str
    objective_text: str
    activity_text: str
    image: str | None = None
    glyph_file: str | None = None


class Curriculum(Contract):
    record_id: str
    document_id: str
    file_hash: str
    title: str
    edition: str | None = None
    syllabus_years: str | None = None
    copyright: str | None = None
    source_filename: str
    references: list[str]
    scope: Literal["Core", "Extended", "Core/Extended"]
    objectives: list[Objective]
    objective_text: str
    activity_text: str
    source_notes: str
    fragments: list[Fragment]
    warnings: list[str] = Field(default_factory=list)


class Question(Contract):
    question_id: str
    question_text: str
    topic: str | None = None
    subtopic: str | None = None
    exam_board: str | None = None
    year: int | str | None = None
    paper: str | None = None
    marks: int | None = None
    question_type: str | None = None
    mark_scheme: Any = None
    source_filename: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class Intent(Contract):
    search_query: str
    interpretation: str


class Selection(Contract):
    curriculum_ids: list[str]
    question_ids: list[str]
    clarification: str | None = None
    sufficient: bool
    rationale: str
    question_notes: list[str] = Field(default_factory=list)


class Alignment(Contract):
    objective: str
    curriculum_id: str
    objective_number: str | None = None


class Activity(Contract):
    title: str
    minutes: int = Field(ge=1)
    teacher_actions: str = Field(min_length=1)
    learner_actions: str = Field(min_length=1)
    assessment: str = Field(min_length=1)
    curriculum_ids: list[str]
    question_ids: list[str] = Field(default_factory=list)


class Draft(Contract):
    title: str = Field(min_length=1)
    scope: Scope
    duration_minutes: int
    objectives: list[Alignment] = Field(min_length=1)
    activities: list[Activity] = Field(min_length=1)
    differentiation: str = Field(min_length=1)
    resources: list[str]
    assumptions: list[str]
    question_ids: list[str] = Field(min_length=1)


class Plan(Draft):
    questions: list[Question] = Field(min_length=1)
    question_ids: list[str] = Field(default_factory=list)


class Review(Contract):
    passed: bool
    issues: list[str]


class Attempt(Contract):
    attempt: int
    issues: list[str]


class Result(Contract):
    outcome: Literal[
        "success", "needs_clarification", "insufficient_evidence", "validation_failed", "error"
    ]
    message: str
    request: Request
    model: str
    plan: Plan | None = None
    curriculum: list[Curriculum] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    selection: Selection | None = None
    attempts: list[Attempt] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
