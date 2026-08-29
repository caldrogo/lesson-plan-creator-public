from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Question(BaseModel):
    model_config = ConfigDict(extra="allow")

    question_id: str
    question_text: str
    topic: str | None = None
    subtopic: str | None = None
    exam_board: str | None = None
    year: int | None = None
    paper: str | None = None
    marks: int | None = Field(default=None, ge=0)
    question_type: str | None = None
    mark_scheme: str | None = None
    source_filename: str
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class CurriculumRecord(BaseModel):
    unit_topic: str | None = None
    lesson_week: str | None = None
    learning_objectives: list[str] = Field(default_factory=list)
    prior_knowledge: list[str] = Field(default_factory=list)
    suggested_activities: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)
    assessment_ideas: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    source_document: str
    source_location: str
    text: str


class RetrievedResource(BaseModel):
    resource_type: Literal["question", "curriculum"]
    resource_id: str
    title: str
    excerpt: str
    score: float | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    resources: list[RetrievedResource] = Field(default_factory=list)
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
