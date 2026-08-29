from pydantic import BaseModel, Field, field_validator

from lesson_plan_ai.models.evidence import RetrievedResource


class LessonObjective(BaseModel):
    objective: str = Field(min_length=1)
    success_criteria: list[str] = Field(default_factory=list)


class Activity(BaseModel):
    name: str = Field(min_length=1)
    minutes: int = Field(gt=0)
    teacher_actions: str
    student_actions: str
    resources: list[str] = Field(default_factory=list)


class Assessment(BaseModel):
    method: str = Field(min_length=1)
    questions: list[str] = Field(default_factory=list)
    expected_evidence: str


class Differentiation(BaseModel):
    support: list[str] = Field(default_factory=list)
    stretch: list[str] = Field(default_factory=list)


class LessonPlan(BaseModel):
    title: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    year_group: str = Field(min_length=1)
    duration_minutes: int = Field(gt=0)
    learning_objectives: list[LessonObjective] = Field(min_length=1)
    prior_knowledge: list[str] = Field(default_factory=list)
    activities: list[Activity] = Field(min_length=1)
    differentiation: Differentiation
    assessment: Assessment
    homework: str | None = None
    sources: list[RetrievedResource] = Field(default_factory=list)

    @field_validator("activities")
    @classmethod
    def timings_match_duration(cls, activities: list[Activity], info):
        duration = info.data.get("duration_minutes")
        if duration is not None and sum(item.minutes for item in activities) != duration:
            raise ValueError("activity timings must sum exactly to duration_minutes")
        return activities
