from pydantic import BaseModel, Field


class CriticIssue(BaseModel):
    severity: str
    section: str
    problem: str
    suggested_fix: str


class CriticResult(BaseModel):
    passed: bool
    issues: list[CriticIssue] = Field(default_factory=list)
