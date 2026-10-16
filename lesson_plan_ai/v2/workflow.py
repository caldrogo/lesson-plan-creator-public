import asyncio

from pydantic import ValidationError

from .models import Attempt, Result
from .runtime import public_error


def validate(plan, request, scope, curriculum, questions):
    issues = []
    by_c = {c.record_id: c for c in curriculum}
    by_q = {q.question_id: q for q in questions}
    if plan.scope != scope:
        issues.append("Plan scope differs from resolved scope")
    if (
        plan.duration_minutes != request.duration_minutes
        or sum(a.minutes for a in plan.activities) != request.duration_minutes
    ):
        issues.append("Activity minutes and plan duration must equal requested duration")
    for aligned in plan.objectives:
        source = by_c.get(aligned.curriculum_id)
        if source is None:
            issues.append("Objective cites unknown curriculum ID")
        elif aligned.objective_number not in {o.number for o in source.objectives}:
            issues.append("Objective number does not exist in cited curriculum record")
    for source in curriculum:
        if scope == "Core" and source.scope == "Extended":
            issues.append("Extended-only curriculum cannot support a Core lesson")
    used = set()
    for activity in plan.activities:
        if not activity.curriculum_ids or not set(activity.curriculum_ids) <= by_c.keys():
            issues.append("Every activity must cite supplied curriculum evidence")
        used.update(activity.question_ids)
    included = [q.question_id for q in plan.questions]
    if plan.question_ids and plan.question_ids != included:
        issues.append("Declared question IDs differ from attached source questions")
    if len(set(included)) != len(included):
        issues.append("Duplicate question IDs in plan")
    if not used or used != set(included):
        issues.append("Plan questions must exactly match questions used in activities")
    for question in plan.questions:
        if by_q.get(question.question_id) != question:
            issues.append("Question provenance or wording changed, or unknown question ID")
    return list(dict.fromkeys(issues))


async def generate(request, settings, corpus, runtime, progress=lambda _: None):
    assumptions = (
        [f"Duration assumed: {request.duration_minutes} minutes."]
        if request.duration_assumed
        else []
    )
    result_args = {
        "request": request,
        "model": settings.generation_model,
        "assumptions": assumptions,
    }
    progress("Verifying your Gemini key and selected model")
    await runtime.verify()
    progress("Interpreting topic and constraints")
    intent = await runtime.call("interpreter", {"request": request.model_dump()})
    progress("Retrieving curriculum and question candidates from Qdrant")
    curriculum = await asyncio.to_thread(
        corpus.search, intent.search_query, "curriculum", limit=12, scope=request.scope
    )
    questions = await asyncio.to_thread(corpus.search, intent.search_query, "question", limit=24)
    result_args.update(curriculum=curriculum, questions=questions)
    if not curriculum or not questions:
        return Result(
            **result_args,
            outcome="insufficient_evidence",
            message="No curriculum or question evidence matched the available corpus and scope.",
        )
    payload = {
        "request": request.model_dump(),
        "intent": intent.model_dump(),
        "curriculum": [c.model_dump() for c in curriculum],
        "questions": [q.model_dump() for q in questions],
    }
    progress("Assessing evidence relevance, scope and question completeness")
    allowed_c = {c.record_id for c in curriculum}
    allowed_q = {q.question_id for q in questions}
    # Evidence repair is bounded separately from the three lesson-draft attempts.
    selection = None
    for evidence_attempt in range(3):
        try:
            selection = await runtime.call("evidence", payload)
            if (
                set(selection.curriculum_ids) <= allowed_c
                and set(selection.question_ids) <= allowed_q
            ):
                break
        except (ValidationError, ValueError):
            selection = None
        if evidence_attempt == 2:
            return Result(
                **result_args,
                outcome="validation_failed",
                message="Evidence selection could not be validated after two automatic retries. "
                "No unsupported sources were used. Please retry the lesson request.",
            )
        progress("Repairing evidence selection: use exact IDs from retrieved candidates")
        payload["evidence_validation_feedback"] = {
            "instruction": "Return a complete corrected selection. Copy only the allowed IDs; "
            "syllabus references, document IDs and filenames are not record IDs.",
            "allowed_curriculum_ids": sorted(allowed_c),
            "allowed_question_ids": sorted(allowed_q),
        }
    result_args["selection"] = selection
    payload.pop("evidence_validation_feedback", None)
    selected_c = [c for c in curriculum if c.record_id in selection.curriculum_ids]
    selected_q = [q for q in questions if q.question_id in selection.question_ids]
    result_args.update(curriculum=selected_c, questions=selected_q)
    scopes = {c.scope for c in selected_c}
    if request.scope is None and (
        selection.clarification or "Core/Extended" in scopes or len(scopes) > 1
    ):
        return Result(
            **result_args,
            outcome="needs_clarification",
            message=selection.clarification
            or "The retrieved curriculum covers different Core and Extended expectations. "
            "Choose Core or Extended to align the lesson.",
        )
    if selection.clarification:
        return Result(**result_args, outcome="needs_clarification", message=selection.clarification)
    if not selection.sufficient or not selected_c or not selected_q:
        return Result(**result_args, outcome="insufficient_evidence", message=selection.rationale)
    scope = request.scope or selected_c[0].scope
    if request.scope is None:
        assumptions.append(f"Scope inferred from retrieved evidence: {scope}.")
    payload.update(
        curriculum=[c.model_dump() for c in selected_c],
        questions=[q.model_dump() for q in selected_q],
        resolved_scope=scope,
        assumptions=assumptions,
    )
    attempts, plan = [], None
    for attempt in range(1, 4):
        progress(f"Drafting and validating lesson: attempt {attempt} of 3")
        plan = None
        try:
            plan = await runtime.call("designer", payload)
            issues = validate(plan, request, scope, selected_c, selected_q)
        except (ValidationError, ValueError):
            issues = [
                "Draft did not match the required JSON schema; return a complete valid object."
            ]
        if not issues:
            progress("Independent teaching and mathematics review")
            try:
                review = await runtime.call("critic", {**payload, "draft": plan.model_dump()})
                issues = review.issues or (
                    [] if review.passed else ["Critic did not approve the draft"]
                )
            except (ValidationError, ValueError):
                issues = ["Critic returned invalid structured output; draft has not passed review."]
        attempts.append(Attempt(attempt=attempt, issues=issues))
        if not issues:
            return Result(
                **result_args,
                outcome="success",
                message="Passed automated validation. "
                "Review mathematical content and suitability before teaching.",
                plan=plan,
                attempts=attempts,
            )
        payload["validation_feedback"] = issues
        payload["previous_draft"] = plan.model_dump() if plan else None
    return Result(
        **result_args,
        outcome="validation_failed",
        message="Draft still has issues after two revision attempts.",
        plan=plan,
        attempts=attempts,
    )


async def run(request, settings, corpus, runtime, *, progress=lambda _: None, secret=""):
    try:
        result = await asyncio.wait_for(
            generate(request, settings, corpus, runtime, progress), timeout=settings.timeout_seconds
        )
    except Exception as exc:  # noqa: BLE001 — sanitize all provider errors at the boundary
        result = Result(
            outcome="error",
            message=public_error(exc),
            request=request,
            model=settings.generation_model,
        )
    finally:
        try:
            await runtime.close()
        except Exception:  # noqa: BLE001, S110 — never log credential-bearing SDK exceptions
            pass  # Do not expose SDK cleanup exceptions or credentials.
    # Defense in depth if a teacher pastes their own key into free text or it is echoed by a model.
    if secret:
        result = Result.model_validate_json(result.model_dump_json().replace(secret, "[REDACTED]"))
    return result
