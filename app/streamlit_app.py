"""Streamlit dashboard for evidence-grounded mathematics lesson planning."""

import asyncio

import streamlit as st

from lesson_plan_ai.v2.models import Request, Result
from lesson_plan_ai.v2.settings import Settings

st.set_page_config(page_title="IGCSE Math Lesson Planner", page_icon="📐", layout="wide")
settings = Settings()
st.title("IGCSE Math Lesson Planner")
st.caption("Evidence-grounded mathematics lessons")
with st.sidebar:
    st.header("Your Gemini connection")
    api_key = st.text_input("Gemini API key", type="password", key="teacher_api_key")
    model = st.text_input("Generation model", value=settings.generation_model)
    st.caption(
        "Your key stays in this browser session. Requests and selected evidence are sent to Gemini."
    )
    if st.button("Clear session"):
        for key in list(st.session_state):
            del st.session_state[key]
        st.rerun()
with st.form("lesson_request"):
    topic = st.text_input("Lesson topic", placeholder="e.g. Rearranging formulae")
    left, right = st.columns(2)
    scope = left.selectbox("Syllabus scope", ["Use retrieved evidence", "Core", "Extended"])
    use_default = right.checkbox(
        f"Use default duration ({settings.default_duration} minutes)", value=True
    )
    duration = right.number_input(
        "Duration (minutes)", min_value=10, max_value=240, value=settings.default_duration
    )
    with st.expander("Optional class context"):
        level = st.text_input("Learner level")
        prior = st.text_area("Prior knowledge")
        ability = st.text_input("Ability / attainment")
        needs = st.text_area("Learning needs")
        resources = st.text_area("Available resources")
        instructions = st.text_area("Additional instructions")
    submitted = st.form_submit_button("Create lesson plan", type="primary")

if submitted:
    st.session_state.pop("lesson_result", None)
    if not topic.strip() or not api_key.strip():
        st.warning("Enter a lesson topic and your Gemini API key.")
    else:
        from lesson_plan_ai.v2.corpus import Corpus, CorpusError, connect
        from lesson_plan_ai.v2.embeddings import Embedder
        from lesson_plan_ai.v2.runtime import ADKRuntime
        from lesson_plan_ai.v2.workflow import run

        request = Request(
            topic=topic.strip(),
            scope=None if scope == "Use retrieved evidence" else scope,
            duration_minutes=settings.default_duration if use_default else int(duration),
            duration_assumed=use_default,
            learner_level=level,
            prior_knowledge=prior,
            ability=ability,
            learning_needs=needs,
            resources=resources,
            additional_instructions=instructions,
        )
        settings.generation_model = model.strip()

        # Cache only the public local embedding model, never credentials, runtimes or requests.
        @st.cache_resource
        def embedding_model(name, revision, dimension):
            return Embedder(
                Settings(
                    embedding_model=name, embedding_revision=revision, embedding_dimension=dimension
                )
            )

        client = None
        with st.status("Preparing lesson", expanded=True) as status:
            try:
                client = connect(settings)
                corpus = Corpus(
                    client,
                    settings.collection,
                    embedding_model(
                        settings.embedding_model,
                        settings.embedding_revision,
                        settings.embedding_dimension,
                    ),
                )
                corpus.metadata()  # Read-only; never ingest on a teacher request.
                runtime = ADKRuntime(api_key, settings.generation_model, settings.corpus_directory)
                result = asyncio.run(
                    run(request, settings, corpus, runtime, progress=st.write, secret=api_key)
                )
                st.session_state.lesson_result = result.model_dump(mode="json")
                status.update(
                    label=result.outcome.replace("_", " ").title(),
                    state="complete" if result.outcome == "success" else "error",
                )
            except CorpusError as exc:
                st.session_state.lesson_result = Result(
                    outcome="error",
                    message=str(exc),
                    request=request,
                    model=settings.generation_model,
                ).model_dump(mode="json")
                st.error(str(exc))  # CorpusError messages are fixed, contain no client details.
                status.update(label="Corpus setup required", state="error")
            except Exception:  # noqa: BLE001 — never log credential-bearing SDK exceptions
                message = (
                    "Could not connect to the corpus or load embeddings. Check deployment setup."
                )
                st.session_state.lesson_result = Result(
                    outcome="error",
                    message=message,
                    request=request,
                    model=settings.generation_model,
                ).model_dump(mode="json")
                st.error(message)
                status.update(label="Setup error", state="error")
            finally:
                if client:
                    client.close()

if data := st.session_state.get("lesson_result"):
    result = Result.model_validate(data)
    renderer = st.success if result.outcome == "success" else st.warning
    renderer(result.message)
    for assumption in result.assumptions:
        st.info(assumption)
    if result.outcome == "needs_clarification":
        st.info("Update the scope or additional instructions above and submit again.")
    if result.plan:
        plan = result.plan
        st.subheader(plan.title)
        st.caption(f"{plan.scope} • {plan.duration_minutes} minutes")
        if result.outcome != "success":
            st.error("Unvalidated draft — unresolved issues are listed below.")
        for objective in plan.objectives:
            st.markdown(f"- {objective.objective}")
        for activity in plan.activities:
            with st.expander(f"{activity.minutes} min · {activity.title}", expanded=True):
                st.markdown("**Teacher:** " + activity.teacher_actions)
                st.markdown("**Learners:** " + activity.learner_actions)
                st.markdown("**Assessment:** " + activity.assessment)
                st.caption("Question IDs: " + ", ".join(activity.question_ids))
        st.markdown("**Differentiation:** " + plan.differentiation)
        st.write("Resources:", plan.resources)
        st.write("Planning assumptions:", plan.assumptions)
        with st.expander("Selected questions"):
            for q in plan.questions:
                st.markdown(f"**{q.question_id}** · Marks: {q.marks}")
                st.write(q.question_text)
    with st.expander("Evidence and validation history"):
        st.json(
            {
                "curriculum": data["curriculum"],
                "questions": data["questions"],
                "selection": data["selection"],
                "attempts": data["attempts"],
            }
        )
    st.subheader("JSON output")
    st.json(result.model_dump(mode="json"), expanded=1)
    st.download_button(
        "Download result JSON",
        result.model_dump_json(indent=2),
        file_name="lesson-result.json",
        mime="application/json",
    )
