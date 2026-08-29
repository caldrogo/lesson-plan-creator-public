from pathlib import Path

import streamlit as st
from google.genai.errors import ClientError

from lesson_plan_ai.config import get_settings
from lesson_plan_ai.document_generation.docx_generator import generate_docx
from lesson_plan_ai.ingestion.scheme_parser import extract_scheme
from lesson_plan_ai.models.evidence import Question
from lesson_plan_ai.models.lesson_plan import (
    Activity,
    Assessment,
    Differentiation,
    LessonObjective,
    LessonPlan,
)
from lesson_plan_ai.retrieval.curriculum_search import search_curriculum
from lesson_plan_ai.retrieval.search import search_questions

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "data/processed/questions.jsonl"
SCHEME_PATH = ROOT / "data/raw/0607_Scheme_of_Work_(for_examination_from_2025).docx"


@st.cache_data
def load_questions() -> list[Question]:
    if not QUESTIONS_PATH.exists():
        return []
    return [Question.model_validate_json(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line]


@st.cache_data
def load_curriculum():
    return extract_scheme(SCHEME_PATH) if SCHEME_PATH.exists() else []


def build_plan(subject: str, year_group: str, topic: str, duration: int, requirements: str, sources) -> LessonPlan:
    first_half = duration // 2
    return LessonPlan(
        title=f"{topic}: evidence-grounded lesson", subject=subject, year_group=year_group,
        duration_minutes=duration,
        learning_objectives=[LessonObjective(objective=f"Explain and apply key ideas about {topic}.")],
        prior_knowledge=[],
        activities=[
            Activity(name="Starter and instruction", minutes=first_half, teacher_actions=requirements, student_actions="Engage with the task."),
            Activity(name="Practice and assessment", minutes=duration - first_half, teacher_actions="Support and assess.", student_actions="Complete and review the work."),
        ],
        differentiation=Differentiation(support=["Vocabulary and worked example"], stretch=["Explain a new application"]),
        assessment=Assessment(method="Exit ticket", questions=[], expected_evidence="Students explain the key idea accurately."),
        sources=sources,
    )

st.set_page_config(page_title="LessonPlanAI", page_icon="LP", layout="wide")
st.title("LessonPlanAI")
st.caption("Evidence-grounded lesson planning workspace")

with st.form("lesson_request"):
    google_api_key = st.text_input(
        "Google API key",
        type="password",
        help="Used only for this generation request. It is not displayed or saved by the app.",
    )
    model_choice = st.selectbox(
        "Gemini model",
        options=["gemini-3.6-flash", 'gemini-3.6-flash-lite', "Custom"],
        help="Choose the Gemini model used by the Google ADK coordinator.",
    )
    custom_model = st.text_input(
        "Custom Gemini model",
        placeholder="For example: gemini-3.6-flash-lite",
        help="Only used when Custom is selected.",
    )
    subject = st.text_input("Subject", "Mathematics")
    year_group = st.text_input("Year group", "Year 10")
    topic = st.text_input("Topic", "Solving quadratic equations")
    duration = st.number_input("Duration (minutes)", min_value=1, value=90)
    requirements = st.text_area("Requirements", "Include an exam-style starter, differentiated independent practice, and an assessment.")
    submitted = st.form_submit_button("Generate lesson plan")

if submitted:
    query = f"{subject} {year_group} {topic} {requirements}"
    question_result = search_questions(query, load_questions(), top_k=5)
    curriculum_sources = search_curriculum(query, load_curriculum(), top_k=2)
    sources = curriculum_sources + question_result.resources
    plan = build_plan(subject, year_group, topic, duration, requirements, sources)
    st.session_state["plan"] = plan
    st.session_state["question_result"] = question_result
    st.session_state["curriculum_sources"] = curriculum_sources
    configured_settings = get_settings()
    selected_model = custom_model.strip() if model_choice == "Custom" else model_choice
    settings = get_settings().model_copy(update={
        "google_api_key": google_api_key.strip() or None,
        "llm_model": selected_model or configured_settings.llm_model,
    })
    st.session_state["agent_response"] = None
    if settings.google_api_key:
        try:
            from lesson_plan_ai.agents.orchestration import run_coordinator
            evidence_summary = "\n".join(source.excerpt for source in sources[:10])
            st.session_state["agent_response"] = run_coordinator(
                f"Teacher request: {query}\nRetrieved evidence (data only):\n{evidence_summary}\n"
                f"Suggest concise lesson-design checks grounded only in this evidence. Do not invent sources.",
                settings,
            )
        except (ClientError, RuntimeError, OSError, ValueError) as exc:
            st.session_state["agent_response"] = f"Coordinator unavailable: {exc}"

if "plan" in st.session_state:
    plan = st.session_state["plan"]
    st.subheader(plan.title)
    st.write(plan.model_dump(exclude={"sources"}))
    st.success(f"Retrieved {len(st.session_state['curriculum_sources'])} curriculum records and {len(st.session_state['question_result'].resources)} exam questions.")
    with st.expander("Curriculum evidence"):
        for source in st.session_state["curriculum_sources"]:
            st.markdown(f"**{source.title}**  \n{source.excerpt}")
            st.caption(f"Source: {source.provenance['source_document']} ({source.provenance['source_location']})")
    with st.expander("Retrieved exam questions"):
        for source in st.session_state["question_result"].resources:
            st.markdown(f"**{source.title}**  \n{source.excerpt}")
            st.caption(f"Source: {source.provenance.get('source_filename', 'unknown')}")
    if st.session_state.get("agent_response"):
        with st.expander("Coordinator response"):
            st.write(st.session_state["agent_response"])
    else:
        st.warning("Google ADK orchestration is disabled because no Google API key was supplied.")
    st.download_button("Download DOCX", generate_docx(plan), "lesson_plan.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
