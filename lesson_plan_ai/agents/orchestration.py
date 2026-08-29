import asyncio
import os

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from lesson_plan_ai.agents.root_agent import build_root_agent
from lesson_plan_ai.config import Settings


async def _run_agent(prompt: str, settings: Settings) -> str:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured")
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    session_service = InMemorySessionService()
    app_name = "lesson_plan_ai"
    user_id = "streamlit_user"
    session = await session_service.create_session(app_name=app_name, user_id=user_id)
    runner = Runner(
        app_name=app_name,
        agent=build_root_agent(settings.llm_model),
        session_service=session_service,
        auto_create_session=False,
    )
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    response = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response = "".join(part.text or "" for part in event.content.parts)
    return response


def run_coordinator(prompt: str, settings: Settings) -> str:
    """Run one bounded coordinator turn and return its final text response."""
    return asyncio.run(_run_agent(prompt, settings))