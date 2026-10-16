FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md requirements.lock ./
# The lock includes CPU-only PyTorch on Linux, with hashes for every dependency.
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY lesson_plan_ai ./lesson_plan_ai
COPY app ./app
COPY .streamlit ./.streamlit
COPY data ./data
RUN pip install --no-cache-dir --no-deps --no-build-isolation . \
    && python -m lesson_plan_ai.v2.cli extract
EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
