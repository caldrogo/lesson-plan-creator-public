FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY lesson_plan_ai ./lesson_plan_ai
COPY app ./app
COPY data ./data
# Pin CPU-only PyTorch so the image does not download CUDA runtimes.
RUN pip install --no-cache-dir \
	--index-url https://download.pytorch.org/whl/cpu \
	torch==2.7.1+cpu \
	&& pip install --no-cache-dir \
	--extra-index-url https://download.pytorch.org/whl/cpu \
	.
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
