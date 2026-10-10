# The REST API in a container, on CPU.
#
#   docker build -t hotel-rag .
#   docker run --rm -p 8000:8000 -v hotel-rag-models:/home/app/.cache/huggingface hotel-rag
#
# The models (about 3.6 GB) are not in the image: they are downloaded on the first start and kept in the
# volume, so later starts take seconds. Settings: -e HOTEL_RAG_MODEL=Qwen/Qwen2.5-0.5B-Instruct, etc.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1

RUN useradd --create-home app
WORKDIR /app

# The dependencies first, in their own layer: rebuilt only when the lock file changes. uv's download cache
# is mounted for the build only, so it does not end up in the image (it would add about 1 GB).
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra api --no-install-project

COPY README.md LICENSE ./
COPY src ./src
COPY data ./data
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra api

USER app
RUN mkdir -p /home/app/.cache/huggingface
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/home/app/.cache/huggingface
EXPOSE 8000
# Loading the models takes a while: minutes on the first start, when they are downloaded
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "hotel_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
