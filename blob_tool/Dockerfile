# CPU-only image (Docker on macOS cannot reach the Apple GPU). For speed on a Mac use `make run`.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    BLOB_WORKSPACE=/data/workspace \
    BLOB_MODELS_DIR=/data/models \
    YOLO_CONFIG_DIR=/data/ultralytics \
    HF_HOME=/data/hf

WORKDIR /app
# deps first so code edits don't reinstall torch (on Apple Silicon hosts the linux/arm64 torch wheel is CPU-only)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev
COPY src ./src
COPY web ./web
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uvicorn", "blob_tool.app:app", "--host", "0.0.0.0", "--port", "8000"]
