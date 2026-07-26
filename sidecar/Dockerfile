# SAM3 inference sidecar (CUDA). Target: linux/arm64 or amd64 with NVIDIA GPU.
FROM nvidia/cuda:13.0.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    SAM3_BACKEND=transformers \
    PORT=9001 \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY app ./app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 \
    && uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 9001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9001"]
