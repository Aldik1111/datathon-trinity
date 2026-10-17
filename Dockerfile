FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 service
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt
# Explicit allowlist: never bake uploads, keys, tests, weights or local data in.
COPY src ./src
COPY configs ./configs
COPY .streamlit ./.streamlit
RUN mkdir -p /app/artifacts/uploads && chown -R service:service /app/artifacts
USER service
CMD ["python", "-m", "uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "16", "--no-access-log"]
