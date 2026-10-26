# AgentShield API image.
#   docker build -t agentshield .
#   docker run -p 8000:8000 agentshield          # SQLite + in-memory sessions
# For PostgreSQL + Redis use docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# the trained models are package data, so they are installed with the code
COPY pyproject.toml README.md ./
COPY agentshield ./agentshield
RUN pip install ".[api,postgres,redis]"

# run unprivileged; /data holds the SQLite file when no PostgreSQL is configured,
# /workspace is the only directory the file tools may touch
RUN useradd --create-home --uid 10001 shield \
    && mkdir /data /workspace && chown shield:shield /data /workspace
USER shield

ENV AGENTSHIELD_API_HOST=0.0.0.0 \
    AGENTSHIELD_API_PORT=8000 \
    AGENTSHIELD_DATABASE_URL=sqlite:////data/agentshield.db \
    AGENTSHIELD_WORKSPACE_DIR=/workspace

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["agentshield-api"]
