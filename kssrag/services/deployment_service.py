"""
Deployment scaffolding.

Generates the files needed to run a KSS RAG server in a container. Kept in the
service layer so the CLI stays a transport: the CLI parses flags and prints the
result, this module decides what the files contain.

The templates deliberately install the base package only. `pip install kssrag`
pulls no torch, no FAISS, and no OCR stack; the heavy extras are opt-in and a
deployment that does not use them should not pay for them.
"""
import os
from dataclasses import dataclass
from typing import List

DOCKERFILE = """\
# Production image for a KSS RAG server.
#
# Installs the base package only. If you use a semantic or OCR vector store,
# add the extra you need to the pip line below (for example kssrag[faiss]),
# and expect a much larger image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PORT=8000

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser \\
    && pip install --no-cache-dir --upgrade pip \\
    && pip install --no-cache-dir kssrag

# The knowledge directory and prompt are mounted at run time rather than copied,
# so secrets and documents never enter an image layer and never appear in the
# build history. See docker-compose.yml.
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \\
    CMD python -c "import sys,urllib.request;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"

CMD ["python", "-m", "kssrag.cli", "server", \
"--host", "0.0.0.0", \
"--port", "8000", \
"--file", "/app/knowledge/document.txt", \
"--system-prompt", "/app/prompt.txt", \
"--vector-store", "bm25s"]
"""

DOCKERIGNORE = """\
.env
.git
.gitignore
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
knowledge/
prompt.txt
"""

COMPOSE_FILE = """\
services:
  kssrag:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      # Read-only mounts: the container never writes to your documents.
      - ./knowledge:/app/knowledge:ro
      - ./prompt.txt:/app/prompt.txt:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import sys,urllib.request;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s
"""

ENV_EXAMPLE = """\
# Copy to .env and fill in. Never commit .env.
PROVIDER=openrouter
OPENROUTER_API_KEY=

# Optional overrides
# DEFAULT_MODEL=deepseek/deepseek-chat
# VECTOR_STORE_TYPE=bm25s
# LOG_LEVEL=INFO
# RATE_LIMIT_PER_MINUTE=120
"""

TEMPLATES = {
    "Dockerfile": DOCKERFILE,
    ".dockerignore": DOCKERIGNORE,
    "docker-compose.yml": COMPOSE_FILE,
    ".env.example": ENV_EXAMPLE,
}


@dataclass(frozen=True)
class ScaffoldResult:
    """Files written and files skipped because they already existed."""

    written: List[str]
    skipped: List[str]


class DeploymentService:
    """Write container deployment files into a directory."""

    def scaffold(self, directory: str, overwrite: bool = False) -> ScaffoldResult:
        """Write the deployment templates into ``directory``.

        Existing files are skipped unless ``overwrite`` is true, so running this
        twice never destroys edits. Raises FileNotFoundError when ``directory``
        does not exist.
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        written: List[str] = []
        skipped: List[str] = []

        for name, content in TEMPLATES.items():
            target = os.path.join(directory, name)
            if os.path.exists(target) and not overwrite:
                skipped.append(name)
                continue
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(content)
            written.append(name)

        return ScaffoldResult(written=written, skipped=skipped)
