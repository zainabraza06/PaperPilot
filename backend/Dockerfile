# PaperPilot backend.
#
# Two things dominate this image and both are handled deliberately.
#
# **Torch.** sentence-transformers pulls in PyTorch, and the default wheel
# carries CUDA libraries worth roughly 2 GB that this service will never
# use — nothing here touches a GPU. Installing from PyTorch's CPU index
# instead takes the image from ~2.5 GB to ~1.2 GB.
#
# **Models.** The embedding model (~90 MB) and the spaCy pipeline (~12 MB)
# are downloaded at *build* time, not first-request time. Otherwise the
# first search after every deploy pays for a download, and the container
# cannot start at all without network access to Hugging Face — which is a
# bad property for something meant to be reproducible.

# ---------------------------------------------------------------- builder
FROM python:3.12-slim AS builder

# Build-time only: some wheels still compile, and the model download needs
# certificates. None of this survives into the runtime image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# The retry and timeout settings are not boilerplate. This image pulls
# ~600 MB of wheels, and on a slow or flaky connection pip's defaults give
# up partway through and fail the whole build after twenty minutes of
# downloading. Retrying is much cheaper than starting over.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN python -m venv "$VIRTUAL_ENV"

# CPU-only torch first, so the dependency resolver never considers the
# CUDA build when sentence-transformers asks for torch below.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the models in. HF_HOME is set explicitly so the cache lands
# somewhere copyable rather than in a home directory that changes with the
# user in the runtime stage.
ENV HF_HOME=/opt/models/huggingface
RUN python -m spacy download en_core_web_sm \
    && python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

# curl is here for the healthcheck below and nothing else; without it the
# container has no way to report that it is actually serving.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 paperpilot

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/models/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models

WORKDIR /app
COPY --chown=paperpilot:paperpilot app ./app
COPY --chown=paperpilot:paperpilot scripts ./scripts
COPY --chown=paperpilot:paperpilot eval ./eval

# The SQLite files live here. Declared as a volume in compose so summaries
# and stored papers survive a rebuild — regenerating them costs API calls
# and, with a key configured, money.
RUN mkdir -p /app/data && chown paperpilot:paperpilot /app/data

USER paperpilot
EXPOSE 8000

# Generous start period: the embedding model is loaded during startup, in
# a worker thread, and that takes several seconds on a cold container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
