# Container image for the SHL Assessment Recommender.
#
# Targets a Hugging Face Docker Space (the deploy host), which runs the container as
# a non-root user with UID 1000 and expects the app on port 7860. The same image runs
# anywhere Docker does, so this is not HF-specific beyond the port and user setup.
#
# Retrieval is lexical-only, so the image needs no model download at build time.

FROM python:3.12-slim

# System build deps kept minimal. Nothing beyond what pip needs for the wheels we
# install; torch and scikit-learn ship manylinux wheels, so no compiler is required.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run as the UID 1000 user the Space provides, with a home-local install path.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install dependencies first, as their own layer, so code changes do not reinvalidate
# the (slow) dependency install.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application. Only what the service needs at runtime; see .dockerignore.
COPY --chown=user shl_recommender ./shl_recommender
COPY --chown=user scripts ./scripts
COPY --chown=user data ./data
COPY --chown=user pyproject.toml ./

# Configuration for the deployed service. Logs are JSON for a real collector.
ENV SHL_LOG_FORMAT=json

EXPOSE 7860

# Bind to the port the Space routes to. One worker: the service is stateless but the
# free tier is small, and the model call is the latency bottleneck, not CPU.
CMD ["uvicorn", "shl_recommender.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
