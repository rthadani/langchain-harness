# syntax=docker/dockerfile:1
#
# Reference image for a chatbot backed by langchain-harness.
#
# Build it with a prebuilt wheel and a hash-locked dependency set:
#
#   make wheel          # writes dist/*.whl and dist/requirements.lock
#   podman build -t langchain-harness:0.1.0 .
#
# The chatbot's persona, memory, MCP servers, and harness settings come from
# the profile directory below, copied into the image at /app. Swap it per bot:
#
#   podman build --build-arg PROFILE_DIR=chatbots/support -t support-bot .
#
# Everything under /app/.harness is runtime state and must be a volume.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Stage 1: install the wheel and its locked dependencies into a venv
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS wheeldeps

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Copy the lock first so dependency layers cache independently of the wheel.
COPY dist/requirements.lock /build/requirements.lock

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --require-hashes -r /build/requirements.lock

# --no-deps: the lock already pinned every transitive dependency. pip check
# catches drift between the lock and the wheel's own metadata.
COPY dist/*.whl /wheels/
RUN /opt/venv/bin/pip install --no-deps /wheels/*.whl \
    && /opt/venv/bin/pip check

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG PROFILE_DIR=deploy/agent

# HARNESS_SERVER_HOST is what makes the server reachable from outside the
# container; the CLI default is 127.0.0.1. Port stays unset so the entrypoint
# can honour a platform-injected PORT and otherwise fall back to 8000.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HARNESS_SERVER_HOST=0.0.0.0 \
    HARNESS_SERVER_ENABLE_SHELL=false

# Optional: install Node for npx-based MCP servers. Off by default to keep the
# image small; Debian's nodejs may be older than some servers require.
ARG INSTALL_NODE=0
RUN if [ "$INSTALL_NODE" = "1" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends nodejs npm \
        && rm -rf /var/lib/apt/lists/*; \
    fi

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --home-dir /home/app \
        --shell /usr/sbin/nologin app \
    && mkdir -p /app/.harness \
    && chown -R 1000:1000 /app

COPY --from=wheeldeps /opt/venv /opt/venv
COPY deploy/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY --chown=1000:1000 ${PROFILE_DIR}/ /app/

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /app
USER 1000:1000

EXPOSE 8000

# Podman/Docker health only; Kubernetes uses the probes on /healthz and /readyz.
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.environ.get('HARNESS_SERVER_PORT') or os.environ.get('PORT') or '8000'; urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2)"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve"]
