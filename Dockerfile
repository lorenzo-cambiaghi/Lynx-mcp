# syntax=docker/dockerfile:1
#
# Lynx — a 100% local MCP server for semantic code search.
#
# This image is for SELF-HOSTING Lynx on your own server: the code you index
# lives inside the container (mounted from the host), the embedding model runs
# locally, and nothing leaves the machine. It is NOT a hosted SaaS — there is
# no Lynx cloud. "Local" here means "local to your server".
#
# Two run modes (see the bottom of this file):
#   1. Web UI + local JSON API  ->  lynx manager ui   (default; long-running HTTP)
#   2. MCP stdio server         ->  lynx serve        (for an MCP client over stdio)

FROM python:3.12-slim

# git: the codebase source uses it for change detection and .gitignore rules.
# Everything else (tree-sitter parsers, chromadb, onnxruntime) ships as wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Caches and config live under fixed paths so a single named volume per mount
# point persists the index (/data) across container restarts.
ENV HF_HOME=/data/hf \
    RAG_CONFIG_PATH=/config/config.json \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    LYNX_TOOL_PROFILE=full
# LYNX_TOOL_PROFILE=full: a container serves many clients and directory
# scanners (Glama runs this image to list the tools), so it exposes all 17.
# A desktop install defaults to the 10-tool `standard` profile instead.

WORKDIR /app

# Install Lynx from the source in the build context — this is what Glama builds,
# so the image always matches the committed code (not a floating PyPI version).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# Optional: bake the ~130MB embedding model into the image so the container is
# fully offline from first boot (no model download on first query). Off by
# default to keep the image lean and the automated build fast:
#   docker build --build-arg PREFETCH_MODEL=true -t lynx .
ARG PREFETCH_MODEL=false
RUN if [ "$PREFETCH_MODEL" = "true" ]; then \
        lynx manager install --model BAAI/bge-small-en-v1.5; \
    fi

# Baked default config (indexes /workspace into /data/rag_storage). Mounting
# your own /config/config.json or /workspace overrides these defaults.
COPY docker/config.docker.json /config/config.json

# Run unprivileged. Create the mount points before declaring them as volumes so
# the baked config is preserved as the volume's initial content.
RUN mkdir -p /workspace /data \
    && useradd --create-home --uid 1000 lynx \
    && chown -R lynx:lynx /config /workspace /data /app
USER lynx

VOLUME ["/config", "/workspace", "/data"]
EXPOSE 8765

# Liveness: the local JSON API answers once the UI is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/api/v1/sources', timeout=3).status==200 else 1)"

# Default: web UI + local JSON API (/api/v1), bound on all interfaces so it is
# reachable from outside the container. To keep it localhost-only, publish the
# port only to loopback on the host:  -p 127.0.0.1:8765:8765
ENTRYPOINT ["lynx"]
CMD ["manager", "ui", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]

# --- Alternative: run as an MCP stdio server -------------------------------
# docker run -i --rm \
#   -v "$PWD:/workspace:ro" \
#   -v lynx-data:/data \
#   lynx serve
# ---------------------------------------------------------------------------
