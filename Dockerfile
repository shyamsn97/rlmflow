# Sandbox image for rlmflow's DockerRuntime.
#
# Build:
#   docker build -t rlmflow:local .
#
# Use:
#   from rflow.runtime.docker import DockerRuntime
#   runtime = DockerRuntime("rlmflow:local")
#
# Or via any of the bundled examples:
#   python examples/summarizer.py --runtime docker --docker-image rlmflow:local

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/rlmflow
COPY pyproject.toml README.md ./
COPY rflow ./rflow
RUN pip install ".[openai,anthropic,cloudpickle]"

# DockerRuntime bind-mounts the host workspace at /workspace.
WORKDIR /workspace

# DockerRuntime spawns: `docker run -i --rm <image> python -m rflow.runtime.repl`.
# Setting it as CMD also makes `docker run -i rlmflow:local` work standalone.
CMD ["python", "-m", "rflow.runtime.repl"]
