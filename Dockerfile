# Sandbox image for rlmflow's DockerRuntime.
#
# Build:
#   docker build -t rlmflow:local .
#
# Use:
#   from rlmflow.runtime.docker import DockerRuntime
#   runtime = DockerRuntime("rlmflow:local")
#
# Or via any of the bundled examples:
#   python examples/summarizer.py --runtime docker --docker-image rlmflow:local

FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/rlmflow
COPY pyproject.toml README.md ./
COPY rlmflow ./rlmflow
RUN pip install .

RUN groupadd --gid 1000 rlmflow \
    && useradd --uid 1000 --gid 1000 --create-home rlmflow

# DockerRuntime bind-mounts the host workspace at /workspace.
WORKDIR /workspace
USER 1000:1000

# DockerRuntime attaches to this worker over stdin/stdout.
CMD ["python", "-u", "-m", "rlmflow.runtime.repl_server"]
