FROM node:22-bookworm-slim AS node

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/PrimeIntellect-ai/prime-agent.git /opt/prime-agent \
    && cd /opt/prime-agent \
    && git checkout e260085dd8f742e0def3d871860c9a888b114851 \
    && npm ci \
    && npm run build

FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /opt/prime-agent/packages/coding-agent/dist /opt/prime-agent/dist
COPY --from=node /opt/prime-agent/packages/coding-agent/package.json /opt/prime-agent/package.json
COPY --from=node /opt/prime-agent/node_modules/undici /opt/prime-agent/node_modules/undici
RUN ln -s /opt/prime-agent/dist/bundle/cli.js /usr/local/bin/prime-agent

WORKDIR /opt/rlmflow
COPY pyproject.toml README.md ./
COPY rlmflow ./rlmflow
RUN pip install ".[eval,eval-environments,official-rlm,openai,anthropic]"
RUN python -m venv --system-site-packages /opt/jaz \
    && /opt/jaz/bin/pip install "jaz-lang==0.2.0a4"
RUN python -m venv --system-site-packages /opt/prime-agent/kernel \
    && /opt/prime-agent/kernel/bin/pip install \
        /opt/prime-agent/dist/prime-agent-runtime \
        beautifulsoup4 \
        lxml \
        scipy \
        tomli
ENV PRIME_AGENT_KERNEL_PYTHON=/opt/prime-agent/kernel/bin/python

WORKDIR /workspace
