FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Shared libraries required by opencv-python (a pipecat[webrtc] dependency)
# that the slim image lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxcb1 \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

# System libraries for headless Chromium (the PWA browser E2E, `pytest -m e2e`).
# Only the shared libs live in the image; the Chromium binary is fetched by
# `playwright install chromium` into /root/.cache (the persisted model-cache
# volume), so it is downloaded once and survives rebuilds. Test-only — the
# production runtime never launches a browser.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
        libxkbcommon0 libatspi2.0-0 libxcomposite1 libxdamage1 libxfixes3 \
        libxrandr2 libgbm1 libasound2 libpango-1.0-0 libcairo2 libxext6 \
        libxi6 libxtst6 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000
# The watchdog supervises uvicorn: restarts it on crash AND on a hung event
# loop (health probe failures) — docker restart policies only cover exits.
CMD ["uv", "run", "--no-dev", "python", "scripts/watchdog.py"]
