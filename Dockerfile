FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    DERMACARE_DB=/data/dermacare_history.sqlite3 \
    DERMACARE_MODEL_PATH=/app/models/skin_disease_cnn.keras \
    DERMACARE_LABELS_PATH=/app/models/class_names.json \
    OLLAMA_HOST=http://ollama:11434

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . ./
RUN mkdir -p /data /app/models \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin dermacare \
    && chown -R dermacare:dermacare /app /data

USER dermacare
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"

CMD ["streamlit", "run", "app.py", "--server.headless=true"]
