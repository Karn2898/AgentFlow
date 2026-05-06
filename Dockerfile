FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY agentflow ./agentflow
COPY backend.py streaming.py run-local.sh ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install .

EXPOSE 8501

CMD ["agentflow", "ui"]