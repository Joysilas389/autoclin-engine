FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend-req.txt
COPY ml/requirements.txt ./ml-req.txt
RUN pip install --no-cache-dir -r backend-req.txt -r ml-req.txt

COPY backend/ ./backend/
COPY ml/ ./ml/

ENV PYTHONPATH=/app

CMD ["celery", "-A", "backend.app.worker", "worker", "--loglevel=info", "--concurrency=2"]
