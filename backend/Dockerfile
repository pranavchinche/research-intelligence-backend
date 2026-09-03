# Backend Dockerfile - Research Intelligence Platform

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libmagic1 poppler-utils && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Listen on all interfaces. PORT is overridable by the platform.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD python -c "import httpx,os; httpx.get('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health',timeout=5).raise_for_status()"

CMD ["python", "run.py"]