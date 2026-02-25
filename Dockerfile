FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD uvicorn agent_api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 30
