FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader -d /usr/share/nltk_data vader_lexicon punkt

ENV NLTK_DATA=/usr/share/nltk_data

COPY . .

RUN mkdir -p /app/src/webapp/instance /app/models /app/data \
    && chown -R app:app /app

USER app

ENV FLASK_ENV=production \
    FLASK_DEBUG=false \
    PORT=5000

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "src.webapp.app:app"]
