FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system app && adduser --system --ingroup app --home /home/app app \
    && mkdir -p /home/app && chown app:app /home/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader -d /usr/share/nltk_data vader_lexicon punkt \
    && python -c "import zipfile, glob; [zipfile.ZipFile(z).extractall(z.rsplit('/', 1)[0]) for z in glob.glob('/usr/share/nltk_data/**/*.zip', recursive=True)]" \
    && chmod -R a+rX /usr/share/nltk_data

ENV NLTK_DATA=/usr/share/nltk_data

COPY . .

RUN mkdir -p /app/src/webapp/instance /app/models /app/data \
    && chown -R app:app /app

USER app

ENV FLASK_ENV=production \
    FLASK_DEBUG=false \
    PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).status == 200 else 1)"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "src.webapp.app:app"]
