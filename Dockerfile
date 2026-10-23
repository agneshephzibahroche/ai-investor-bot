FROM python:3.12-slim AS base

# Every dependency ships prebuilt wheels, so no compiler toolchain is needed;
# just pull in OS security fixes released since the base image was built.
RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system app && adduser --system --uid 1000 --ingroup app --home /home/app app \
    && mkdir -p /home/app && chown app:app /home/app

WORKDIR /app

COPY requirements.txt .
# Install the app's dependencies, then remove the packaging tools (pip,
# setuptools and ensurepip's bundled wheels). The app never uses them at
# runtime, and their vendored copies of urllib3/msgpack/setuptools are what
# image scanners flag.
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m pip uninstall -y setuptools wheel pip \
    && rm -rf /usr/local/lib/python3.*/ensurepip

COPY . .

RUN mkdir -p /app/src/webapp/instance /app/models /app/data \
    && chown -R app:app /app

USER app

ENV FLASK_ENV=production \
    FLASK_DEBUG=false \
    PORT=5000 \
    WEB_CONCURRENCY=2

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/health', timeout=3).status == 200 else 1)"

# Hosts such as Render set PORT; gunicorn reads the worker count from WEB_CONCURRENCY
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:$PORT --timeout 120 src.webapp.app:app"]
