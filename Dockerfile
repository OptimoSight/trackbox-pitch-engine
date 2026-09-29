FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: this layer is reused until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .

COPY main.py synthetic_generator.py ./
COPY config ./config

# Run as an unprivileged user; /app stays writable for the generated feed.
RUN useradd --create-home --uid 10001 runner && chown -R runner /app
USER runner

ENTRYPOINT ["python", "main.py"]
CMD ["--config", "config/default.json", "--generate-video"]