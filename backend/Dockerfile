FROM python:3.12.3-slim-bullseye

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY uv.lock pyproject.toml /app/

RUN uv sync --frozen --no-dev --no-install-project

COPY ./ /app/
