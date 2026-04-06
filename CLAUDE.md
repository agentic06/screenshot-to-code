# Project Agent Instructions

Python environment:

- Always use `uv` for Python package management in the backend.
- Preferred invocation: `cd backend && uv run <command>`.
- Install dependencies: `cd backend && uv sync --all-groups`.

Testing policy:

- Always run backend tests after every code change: `cd backend && uv run pytest`.
- Always run type checking after every code change: `cd backend && uv run pyright`.
- Type checking policy: no new warnings in changed files (`pyright`).

## Frontend

- Frontend: `cd frontend && yarn lint`

If changes touch both, run both sets.

## Prompt formatting

- Prefer triple-quoted strings (`"""..."""`) for multi-line prompt text.
- For interpolated multi-line prompts, prefer a single triple-quoted f-string over concatenated string fragments.

# Hosted

The hosted version is on the `hosted` branch. The `hosted` branch connects to a saas backend, which is a seperate codebase at ../screenshot-to-code-saas
