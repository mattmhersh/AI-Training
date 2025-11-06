# Copilot Instructions for AI-Training

## Project Overview
- This is a 6-week AI infrastructure learning project focused on Python, FastAPI, Docker, Linux, and RunPod deployment.
- The codebase is organized by week, with each week introducing new concepts and files (see `README.md` for roadmap).
- Major components: Python scripts (data processing, OOP), FastAPI REST API (`main.py`), Docker containerization, CI/CD with GitHub Actions, and serverless deployment on RunPod.

## Key Files & Structure
- `main.py`: FastAPI app with endpoints for health, deployments, and cost calculation. Uses in-memory data for demo purposes.
- `classes.py`, `functions.py`, `file-ops.py`, `csv_processor.py`: Python basics, OOP, file I/O, and CSV/JSON processing.
- `Dockerfile`, `docker-compose.yml`: Containerization and multi-service orchestration.
- `.github/workflows/docker-build-push.yml`: CI/CD pipeline for build, test, lint, and Docker Hub push.
- `tests/`: Pytest-based unit and integration tests for API endpoints.
- `Runpod2/handler.py`, `Runpod/rp_handler.py`: RunPod-compatible handler scripts for serverless deployment.
- `.env.example`, `.gitignore`: Environment variable templates and ignore rules.

## Developer Workflows
- **Run locally:** `python main.py` (for scripts), `uvicorn main:app --reload` (for FastAPI)
- **Test:** `pytest tests/ -v --cov=. --cov-report=html`
- **Lint/Format:** `ruff check .` and `black .`
- **Build Docker:** `docker build -t ai-api:v1 .` (context may be `Runpod2` for custom handler)
- **Run Docker:** `docker run -d -p 8000:8000 ai-api:v1`
- **CI/CD:** On push to `main`, GitHub Actions runs tests, lints, builds, and pushes Docker images.
- **Deploy:** Push Docker image to Docker Hub, then pull/run on DigitalOcean or RunPod.

## Project-Specific Patterns & Conventions
- API endpoints use Pydantic models for input validation.
- Cost calculation logic is centralized in API and scripts for consistency.
- Logging is structured using Python's `logging` module for monitoring and cost tracking.
- API key authentication and rate limiting are implemented in handler scripts for security.
- All environment variables are loaded from `.env` or set in CI/CD secrets.
- Data flows: Scripts process CSV/JSON, API exposes endpoints, Docker containers encapsulate services, CI/CD automates builds/tests.
- Use `Runpod2` (not `runpod2`) for custom handler context in workflows.

## Integration Points & External Dependencies
- FastAPI, Uvicorn, Pydantic, Docker, Pytest, Ruff, Black, RunPod API.
- External services: Docker Hub, DigitalOcean, RunPod.
- Secrets for Docker Hub and API keys must be set in GitHub Actions and `.env` files.

## Examples
- See `README.md` for code samples, API usage, and workflow commands.
- See `tests/test_api.py` for API endpoint test patterns.
- See `Runpod2/handler.py` for RunPod handler conventions.

## Quickstart for AI Agents
- Follow the roadmap in `README.md` for week-by-week progression.
- Use provided scripts and API patterns as templates for new features.
- Always run tests and lint before pushing changes.
- Reference Docker and CI/CD files for build/deploy automation.
- Use structured logging and environment variables for all new integrations.

---

If any section is unclear or missing, please provide feedback for further iteration.
