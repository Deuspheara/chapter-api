# Chapter API

An Axum-based Rust web service that serves book chapters with optional “immersive” AI features. It integrates Redis (Redis Stack) for storage/caching and can call an external RAG service for contextual Q&A, mood analysis, and mystery exploration.

## Highlights
- Fast chapter retrieval with summaries or full paragraphs
- Search by title
- Immersive mode: mood analysis, voice narration hints, emotional journey, mystery elements
- Contextual Q&A via RAG (configurable via `RAG_SERVICE_URL`)
- Smart cache and fallbacks when RAG is unavailable
- Docker and Docker Compose included

## Tech Stack
- Rust, Tokio, Axum, Tower HTTP
- Redis (redis/redis-stack) for JSON storage and RedisInsight
- Reqwest (rustls) for outbound HTTP to RAG service

## Quickstart

### With Docker Compose (recommended)
1. Ensure a Redis-compatible port is free locally and (optionally) you have a RAG service reachable.
2. From the project root:
   ```bash
   docker compose up --build
   ```
3. The API will be available at:
   - API: http://localhost:3000
   - RedisInsight UI: http://localhost:8002

The compose file mounts `./chapters` into the container and sets `AUTO_LOAD=true`, so chapters are loaded into Redis at startup.

RAG connectivity is configured through `RAG_SERVICE_URL` (see “Environment” below). On startup, the service logs which RAG URL it resolved to.

### Local (without Docker)
- Prereqs: Rust toolchain, a running Redis (Redis Stack preferred)
- Recommended env vars:
  ```bash
  export REDIS_URL=redis://127.0.0.1:6379
  export CHAPTERS_DIR=./chapters
  export PORT=3000
  export AUTO_LOAD=true
  # Optional, if you run a RAG service locally or elsewhere:
  export RAG_SERVICE_URL=http://127.0.0.1:8001
  ```
- Run:
  ```bash
  cargo run
  ```

## Environment
- `REDIS_URL` (default `redis://127.0.0.1:6379`)
- `CHAPTERS_DIR` (default `./chapters`)
- `PORT` (default `3000`)
- `AUTO_LOAD` (default `true`): load chapters from `CHAPTERS_DIR` into Redis at startup
- `RAG_SERVICE_URL` (default `http://127.0.0.1:8001`): base URL for the external RAG service; the app logs the resolved value at startup
- `RUST_LOG` (example `info,tower_http=info`)

## Endpoints (overview)
- Health & Stats
  - `GET /health`
  - `GET /stats`
- Chapters
  - `GET /chapters/{number}` — `?summary=true` to omit paragraphs
  - `GET /chapters?start=1&end=10&limit=50`
  - `GET /chapters?start=1&end=10&summary=true`
- Search
  - `GET /search?q=crimson&limit=10`
- Immersive
  - `GET /immersive/chapters/{number}` — `?include_recap=true` optional
  - `POST /immersive/question` — contextual Q&A (JSON body)
  - `GET /immersive/emotions/{number}`
  - `GET /immersive/mystery/{number}`
  - `POST /immersive/cache/refresh`
  - `GET /immersive/cache/status`
- Admin
  - `POST /admin/reload`
  - `POST /admin/flush`

See full API docs in `docs/`.

## API Documentation
- Human-friendly guide: `docs/API-GUIDE.md`
- OpenAPI spec: `docs/api-documentation.yaml`

You can render the OpenAPI spec using Swagger UI or Redoc. Example with Docker:
```bash
docker run -p 8080:8080 -e SWAGGER_JSON=/spec/openapi.yaml -v $(pwd)/docs/api-documentation.yaml:/spec/openapi.yaml swaggerapi/swagger-ui
# Then open http://localhost:8080
```

## Example requests
```bash
# Single chapter summary
curl 'http://localhost:3000/chapters/1?summary=true'

# Immersive chapter
curl 'http://localhost:3000/immersive/chapters/1'

# Ask a question
curl -X POST 'http://localhost:3000/immersive/question' \
  -H 'Content-Type: application/json' \
  -d '{"question":"Who is the protagonist?","current_chapter":5}'
```

## Development notes
- The service logs key configuration at startup (e.g., `Using RAG service at ...`).
- Redis Stack is used to support RedisJSON operations (see `docker-compose.yml`).
- The `chapters/` directory contains the source chapter files loaded when `AUTO_LOAD=true`.
- An example web client is available in `index.html` (update the `API_BASE` constant near the bottom to point to your API).

## Project structure (key paths)
- `src/` — application code
- `chapters/` — input chapter files
- `docs/` — documentation and OpenAPI (see below)
- `docker-compose.yml` — local dev stack (Redis + API)
- `Dockerfile` — production container build

## Docs location
All docs have been consolidated under `docs/`:
- `docs/API-GUIDE.md` — mobile-friendly API guide with examples
- `docs/api-documentation.yaml` — machine-readable OpenAPI spec
- `docs/IMPLEMENTATION_GUIDE.md` — internal implementation guidance
- `docs/REFINED_IMPLEMENTATION.md` — refined architecture/approach notes
- `docs/test_immersive_api.md` — manual testing notes for immersive endpoints
