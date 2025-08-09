# RAG API

Placeholder for the Retrieval-Augmented Generation service that will be used by Chapter API.

Planned next steps:
- Implement a simple HTTP endpoint at `/query` that accepts `{ "question": string }`
- Return a JSON payload `{ "answer": string, "sources": [{"title": string, "url": string}] }`
- Add a Dockerfile and include this service in `docker-compose.yml`

Until implemented, Chapter API can be configured to point to an external RAG service via `RAG_SERVICE_URL`.
