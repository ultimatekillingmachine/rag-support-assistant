# Support RAG Assistant

[![CI](https://github.com/ultimatekillingmachine/rag-support-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/ultimatekillingmachine/rag-support-assistant/actions/workflows/ci.yml)

**English** · [Русский](README.ru.md)

An assistant that answers customer-support questions **only from the company's help articles**, shows which articles it used, and says "I don't know" instead of inventing an answer.

It runs offline out of the box: local embeddings, SQLite, and an offline answer stub. Point it at any OpenAI-compatible model (DeepSeek, OpenAI, a local Ollama) for real answers.

## How it works

```
help articles ─► split into chunks ─► embeddings ─► vector store
                                                        ▲
question ─► retrieval (vector │ + BM25, fused by RRF) ──┘
         ─► relevance gate (nothing close enough ─► refusal)
         ─► prompt: "answer only from these fragments" ─► LLM
         ─► answer + numbered sources
```

## Features

| Area | What is included |
|---|---|
| Retrieval | meaning search; optional keyword (BM25) search fused with it; confidence gate; at most one chunk per article |
| Answers | grounded in the retrieved fragments, numbered citations, honest refusal |
| Dialogue | follow-up questions resolved from the previous turn |
| Streaming | `POST /ask/stream` (server-sent events) + a small chat page |
| Access control | bearer tokens; internal articles are invisible to customers |
| Performance | response cache; latency percentiles via `GET /stats` |
| Indexing | incremental re-indexing (content + settings fingerprints, schema migration) |
| Deployment | `Dockerfile` and `docker-compose.yml` (application + Postgres/pgvector) |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
copy .env.example .env            # defaults: SQLite + offline stub
python -m scripts.ingest_kb       # build the search index
uvicorn app.main:app --reload     # then open http://127.0.0.1:8000
```

The chat page asks for a token (`customer-token` by default). API example:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Authorization: Bearer customer-token" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Сколько идёт доставка в регионы?\"}"
```

## Configuration

Settings live in `.env` (see `.env.example` for the full list):

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | local embedding model (~2.25 GB, cached) |
| `LLM_PROVIDER` | `mock` | `mock`, `openai_compatible` or `ollama` |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | — | credentials of the answer model |
| `STORAGE_BACKEND` | `sqlite` | `sqlite` (zero setup) or `postgres` (pgvector) |
| `API_TOKEN_CUSTOMER`, `API_TOKEN_OPERATOR` | sample values | bearer tokens; change before exposing |
| `HYBRID_ENABLED` | `true` | add keyword search to meaning search |
| `RERANK_ENABLED` | `false` | second-stage reranking (measured slower and worse on this data) |
| `MIN_RELEVANCE_SCORE` | `0.80` | below this the assistant refuses |
| `CACHE_ENABLED`, `CACHE_TTL_SECONDS` | `true`, `900` | response cache |

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | — | liveness and index size |
| `GET /stats` | — | cache hit rate and latency percentiles |
| `POST /ask` | bearer | one grounded answer (JSON) |
| `POST /ask/stream` | bearer | the same, streamed as server-sent events |
| `GET /` | — | chat page |
| `GET /docs` | — | generated API documentation |

`POST /ask` body: `{"question": "...", "top_k": 5, "history": [{"role": "user", "content": "..."}]}`.
`history` is optional and limited to 12 messages; the role is taken from the token, never from the body.

## Quality

The evaluation set is 56 hand-written questions: 51 answerable and 5 deliberately outside the help articles.

```bash
python -m scripts.run_eval      # compares retrieval modes
python -m scripts.gate_report   # shows which confidence threshold to pick
```

| Metric | Result |
|---|---|
| hit@3 / hit@5 | **1.000** |
| hit@1 | 0.941 |
| MRR | 0.971 |
| wrongly refused real questions | **0 of 51** |

Measured decisions, in short:

* Hybrid keyword search needed **Russian stemming** to reach parity with meaning search; without it, word forms such as "треснул"/"трещины" pulled the wrong article up.
* A **cross-encoder reranker** made ranking worse (hit@1 0.941 → 0.843) and was 50× slower, so it is off by default; a threshold tuned for one scoring model does not transfer to another.
* Refusing an answer **skips the model call entirely**, so refusals cost nothing.

## Re-indexing

```bash
python -m scripts.ingest_kb          # incremental: only new/changed articles
python -m scripts.ingest_kb --full   # rebuild everything
```

| Scenario (29-article corpus) | Re-embedded | Time |
|---|---|---|
| First build | 29 articles | 202 s |
| No changes | **0** | **0.1 s** |
| One article edited | 1 | 15 s |

## Deployment

```bash
docker compose up -d          # application + Postgres with pgvector
python -m scripts.smoke_api http://127.0.0.1:8000   # end-to-end check
```

`smoke_api` verifies the whole contract: no token → 401, customer sees public articles only, operator reaches internal ones, streaming emits sources → tokens → done, chat page is served.

## Project layout

```
app/                service code (articles, search, prompt, API, chat page)
knowledge_base/     help articles (sample content, 29 articles / 8 sections)
data/eval/          56 labelled questions
scripts/            ingest, ask, evaluate, gate report, smoke test
tests/              90 tests; the model is replaced by fakes, so CI needs no downloads
```

## Notes

The help articles are fictional sample content written for this project; no third-party documentation is redistributed. Code is MIT-licensed.
