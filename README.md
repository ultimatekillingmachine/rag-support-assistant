# Support RAG Assistant

A production-style Retrieval-Augmented Generation assistant that answers customer-support questions **strictly from a product knowledge base**, with citations, honest "not found" behavior, and measurable retrieval quality.

Built as a portfolio project demonstrating the full lifecycle of an LLM application: ingestion → chunking → embeddings → retrieval → grounded generation → evaluation → API/UI/deploy.

> **Demo corpus:** fictional e-commerce store "ТехноМаркет" (Russian-language KB, ~30 articles across 8 categories). The Russian corpus is intentional: it exercises multilingual retrieval, which most toy RAG demos skip.

## Why this project

- Customer support over a knowledge base is one of the most common commercial RAG use cases (e.g., Alfa-Bank × KTS built a contact-center RAG for 12,000+ operators: search time reduced 20×, 93% positive ratings).
- The hard parts of RAG are not "call an LLM" — they are retrieval quality, grounding, citations, freshness, access rights, and evaluation. This project implements and measures exactly those.

## Architecture

```
knowledge_base/*.md
      │  parse + markdown-aware chunking
      ▼
embeddings (fastembed, multilingual-e5) ──► vector store (SQLite dev │ Postgres+pgvector prod)
                                                             ▲
question ──► retrieve top-k (vector │ hybrid BM25+RRF; relevance gate; dedup per article)
                                                             │
                                                             ▼
                     grounded prompt + citations ──► LLM (OpenAI-compatible │ Ollama │ mock)
                                                             │
                                                             ▼
                              answer + sources[] + latency_ms + model
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | async, typed, auto-generated docs |
| Vector store | SQLite (dev) / Postgres + pgvector (prod) | zero-setup dev, production path in `docker-compose.yml` |
| Embeddings | fastembed (ONNX, multilingual-e5) | no torch, no API keys, runs offline |
| LLM | OpenAI-compatible endpoint (OpenAI / OpenRouter / Ollama) + `mock` provider | provider-agnostic; CI runs without keys |
| Quality | pytest, ruff, eval harness (hit@k, MRR, citation rate) | measurable quality instead of vibes |

## Quickstart (dev mode: SQLite + mock LLM)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env            # defaults: storage=sqlite, llm=mock
python -m scripts.ingest_kb
uvicorn app.main:app --reload
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Сколько идёт доставка в регионы?\"}"
```

Or one-shot from the CLI (no server needed):

```bash
python -m scripts.ask_cli "Сколько идёт доставка в регионы?"
```

Out-of-scope questions (e.g. «В каком году основан ТехноМаркет?») are refused
by the relevance gate with an honest «нет информации» answer instead of a
hallucination — see the evaluation section for calibration details.

### LLM modes

Set in `.env`:
- `LLM_PROVIDER=mock` — extractive answer built from retrieved context (no keys, used by tests/CI);
- `LLM_PROVIDER=openai_compatible` — any OpenAI-compatible API: OpenAI, OpenRouter, or local Ollama (`LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=qwen2.5:7b`).

### Production mode (Postgres + pgvector)

```bash
docker compose up -d
# set STORAGE_BACKEND=postgres in .env
```

## Evaluation

```bash
python -m scripts.run_eval      # retrieval modes comparison
python -m scripts.gate_report   # relevance-gate calibration data
```

The eval set: `data/eval/questions.jsonl` — 56 questions: 51 answerable
(question → expected article slugs) and 5 deliberately out-of-scope.

**Final results** (`intfloat/multilingual-e5-large`, top_k=5, default config):

| mode | hit@1 | hit@3 | hit@5 | MRR | out-of-scope refused | over-refusals |
|---|---|---|---|---|---|---|
| **vector + gate (default)** | **0.941** | **1.000** | **1.000** | **0.971** | 2/5 | 0/51 |
| vector, gate off | 0.941 | 1.000 | 1.000 | 0.971 | 0/5 | 0/51 |
| hybrid (weighted RRF) | 0.941 | 0.961 | 1.000 | 0.960 | — | — |
| hybrid + gate | 0.941 | 0.961 | 1.000 | 0.960 | 2/5 | 0/51 |

The harness deliberately skips the LLM: retrieval is the cheap-to-measure core
of a RAG system and runs locally on every commit (no API keys, no network).

### Retrieval experiments log (why the defaults are what they are)

1. **Equal-weight hybrid regressed retrieval** against the vector baseline
   (hit@1 0.958 → 0.854 on the first 48 questions): BM25 promotes lexically
   similar but semantically wrong chunks — for «треснул экран» it ranked the
   *how-to-file-a-claim* article above the correct one (which says «трещины,
   следы падения»), a textbook Russian-morphology mismatch.
2. **Weighted RRF** (vector 1.0 / BM25 0.5) recovered most of the loss
   (hit@1 0.896); lowering the lexical weight to **0.25** brought hybrid to
   hit@1 parity with vector on the final 56-question set.
3. **Per-article dedup** (`MAX_CHUNKS_PER_SLUG=1`) fixed context flooding:
   before it, a single article occupied 4 of 5 context slots and crowded the
   correct document out of the window.
4. **Decision:** the default is vector-only with the gate on; hybrid ships as
   a pre-tuned configurable mode for exact-term workloads (codes, SKUs) or
   lighter embedding models. Roadmap: Russian stemming / char n-grams for the
   lexical branch, then re-run the eval.

### Relevance gate calibration (`scripts/gate_report`)

The best separating threshold between answerable (min top-1 cosine 0.802) and
out-of-scope (max 0.830) questions is 0.816 — but the classes overlap:

| MIN_RELEVANCE_SCORE | out-of-scope refused | over-refusals | notes |
|---|---|---|---|
| 0 (disabled) | 0/5 | 0/51 | baseline |
| **0.80 (default)** | 2/5 | 0/51 | conservative: never refuses a valid question |
| 0.816 | 3/5 | 1/51 | balanced |
| 0.82 | 3/5 | 2/51 | aggressive |

The gate is only the first line of defense: the system prompt additionally
instructs the LLM to state «информация не найдена» when the context lacks the
answer, and a refused question skips the LLM call entirely (no tokens spent).

## Troubleshooting

**Windows: `ONNXRuntimeError: External data path escapes model directory`**

When Windows Developer Mode is enabled, huggingface_hub caches model files as
symlinks pointing outside the model directory, and ONNX Runtime (>= 1.24)
refuses to load models with external data files. The app sets
`HF_HUB_DISABLE_SYMLINKS=1` automatically (see `app/__init__.py`), so fresh
downloads are unaffected. If you hit this error with an already-downloaded
cache, purge it and re-run ingestion:

```powershell
Remove-Item "$env:TEMP\fastembed_cache" -Recurse -Force
python -m scripts.ingest_kb
```

**First run downloads the embedding model** (~2.25 GB for `intfloat/multilingual-e5-large`).
It is cached by fastembed (in `%TEMP%\fastembed_cache` on Windows) and reused
afterwards. For a lighter setup, switch `EMBEDDING_MODEL` in `.env` to
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384d, ~0.5 GB).

## Roadmap

- [x] Project spec & scaffold
- [x] **Phase 1 (MVP):** KB ingestion, vector retrieval, grounded answers with citations, `POST /ask`
- [x] **Phase 2a:** retrieval eval harness (`python -m scripts.run_eval`) + `ask_cli`
- [x] **Phase 2b (retrieval quality):** BM25 + weighted RRF (configurable hybrid), relevance gate with calibration report (`scripts/gate_report`), per-article dedup — all measured on the eval set
- [ ] **Phase 2c:** cross-encoder rerank (multilingual), streaming, multi-turn dialog, lexical branch improvements (Russian stemming)
- [ ] **Phase 3:** incremental re-index, response cache, audience-based access to articles, p95 metrics, chat UI, deploy

## Corpus note

The knowledge base is fictional content written specifically for this project (no third-party documentation is redistributed). Code: MIT.
