# Support RAG Assistant

**English** · [Русский](README.ru.md)

[![CI](https://github.com/ultimatekillingmachine/rag-support-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/ultimatekillingmachine/rag-support-assistant/actions/workflows/ci.yml)

An assistant that answers customer-support questions **using only the company's help articles**, shows which articles it used, and says "I don't know" instead of making things up.

It was built as a portfolio project to show the whole path of an AI feature: from documents to a working service — with quality measured, not just claimed.

> **Demo data:** a fictional electronics store, "ТехноМаркет", and its support articles (29 articles, 8 sections). The texts are in Russian on purpose — most demo projects only handle English, while real Russian companies need their own language to work well.

## What problem it solves

- **Answers must be trustworthy.** A customer-support bot may not invent delivery times or refund rules. So the assistant answers only from the help articles and always shows its sources.
- **A plain "search" is not enough.** Customers ask in their own words ("my phone fell and the screen cracked — is that covered?") while the article says "cracks, signs of a fall". The service understands the meaning, not just the exact words.
- **Honesty beats guessing.** If the question is outside the help articles, the assistant says so and offers to pass the question to a human — instead of producing a confident wrong answer.

## How it works

```
help articles (knowledge_base/*.md)
      │  read + split into small pieces
      ▼
numbers that represent meaning ──► storage of all pieces (SQLite for development │ Postgres for production)
                                                             ▲
customer question ──► find the most relevant pieces ─────────┘
                      (meaning search │ + exact-word search; confidence check;
                       no more than one piece per article)
                                                             │
                                                             ▼
                     "answer only from these pieces" ──► AI model (any OpenAI-compatible │ Ollama │ offline mock)
                                                             │
                                                             ▼
                              answer + list of sources + response time
```


## Words used above, in plain language

| Term | What it means here |
|---|---|
| **RAG** (Retrieval-Augmented Generation) | The AI first **finds** relevant documents, then **writes** an answer based on them. This is what keeps answers grounded. |
| **Embedding** | A list of numbers that represents the *meaning* of a text. Texts with similar meaning get similar numbers, so we can compare them mathematically. |
| **Vector search** | Finding pieces by comparing these numbers. Understands paraphrases. |
| **Keyword / BM25 search** | Classic search by exact words. Good for codes and exact terms, blind to paraphrases. |
| **Hybrid search** | Using both searches and merging their results. |
| **Relevance gate** | A confidence check: if nothing in the help articles is close enough, the assistant refuses to answer. |
| **Chunk** | A small piece of an article (a few paragraphs). Answers are assembled from pieces, not whole articles. |
| **LLM** (Large Language Model) | The AI that writes the final answer: here any OpenAI-compatible model (e.g. DeepSeek, OpenAI), a local Ollama model, or an offline test stub. |
| **Mock** | A simple stand-in used in tests instead of a real AI model, so tests are free and always available. |
| **CI** (Continuous Integration) | A robot that runs tests automatically after every code change and shows a green/red mark. |

## What is inside

| Part | What was chosen | Why |
|---|---|---|
| Web service | FastAPI | Fast, modern, automatically documents its own API |
| Meaning search | `intfloat/multilingual-e5-large` (runs locally) | Good Russian quality, works without paid APIs, no data leaves the machine |
| Storage | SQLite (development) / Postgres with pgvector (production) | Works right after download; production path prepared in `docker-compose.yml` |
| Exact-word search | Own small BM25 implementation | No extra dependency; the ranking maths stays readable and testable |
| AI model | Any OpenAI-compatible endpoint (DeepSeek, OpenAI, OpenRouter), local Ollama, or offline `mock` | The project runs and is tested **without any API key** |
| Quality control | pytest, ruff, and our own evaluation script | Quality is a number we can watch, not a feeling |

## Quick start (works offline, no API key)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env            # defaults: SQLite storage, offline mock AI
python -m scripts.ingest_kb       # read the help articles and prepare the search index
uvicorn app.main:app --reload
```

Then ask a question (or use the automatic API page at http://127.0.0.1:8000/docs):

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Сколько идёт доставка в регионы?\"}"
```

Or without running a server:

```bash
python -m scripts.ask_cli "Сколько идёт доставка в регионы?"
```

Questions outside the help articles (for example "В каком году основан ТехноМаркет?") get an honest "нет информации" reply instead of an invented one.

### Switching to a real AI model

Set these lines in `.env`:

```ini
# DeepSeek
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-...            # your key
LLM_MODEL=deepseek-chat

# or a model running on your own computer (Ollama)
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=qwen2.5:7b
```

Everything else stays the same: the service, the search, the evaluation.

### Production storage (Postgres)

```bash
docker compose up -d
# then set STORAGE_BACKEND=postgres in .env
```


## Measuring quality

```bash
python -m scripts.run_eval      # compares search modes, prints hit@1/3/5 and MRR
python -m scripts.gate_report   # shows which confidence threshold to choose
```

The evaluation set is 56 questions written by hand: 51 with a known correct
article and 5 that are deliberately **outside** the help articles.

Results with the default settings (`intfloat/multilingual-e5-large`, 5 pieces in context):

| Metric | Result | In plain words |
|---|---|---|
| hit@3 | **1.000** | The correct article is among the top 3 results for **every** question |
| hit@1 | 0.941 | The correct article is the very first result for 48 of 51 questions |
| MRR | 0.971 | On average the correct article appears almost at the very top |
| out-of-scope questions | 2 of 5 refused | The confidence check stops some of the trick questions |
| wrong refusals | **0 of 51** | Not a single real question was wrongly refused |

These numbers come from `python -m scripts.run_eval` and can be reproduced on any machine.

### Why the defaults are what they are

1. **Adding keyword search made things worse at first** (hit@1 fell from 0.958 to 0.854). Reason: exact-word search without Russian word-form handling ("треснул" vs "трещины") pushed the *wrong* article up. This is a known weakness of the simple approach.
2. **Giving the meaning search more weight** (and the keyword search less) brought the combined mode back to hit@1 = 0.941 — on par with meaning search alone.
3. **Limiting to one piece per article** fixed a different problem: one long article used to fill 4 of the 5 context slots and push the correct article out.
4. **Decision:** the default is meaning search + confidence check; the combined mode is available as a ready-to-use option for cases where exact terms matter more (product codes, model names) or when a lighter search model is used.

### Choosing the confidence threshold

The best value sits between the worst good question (0.802) and the best
out-of-scope question (0.830) — but the two groups overlap, so there is no
perfect number. The chosen default (0.80) never wrongly refuses a real
question and stops some of the trap questions. Refused questions skip the AI
call entirely, so they cost nothing. This is deliberate: in customer support a
wrong answer is worse than asking a colleague.

## Project layout

```
app/                 service code (reading articles, search, answer building, API)
knowledge_base/      the help articles themselves (demo data)
data/eval/           56 hand-written test questions
scripts/             command-line helpers (index building, asking, evaluation)
tests/               32 automatic tests
```

## Troubleshooting

**Windows: `ONNXRuntimeError: External data path escapes model directory`**

Windows can store downloaded AI models as shortcuts (symlinks) pointing outside
their folder, and the local model engine then refuses to load them. The project
sets `HF_HUB_DISABLE_SYMLINKS=1` for you, so a fresh download is not affected.
If you already hit the error, delete the downloaded cache and rebuild the index:

```powershell
Remove-Item "$env:TEMP\fastembed_cache" -Recurse -Force
python -m scripts.ingest_kb
```

**First run downloads the search model** (~2.25 GB for `intfloat/multilingual-e5-large`).
It is cached and reused afterwards. For a lighter setup set
`EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~0.5 GB).

## Roadmap

- [x] Read the help articles, split them, build the search index, answer with sources (`POST /ask`)
- [x] Quality measurement: comparison of search modes, confidence-threshold report
- [x] Combined meaning + keyword search, confidence check, one piece per article
- [ ] Better re-ranking of results, streaming answers, follow-up questions
- [ ] Automatic index refresh, response cache, access rules for internal articles, slow-response statistics, chat interface, public deployment
- [ ] Russian word-form handling for the keyword search, then re-run the evaluation

## Notes

The help articles are fictional texts written for this project; no third-party documentation is redistributed. Code license: MIT.

