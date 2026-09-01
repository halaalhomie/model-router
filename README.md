# Intelligent LLM Model Router & Evaluation Platform

Route each request to the model that suits it, instead of sending everything to
the most expensive model available.

## Current pipeline

```
CLI request
    |
    v
ANALYZER  (cheap model, structured output -> TaskProfile)
    |
    v
ROUTER    (pure function: TaskProfile + ModelCatalog -> RoutingDecision)
    |
    v
EXECUTOR  (calls the selected model, captures token usage)
    |
    v
PipelineResult -> printed
```

## Setup

1. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
2. Install dependencies:
   `venv\Scripts\python.exe -m pip install -r requirements-dev.txt`

Keep `.env` private. Every `.env*` file except `.env.example` is ignored by Git.

## Run

```
venv\Scripts\python.exe -m app.main "Write a Python function that sorts a list."
```

Reads from a pipe when no argument is given.

## Test

```
venv\Scripts\python.exe -m pytest
```

No test makes a network call. The Gemini client is injected into every function
that needs it, so tests pass fake clients instead (see `tests/fakes.py`).

## Modules

| Module | Responsibility |
| --- | --- |
| `app/schemas.py` | Pydantic contracts shared by every stage |
| `app/config.py` | `.env` -> `Settings` (API key, analyzer model, catalog) |
| `app/analyzer.py` | Request -> validated `TaskProfile` |
| `app/router.py` | `TaskProfile` -> `RoutingDecision` (pure, no I/O) |
| `app/executor.py` | Runs the selected model, captures token usage |
| `app/pipeline.py` | Orchestrates analyze -> route -> execute |
| `app/main.py` | CLI adapter over the pipeline |

## Design notes

**Why structured output plus Pydantic?** The analyzer asks Gemini for JSON
matching `TaskProfile`. Pydantic then validates it, so a hallucinated task type
raises an error at the boundary instead of silently reaching the router.

**Why a separate analyzer model?** The classifier runs on every single request.
Paying for an expensive model just to decide which model to use would defeat the
purpose of routing, so `GEMINI_ANALYZER_MODEL` is configured independently of the
catalog.

**Why is the router a pure function?** Routing policy is the part most likely to
change. Keeping it free of I/O means its rules are tested exhaustively without
network calls or API keys.

**Why is orchestration in `pipeline.py` rather than `main.py`?** `main.py` is one
transport. The FastAPI handler will be another. Both call the same function, so
neither owns the logic.

**Why pinned model versions instead of `-latest` aliases?** A floating alias
would change the underlying model without warning, which would invalidate any
historical evaluation metrics collected against it.

## Model availability

`client.models.list()` is not a reliable guide to what a key can actually call.
Measured against this project's free-tier key with one identical prompt:

| Model | Latency | Notes |
| --- | --- | --- |
| `gemini-3.5-flash-lite` | ~2s | no thinking tokens |
| `gemini-3.5-flash` | ~6-14s | ~600-1200 thinking tokens |
| `gemini-3.6-flash` | ~19s | ~900 thinking tokens |
| `gemini-3.7-flash` | 504 | `DEADLINE_EXCEEDED`, unusable |
| `gemini-2.5-pro`, `gemini-2.5-flash` | 404 | listed but `NOT_FOUND` |
| `gemini-pro-latest`, `gemini-3.1-pro-preview` | 429 | `RESOURCE_EXHAUSTED` |

Two consequences worth understanding:

1. **No pro-tier model is reachable without billing enabled.** The catalog is
   therefore three flash tiers. They differ in latency and thinking-token usage,
   which is enough to build and demonstrate routing, but Phase 5 will not be able
   to show a large *quality* gap between tiers until a stronger model is
   reachable.
2. **Models fail in normal operation**, not just in theory. A route to a model
   returning 504 or 429 currently kills the request. That makes the Phase 3
   fallback work concrete rather than hypothetical.

## Roadmap

Phase 1 Basic LLM integration — done
Phase 2 Intelligent routing — done
Phase 3 Reliability: FastAPI, retries, timeouts, fallbacks, confidence routing
Phase 4 Kafka event pipeline
Phase 5 Evaluation
Phase 6 PostgreSQL
Phase 7 LangChain
Phase 8 LangGraph
Phase 9 Dashboard
Phase 10 Productionization
