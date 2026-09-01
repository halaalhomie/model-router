# Intelligent LLM Model Router & Evaluation Platform

Route each request to the model that suits it, instead of sending everything to
the most expensive model available.

## Current pipeline

```
CLI request
    |
    v
ANALYZER  (cheap model, structured output -> TaskProfile, retries transient errors)
    |
    v
ROUTER    (pure function: TaskProfile + ModelCatalog -> RoutingDecision)
    |
    v
EXECUTOR  (calls the selected model, retries transient errors, captures usage)
    |
    v
[failed after retries?] -- yes --> retry once on fast_model, mark fallback_used
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
| `app/retry.py` | Exponential-backoff retry for transient provider errors |
| `app/pipeline.py` | Orchestrates analyze -> route -> execute, with fallback |
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

**What retries, and what falls back?** `app/retry.py` retries one call (with
exponential backoff) on 5xx errors, 429 rate limits, and network timeouts --
failures worth waiting out. A 400/404 is a defect in the request itself, so it
is not retried; retrying it three times would only waste the timeout budget on
an outcome that was never in doubt. If the *routed* model still fails after its
own retries, `pipeline.py` retries the whole request once on `fast_model` and
marks the result `fallback_used` -- because in this project's own measurements
(below), a model failing mid-request is routine, not exceptional. A request
already routed to `fast_model` has nowhere safer to fall back to, so its
failure propagates. The analyzer has no such fallback yet: a persistent
classification failure still fails the whole request.

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
2. **Models fail in normal operation**, not just in theory. `gemini-3.7-flash`
   reliably 504s and every pro-tier model 429s -- confirmed live, not just in
   this table. Retry and fallback (see "What retries, and what falls back?"
   above) exist because of this measurement, not on general principle.

## Roadmap

Phase 1 Basic LLM integration — done
Phase 2 Intelligent routing — done
Phase 3 Reliability — in progress: retries, timeouts, and fallback done;
         FastAPI and confidence-aware routing still open
Phase 4 Kafka event pipeline
Phase 5 Evaluation
Phase 6 PostgreSQL
Phase 7 LangChain
Phase 8 LangGraph
Phase 9 Dashboard
Phase 10 Productionization
