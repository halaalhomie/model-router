# Intelligent LLM Model Router & Evaluation Platform

Route each request to the model that suits it, instead of sending everything to
the most expensive model available.

## Current pipeline

```
CLI request          HTTP POST /route
    |                     |
    +----------+----------+
               |
               v
ANALYZER  (cheap model, structured output -> TaskProfile, retries transient errors)
    |
    v
ROUTER    (pure function: TaskProfile + ModelCatalog -> RoutingDecision;
           low confidence overrides every other rule -- see below)
    |
    v
EXECUTOR  (calls the selected model, retries transient errors, captures usage)
    |
    v
[failed after retries?] -- yes --> retry once on fast_model, mark fallback_used
    |
    v
PipelineResult -+-> printed, or returned as JSON   (synchronous: the caller waits)
                |
                +-> Kafka topic model-router.requests
                       (asynchronous: best-effort, never blocks the caller)
```

Two transports, one pipeline: `app/main.py` (CLI) and `app/api.py` (HTTP) both
call the same `run_pipeline()`. Neither owns the analyze/route/execute logic.

## Setup

1. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
2. Install dependencies:
   `venv\Scripts\python.exe -m pip install -r requirements-dev.txt`
3. Start Kafka and Postgres (needs Docker Desktop running):
   `docker compose up -d`

Keep `.env` private. Every `.env*` file except `.env.example` is ignored by Git.

Step 3 is optional for running the router itself: if Kafka is unreachable,
requests still succeed and only event publishing is skipped (with a logged
warning). Nothing uses Postgres until Phase 6.

## Run

```
venv\Scripts\python.exe -m app.main "Write a Python function that sorts a list."
```

Reads from a pipe when no argument is given.

Or run it as an HTTP service:

```
venv\Scripts\python.exe -m uvicorn app.api:app --reload
```

Then, in another terminal:

```
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/route -H "Content-Type: application/json" -d "{\"text\": \"Write a Python function that sorts a list.\"}"
```

Or open http://127.0.0.1:8000/docs for an interactive UI generated
automatically from the same type hints -- nobody wrote that page by hand.

To watch events arrive, run the analytics consumer in a second terminal
and then make requests in the first:

```
venv\Scripts\python.exe -m app.consumer
```

It prints a line per event and a per-model summary on Ctrl+C. Stopping it
does not affect the router; restarting it resumes from its last committed
offset rather than replaying everything.

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
| `app/client.py` | Builds the one Gemini client both transports share |
| `app/events.py` | Publishes one Kafka event per completed request |
| `app/consumer.py` | Reads those events and aggregates per-model stats |
| `app/console.py` | Console helpers shared by both CLI entry points |
| `app/main.py` | CLI adapter over the pipeline |
| `app/api.py` | HTTP adapter over the pipeline (FastAPI + uvicorn) |

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

**Why is orchestration in `pipeline.py` rather than `main.py`?** `main.py` is
one transport, `app/api.py` is another. Both call the same `run_pipeline()`,
so neither owns the logic -- adding HTTP required zero changes to analyze,
route, or execute.

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

**Why does low confidence override every other routing rule, not just add
one more rule?** `TaskProfile.confidence` describes the classification as a
whole -- there's one score, not one per field. A profile with confidence
0.5 doesn't mean "context_size is uncertain but task_type is fine"; every
field came from the same uncertain classification. So `router.py` checks
confidence *first*: below `CONFIDENCE_THRESHOLD` (0.6, an unmeasured
starting guess -- Phase 5 should tune it from real accuracy data), routing
falls back to `reasoning_model` regardless of what `context_size` or
`task_type` say, because trusting those fields to pick a specialized model
would be trusting a coin flip. Confirmed live: a deliberately vague prompt
("so like, idk, maybe do the thing with the stuff from before") scored
confidence 0.50, correctly routed to the reasoning model over the normal
rules -- and that model then failed on its own, correctly triggering the
*separate* fallback mechanism above. Both Phase 3 safety nets, one request.

## FastAPI concepts used here

**FastAPI vs. uvicorn.** FastAPI defines *what* exists: which URLs are
routes, what Python function each one calls, what shape the request/response
are. It does not open a network socket. uvicorn is the ASGI server -- the
actual program that binds a port, speaks HTTP, and calls into FastAPI per
request. Every "run a FastAPI app" command is really "run uvicorn, pointed
at a FastAPI app object."

**Path operations.** `@app.get("/health")` / `@app.post("/route")` register a
function to run for that method + URL. FastAPI reads the function's type
hints to know what to validate on the way in and how to serialize on the way
out -- the same idea `TaskProfile` already uses for structured LLM output,
just applied to HTTP instead.

**Request/response models are Pydantic.** `RouteRequestBody` in `app/api.py`
is a normal `BaseModel`. FastAPI validates the incoming JSON against it
*before* `route_request()` runs -- a request that fails validation (missing
`text`, or blank after our `field_validator`) never reaches our code; FastAPI
returns `422 Unprocessable Entity` itself. `response_model=PipelineResult`
does the same thing in reverse: return a `PipelineResult` and FastAPI
serializes it to JSON, using the schema already defined in `app/schemas.py`.

**Dependency injection (`Depends`).** `get_settings`/`get_client` in
`app/api.py` are dependencies: FastAPI calls them and passes the result into
the handler. The payoff is in `tests/test_api.py` --
`app.dependency_overrides[get_client] = lambda: scripted_fake` swaps in a
fake for the whole test, with no monkeypatching and no real network access,
the same way `client=` injection already let `analyzer.py`/`executor.py` be
tested without a real Gemini client.

**`lifespan`.** Settings and the Gemini client are built once, when the
server starts (`app/api.py`'s `lifespan` function), not on every request.
A CLI run is one process per request, so `main.py` rebuilding them each time
is fine; a server handles many requests over one process's lifetime, so
doing this per-request would mean reconnecting on every call and only
discovering a broken `.env` on the *first* request instead of before the
server accepts any traffic.

**Sync `def`, not `async def`.** `route_request` is declared as a plain
`def`. `run_pipeline` makes blocking network calls (google-genai is a
synchronous SDK). FastAPI automatically runs sync path operations in a
worker thread pool, so one slow request doesn't stall every other request
being served concurrently. Writing `async def` around a blocking call would
run it directly on the event loop and serialize every request behind
whichever one is mid-call -- the opposite of what async is for.

**HTTP status codes carry meaning.** `422` = your request body failed
validation (client's fault). `502` = we tried to call an upstream service
(Gemini) and it failed even after retry and fallback (not our bug, not the
caller's fault either). An uncaught bug becomes FastAPI's own `500`. Mapping
these deliberately, rather than returning `200` with an error message in the
body, is what lets a caller branch on `response.status_code` instead of
parsing text.

## Kafka concepts used here

**Why Kafka at all, and why not in the request path?** The router produces
data worth analyzing -- which model was chosen, what it cost, how long it
took, whether fallback fired. Analytics, evaluation, and logging all want
that same data, at their own pace, without any of them slowing down the
person waiting for an answer. That is the actual problem Kafka solves here:
one producer, many independent consumers, none of them in the caller's way.
If it were in the synchronous path, a slow or down Kafka would make the
whole router slow or down -- strictly worse than not having it.

**Broker, topic, partition.** The broker is the server (our `kafka`
container). A topic (`model-router.requests`) is a named stream. Each topic
is split into partitions, and each partition is an independent append-only
log. Kafka guarantees ordering *within* a partition, not across a topic --
that is the tradeoff that buys parallelism, since partitions can be
consumed independently.

**Keys decide partitions.** `KafkaEventPublisher` keys each event by
`request_id`. Kafka hashes the key to pick a partition, so all events
sharing a key land in the same partition and keep their relative order.
With one event per request today this mainly spreads load, but it is what
makes ordering work if a request ever emits several events.

**Consumer groups and offsets.** An offset is how far a consumer group has
read into a partition. Kafka stores it server-side, per group -- not in the
consumer process -- which is why a restarted consumer resumes where it left
off, and why two different groups can read the same topic completely
independently (analytics and evaluation will each be their own group).
`kafka-consumer-groups.sh --describe --group <name>` shows
`CURRENT-OFFSET` / `LOG-END-OFFSET` / `LAG` per partition; lag is the
single most useful "is my consumer keeping up" number.

**Retention.** Kafka keeps messages for a configured time whether or not
anyone consumed them (default `log.retention.hours=168`, one week). This is
the real difference from a traditional queue, where a message disappears
once acknowledged -- and it is what lets a new consumer group be added
later and still read last week's events.

**produce vs. poll vs. flush** -- the easiest thing to get wrong:
`produce()` queues locally and returns immediately (this is what makes
publishing async); `poll(0)` services already-completed delivery callbacks
without blocking; `flush()` blocks until everything queued is actually
sent. `flush()` belongs only where a process is about to exit and would
otherwise drop queued messages -- `main.py` after printing the answer, and
`api.py`'s lifespan shutdown. Calling it per request would convert this
whole async design back into a synchronous one.

**Delivery semantics -- and why the two sides differ.** They are chosen
independently, and here they are deliberately opposite:

*Producing is at-most-once.* `publish_safely()` swallows failures, so an
event can be lost (Kafka down, or the process dies with messages still
queued) and is never retried. Telemetry loss is cheaper than failing a
user's request, and cheaper than an outbox table. At-least-once producing
would mean persisting the event locally first and retrying delivery -- a
real option once Phase 6 provides a database to put an outbox in.

*Consuming is at-least-once.* `app/consumer.py` commits the offset
**after** processing, so a crash in between redelivers the event rather
than dropping it. Committing before processing would be at-most-once and
lose it. The cost is that processing must tolerate duplicates: the
in-memory counters would double-count, which is fine for a restartable
local aggregate but is exactly why Phase 6's Postgres writes must be
idempotent -- upsert on `request_id`, never a blind insert.

**Poison messages.** A malformed event is logged and its offset committed
past, rather than retried forever. Without that, one bad message blocks
its partition permanently, because the offset would never advance.

**Client errors are callbacks, not exceptions.** A connection failure
never raises from `produce()` or `poll()` -- the client retries in the
background and reports through the `error_cb` both clients configure in
`base_client_config()`. Without that callback, an unreachable broker
looks exactly like an idle one.

**Verified end to end**, not just unit-tested: a CLI run and an HTTP
`POST /route` each produced an event on `model-router.requests` whose
`request_id` matched the response, read back with
`kafka-console-consumer.sh`. With Kafka deliberately stopped, both
transports still returned complete, correct answers -- only a warning
line differed.

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
Phase 3 Reliability — done: retries, timeouts, fallback, the FastAPI
         transport, and confidence-aware routing
Phase 4 Kafka — in progress: infrastructure and the producer are done
         (one event per request); consumers are next
Phase 5 Evaluation
Phase 6 PostgreSQL
Phase 7 LangChain
Phase 8 LangGraph
Phase 9 Dashboard
Phase 10 Productionization
