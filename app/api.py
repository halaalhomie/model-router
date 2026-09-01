"""HTTP transport for the pipeline.

This is the second "adapter" pipeline.py's docstring promised: main.py reads
argv and prints to stdout, this module reads an HTTP request body and
returns an HTTP response. Both call the exact same run_pipeline(). Nothing
about analyze/route/execute changes because HTTP exists now.

Run it with:  uvicorn app.api:app --reload
Then open:    http://127.0.0.1:8000/docs   (interactive API docs, generated
                                             automatically from the type
                                             hints below -- nothing here
                                             writes documentation by hand)

FastAPI itself only *defines* the app (which URLs exist, what Python
function each one calls). It does not listen on a network socket -- that is
what uvicorn is: an ASGI server, the actual program that accepts TCP
connections, speaks HTTP, and calls into FastAPI for each request.
--reload restarts that server whenever a source file changes, for local dev.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from google import genai
from pydantic import BaseModel, field_validator

from app.client import build_client
from app.config import Settings, load_settings
from app.pipeline import EXECUTION_FAILURES, run_pipeline
from app.schemas import PipelineResult


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build Settings and the Gemini client once, when the server starts.

    A CLI run is one process per request, so main.py rebuilding the client
    every time is fine. A server process handles many requests over its
    whole lifetime, so doing this per-request here would reopen a fresh
    connection for every call instead of reusing one -- and it would mean a
    broken .env is only discovered on the *first* request instead of before
    the server accepts any traffic at all. app.state is FastAPI's place to
    stash things built once and read by every request afterwards.
    """
    app.state.settings = load_settings()
    app.state.client = build_client(app.state.settings)
    yield
    # Nothing to release on shutdown yet -- the Gemini client holds no
    # connection pool of its own worth closing.


app = FastAPI(title="Model Router", lifespan=lifespan)


def get_settings(request: Request) -> Settings:
    """A FastAPI dependency: request.app is the same `app` object above, so
    this just hands back what lifespan() built at startup. Declaring it as
    a dependency (rather than importing app.state directly in the handler)
    is what lets tests swap in fake settings -- see tests/test_api.py."""
    return request.app.state.settings


def get_client(request: Request) -> genai.Client:
    return request.app.state.client


class RouteRequestBody(BaseModel):
    """The JSON body POST /route expects: {"text": "..."}.

    FastAPI validates every request against this model before your handler
    ever runs. A request that fails validation never reaches route_request
    below -- FastAPI returns 422 Unprocessable Entity itself, with a body
    describing which field failed and why.
    """

    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank.")
        return value


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check: is the process up and serving requests at all.

    Says nothing about whether Gemini itself is reachable -- that's a
    separate, more expensive question a load balancer or container
    orchestrator (Phase 10) would not want answered on every heartbeat.
    """
    return {"status": "ok"}


@app.post("/route", response_model=PipelineResult)
def route_request(
    body: RouteRequestBody,
    settings: Settings = Depends(get_settings),
    client: genai.Client = Depends(get_client),
) -> PipelineResult:
    """Run one request through the pipeline and return the full result.

    Declared as a plain `def`, not `async def`, on purpose: run_pipeline
    makes blocking network calls (the Gemini SDK is synchronous). FastAPI
    runs sync path operations in a worker thread pool automatically, so
    one slow request does not stall every other request being served
    concurrently. Writing `async def` here without an async Gemini client
    would run those blocking calls directly on the event loop and defeat
    that -- every request would queue behind whichever one is mid-call.
    """
    try:
        return run_pipeline(body.text, client=client, settings=settings)
    except EXECUTION_FAILURES as error:
        # EXECUTION_FAILURES is the same tuple pipeline.py uses to decide
        # "was this worth falling back for" -- reused here rather than
        # redefined, so this stays in sync with pipeline.py by construction.
        # Anything else (a real bug) is deliberately NOT caught: it becomes
        # FastAPI's own 500, instead of being mislabeled "upstream failed."
        raise HTTPException(
            status_code=502,
            detail=f"The model provider failed and fallback did not "
            f"recover: {error}",
        ) from error
