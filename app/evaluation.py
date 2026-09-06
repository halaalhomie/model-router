"""Runs a fixed dataset through two strategies and compares them.

The project's claim is that routing beats always using the strongest
model. This is the thing that tests that claim, so its fairness matters
more than its features:

1. Both strategies go through execute_request, so both get the same
   retries and the same timeout. Comparing a careful router against a
   naive baseline would prove nothing.
2. The router's cost includes what the classifier cost. The router pays
   for a call the baseline never makes; omitting it would flatter the
   router by exactly that much (see app/schemas.py, AnalysisResult).
3. The dataset is fixed and versioned. Editing a case changes what the
   numbers mean, which is why comparisons cite a dataset version.

What it does NOT measure is answer quality. Cost, latency and failures
are objective; deciding whether a cheaper model answered *well enough*
needs either human judgement or an LLM judge, with its own biases and
costs. Until that exists, read a cost saving as "cheaper", never as
"better" -- a router that always picked the worst model would look
excellent by every number below.

Run:  python -m app.evaluation                 (whole dataset)
      python -m app.evaluation --limit 3       (first 3 cases)
      python -m app.evaluation --dry-run       (no API calls; plan only)
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from pydantic import BaseModel

from app.client import build_client
from app.config import ConfigurationError, Settings, load_settings
from app.console import use_utf8_stdio
from app.executor import execute_request
from app.pipeline import EXECUTION_FAILURES, run_pipeline
from app.schemas import TaskType


DEFAULT_DATASET = (
    Path(__file__).resolve().parent.parent / "evaluation" / "dataset-v1.json"
)

ROUTER = "router"
BASELINE = "baseline"


class EvaluationCase(BaseModel):
    id: str
    text: str
    expected_task_type: TaskType
    note: str | None = None


class Dataset(BaseModel):
    name: str
    version: int
    cases: list[EvaluationCase]

    @property
    def label(self) -> str:
        return f"{self.name} v{self.version}"


@dataclass
class CaseOutcome:
    """One case run under one strategy."""

    case_id: str
    ok: bool
    model_name: str | None = None
    cost_usd: float | None = None
    latency_ms: float = 0.0
    error: str | None = None
    # Router only: what the classifier said, and whether fallback fired.
    predicted_task_type: str | None = None
    expected_task_type: str | None = None
    fallback_used: bool = False

    @property
    def classified_correctly(self) -> bool | None:
        if self.predicted_task_type is None:
            return None
        return self.predicted_task_type == self.expected_task_type


@dataclass
class StrategySummary:
    name: str
    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def succeeded(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failures(self) -> int:
        return len(self.outcomes) - len(self.succeeded)

    @property
    def failure_rate(self) -> float:
        return self.failures / len(self.outcomes) if self.outcomes else 0.0

    @property
    def priced(self) -> list[CaseOutcome]:
        return [o for o in self.succeeded if o.cost_usd is not None]

    @property
    def total_cost_usd(self) -> float | None:
        """None if any successful case could not be priced -- a partial
        total would understate, and understating cost is the one error
        this whole exercise exists to avoid."""
        if len(self.priced) != len(self.succeeded):
            return None
        return sum(o.cost_usd for o in self.priced)

    @property
    def avg_latency_ms(self) -> float:
        if not self.succeeded:
            return 0.0
        return sum(o.latency_ms for o in self.succeeded) / len(self.succeeded)

    @property
    def fallbacks(self) -> int:
        return sum(1 for o in self.succeeded if o.fallback_used)

    @property
    def routing_accuracy(self) -> float | None:
        graded = [
            o for o in self.succeeded if o.classified_correctly is not None
        ]
        if not graded:
            return None
        correct = sum(1 for o in graded if o.classified_correctly)
        return correct / len(graded)

    @property
    def model_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for outcome in self.succeeded:
            if outcome.model_name:
                counts[outcome.model_name] = (
                    counts.get(outcome.model_name, 0) + 1
                )
        return counts


def load_dataset(path: Path = DEFAULT_DATASET) -> Dataset:
    return Dataset.model_validate_json(path.read_text(encoding="utf-8"))


def run_router_case(
    case: EvaluationCase, *, client: genai.Client, settings: Settings
) -> CaseOutcome:
    """The full pipeline: classify, route, execute.

    No publisher is passed on purpose -- an evaluation run would otherwise
    pour synthetic events into the same topic real traffic uses, and
    quietly corrupt the analytics it is meant to inform.
    """
    try:
        result = run_pipeline(case.text, client=client, settings=settings)
    except EXECUTION_FAILURES as error:
        return CaseOutcome(
            case_id=case.id,
            ok=False,
            error=f"{type(error).__name__}: {error}",
            expected_task_type=case.expected_task_type,
        )

    return CaseOutcome(
        case_id=case.id,
        ok=True,
        model_name=result.response.model_name,
        cost_usd=result.total_cost_usd,
        latency_ms=result.latency_ms,
        predicted_task_type=result.profile.task_type,
        expected_task_type=case.expected_task_type,
        fallback_used=result.response.fallback_used,
    )


def run_baseline_case(
    case: EvaluationCase,
    *,
    client: genai.Client,
    model_name: str,
) -> CaseOutcome:
    """Always the same model, no classification step.

    This is the null hypothesis the router has to beat: the obvious thing
    you would do if you had never built a router at all.
    """
    started = time.perf_counter()
    try:
        response = execute_request(
            case.text, client=client, model_name=model_name
        )
    except EXECUTION_FAILURES as error:
        return CaseOutcome(
            case_id=case.id,
            ok=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(error).__name__}: {error}",
            expected_task_type=case.expected_task_type,
        )

    return CaseOutcome(
        case_id=case.id,
        ok=True,
        model_name=response.model_name,
        cost_usd=response.estimated_cost_usd,
        latency_ms=(time.perf_counter() - started) * 1000,
        expected_task_type=case.expected_task_type,
    )


def money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def format_report(
    dataset: Dataset,
    router: StrategySummary,
    baseline: StrategySummary,
    baseline_model: str,
) -> str:
    lines = [
        f"Dataset      : {dataset.label} ({len(dataset.cases)} cases run)",
        f"Baseline     : always {baseline_model}",
        "",
        f"{'':<16} {'router':>14} {'baseline':>14}",
        f"{'total cost':<16} {money(router.total_cost_usd):>14} "
        f"{money(baseline.total_cost_usd):>14}",
        f"{'avg latency':<16} {router.avg_latency_ms:>13.0f}m "
        f"{baseline.avg_latency_ms:>13.0f}m",
        f"{'failures':<16} {router.failures:>14} {baseline.failures:>14}",
    ]

    accuracy = router.routing_accuracy
    if accuracy is not None:
        lines.append(f"{'classifier':<16} {accuracy:>13.0%} {'-':>14}")
    if router.fallbacks:
        lines.append(
            f"{'fallbacks':<16} {router.fallbacks:>14} {'-':>14}"
        )

    router_cost = router.total_cost_usd
    base_cost = baseline.total_cost_usd
    lines.append("")
    if router_cost is None or base_cost is None:
        lines.append(
            "Cost comparison unavailable: at least one run had an "
            "unpriced model."
        )
    elif base_cost == 0:
        lines.append("Baseline cost was zero; nothing to compare against.")
    else:
        saved = base_cost - router_cost
        pct = saved / base_cost
        verdict = "cheaper" if saved > 0 else "MORE EXPENSIVE"
        lines.append(
            f"Routing was {verdict} by {money(abs(saved))} "
            f"({abs(pct):.1%}) across {len(router.succeeded)} answered cases."
        )
        lines.append(
            "Cost only. This says nothing about whether the answers were "
            "as good."
        )

    # Per case, because a single case routed to an expensive model can
    # outweigh savings on every other case -- and a total alone hides that.
    by_case = {o.case_id: o for o in baseline.outcomes}
    lines.extend([
        "",
        f"{'case':<18} {'routed to':<24} {'router':>10} {'baseline':>10} "
        f"{'delta':>10}",
    ])
    for routed in router.outcomes:
        base = by_case.get(routed.case_id)
        if not routed.ok or base is None or not base.ok:
            lines.append(f"  {routed.case_id:<16} {'(failed)':<24}")
            continue
        delta = (
            None
            if routed.cost_usd is None or base.cost_usd is None
            else base.cost_usd - routed.cost_usd
        )
        delta_text = "n/a" if delta is None else f"{delta:+.6f}"
        lines.append(
            f"  {routed.case_id:<16} {routed.model_name or '-':<24} "
            f"{money(routed.cost_usd):>10} {money(base.cost_usd):>10} "
            f"{delta_text:>10}"
        )

    lines.extend(["", "Model distribution (router):"])
    for name, count in sorted(router.model_distribution.items()):
        lines.append(f"  {name:<26} {count:>3}")

    misrouted = [
        o
        for o in router.succeeded
        if o.classified_correctly is False
    ]
    if misrouted:
        lines.extend(["", "Classified differently than expected:"])
        for outcome in misrouted:
            lines.append(
                f"  {outcome.case_id:<20} expected "
                f"{outcome.expected_task_type}, got "
                f"{outcome.predicted_task_type}"
            )

    failures = [o for o in router.outcomes + baseline.outcomes if not o.ok]
    if failures:
        lines.extend(["", "Failures:"])
        for outcome in failures:
            lines.append(f"  {outcome.case_id:<20} {outcome.error}")

    return "\n".join(lines)


def evaluate(
    dataset: Dataset,
    *,
    client: genai.Client,
    settings: Settings,
    baseline_model: str,
    on_progress: object = None,
) -> tuple[StrategySummary, StrategySummary]:
    router = StrategySummary(ROUTER)
    baseline = StrategySummary(BASELINE)

    for case in dataset.cases:
        if on_progress is not None:
            on_progress(case)
        router.outcomes.append(
            run_router_case(case, client=client, settings=settings)
        )
        baseline.outcomes.append(
            run_baseline_case(
                case, client=client, model_name=baseline_model
            )
        )

    return router, baseline


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation",
        description="Compare routing against always using one model.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N cases."
    )
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="Defaults to the catalog's reasoning model (the strongest tier).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run, and how many API calls it costs, then stop.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdio()
    args = parse_args(sys.argv[1:] if argv is None else argv)

    dataset = load_dataset(args.dataset)
    if args.limit is not None:
        dataset = dataset.model_copy(
            update={"cases": dataset.cases[: args.limit]}
        )

    try:
        settings = load_settings()
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    baseline_model = args.baseline_model or settings.catalog.reasoning_model

    if args.dry_run:
        # Each case costs two calls on the router path (classify, execute)
        # and one on the baseline. Worth stating before spending it.
        print(f"Dataset      : {dataset.label}")
        print(f"Cases        : {len(dataset.cases)}")
        print(f"Baseline     : always {baseline_model}")
        print(f"API calls    : {len(dataset.cases) * 3} "
              f"({len(dataset.cases)} classify + "
              f"{len(dataset.cases)} routed + "
              f"{len(dataset.cases)} baseline)")
        return 0

    client = build_client(settings)
    print(
        f"Running {dataset.label}: {len(dataset.cases)} cases, "
        f"{len(dataset.cases) * 3} API calls.\n"
    )

    router, baseline = evaluate(
        dataset,
        client=client,
        settings=settings,
        baseline_model=baseline_model,
        on_progress=lambda case: print(f"  {case.id} ...", flush=True),
    )

    print()
    print(format_report(dataset, router, baseline, baseline_model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
