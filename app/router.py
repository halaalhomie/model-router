from app.schemas import ModelCatalog, RoutingDecision, TaskProfile


# TaskProfile.confidence describes the classification as a whole, not any
# one field -- there's no separate confidence for task_type vs context_size.
# So a low score means every field is suspect, not just one, which is why
# this check has to run before (and override) every rule below it: routing
# on context_size == "large" is pointless if we don't trust context_size.
#
# 0.6 is a starting guess, not a measured value -- live runs so far have all
# scored 0.8-1.0, so this has never actually fired yet. Once Phase 5 gives
# us real accuracy-vs-confidence data, this should be tuned from that, not
# left as a guess.
CONFIDENCE_THRESHOLD = 0.6


def select_model(profile: TaskProfile, catalog: ModelCatalog) -> RoutingDecision:
    """Select a model using a small, transparent first-pass policy."""
    if profile.confidence < CONFIDENCE_THRESHOLD:
        return RoutingDecision(
            model_name=catalog.reasoning_model,
            reason=(
                f"Classification confidence ({profile.confidence:.2f}) is "
                f"below {CONFIDENCE_THRESHOLD:.2f}, so the rest of the "
                f"profile isn't trusted enough to route on -- using the "
                f"strongest model as the safe default instead."
            ),
        )

    if profile.context_size == "large":
        return RoutingDecision(
            model_name=catalog.long_context_model,
            reason="The request needs a model configured for large context.",
        )

    if profile.task_type == "coding":
        return RoutingDecision(
            model_name=catalog.code_model,
            reason="The request is primarily a coding task.",
        )

    if profile.difficulty == "high" or profile.reasoning_required == "high":
        return RoutingDecision(
            model_name=catalog.reasoning_model,
            reason="The request needs high-level reasoning or has high difficulty.",
        )

    return RoutingDecision(
        model_name=catalog.fast_model,
        reason="The request does not require a specialized model.",
    )
