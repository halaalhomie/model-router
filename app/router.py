from app.schemas import ModelCatalog, RoutingDecision, TaskProfile


def select_model(profile: TaskProfile, catalog: ModelCatalog) -> RoutingDecision:
    """Select a model using a small, transparent first-pass policy."""
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
