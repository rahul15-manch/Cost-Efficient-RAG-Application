from app.models import MetricsSnapshot

# In-memory metrics store
_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "sum_retrieval_ms": 0.0,
    "sum_generation_ms": 0.0,
    "sum_total_ms": 0.0,
    "sum_confidence": 0.0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0
}

def record_success(retrieval_ms: float, generation_ms: float, total_ms: float, confidence: float, prompt_tokens: int, completion_tokens: int):
    _metrics["total_requests"] += 1
    _metrics["successful_requests"] += 1
    _metrics["sum_retrieval_ms"] += retrieval_ms
    _metrics["sum_generation_ms"] += generation_ms
    _metrics["sum_total_ms"] += total_ms
    _metrics["sum_confidence"] += confidence
    _metrics["total_prompt_tokens"] += prompt_tokens
    _metrics["total_completion_tokens"] += completion_tokens

def record_failure():
    _metrics["total_requests"] += 1
    _metrics["failed_requests"] += 1

def snapshot() -> MetricsSnapshot:
    succ = _metrics["successful_requests"]
    return MetricsSnapshot(
        total_requests=_metrics["total_requests"],
        successful_requests=succ,
        failed_requests=_metrics["failed_requests"],
        avg_retrieval_ms=round(_metrics["sum_retrieval_ms"] / succ, 2) if succ > 0 else 0.0,
        avg_generation_ms=round(_metrics["sum_generation_ms"] / succ, 2) if succ > 0 else 0.0,
        avg_total_ms=round(_metrics["sum_total_ms"] / succ, 2) if succ > 0 else 0.0,
        avg_confidence=round(_metrics["sum_confidence"] / succ, 4) if succ > 0 else 0.0,
        total_prompt_tokens=_metrics["total_prompt_tokens"],
        total_completion_tokens=_metrics["total_completion_tokens"]
    )

def reset_metrics():
    for k in _metrics:
        _metrics[k] = 0.0 if "sum" in k else 0
