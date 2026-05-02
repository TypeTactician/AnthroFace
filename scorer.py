"""Scoring engine: aggregates per-metric scores into category and overall scores."""

CATEGORY_WEIGHTS = {
    "Symmetry": 0.20,
    "Proportions": 0.30,
    "Profile": 0.20,
    "Golden Ratio": 0.15,
    "Skin": 0.10,
}


def compute_category_scores(metrics: list[dict]) -> dict[str, float]:
    """Compute weighted average score per category."""
    categories: dict[str, list[float]] = {}
    for m in metrics:
        cat = m["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(m["score"])

    scores = {}
    for cat, vals in categories.items():
        scores[cat] = round(sum(vals) / len(vals), 1)
    return scores


def compute_overall_score(metrics: list[dict], category_scores: dict[str, float]) -> float:
    """Compute weighted overall score from category scores."""
    total = 0.0
    weight_sum = 0.0
    for cat, weight in CATEGORY_WEIGHTS.items():
        if cat in category_scores:
            total += category_scores[cat] * weight
            weight_sum += weight
    if weight_sum == 0:
        return 0.0
    return round(total / weight_sum, 1)


def run_scoring(metrics: list[dict]) -> dict:
    """Run the full scoring pipeline.

    Metrics should already have 'score' populated from metrics.py.
    This function computes category and overall scores.
    """
    category_scores = compute_category_scores(metrics)
    overall = compute_overall_score(metrics, category_scores)

    return {
        "overall": overall,
        "category_scores": category_scores,
        "metrics": metrics,
    }
