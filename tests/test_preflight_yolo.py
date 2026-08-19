from preflight_yolo import BASELINE_GFLOPS, BASELINE_PARAMETERS, budget_failures


def test_candidate_within_both_baseline_budgets_passes() -> None:
    assert budget_failures(BASELINE_PARAMETERS, BASELINE_GFLOPS) == []


def test_budget_failures_report_each_exceeded_limit() -> None:
    failures = budget_failures(BASELINE_PARAMETERS + 1, BASELINE_GFLOPS + 0.001)

    assert failures == [
        "parameters 18,950,730 exceed 18,950,729",
        "GFLOPs 23.826000 exceed 23.825000",
    ]
