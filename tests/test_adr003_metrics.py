from pathlib import Path

from dev.verify_adr003_metrics import verify_adr003_metrics


def test_adr003_calculated_tables_match_source_metric_tables() -> None:
    errors = verify_adr003_metrics(Path("docs/ADR-003-fair-batch16-lr001-experiment-plan.md"))

    assert errors == []
