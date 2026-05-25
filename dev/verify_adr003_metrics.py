# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Verify calculated comparison tables in ADR-003 against source metric tables."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


METRIC_COLUMNS = ("P", "R", "mAP50", "mAP50-95")
CLASSES = ("all", "smoke", "fire", "person")
F_COMPARISONS = {
    "F1 - F0": ("F1", "F0"),
    "F2 - F1": ("F2", "F1"),
    "F4 - F3": ("F4", "F3"),
    "F3 - F1": ("F3", "F1"),
    "F2 - F0": ("F2", "F0"),
    "F4 - F0": ("F4", "F0"),
}
D_EXPERIMENTS = ("D1", "D2.1", "D2.2", "D2.3", "D3")


@dataclass(frozen=True)
class TableRow:
    cells: list[str]
    line_number: int


def _clean_cell(cell: str) -> str:
    return cell.strip().replace("**", "").replace("\\@", "@")


def _parse_decimal(text: str) -> Decimal:
    return Decimal(_clean_cell(text).replace("+", ""))


def _format_delta(value: Decimal) -> str:
    value = value.quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)
    return f"{value:+.5f}"


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", cell.strip()) for cell in cells)


def _iter_tables(lines: list[str]) -> list[tuple[int, list[TableRow]]]:
    tables: list[tuple[int, list[TableRow]]] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue

        start = index
        rows: list[TableRow] = []
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            cells = [_clean_cell(cell) for cell in lines[index].strip().strip("|").split("|")]
            rows.append(TableRow(cells, index + 1))
            index += 1
        tables.append((start + 1, rows))
    return tables


def _heading_before(lines: list[str], line_number: int) -> str:
    for index in range(line_number - 1, -1, -1):
        if lines[index].startswith("### "):
            return lines[index].strip()
    return ""


def _experiment_from_heading(heading: str) -> str | None:
    match = re.search(r"\b(F[0-4]|D2\.[123]|D1|D3)\b\s+验证结果", heading)
    return match.group(1) if match else None


def _metric_values_from_row(cells: list[str]) -> dict[str, Decimal]:
    values = [_parse_decimal(cell) for cell in cells[-4:]]
    return dict(zip(METRIC_COLUMNS, values))


def parse_source_metrics(markdown: str) -> dict[str, dict[str, dict[str, Decimal]]]:
    lines = markdown.splitlines()
    metrics: dict[str, dict[str, dict[str, Decimal]]] = {}

    for line_number, rows in _iter_tables(lines):
        heading = _heading_before(lines, line_number)
        experiment = _experiment_from_heading(heading)
        if not experiment:
            continue

        for row in rows[2:]:
            if not row.cells or _is_separator_row(row.cells):
                continue
            class_name = row.cells[0]
            if class_name in CLASSES:
                metrics.setdefault(experiment, {})[class_name] = _metric_values_from_row(row.cells)

    return metrics


def _table_after_heading(markdown: str, heading_pattern: str) -> list[TableRow]:
    lines = markdown.splitlines()
    start_index = next(
        (index for index, line in enumerate(lines) if re.search(heading_pattern, line)),
        None,
    )
    if start_index is None:
        raise ValueError(f"Heading not found: {heading_pattern}")

    for line_number, rows in _iter_tables(lines[start_index + 1 :]):
        return [TableRow(row.cells, row.line_number + start_index + 1) for row in rows]
    raise ValueError(f"No table found after heading: {heading_pattern}")


def _check_equal(errors: list[str], label: str, expected: Decimal, actual_text: str, line_number: int) -> None:
    actual = _parse_decimal(actual_text)
    expected = expected.quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)
    if actual != expected:
        errors.append(f"line {line_number}: {label} expected {_format_delta(expected)}, found {actual_text}")


def verify_f_comparison_table(markdown: str, metrics: dict[str, dict[str, dict[str, Decimal]]]) -> list[str]:
    errors: list[str] = []
    rows = _table_after_heading(markdown, r"简要分析")
    seen: set[str] = set()

    for row in rows[2:]:
        if len(row.cells) < 2:
            continue
        comparison = row.cells[0]
        if comparison not in F_COMPARISONS:
            continue
        left, right = F_COMPARISONS[comparison]
        expected = metrics[left]["all"]["mAP50-95"] - metrics[right]["all"]["mAP50-95"]
        _check_equal(errors, f"{comparison} all mAP50-95 change", expected, row.cells[1], row.line_number)
        seen.add(comparison)

    missing = set(F_COMPARISONS) - seen
    errors.extend(f"missing F comparison row: {comparison}" for comparison in sorted(missing))
    return errors


def verify_d_delta_table(markdown: str, metrics: dict[str, dict[str, dict[str, Decimal]]]) -> list[str]:
    errors: list[str] = []
    rows = _table_after_heading(markdown, r"相对 F2")
    expected_rows = {(experiment, class_name) for experiment in D_EXPERIMENTS for class_name in CLASSES}
    seen: set[tuple[str, str]] = set()

    for row in rows[2:]:
        if len(row.cells) < 6:
            continue
        experiment, class_name = row.cells[0], row.cells[1]
        if experiment not in D_EXPERIMENTS or class_name not in CLASSES:
            continue

        for index, metric_name in enumerate(METRIC_COLUMNS, start=2):
            expected = metrics[experiment][class_name][metric_name] - metrics["F2"][class_name][metric_name]
            _check_equal(errors, f"{experiment} {class_name} delta {metric_name}", expected, row.cells[index], row.line_number)
        seen.add((experiment, class_name))

    missing = expected_rows - seen
    errors.extend(f"missing D delta row: {experiment} {class_name}" for experiment, class_name in sorted(missing))
    return errors


def verify_d_summary_table(markdown: str, metrics: dict[str, dict[str, dict[str, Decimal]]]) -> list[str]:
    errors: list[str] = []
    rows = _table_after_heading(markdown, r"D 系列完整分析")
    expected_experiments = ("F2", *D_EXPERIMENTS)
    seen: set[str] = set()

    for row in rows[2:]:
        if len(row.cells) < 5:
            continue
        experiment = row.cells[0]
        if experiment not in expected_experiments:
            continue

        expected_values = (
            metrics[experiment]["all"]["mAP50-95"],
            metrics[experiment]["fire"]["mAP50-95"],
            metrics[experiment]["person"]["mAP50-95"],
        )
        for metric_label, expected, actual_text in zip(("all", "fire", "person"), expected_values, row.cells[2:5]):
            _check_equal(errors, f"{experiment} summary {metric_label} mAP50-95", expected, actual_text, row.line_number)
        seen.add(experiment)

    missing = set(expected_experiments) - seen
    errors.extend(f"missing D summary row: {experiment}" for experiment in sorted(missing))
    return errors


def verify_adr003_metrics(path: Path) -> list[str]:
    markdown = path.read_text(encoding="utf-8")
    metrics = parse_source_metrics(markdown)
    required = {"F0", "F1", "F2", "F3", "F4", "D1", "D2.1", "D2.2", "D2.3", "D3"}
    errors = [f"missing source metric table: {experiment}" for experiment in sorted(required - set(metrics))]

    for experiment, class_metrics in metrics.items():
        missing_classes = set(CLASSES) - set(class_metrics)
        errors.extend(f"missing source row: {experiment} {class_name}" for class_name in sorted(missing_classes))

    if errors:
        return errors

    errors.extend(verify_f_comparison_table(markdown, metrics))
    errors.extend(verify_d_delta_table(markdown, metrics))
    errors.extend(verify_d_summary_table(markdown, metrics))
    return errors


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/ADR-003-fair-batch16-lr001-experiment-plan.md")
    errors = verify_adr003_metrics(path)
    if errors:
        print("ADR-003 metric verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("ADR-003 metric verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
