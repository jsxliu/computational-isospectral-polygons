# This file keeps polygon generation and spectral screening consistent.
# It counts saved candidates, chooses the grid refinement range, records
# what happens at each spectral stage, and prints the final run summary.

from __future__ import annotations

from dataclasses import dataclass, field


# Start at h=1/3 instead of h=1/2 for inputs at or above this candidate count.
LARGE_CANDIDATE_COUNT = 1_000_000


# Count data lines after the header in a generated polygon CSV.
def count_csv_rows(path: str) -> int:
    with open(path, encoding="utf-8") as source:
        return max(0, sum(1 for _ in source) - 1)


# Choose the starting grid spacing, h = 1 / denominator.
def first_grid_denominator(candidate_count: int) -> int:
    return 3 if candidate_count >= LARGE_CANDIDATE_COUNT else 2


# Choose the final grid denominator from the maximum step length.
def final_grid_denominator(max_length: int) -> int:
    return 6 + max(0, max_length - 2)


# Name the candidate CSV for a polygon class.
def candidate_filename(n_sides: int, max_length: int) -> str:
    return f"candidate_polygons_{n_sides}_{max_length}.csv"


# Record counts and time data for one spectral screening stage.
@dataclass
class Phase2Stage:
    denominator: int
    mode: str
    input_polygons: int
    output_count: int
    seconds: float
    output_label: str


# Collect spectral results for the final run summary.
@dataclass
class Phase2Summary:
    stages: list[Phase2Stage] = field(default_factory=list)
    accepted: bool = False
    final_denominator: int | None = None
    planned_final_denominator: int = 6
    final_pairs_csv: str | None = None
    eigenvalues_csv: str | None = None
    plot_dir: str | None = None
    seconds: float = 0.0
    final_match_count: int | None = None
    ambiguous: bool = False


# Print polygon counts, spectral results, output paths, and time data.
def print_pipeline_summary(
    *,
    n_sides: int,
    max_length: int,
    phase2: Phase2Summary,
    total_seconds: float,
    unique_polygons: int | None = None,
    candidate_count: int | None = None,
    generation_seconds: float | None = None,
) -> None:
    lines = [
        "",
        "=" * 60,
        "  RUN SUMMARY",
        "=" * 60,
        f"  Case: ({n_sides}, {max_length})",
    ]

    if unique_polygons is not None:
        lines.extend(["", f"  Generated unique polygons: {unique_polygons:,}"])
        if generation_seconds is not None:
            lines.append(f"  Generation time:           {generation_seconds:.1f}s")

    if candidate_count is not None:
        lines.append(f"  Invariant-filter candidates: {candidate_count:,}")

    if phase2.stages:
        lines.extend(["", "  Spectral refinement:"])
        for stage in phase2.stages:
            lines.append(
                f"    h=1/{stage.denominator:<2} {stage.mode:<9} "
                f"in={stage.input_polygons:>9,}  "
                f"{stage.output_label}={stage.output_count:>9,}  "
                f"{stage.seconds:.1f}s"
            )

    if phase2.accepted:
        lines.append(
            f"\n  Numerical candidate pairs accepted at h=1/"
            f"{phase2.final_denominator}: {phase2.final_match_count}"
        )
    elif phase2.ambiguous:
        lines.append(
            f"\n  Ambiguous at h=1/{phase2.planned_final_denominator}: "
            f"{phase2.final_match_count} pairs remain"
        )
    else:
        lines.append("\n  No pair survived the full numerical screening.")

    if phase2.final_pairs_csv:
        lines.extend(["", f"  Pair CSV:       {phase2.final_pairs_csv}"])
    if phase2.eigenvalues_csv:
        lines.append(f"  Eigenvalue CSV: {phase2.eigenvalues_csv}")
    if phase2.plot_dir:
        lines.append(f"  Plots:          {phase2.plot_dir}")

    lines.extend(
        [
            "",
            f"  Spectral time: {phase2.seconds:.1f}s",
            f"  Total time:    {total_seconds:.1f}s",
            "",
            "=" * 60,
            "",
        ]
    )
    print("\n".join(lines), flush=True)
