# This file makes spectral search results easier to save and inspect by
# formatting polygons, candidate pairs, and eigenvalues for CSV output.
# It also reads saved pairs back into polygon coordinates and creates
# the comparison plots of candidate polygons.

import csv
import os


# Build CSV column names for polygon invariants and coordinates.
def geometry_header(n_sides):
    return ["Area2", "Axial", "Diagonal"] + [
        label
        for index in range(1, n_sides + 1)
        for label in (f"X{index}", f"Y{index}")
    ]


# Format a polygon and its invariants as a CSV row.
def polygon_row(invariants, polygon):
    return [*invariants] + [
        coordinate for point in polygon for coordinate in point
    ]


# Build CSV column names for matched polygon pairs.
def pair_header(n_sides):
    return ["bench_idx", "cand_idx", "Area2", "Axial", "Diagonal"] + [
        f"{prefix}{axis}{index}"
        for prefix in ("Bench", "Cand")
        for index in range(1, n_sides + 1)
        for axis in ("X", "Y")
    ]


# Format a polygon pair and its invariants as a CSV row.
def pair_row(first_id, second_id, invariants, first, second):
    return (
        [first_id, second_id, *invariants]
        + [value for point in first for value in point]
        + [value for point in second for value in point]
    )


# Save eigenvalue comparisons for final pairs to a CSV file.
def write_eigenvalue_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "bench_idx",
                "cand_idx",
                "h_denom",
                "eigenvalue_index",
                "benchmark_value",
                "candidate_value",
            ]
        )
        writer.writerows(rows)


# Read saved pairs and rebuild their polygon coordinates.
def iter_pair_rows(path, n_sides):
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        next(reader, None)
        for row_number, row in enumerate(reader, 2):
            values = tuple(map(int, row))
            if len(values) != 5 + 4 * n_sides:
                raise ValueError(f"invalid pair row {row_number} in {path}")
            split = 5 + 2 * n_sides
            benchmark_values = values[5:split]
            candidate_values = values[split:]
            benchmark = list(
                zip(benchmark_values[::2], benchmark_values[1::2])
            )
            candidate = list(
                zip(candidate_values[::2], candidate_values[1::2])
            )
            yield values[0], values[1], values[2:5], benchmark, candidate


# Save a comparison plot of two polygons.
def plot_pair(
    candidate,
    benchmark,
    benchmark_id,
    candidate_id,
    output_directory,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_directory, exist_ok=True)

    candidate = candidate + [candidate[0]]
    benchmark = benchmark + [benchmark[0]]

    figure, axes = plt.subplots()
    axes.plot(*zip(*candidate), "o-", label=f"Polygon {candidate_id}")
    axes.plot(*zip(*benchmark), "x--", label=f"Polygon {benchmark_id}")
    axes.set_aspect("equal")
    axes.grid()
    axes.legend()

    output_path = os.path.join(
        output_directory,
        f"match_b{benchmark_id:03d}_c{candidate_id:03d}.png",
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


# Save comparison plots for all pairs in a CSV file.
def plot_pairs(pair_csv, n_sides, output_directory):
    for benchmark_id, candidate_id, _, benchmark, candidate in iter_pair_rows(
        pair_csv, n_sides
    ):
        plot_pair(
            candidate,
            benchmark,
            benchmark_id,
            candidate_id,
            output_directory,
        )
