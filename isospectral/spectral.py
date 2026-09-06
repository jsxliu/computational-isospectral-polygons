# This file screens polygons using finite-difference Dirichlet eigenvalues.
# For each grid spacing, the search follows five steps:
# 1. Build a discrete Dirichlet Laplacian for every polygon.
# 2. Group polygons by spectral invariants and their first five eigenvalues.
# 3. Compute up to 40 eigenvalues for every polygon in a shared group for more detailed comparison.
# 4. Keep polygons whose detailed spectra agree within a given tolerance.
# 5. Repeat on finer grids, then enumerate and report the final candidate pairs.

import csv
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice, repeat
from .config import pool_kwargs

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import ArpackError, eigsh
from shapely import contains_xy
from shapely.geometry import Polygon

from .pipeline_common import Phase2Stage, Phase2Summary, count_csv_rows
from .spectral_output import (
    geometry_header,
    iter_pair_rows,
    pair_header,
    pair_row,
    plot_pairs,
    polygon_row,
    write_eigenvalue_csv,
)


SIGNATURE_EIGENVALUES = 5  # Maximum number of eigenvalues per coarse signature.
VERIFICATION_EIGENVALUES = 40  # Maximum number of eigenvalues per detailed comparison.
SIGNATURE_DECIMALS = 6  # Decimal places retained in coarse signatures.
EIGENVALUE_MATCH_ATOL = 1e-8  # Comparison absolute tolerance.
FINAL_PAIR_ENUMERATION_LIMIT = 20_000 # Maximum polygons allowed for final pair enumeration.


# Create a Boolean grid to mark which grid points lie inside the polygon.
# If a gridpoint is inside the polygon, it is marked as True. Otherwise, it is False.
def interior_mask(polygon, grid_spacing):
    coordinates = np.asarray(polygon, dtype=float)
    minimum_x, minimum_y = coordinates.min(axis=0) - 1
    maximum_x, maximum_y = coordinates.max(axis=0) + 1

    x_values = np.round(
        np.arange(minimum_x, maximum_x + 0.5 * grid_spacing, grid_spacing), 12
    )
    y_values = np.round(
        np.arange(minimum_y, maximum_y + 0.5 * grid_spacing, grid_spacing), 12
    )
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    shrunk = Polygon(polygon).buffer(-grid_spacing / 2.0)
    return contains_xy(shrunk, x_grid, y_grid)


# Turn the selected interior grid points into the discrete Dirichlet matrix.
def build_laplacian(inside, grid_spacing):
    size = int(inside.sum())
    point_ids = -np.ones_like(inside, dtype=np.int64)
    point_ids[inside] = np.arange(size)

    vertical = inside[:-1, :] & inside[1:, :]
    horizontal = inside[:, :-1] & inside[:, 1:]
    vertical_first = point_ids[:-1, :][vertical]
    vertical_second = point_ids[1:, :][vertical]
    horizontal_first = point_ids[:, :-1][horizontal]
    horizontal_second = point_ids[:, 1:][horizontal]

    # Five-point stencil for -Δ: (4u - u_left - u_right - u_up - u_down) / h².
    # Neighbors outside the selected points contribute zero (Dirichlet condition).
    off_rows = np.concatenate((
        vertical_first, vertical_second, horizontal_first, horizontal_second,
    ))
    off_columns = np.concatenate((
        vertical_second, vertical_first, horizontal_second, horizontal_first,
    ))
    diagonal = np.arange(size)
    rows = np.concatenate((diagonal, off_rows))
    columns = np.concatenate((diagonal, off_columns))
    values = np.concatenate(
        (np.full(size, 4.0), np.full(len(off_rows), -1.0))
    )
    matrix = coo_matrix((values, (rows, columns)), shape=(size, size)).tocsr()
    return matrix * (1.0 / grid_spacing**2)


# Return the smallest eigenvalues using the appropriate matrix solver for comparison.
# For a coarse signature, we use 5 eigenvalues; for detailed verification we use 40.
def _solve_lowest_eigenvalues(matrix, count):
    size = matrix.shape[0]
    count = min(count, size)
    if 2 * count >= size:
        values = eigh(matrix.toarray(), subset_by_index=(0, count - 1),
                      eigvals_only=True)
    else:
        try:
            values = eigsh(matrix, k=count, sigma=0, which="LM",
                           return_eigenvectors=False)
        except ArpackError as initial_error:
            last_error = initial_error
            max_iterations = max(1_000, 10 * size)
            for shift in (0.0, 1e-10, 1e-8, 1e-6):
                for multiplier in (2, 3, 4):
                    search_vectors = min(size,
                                         max(multiplier * count + 20, count + 2))
                    try:
                        values = eigsh(
                            matrix, k=count, sigma=shift, which="LM",
                            return_eigenvectors=False, ncv=search_vectors,
                            maxiter=max_iterations, tol=1e-10)
                        return np.sort(values)
                    except ArpackError as error:
                        last_error = error
            raise last_error
    return np.sort(values)


# Create a coarse signature from up to five rounded discrete eigenvalues.
def create_coarse_signature(polygon, grid_spacing):
    matrix = build_laplacian(interior_mask(polygon, grid_spacing), grid_spacing)
    if matrix.shape[0] == 0:
        return ()
    eigenvalues = _solve_lowest_eigenvalues(matrix, SIGNATURE_EIGENVALUES)
    return tuple(np.round(eigenvalues, SIGNATURE_DECIMALS))


# Produce the detailed spectrum used to verify a coarse match. Shapes with fewer than
# three interior points are discarded due to weak approximations.
def compute_detailed_spectrum(polygon, grid_spacing):
    matrix = build_laplacian(interior_mask(polygon, grid_spacing), grid_spacing)
    if matrix.shape[0] < 3:
        return None
    return _solve_lowest_eigenvalues(matrix, VERIFICATION_EIGENVALUES)


# Read polygons from the CSV one at a time and assign each an ID.
def read_polygon_records(path):
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if (
            header is None
            or header[:3] != ["Area2", "Axial", "Diagonal"]
            or (len(header) - 3) % 2
        ):
            raise ValueError(f"invalid polygon header in {path}")
        for polygon_id, row in enumerate(reader, 1):
            if len(row) != len(header):
                raise ValueError(f"invalid polygon row {polygon_id} in {path}")
            values = tuple(map(int, row))
            coordinates = values[3:]
            polygon = list(zip(coordinates[::2], coordinates[1::2]))
            yield polygon_id, values[:3], polygon


# Compute and save coarse signature records for every candidate polygon.
def _write_coarse_signature_records(input_csv, output_tsv,
                                     grid_spacing, workers):
    started = time.time()
    written = 0
    print(f"  Computing signatures with {workers} workers", flush=True)

    with open(output_tsv, "w", encoding="utf-8", newline="") as output:
        with ProcessPoolExecutor(max_workers=workers, **pool_kwargs()) as pool:
            records = read_polygon_records(input_csv)
            while True:
                chunk = list(islice(records, 2_000))
                if not chunk:
                    break
                polygons = [record[2] for record in chunk]
                signatures = pool.map(
                    create_coarse_signature, polygons, repeat(grid_spacing)
                )
                for (polygon_id, invariants, polygon), signature in zip(
                    chunk, signatures
                ):
                    # Create a key from twice the area, boundary lengths, and five coarse eigenvalues.
                    area2, axial, diagonal = invariants
                    eigenvalues = "|".join(f"{value:.6f}" for value in signature)
                    key = f"{area2:010d}:{axial:06d}:{diagonal:06d}:{eigenvalues}"
                    coordinates = ",".join(
                        str(value) for point in polygon for value in point
                    )
                    output.write(
                        f"{key}\t{polygon_id}\t{coordinates}\n"
                    )
                written += len(chunk)
                if written % 20_000 == 0:
                    print(f"    signatures: {written:,}", flush=True)

    print(
        f"  Computed {written:,} signatures in "
        f"{time.time() - started:.1f}s",
        flush=True,
    )


# Sort polygons by their coarse signatures so possible matches appear together.
def _sort_candidates_by_coarse_signature(input_csv, denominator, workers):
    work_directory = tempfile.mkdtemp()
    dump_path = os.path.join(work_directory, "signatures.tsv")
    sorted_path = os.path.join(work_directory, "signatures_sorted.tsv")
    try:
        _write_coarse_signature_records(input_csv, dump_path, 1.0 / denominator, workers)
        environment = dict(os.environ, LC_ALL="C")
        subprocess.run(["sort", "-T", work_directory, "-t", "\t", "-k", "1,1", dump_path, "-o", sorted_path], check=True, env=environment)
    except BaseException:
        shutil.rmtree(work_directory, ignore_errors=True)
        raise
    return sorted_path, work_directory


# Creates and returns groups of polygons that share a coarse signature key.
def _group_polygons_with_matching_signatures(sorted_tsv):
    current_key = None
    group = []
    with open(sorted_tsv, encoding="utf-8") as source:
        for line in source:
            key, polygon_id, coordinates = line.rstrip("\n").split("\t", 2)
            if current_key is not None and key != current_key:
                if len(group) >= 2:
                    yield group
                group = []
            current_key = key
            group.append((int(polygon_id), coordinates))
    if len(group) >= 2:
        yield group


# Check which polygons in one coarse group have detailed matching spectra.
# In particular, the first 40 eigenvalues are compared and returns pairs that survive.
def _find_detailed_matches(arguments):
    group, grid_spacing, collect_pairs = arguments
    spectra_by_length = {}
    for polygon_id, coordinate_text in group:
        values = tuple(map(int, coordinate_text.split(",")))
        polygon = list(zip(values[::2], values[1::2]))
        spectrum = compute_detailed_spectrum(polygon, grid_spacing)
        if spectrum is not None:
            spectra_by_length.setdefault(len(spectrum), []).append(
                (polygon_id, spectrum)
            )

    pair_count = 0
    matches = [] if collect_pairs else set()
    for spectra in spectra_by_length.values():
        spectra.sort(key=lambda item: item[1][0])
        for first_index, (first_id, first_spectrum) in enumerate(spectra):
            for second_id, second_spectrum in spectra[first_index + 1 :]:
                if second_spectrum[0] - first_spectrum[0] > EIGENVALUE_MATCH_ATOL:
                    break
                if np.allclose(first_spectrum, second_spectrum,
                               atol=EIGENVALUE_MATCH_ATOL, rtol=0.0):
                    pair_count += 1
                    if collect_pairs:
                        matches.append((first_id, second_id))
                    else:
                        matches.update((first_id, second_id))
    return pair_count, matches


# Save the surviving polygons for testing on the next finer grid.
def _write_surviving_polygons(output_csv, input_csv, polygon_ids, n_sides):
    with open(output_csv, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(geometry_header(n_sides))
        for polygon_id, invariants, polygon in read_polygon_records(input_csv):
            if polygon_id in polygon_ids:
                writer.writerow(polygon_row(invariants, polygon))
    return len(polygon_ids)


# Group candidates, verify possible matches, and save survivors for the next grid.
def _screen_grid_for_survivors(input_csv, output_csv, denominator,
                               n_sides, workers):
    sorted_tsv, work_directory = _sort_candidates_by_coarse_signature(
        input_csv, denominator, workers
    )
    spacing = 1.0 / denominator
    survivor_ids = set()
    verified_pair_count = 0
    processed_group_count = 0
    # Process verification jobs in batches to limit memory use.
    batch_limit = 32 * workers

    try:
        with ProcessPoolExecutor(max_workers=workers, **pool_kwargs()) as pool:
            futures = []

            def consume_batch(batch):
                nonlocal verified_pair_count, processed_group_count
                for future in as_completed(batch):
                    pair_count, polygon_ids = future.result()
                    verified_pair_count += pair_count
                    survivor_ids.update(polygon_ids)
                    processed_group_count += 1

            for group in _group_polygons_with_matching_signatures(sorted_tsv):
                # Keep only the polygon IDs needed for the next finer grid.
                futures.append(pool.submit(
                    _find_detailed_matches, (group, spacing, False)
                ))
                if len(futures) == batch_limit:
                    consume_batch(futures)
                    futures = []
                    print(
                        f"    verified groups: {processed_group_count:,}; "
                        f"survivors: {len(survivor_ids):,}",
                        flush=True,
                    )
            consume_batch(futures)

        surviving_polygon_count = len(survivor_ids)
        if output_csv is not None:
            surviving_polygon_count = _write_surviving_polygons(
                output_csv, input_csv, survivor_ids, n_sides
            )
    finally:
        shutil.rmtree(work_directory, ignore_errors=True)

    print(
        f"  Verified pairs: {verified_pair_count:,}; "
        f"survivors: {surviving_polygon_count:,}",
        flush=True,
    )
    return verified_pair_count, surviving_polygon_count


# Find and save every matching polygon pair at the final grid level.
def _save_final_matching_pairs(input_csv, output_csv, denominator,
                               n_sides, workers):
    sorted_tsv, work_directory = _sort_candidates_by_coarse_signature(
        input_csv, denominator, workers
    )
    spacing = 1.0 / denominator
    polygons = {
        polygon_id: (invariants, polygon)
        for polygon_id, invariants, polygon in read_polygon_records(input_csv)
    }
    verification_tasks = (
        (group, spacing, True)
        for group in _group_polygons_with_matching_signatures(sorted_tsv)
    )
    verified_pair_count = 0

    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(pair_header(n_sides))
            # Compare groups in parallel, then save their matching pairs in order.
            with ProcessPoolExecutor(max_workers=workers,
                                     **pool_kwargs()) as pool:
                verified_groups = pool.map(
                    _find_detailed_matches, verification_tasks
                )
                for pair_count, pairs in verified_groups:
                    verified_pair_count += pair_count
                    for first_id, second_id in sorted(pairs):
                        invariants, first = polygons[first_id]
                        _, second = polygons[second_id]
                        writer.writerow(pair_row(
                            first_id, second_id, invariants, first, second
                        ))
    finally:
        shutil.rmtree(work_directory, ignore_errors=True)

    print(f"  Verified pairs: {verified_pair_count:,}", flush=True)
    return verified_pair_count


# Record the detailed spectra supporting each final candidate pair.
def write_final_spectra(pair_csv, n_sides, denominator):
    spacing = 1.0 / denominator
    output_path = pair_csv.replace(".csv", "_eigenvalues.csv")

    def eigenvalue_rows():
        for benchmark_id, candidate_id, _, benchmark, candidate in iter_pair_rows(
            pair_csv, n_sides
            ):
            benchmark_spectrum = compute_detailed_spectrum(benchmark, spacing)
            candidate_spectrum = compute_detailed_spectrum(candidate, spacing)
            for index, (benchmark_value, candidate_value) in enumerate(
                zip(benchmark_spectrum, candidate_spectrum), 1
            ):
                yield [benchmark_id, candidate_id, denominator, index, benchmark_value, candidate_value]

    write_eigenvalue_csv(output_path, eigenvalue_rows())
    print(f"  Saved verification spectra: {output_path}", flush=True)
    return output_path


# Run the complete spectral search. Summary of algorithm:
# At each grid, group polygons by their spectral invariants and first five eigenvalues,
# verify possible matches using up to 40 eigenvalues, and send the survivors to
# the next finer grid. At the final grid size, save the pairs, spectra, and plot them.
def run_spectral_search(*, n_sides, max_length, candidate_csv,
                        start_denominator, end_denominator, workers, output_dir):
    started = time.time()
    summary = Phase2Summary(planned_final_denominator=end_denominator)
    tag = f"{n_sides}_{max_length}"
    current_candidates_csv = candidate_csv
    current_polygon_count = count_csv_rows(current_candidates_csv)

    for denominator in range(start_denominator, end_denominator + 1):
        spacing = 1.0 / denominator
        pair_csv = os.path.join(
            output_dir, f"isospectral_pairs_{tag}_1over{denominator}.csv"
        )
        input_polygon_count = current_polygon_count
        print(
            f"\n  Grid h=1/{denominator} ({spacing:.6g}): "
            f"{input_polygon_count:,} polygons",
            flush=True,
        )
        stage_started = time.time()
        is_final_grid = denominator == end_denominator
        can_enumerate_final_pairs = (
            is_final_grid and input_polygon_count <= FINAL_PAIR_ENUMERATION_LIMIT
        )

        if can_enumerate_final_pairs:
            # The final small stage retains complete pairs for reporting.
            verified_pair_count = _save_final_matching_pairs(
                current_candidates_csv, pair_csv, denominator, n_sides, workers
            )
            mode = "pairwise"
        else:
            # Other stages retain only IDs needed at the next resolution.
            next_candidates_csv = None
            if not is_final_grid:
                next_candidates_csv = os.path.join(
                    output_dir,
                    f"unique_polygons_{tag}_1over{denominator + 1}.csv",
                )
            verified_pair_count, surviving_polygon_count = (
                _screen_grid_for_survivors(
                    current_candidates_csv, next_candidates_csv,
                    denominator, n_sides, workers
                )
            )
            if next_candidates_csv is not None:
                current_candidates_csv = next_candidates_csv
                current_polygon_count = surviving_polygon_count
            mode = "screen"

        stage_seconds = time.time() - stage_started
        print(
            f"  Grid h=1/{denominator} time: {stage_seconds:.1f}s",
            flush=True,
        )
        summary.stages.append(Phase2Stage(denominator=denominator, mode=mode,
            input_polygons=input_polygon_count, output_count=verified_pair_count, seconds=stage_seconds, output_label="pairs"))

        if verified_pair_count == 0:
            print("  No candidate survives; finer grids cannot restore one.")
            summary.final_match_count = 0
            if can_enumerate_final_pairs:
                summary.final_pairs_csv = pair_csv
            break

        if is_final_grid:
            summary.final_match_count = verified_pair_count
            summary.final_denominator = denominator
            if can_enumerate_final_pairs:
                summary.final_pairs_csv = pair_csv
                summary.accepted = True
                summary.eigenvalues_csv = write_final_spectra(
                    pair_csv, n_sides, denominator
                )
                summary.plot_dir = os.path.join(
                    output_dir,
                    f"iso_plots_{tag}_1over{denominator}",
                )
                plot_pairs(pair_csv, n_sides, summary.plot_dir)
            else:
                summary.ambiguous = True

    summary.seconds = time.time() - started
    return summary
