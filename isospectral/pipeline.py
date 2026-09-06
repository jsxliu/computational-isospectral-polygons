# This file coordinates the pipeline by using polygen.py to generate
# polygons across multiple workers and merging their results. It removes
# duplicates, keeps polygons with matching area and perimeter, and sends
# the candidates to spectral.py for numerical screening. Finally, it saves
# the results and prints the run summary.

import argparse
import csv
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from .config import best_mp_context, njit, recommended_workers
import numpy as np


from .pipeline_common import (
    Phase2Summary,
    candidate_filename,
    final_grid_denominator,
    first_grid_denominator,
    print_pipeline_summary,
)
from .polygen import (
    canonical_hashes,
    decode_direction_pattern,
    ensure_reachability_cache,
    generate_polygons_for_pattern,
    polygon_invariants_batch,
    split_range,
    suffix_direction_counts,
)
from .spectral import run_spectral_search

PROGRESS_INTERVAL = 5_000        # Patterns checked between progress updates.
WORKER_BUFFER_RECORDS = 100_000  # Polygons kept in memory before saving.
BINARY_READ_RECORDS = 500_000    # Polygons read at once for area/perimeter checks.

_CENSUS = {}


# Check whether a polygon is new and record it if so.
@njit
def _add_hash(table, hash_a, hash_b):
    size = np.int64(len(table))
    mask = size - np.int64(1)
    key = np.uint64(hash_a) ^ (
        np.uint64(hash_b) * np.uint64(0x9E3779B97F4A7C15)
    )
    if key == np.uint64(0):
        key = np.uint64(1) 

    index = np.int64(key) & mask
    for _ in range(size):
        existing = table[index]
        if existing == np.uint64(0):
            table[index] = key
            return True
        if existing == key:
            return False
        index = (index + np.int64(1)) & mask
    raise RuntimeError("global polygon hash table is full")


# Check a batch for repeated polygons and mark the new ones.
@njit
def _add_hashes(table, hashes_a, hashes_b, is_new):
    for index in range(len(hashes_a)):
        is_new[index] = _add_hash(table, hashes_a[index], hashes_b[index])


# Define how each polygon's data is arranged and stored.
def _record_dtype(n_sides):
    return np.dtype(
        [
            ("hash_a", "<u8"),
            ("hash_b", "<u8"),
            ("area2", "<i4"),
            ("axial", "<i4"),
            ("diagonal", "<i4"),
            ("coordinates", ("<i2", 2 * n_sides)),
        ]
    )


# Load the pipeline settings and reachability data once per worker.
def _initialize_census(reach_path, reach_radius, n_sides, max_length):
    _CENSUS["reach"] = np.load(reach_path, mmap_mode="r")
    _CENSUS["reach_radius"] = reach_radius
    _CENSUS["n_sides"] = n_sides
    _CENSUS["max_length"] = max_length
    _CENSUS["signed_lengths"] = np.array(
        [length for length in range(-max_length, max_length + 1) if length], dtype=np.int32
    )
    _CENSUS["positive_lengths"] = np.arange(1, max_length + 1, dtype=np.int32)
    _CENSUS["dtype"] = _record_dtype(n_sides)


# Print worker progress, polygon counts, and estimated time remaining.
def _print_worker_progress(done, total, started, unique_count):
    if done % PROGRESS_INTERVAL and done != total:
        return
    elapsed = time.time() - started
    rate = done / elapsed if elapsed else 0.0
    eta = (total - done) / rate if rate else float("inf")
    print(
        f"  [PID {os.getpid()}] {done}/{total} ({done / total:.1%})  "
        f"elapsed={elapsed / 60:.1f}m  ETA={eta / 60:.1f}m  "
        f"local unique={unique_count:,}", flush=True,
    )


# Generate polygons for one task, remove duplicates, and save the records.
def _census_task(index_range, output_path):
    start, end = index_range
    total = end - start
    n_sides = _CENSUS["n_sides"]
    max_length = _CENSUS["max_length"]
    signed_lengths = _CENSUS["signed_lengths"]
    positive_lengths = _CENSUS["positive_lengths"]
    reach = _CENSUS["reach"]
    reach_radius = _CENSUS["reach_radius"]
    record_dtype = _CENSUS["dtype"]

    local_hashes = set()
    buffer = np.empty(WORKER_BUFFER_RECORDS, dtype=record_dtype)
    buffered = 0
    started = time.time()

    with open(output_path, "wb") as output:
        for completed, pattern_index in enumerate(range(start, end), 1):
            pattern = decode_direction_pattern(np.int64(pattern_index), n_sides)

            if pattern[0] == pattern[-1]:
                _print_worker_progress(completed, total, started, len(local_hashes))
                continue

            remaining_counts = suffix_direction_counts(pattern)
            counts = remaining_counts[0]
            if not reach[
                int(counts[0]),
                int(counts[1]),
                int(counts[2]),
                int(counts[3]),
                reach_radius,
                reach_radius,
            ]:
                _print_worker_progress(completed, total, started, len(local_hashes))
                continue

            polygons = generate_polygons_for_pattern(
                pattern, max_length, signed_lengths, positive_lengths,
                remaining_counts, reach, reach_radius,
            )
            if len(polygons) == 0:
                _print_worker_progress(completed, total, started, len(local_hashes))
                continue

            hashes_a, hashes_b = canonical_hashes(polygons)
            areas, axial_lengths, diagonal_lengths = polygon_invariants_batch(polygons)

            for index, polygon in enumerate(polygons):
                signature = (int(hashes_a[index]), int(hashes_b[index]))
                if signature in local_hashes:
                    continue
                local_hashes.add(signature)

                record = buffer[buffered]
                record["hash_a"], record["hash_b"] = signature
                record["area2"] = int(areas[index])
                record["axial"] = int(axial_lengths[index])
                record["diagonal"] = int(diagonal_lengths[index])
                record["coordinates"] = polygon.reshape(-1)
                buffered += 1

                if buffered == len(buffer):
                    buffer.tofile(output)
                    buffered = 0

            _print_worker_progress(completed, total, started, len(local_hashes))

        if buffered:
            buffer[:buffered].tofile(output)

    return len(local_hashes), output_path


# Estimate a hash table size, capped at 2 GiB.
def _hash_table_size(direction_pattern_count):
    estimated_unique = max(direction_pattern_count * 200, 1 << 20)
    bits = 20
    while (1 << bits) < 2 * estimated_unique:
        bits += 1
    return 1 << min(bits, 28)


# Generate polygons in parallel and merge unique records into a single file.
def _generate_unique_records(n_sides, max_length, workers, temporary_dir):
    started = time.time()
    reach_path, reach_radius = ensure_reachability_cache(n_sides, max_length)
    direction_pattern_count = 2 * 3 ** (n_sides - 1)
    task_count = (
        max(workers, 8)
        if direction_pattern_count <= 50_000
        else workers * 32
    )
    index_ranges = split_range(direction_pattern_count, task_count)

    table_size = _hash_table_size(direction_pattern_count)
    print(f"  Direction patterns: {direction_pattern_count:,}", flush=True)
    print(f"  Generation tasks:   {len(index_ranges):,}", flush=True)
    print(
        f"  Dedup table:        {table_size:,} fingerprints "
        f"({table_size * 8 / 2 ** 30:.2f} GiB)", flush=True,
    )

    global_hashes = np.zeros(table_size, dtype=np.uint64)
    _add_hashes(
        np.zeros(16, dtype=np.uint64), np.array([3], dtype=np.uint64),
        np.array([4], dtype=np.uint64), np.empty(1, dtype=np.bool_),
    )

    record_dtype = _record_dtype(n_sides)
    os.makedirs(temporary_dir, exist_ok=True)
    all_polygons_path = os.path.join(temporary_dir, "unique_polygons.bin")
    invariant_counts = {}
    unique_count = 0

    try:
        with open(all_polygons_path, "wb") as merged_output:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=best_mp_context(), initializer=_initialize_census,
                initargs=(reach_path, reach_radius, n_sides, max_length),
            ) as pool:
                futures = []
                for index, index_range in enumerate(index_ranges):
                    task_path = os.path.join(temporary_dir, f"task_{index:04d}.bin")
                    futures.append(pool.submit(_census_task, index_range, task_path))

                for task_number, future in enumerate(as_completed(futures), 1):
                    local_count, task_path = future.result()
                    print(
                        f"  Task {task_number}/{len(index_ranges)}: "
                        f"{local_count:,} locally unique", flush=True,
                    )

                    records = np.fromfile(task_path, dtype=record_dtype)
                    os.remove(task_path)
                    if len(records) == 0:
                        continue

                    is_new = np.empty(len(records), dtype=np.bool_)
                    _add_hashes(global_hashes, records["hash_a"], records["hash_b"], is_new)
                    new_records = records[is_new]
                    del records

                    for record in new_records:
                        key = (
                            int(record["area2"]),
                            int(record["axial"]),
                            int(record["diagonal"]),
                        )
                        # Only track whether a key occurs once or more than once.
                        invariant_counts[key] = min(2, invariant_counts.get(key, 0) + 1)

                    new_records.tofile(merged_output)
                    unique_count += len(new_records)
                    del new_records
                    print(f"    global unique: {unique_count:,}", flush=True)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    finally:
        del global_hashes

    return (
        all_polygons_path,
        invariant_counts,
        unique_count,
        time.time() - started,
    )


# Save polygons whose area/perimeter key is shared by at least one other polygon.
def _write_invariant_candidates(binary_path, invariant_counts, n_sides, output_path):
    record_dtype = _record_dtype(n_sides)
    repeated_keys = {
        key for key, count in invariant_counts.items() if count >= 2
    }
    header = ["Area2", "Axial", "Diagonal"] + [
        label
        for index in range(1, n_sides + 1)
        for label in (f"X{index}", f"Y{index}")
    ]
    candidate_count = 0

    with open(binary_path, "rb") as binary, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as csv_output:
        writer = csv.writer(csv_output)
        writer.writerow(header)
        while True:
            records = np.fromfile(binary, dtype=record_dtype, count=BINARY_READ_RECORDS)
            if len(records) == 0:
                break
            keep = np.fromiter(
                (
                    (
                        int(record["area2"]),
                        int(record["axial"]),
                        int(record["diagonal"]),
                    )
                    in repeated_keys
                    for record in records
                ),
                dtype=np.bool_, count=len(records),
            )
            kept_records = records[keep]
            writer.writerows(
                [
                    int(record["area2"]),
                    int(record["axial"]),
                    int(record["diagonal"]),
                    *record["coordinates"].tolist(),
                ]
                for record in kept_records
            )
            candidate_count += len(kept_records)

    print(f"  Repeated area/perimeter groups: {len(repeated_keys):,}")
    print(f"  Candidate polygons:             {candidate_count:,}")
    return candidate_count


# Choose the grid densites for the spectral search and run phase 2.
def _run_spectral_phase(n_sides, max_length, candidate_csv, candidate_count, workers, output_dir):
    start_denominator = first_grid_denominator(candidate_count)
    end_denominator = final_grid_denominator(max_length)
    print(
        f"\nPhase 2 — numerical spectral screening "
        f"(h=1/{start_denominator} through 1/{end_denominator})", flush=True,
    )

    return run_spectral_search(
        n_sides=n_sides, max_length=max_length, candidate_csv=candidate_csv, workers=workers,
        start_denominator=start_denominator, end_denominator=end_denominator, output_dir=output_dir,
    )


# Run both search phases, save results, and print the summary.
def run_full_search(n_sides, max_length, workers):
    tag = f"{n_sides}_{max_length}"
    results_dir = os.path.join("results", tag)
    temporary_dir = f".stream_tmp_{tag}"
    started = time.time()

    if os.environ.get("ISOSPECTRAL_RESULTS_PREPARED") != "1":
        shutil.rmtree(results_dir, ignore_errors=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nSearch case ({n_sides}, {max_length}) with {workers} workers")
    print("\nPhase 1 — exhaustive polygon generation", flush=True)

    binary_path, invariant_counts, unique_count, generation_seconds = (
        _generate_unique_records(n_sides, max_length, workers, temporary_dir)
    )

    print(f"\n  Unique polygons: {unique_count:,}", flush=True)
    print(f"  Phase 1 time:    {generation_seconds:.1f}s", flush=True)
    print("\nInvariant filter — exact area and perimeter", flush=True)

    candidate_csv = os.path.join(results_dir, candidate_filename(n_sides, max_length))
    invariant_started = time.time()
    candidate_count = _write_invariant_candidates(binary_path, invariant_counts, n_sides, candidate_csv)
    print(f"  Invariant filter time: {time.time() - invariant_started:.1f}s", flush=True)

    shutil.rmtree(temporary_dir, ignore_errors=True)

    if candidate_count:
        phase2 = _run_spectral_phase(
            n_sides, max_length, candidate_csv, candidate_count, workers, results_dir
        )
    else:
        phase2 = Phase2Summary(planned_final_denominator=final_grid_denominator(max_length))

    print_pipeline_summary(
        n_sides=n_sides, max_length=max_length, unique_polygons=unique_count, candidate_count=candidate_count,
        generation_seconds=generation_seconds, phase2=phase2, total_seconds=time.time() - started,
    )


# Read and validate the search parameters.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate lattice polygons and screen for matching spectra.")
    parser.add_argument("n_sides", type=int)
    parser.add_argument("max_length", type=int)
    parser.add_argument("--workers", type=int, default=0)

    args = parser.parse_args(argv)

    if args.n_sides < 3:
        parser.error("n_sides must be at least 3")
    if args.max_length < 1:
        parser.error("max_length must be at least 1")
    if args.workers < 0:
        parser.error("workers cannot be negative")
    if args.n_sides * args.max_length > np.iinfo(np.int16).max:
        parser.error("coordinates exceed the supported storage range")

    return args


# Read the given inputs, choose a worker count, and run the pipeline.
def main(argv=None):
    # Show progress immediately
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args(argv)
    workers = args.workers or recommended_workers()
    run_full_search(args.n_sides, args.max_length, workers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
