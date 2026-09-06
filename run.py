#!/usr/bin/env python3
# This file runs one polygon class search and saves its output to the terminal and a log file.
# Results are saved under results/<sides>_<length>/.
# Example: python run.py 8 2 runs class (8, 2) and saves results in results/8_2/.
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Show the program's messages in the terminal and save them to a log file.
def tee_run(cmd, env, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    out = getattr(sys.stdout, "buffer", sys.stdout)
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=HERE,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            while True:
                chunk = proc.stdout.read1(65536)
                if not chunk:
                    break
                out.write(chunk)
                out.flush()
                logf.write(chunk)
                logf.flush()
        finally:
            proc.stdout.close()
        return proc.wait()


# Read the given inputs, run the pipeline, and save its output.
def main():
    p = argparse.ArgumentParser(
        description="Run one numerical spectral-candidate search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("n_sides", type=int, metavar="nSides")
    p.add_argument("max_length", type=int, metavar="maxLength")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="worker processes (0 = choose automatically)",
    )
    args = p.parse_args()

    n, m = args.n_sides, args.max_length
    tag = f"{n}_{m}"
    results_dir = os.path.join(HERE, "results", tag)

    # Replace previous results for this case (if it exists).
    shutil.rmtree(results_dir, ignore_errors=True)
    os.makedirs(results_dir, exist_ok=True)

    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
        ISOSPECTRAL_RESULTS_PREPARED="1",
    )
    shutil.rmtree(
        os.path.join(HERE, f".stream_tmp_{tag}"), ignore_errors=True
    )

    log_path = os.path.join(results_dir, f"run_{tag}.log")
    cmd = [
        sys.executable,
        "-m",
        "isospectral.pipeline",
        str(n),
        str(m),
        "--workers",
        str(args.workers),
    ]
    print("=" * 60)
    print("  Numerical Spectral Polygon Search")
    print(f"  sides={n}  max length={m}   results -> results/{tag}/")
    print(f"  python: {sys.executable}")
    print(f"  workers: {'auto' if args.workers == 0 else args.workers}")
    print("=" * 60, flush=True)

    rc = tee_run(cmd, env, log_path)

    print(f"\nDone (exit {rc}). Results in results/{tag}/")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
