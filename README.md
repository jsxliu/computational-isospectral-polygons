# Computational Isospectral Polygons

This project exhaustively generates finite families of lattice polygons and
looks for pairs whose finite-difference Dirichlet spectra match numerically.

The search serves as computational evidence and should not be taken as proof of
isospectrality. Surviving pairs are numerical candidates whose computed
eigenvalues agree across the tested grid refinements.

## Usage

Use Python 3.10 and run these setup commands from the project folder:

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Run class `(8, 2)` (8 sides, maximum lattice step length 2):

```bash
chmod u+x run.sh # Only required for first time
./run.sh 8 2
```

For another class, replace `8` and `2`. Add `--workers n` to use specify the number ofworker
processes. Otherwise, the program chooses the worker count automatically.

## Results

CSV files, any generated plots, and the run log are saved in
`results/<sides>_<length>/`, such as `results/8_2/`. Running the same class again
replaces its previous results.

## Expected Regression Result

For the `(8, 2)` case, a complete run should report:

- `6,089` generated unique polygons
- `6,037` area/perimeter-filter candidates
- `1` final numerical candidate pair at `h=1/6`

Typical output files are written under `results/8_2/`, including:

- `candidate_polygons_8_2.csv`
- `isospectral_pairs_8_2_1over6.csv`
- `isospectral_pairs_8_2_1over6_eigenvalues.csv`
- `iso_plots_8_2_1over6/match_*.png`
- `run_8_2.log`

Generated results, caches, bytecode, and virtual environments are ignored by Git.

## Repository Layout

```text
run.sh                         shell entry point
run.py                         process launcher and log tee
isospectral/config.py          multiprocessing and thread limits
isospectral/pipeline.py        top-level two-phase pipeline
isospectral/pipeline_common.py shared summaries and file helpers
isospectral/polygen.py         lattice polygon generation and invariants
isospectral/spectral.py        numerical spectral screening
isospectral/spectral_output.py spectral CSV formatting and final pair plots
```

## AI assistance

AI tools were used for pair programming during computational code development, including boilerplate generation, refactoring, and efficiency optimization. They also assisted with reviewing and revising comments and documentation, Git workflows, and release preparation. Jonathan Liu is responsible for the code, reported results, and their interpretation.

## Related manuscript

This repository contains the computational code accompanying:

Jonathan Liu. *Computational Discovery of Isospectral Lattice Polygons*.
Research manuscript in preparation.

## Citation

If you use this software in research, please cite the version you used.
Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

The source code and accompanying repository documentation are licensed under
the [MIT License](LICENSE).

The research manuscript is maintained separately and is not covered by this
license. Third-party dependencies remain subject to their own licenses.
