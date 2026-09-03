# Reproduction guide

This guide records the current command sequence from molecular-dynamics
trajectories to publication-facing summaries. It distinguishes supported,
deterministic commands from stages that still depend on unpublished data or
manual rendering.

Exact reproduction is not yet possible from a clean checkout because the
publication trajectory data, fitted pore-network distributions, and reference
PNMs have not been released. Commands below can be used with compatible local
data. The future data archive must provide the missing inputs and checksums.

**Console script vs. `python -m` invocation.** Installed `gas-traj-*` console
scripts (Section 6 onward) are reserved for the deterministic, checkpointed,
tested commands that produce a publication Table or Figure summary; every one
of them can equivalently be run as `python -m scripts.<module>` from an
uninstalled checkout (see each section below). Every other executable module
— one-off plotters, manual visual-inspection tools, and the PyMOL/VMD
renderers under `external_scripts/` — is *only* run as `python -m
scripts.<module>` or invoked directly with `pymol`/`vmd`, and is not wrapped
as an installed command at all.

## 1. Reference environment

Run all commands from a checkout of this repository:

```bash
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment.yml
conda activate kerogen
python -m pytest -q
```

`environment.yml` is the numerical reference environment. A different
Python, NumPy, SciPy, scikit-image, BLAS, or VTK stack may change floating-point
results, fitted parameters, image discretization, or rendering.

## 2. Dataset layout

Define paths once and keep raw input separate from generated files:

```bash
export DATA_DIR="/path/to/dataset"
export INPUT_GRO="$DATA_DIR/input/full_trajectory.gro"
mkdir -p "$DATA_DIR/input"
```

The workflow uses this layout:

```text
<data-dir>/
├── input/
│   └── full_trajectory.gro
├── trj.gro
├── structures/
├── bin_images/
├── raw_images/
├── float_images/
├── pnm/
├── radiuses.npy
├── throat_lengths.npy
├── pi_l_data.npy
├── pi_l_gamma_fitter.pkl
├── throat_lengths_weibull_fitter.pkl
├── errors/
├── traps/
├── figs/
└── ks_stationarity/
```

The full trajectory is a multi-frame GRO-like text file. Each frame needs a
header from which simulation `step` and time in ps can be read, an atom-count
line, fixed-width atom records, and a box-size line. Coordinates are in nm.

Generated structure and image filenames preserve the simulation step, time,
bounding box, and resolution. Do not rename them: later stages recover metadata
from those names.

## 3. Trajectory and structure preparation

Every installed `gas-traj-*` command below has an equivalent `python -m
scripts.<module>` form that runs from a source checkout with nothing
installed (same flags, same behavior — `[project.scripts]` in
`pyproject.toml` just wraps the module's own `argparse` entry point).

Extract the gas atoms by removing the `KRG` residue (the default):

```bash
gas-traj-extract-gas-trajectory "$INPUT_GRO" "$DATA_DIR/trj.gro"
# Without installing the package:
python -m scripts.extract_gas_trajectory_to_file "$INPUT_GRO" "$DATA_DIR/trj.gro"
```

To extract one or more selected kerogen molecules instead:

```bash
gas-traj-extract-krg-trajectory \
  "$INPUT_GRO" "$DATA_DIR/krg_99.gro" \
  --select KRG:99
# Without installing the package:
python -m scripts.extract_krg_trajectory_to_file \
  "$INPUT_GRO" "$DATA_DIR/krg_99.gro" \
  --select KRG:99
```

Preview automatically sampled structure steps before writing files:

```bash
gas-traj-extract-structures \
  "$INPUT_GRO" "$DATA_DIR/structures" \
  --auto-indexes --mode all --count-structures 500 --dry-run
# Without installing the package:
python -m scripts.dynamic_struct_extractor \
  "$INPUT_GRO" "$DATA_DIR/structures" \
  --auto-indexes --mode all --count-structures 500 --dry-run
```

Then remove `--dry-run` to extract the structures. `mode=all` always spreads
indexes evenly and includes both the first and last available step; the
returned count equals `--count-structures` unless the trajectory has fewer
distinct integer positions than requested.

Build binary images and distance maps:

```bash
gas-traj-binarize-structures \
  "$DATA_DIR/structures" \
  "$DATA_DIR/bin_images" \
  "$DATA_DIR/raw_images" \
  --ref-size 250 --mode all --count-slices 500 --num-workers 10

gas-traj-distance-maps \
  "$DATA_DIR/structures" "$DATA_DIR" \
  --ref-size 250 --mode all --count-slices 500

# Without installing the package:
python -m scripts.binarization_structs \
  "$DATA_DIR/structures" \
  "$DATA_DIR/bin_images" \
  "$DATA_DIR/raw_images" \
  --ref-size 250 --mode all --count-slices 500 --num-workers 10

python -m scripts.distance_map_structs \
  "$DATA_DIR/structures" "$DATA_DIR" \
  --ref-size 250 --mode all --count-slices 500
```

The first command writes `.npy` binary volumes and matching headerless
`.raw` volumes. The second writes `.npy` arrays under
`$DATA_DIR/float_images`. Image resolution and cropping are controlled by
`--ref-size` and `--dev`; preserve those values with final results.
`distance_map_structs` also accepts `--mode part --count-slices N` to build a
quick subset instead of the full run.

## 4. Pore-network boundary

The pore-network extractor used by the authors is not distributed. The local
`scripts/pnm_extractor.py` module demonstrates its JSON adapter, but requires
an external executable and is not a portable reproduction dependency. There is
no installed console script for it (not registered in `pyproject.toml`); run
it as a module, pointing at your own extractor binary and its JSON config:

```bash
python -m scripts.pnm_extractor "$DATA_DIR" "$EXTRACTOR_PATH" "$EXTRACTOR_CONFIG"
```

It reads `.raw` volumes from `$DATA_DIR/raw_images` (written by
`gas-traj-binarize-structures`) and writes one PNM per structure step under
`$DATA_DIR/pnm`, named with the pattern expected below.

An alternative extractor can be used if each network is written under
`$DATA_DIR/pnm` with a common prefix and these Statoil files:

```text
<prefix>_node1.dat
<prefix>_node2.dat
<prefix>_link1.dat
<prefix>_link2.dat
```

The prefix must retain a `num=<simulation-step>_` component for time-dependent
PNM analysis. The reader consumes `node1`, `node2`, and `link1`, expects
lengths in metres, and converts them to nm. Keep `link2` in a released
standard-format network even though it is not currently read.

Different extractors can yield different networks. Exact downstream paper
results therefore require the authors' reference PNM files, which are planned
for the publication data archive.

## 5. Fit distributions used by the simulator

After placing PNMs under `$DATA_DIR/pnm`, generate the pore-radius,
pore-intersection-length, and throat-length samples and fitted distributions:

```bash
python -m scripts.generate_pil_distr \
  "$DATA_DIR/pnm" "$DATA_DIR" --x-min 0.025
```

This produces the `radiuses.npy`, `throat_lengths.npy`,
`pi_l_gamma_fitter.pkl`, and `throat_lengths_weibull_fitter.pkl` inputs used
by the synthetic benchmark and probabilistic trajectory analyzers.

These files are caches derived from the selected PNMs. Until cache provenance
is strengthened, do not reuse them after changing PNM inputs or fitting
parameters. Preserve or move old results before recomputing.

## 6. Table I: DM parameter search

The publication profile evaluates 100 trajectories with 1000 points for every
`(k, p)` pair, using base seed 42 and the deterministic candidate grid:

```bash
gas-traj-dm-search \
  "$DATA_DIR" \
  --trajectory-count 100 \
  --trajectory-points 1000 \
  --seed 42 \
  --n-jobs -1
# Without installing the package:
python -m scripts.errors_params \
  "$DATA_DIR" \
  --trajectory-count 100 \
  --trajectory-points 1000 \
  --seed 42 \
  --n-jobs -1
```

Outputs are written to `$DATA_DIR/errors/find_best_params`, including
`parameter_search_manifest.json` and an atomic checkpoint for each
`(k, p)`. To continue a compatible interrupted or legacy calculation:

```bash
gas-traj-dm-search "$DATA_DIR" --resume
# Without installing the package:
python -m scripts.errors_params "$DATA_DIR" --resume
```

Do not repeat `--force-recompute` after an interruption unless a complete
restart is intended. Aggregate completed results with:

```bash
gas-traj-find-dm-params "$DATA_DIR/errors/find_best_params"
# Without installing the package:
python -m scripts.find_best_params "$DATA_DIR/errors/find_best_params"
```

The aggregation writes `table_i_optimized_dm_params.csv` and
`table_i_optimized_dm_params.json` in the input directory. A smaller
`--trajectory-count` is useful for debugging but is not the Table I profile.

## 7. Figures 8 and 13 and Table II: synthetic benchmark

Required files at the root of `$DATA_DIR` are:

```text
radiuses.npy
pi_l_gamma_fitter.pkl
throat_lengths_weibull_fitter.pkl
```

Run the publication profile:

```bash
gas-traj-synthetic-benchmark \
  "$DATA_DIR" \
  --trajectory-count 100 \
  --step-count 3000 \
  --seed 42
# Without installing the package:
python -m scripts.sim_algo_check \
  "$DATA_DIR" \
  --trajectory-count 100 \
  --step-count 3000 \
  --seed 42
```

The command writes:

- `figs/fig08_errors_k=*.svg`: DM/SIB/HYB comparison for Figure 8;
- `figs/fig13_errors_k=*.svg`: DM/NP/SIB comparison for Figure 13;
- `errors/table_ii_synthetic_k_est.csv` and `.json`: Table II;
- `errors/synthetic_benchmark_manifest.json`: seed, grid, and figure series;
- `errors/trajectories` and `errors/checkpoints`: resumable caches.

Both figures show the mean classification error. Their shaded envelope spans
the 20th–80th percentile corridor and is extended to include the mean wherever
a skewed error distribution places the arithmetic mean outside that equal-tail
interval. For each `k`, the figures use the same Y limits and stable
per-algorithm colors, so the shared DM and SIB/NP+Bayesian series are directly
comparable.

The seed deterministically controls both NumPy and Python random generators.
For each trajectory, the estimated trapping probability is
`k_est = N_t / (N_0 + N_t)`.
Here, each maximal run of intra-trap edges contributes one capture event to
`N_t`. Within a run of `L` consecutive inter-trap edges, the `L - 1` adjacent
edge pairs delimit fully observed zero-duration trap visits and contribute
`L - 1` bypass events to `N_0`. Events outside the two trajectory boundaries
are unobserved and are not invented. Counting free and trapped runs instead
would force `k_est` towards 0.5 because those runs necessarily alternate.

## 8. Table III: trapping-time distributions

Each gas dataset needs `trj.gro`, `pi_l_gamma_fitter.pkl`, and
`throat_lengths_weibull_fitter.pkl`. Define two datasets and one shared
summary directory:

```bash
export CH4_DIR="/path/to/ch4-dataset"
export H2_DIR="/path/to/h2-dataset"
export TABLE_III_DIR="/path/to/table-iii-output"

gas-traj-trap-distributions \
  "$CH4_DIR" --label CH4 --num 1 \
  --output "$CH4_DIR/figs/Pt_loglog.svg" \
  --summary-dir "$TABLE_III_DIR"

gas-traj-trap-distributions \
  "$H2_DIR" --label H2 --num 2 \
  --output "$H2_DIR/figs/Pt_loglog.svg" \
  --summary-dir "$TABLE_III_DIR"

# Without installing the package:
python -m scripts.trap_distr_builder \
  "$CH4_DIR" --label CH4 --num 1 \
  --output "$CH4_DIR/figs/Pt_loglog.svg" \
  --summary-dir "$TABLE_III_DIR"

python -m scripts.trap_distr_builder \
  "$H2_DIR" --label H2 --num 2 \
  --output "$H2_DIR/figs/Pt_loglog.svg" \
  --summary-dir "$TABLE_III_DIR"
```

The shared directory receives `table_iii_trapping_summary.csv` and `.json`;
rows are updated by gas and classifier without removing the other gas. Cached
step labels and trap sequences are stored under each dataset's `traps/DM`,
`traps/SIB`, and `traps/HYB` directories.

The event definitions for `N_t`, `N_0`, and `k_est` are the same as in the
synthetic benchmark above. Analyzer labels (`traps_*.pickle`) and derived
event sequences (`seq_*.pickle`) have separate provenance manifests, so a
change to event encoding rebuilds only the inexpensive sequences and summary,
not the DM/SIB/HYB classifications.

To replace only HYB after an algorithm change:

```bash
gas-traj-trap-distributions \
  "$CH4_DIR" --label CH4 --num 1 \
  --summary-dir "$TABLE_III_DIR" --recompute HYB
# Without installing the package:
python -m scripts.trap_distr_builder \
  "$CH4_DIR" --label CH4 --num 1 \
  --summary-dir "$TABLE_III_DIR" --recompute HYB
```

`--force-recompute` replaces caches for all three classifiers.

## 9. Table IV: PNM stationarity

The stationarity command reads `$DATA_DIR/pnm/*_link1.dat` and infers the
simulation-step-to-time mapping from the first two frames in
`$DATA_DIR/trj.gro`:

```bash
gas-traj-stationarity "$DATA_DIR"
# Without installing the package:
python -m scripts.stationarity "$DATA_DIR"
```

To use another trajectory:

```bash
gas-traj-stationarity \
  "$DATA_DIR" --trajectory "$INPUT_GRO"
# Without installing the package:
python -m scripts.stationarity \
  "$DATA_DIR" --trajectory "$INPUT_GRO"
```

Or supply the complete linear mapping without reading a trajectory:

```bash
gas-traj-stationarity \
  "$DATA_DIR" \
  --anchor-step 25000 --anchor-time-ps 50 \
  --step-delta 250000 --time-delta-ps 500
# Without installing the package:
python -m scripts.stationarity \
  "$DATA_DIR" \
  --anchor-step 25000 --anchor-time-ps 50 \
  --step-delta 250000 --time-delta-ps 500
```

The command writes stationarity SVG files and
`ks_stationarity/table_iv_stationarity_summary.csv` and `.json`. The frame
at step 25000 is excluded as pre-equilibration; the first later available PNM
is the baseline.

## 10. Manual structure and trajectory visualization (Figures 1, 2, 4, 7, 10)

These commands render or inspect one structure or trajectory at a time. None
of them save a publication-ready composite figure automatically; each is a
manual step in the Figures 1, 2, 4, 7, and 10 workflow.

### PyMOL and VMD renderers (`external_scripts/`)

PyMOL and VMD are not part of `environment.yml` and must be installed
separately. Every script under `external_scripts/` reads its arguments from
an environment variable instead of CLI flags: PyMOL scripts read
`PYMOL_SCRIPT_ARGS` (shlex-split), VMD scripts read `VMD_SCRIPT_ARGS`
(whitespace-split).

Atom color legend:

```bash
pymol -q external_scripts/visualize_atom_legend.pml
```

One kerogen molecule (single chain/residue) from a PDB:

```bash
export PYMOL_SCRIPT_ARGS="--ker-pdb '$DATA_DIR/ker.pdb' --chain A --resi 1"
pymol -q external_scripts/visualize_kerogen_molecula.pml
```

Full simulation cell from lattice parameters (VMD):

```bash
export VMD_SCRIPT_ARGS="--ker-pdb '$DATA_DIR/ker.pdb' --cell '62.309 74.106 130.150 90 90 90'"
vmd -e external_scripts/visualize_kerogen_cell.tcl
```

A cell fragment without atoms, from an explicit subbox or a `--box-size` cube
around the center:

```bash
export PYMOL_SCRIPT_ARGS="--ker-pdb '$DATA_DIR/ker.pdb' --sim-gro '$INPUT_GRO' --step 25000 --xmin 14.9 --xmax 47.4 --ymin 20.8 --ymax 53.3 --zmin 48.8 --zmax 81.3"
pymol -q external_scripts/visualize_kerogen_part_cell.pml
```

or the VMD equivalent:

```bash
export VMD_SCRIPT_ARGS="--ker-pdb '$DATA_DIR/ker.pdb' --box-size 30"
vmd -e external_scripts/visualize_kerogen_part_cell.tcl
```

A cell fragment with atoms and one highlighted molecule (coordinates are in
Å, i.e. nm × 10):

```bash
export PYMOL_SCRIPT_ARGS="--ker-pdb '$DATA_DIR/ker.pdb' --sim-gro '$INPUT_GRO' --frame 5 --mol-index 85 --box-size 30"
pymol -q external_scripts/visualize_kerogen_part_cell_with_molecula.pml
```

Rotate interactively (`turn y, 45 & turn x, 15`), or pass
`--width`/`--height`/`--dpi`/`--ray`/`--output` for a saved PNG.

### Structure and trajectory scripts

```bash
python -m scripts.atom_visualization
```

Draws a fixed 5-atom color legend from synthetic coordinates; no arguments,
no input data.

```bash
python -m scripts.vis_struct_trajectory \
  "$DATA_DIR" "$DATA_DIR/float_images/<one-image>.npy" \
  --num 2 --start-index 0 --end-index 2061
```

Overlays one molecule's cropped trajectory on a distance-field image.

```bash
python -m scripts.vis_atoms_struct \
  "$DATA_DIR" "$DATA_DIR/float_images/<one-image>.npy" \
  --index 25000 --time-ps 50
```

Overlays MD atoms on the same kind of distance-field image at one frame.

```bash
python -m scripts.vis_struct_pnm \
  "$DATA_DIR/float_images/<one-image>.npy" \
  "$DATA_DIR/pnm/<pnm-prefix>"
```

Overlays the pore-network model on a structure image.

```bash
python -m scripts.vis_slice_struct \
  "$DATA_DIR/bin_images/<one-image>.npy" \
  "$DATA_DIR/figs/img_slice.svg" \
  --ref-size 0.2 --size 1
```

Saves one 2D slice of a binary or float structure image.

```bash
python -m scripts.inv_plot \
  "$INPUT_GRO" --num 0 --mu 2.0 --nu 0.1 \
  --min-index 1000 --max-index 1500 --trap-fill-y-max 0.2
```

Plots the DM invariant along one trajectory with trapping regions shaded.
`--mu` is required; `--diag-percentile`, `--kernel-size`, `--p-value`, and
`--traj-type` select the DM threshold profile; `--output` saves instead of
displaying.

```bash
python -m scripts.vis_traject "$DATA_DIR/trj.gro" 2 --traps "$DATA_DIR/traps/SIB/traps_2.pickle"
```

Displays one molecule's trajectory in 3D. `--traps` marks trap/free
transition points using a cached `traps_<index>.pickle` classification (e.g.
written under `traps/SIB` by `gas-traj-trap-distributions`); omit it to draw
the raw trajectory without transition markers.

```bash
python -m scripts.vis_struct_example \
  --float_image_path "$DATA_DIR/float_images/<one-image>.npy" \
  --isovalue 0.11 --img-opacity 0.5
```

Displays an existing distance-field image (from `gas-traj-distance-maps`) as
an isosurface; used for the Figure 2b–d style structure renders. `--isovalue`
and `--img-opacity` control the surface threshold and opacity. This script
was moved here from the removed `examples/` package — it is a secondary,
exploratory renderer, not the primary reproduction path for any figure.

### Trajectory simulation (Figure 7)

```bash
python -m scripts.simulate_trajectory "$DATA_DIR" --k 0.5 --p 0.5 --steps 1000 --radius 0.02
```

Simulates one gas-molecule trajectory with the algorithm described in the
paper and displays it in 3D. `--k` and `--p` are the trapping/return
probability parameters and `--steps` is the trajectory length; varying them
across separate runs produces the Figure 7 b–e panel variants. `--radius`
controls the rendered trajectory tube/point size.

## 11. Correlation-function and PNM element-size plotting (Figures 3, 6), and power-law fit (Figure 9 / Appendix)

```bash
python -m scripts.corrfunc_krg_mol_plotter \
  "$DATA_DIR/trj_krg/krg_99.gro" \
  "$DATA_DIR/figs/corrfunc_krg_99.svg" \
  "$DATA_DIR/msd/krg_99.pickle"
```

Kerogen-molecule correlation function, for one molecule extracted with
`gas-traj-extract-krg-trajectory`.

```bash
python -m scripts.corrfunc_struct_plotter \
  "$DATA_DIR/bin_images" "$DATA_DIR/ct_pore.npy" "$DATA_DIR/figs/corrfunc.svg" \
  --trj "$CH4_DIR/trj.gro:CH4" --trj "$H2_DIR/trj.gro:H2" \
  --max-t 2.8
```

Time-averaged structural autocorrelation C(t) across one or more gas
trajectories; images are always inverted before computing C(t). `--max-t`,
`--x-max` and `--num-workers` are optional.

```bash
python -m scripts.distr_pnm_element_size_plotter \
  "$DATA_DIR/pnm" "$DATA_DIR/figs" \
  --label CH4 --pnm-step 120 --x-min 0.05 --x-max 1.7
```

Pore/throat size distributions over the PNM time series; the step/time
mapping is inferred from `$DATA_DIR/trj.gro` by default, or set explicitly
with `--anchor-step`/`--anchor-time-ps`/`--step-delta`/`--time-delta-ps`
(same convention as `gas-traj-stationarity`).

```bash
python -m scripts.pil_plotter "$DATA_DIR" --x-min 0.025
```

Pore-intersection-length (PIL) heatmap from the fitted distributions written
by `generate_pil_distr`; `--heatmap-mode`, `--heatmap-interpolation`, and
`--heatmap-dpi` control rendering. `P(r)` is fit on a uniform subsample of
the radii (`--fit-sample-stride`, default `10`) rather than every value —
there are millions of edges, and a stride subsample keeps the same
distribution shape at a fraction of the fit cost.

```bash
python -m scripts.powerlaw_analysis "$DATA_DIR" --prefix SIB --mode xmin --n_synth 2500
```

Clauset-style power-law fit over `trap_distr_builder` trap sequences;
`--prefix` selects `Distance-matrix`/`SIB`/`Hybrid`, `--mode sample` sweeps
sample size instead of `x_min`, publication runs use `--n_synth 2500`.

## 12. Complexity benchmark (Figure 11), PNM connectivity/count statistics, and MSD

```bash
python -m scripts.complexity_estimation "$DATA_DIR" "$DATA_DIR/complexity.pdf"
```

Builds the algorithm-complexity comparison plot (Figure 11).

```bash
python -m scripts.distr_pnm_connectivity \
  --pnm "$CH4_DIR/pnm:type1-300K-CH4" \
  --pnm "$H2_DIR/pnm:type1-300K-H2"
```

Plots pore-network connectivity distributions across one or more PNM
datasets, for supplementary analysis beyond the numbered figures. The plot is
displayed interactively and not saved to a file.

```bash
python -m scripts.distr_pnm_count_pores_throats \
  --pnm "$CH4_DIR/pnm:type1-300K-CH4" \
  --pnm "$H2_DIR/pnm:type1-300K-H2"
```

Plots pore/throat count statistics across one or more PNM datasets, same
supplementary scope and display-only behavior as above.

```bash
python -m scripts.msdt_builder \
  --trj "$DATA_DIR:type1-300K-CH4:1" \
  --trj "${DATA_DIR/ch4/h2}:type1-300K-H2:2"
```

Plots the time-averaged mean-square displacement (log-log) for one or more
gas datasets on a shared axis. `--trj` takes `DATA_DIR:LABEL:STEP`, where
`DATA_DIR` is a data directory containing `trj.gro` (not the trajectory file
itself) and `STEP` subsamples molecules.

## 13. Paper artifact map

| Artifact | Primary implementation | Reproduction status |
|---|---|---|
| Figs. 1–2 | §10 (`external_scripts/`, `vis_struct_trajectory`, `vis_atoms_struct`, `vis_struct_pnm`, `vis_slice_struct`, `vis_struct_example`) | Manual rendering; requires PDB, MD, and reference PNM data |
| Fig. 3 | §11 (`corrfunc_krg_mol_plotter`, `corrfunc_struct_plotter`) | Requires publication trajectories and explicit final recipe |
| Fig. 4 | §10 (`inv_plot`, `vis_traject`) | Confirmed working; panel composition remains manual |
| Fig. 5 | none | No current implementation. The previous example script (`examples/vis_collapsed_traps.py`) has been removed as redundant/unmaintained; collapsing trapping intervals for display is not implemented elsewhere |
| Fig. 6 | image pipeline, external PNM extraction, `generate_pil_distr`, §11 (`distr_pnm_element_size_plotter`, `pil_plotter`) | Blocked on reference PNMs |
| Fig. 7 | §10 (`simulate_trajectory`), VTK visualizer | Confirmed working; `--k`/`--p`/`--steps` produce the panel b–e variants; requires fitted distributions and manual panel composition |
| Fig. 8 | `gas-traj-synthetic-benchmark` | Deterministic command and SVG output implemented |
| Fig. 9 / Appendix | `gas-traj-trap-distributions`, §11 (`powerlaw_analysis`), §10 trajectory visualizers | Summary command implemented; data unavailable |
| Fig. 10 | §10 (`simulate_trajectory`, visualizers) | Confirmed working; panel selection and composition remain manual |
| Fig. 11 | §12 (`complexity_estimation`) | Confirmed working; timing protocol is not yet publication-ready |
| Fig. 12 | `gas-traj-stationarity` | Deterministic summary implemented; PNM data unavailable |
| Fig. 13 | `gas-traj-synthetic-benchmark` | Deterministic command and SVG output implemented |
| Table I | `gas-traj-dm-search`, `gas-traj-find-dm-params` | Deterministic, resumable CSV/JSON workflow implemented |
| Table II | `gas-traj-synthetic-benchmark` | Deterministic CSV/JSON export implemented |
| Table III | `gas-traj-trap-distributions` | CSV/JSON export implemented; data unavailable |
| Table IV | `gas-traj-stationarity` | CSV/JSON export implemented; PNM data unavailable |

This table is deliberately explicit about incomplete stages. A future
publication release should replace every “manual” or “unavailable” entry with
a versioned input, exact command, parameter file, and expected output checksum.

## 14. Result preservation and manifests

Before any force-recompute operation, copy figures, CSV/JSON summaries,
manifests, parameters, and caches to a separate versioned directory. Record the
input dataset, exact command, seed, code revision, environment, and date.

When preparing a data directory for release, build a SHA-256 manifest:

```bash
gas-traj-data-manifest build "$DATA_DIR"
gas-traj-data-manifest verify "$DATA_DIR"
# Without installing the package:
python -m scripts.build_data_release_manifest build "$DATA_DIR"
python -m scripts.build_data_release_manifest verify "$DATA_DIR"
```

Use `gas-traj-data-manifest build --help` to set the archive identifier,
version, creation date, and description. The manifest detects missing,
modified, and unexpected files; it does not download data or decide which
files are scientifically required.
