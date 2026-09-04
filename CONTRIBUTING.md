# Contributing

This is a research codebase accompanying an in-preparation publication. It
welcomes bug reports, reproducibility fixes, and focused improvements; large
architectural changes are more likely to land smoothly if discussed in an
issue first.

## Setting up

The reference environment is Ubuntu x86-64, Python 3.14.4, and the MKL
scientific stack pinned in [`environment.yml`](environment.yml):

```bash
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment.yml
conda activate kerogen
```

This installs the project in editable mode, including the `test` and
`visualization` extras. A plain pip install also works for development
(`python -m pip install -e ".[visualization,test]"`), but is not the
reference environment for reproducing numerical results — see
[`docs/reproduction.md`](docs/reproduction.md).

## Before opening a pull request

Run the full check suite — the same one CI runs on every push:

```bash
bash tools/check.sh
```

This runs Ruff, isort, Black, mypy (strict mode — every function needs type
annotations), and the test suite, over every tracked `*.py` file. To apply
the autofixable parts of that (Ruff autofix, isort, Black) instead of just
checking:

```bash
bash prepare.sh
```

Both scripts read the same file list
([`tools/list_python_files.sh`](tools/list_python_files.sh)), so they never
check a different set of files from each other.

A pull request that doesn't pass `tools/check.sh` won't be merged as-is;
fix it locally first rather than relying on CI to iterate.

## Code conventions

- Every function needs type annotations (`mypy` runs in strict mode:
  `disallow_untyped_defs`, `disallow_any_generics`). Prefer precise
  `numpy.typing.NDArray[...]` element types over bare `np.ndarray`.
- Add a docstring when the signature alone doesn't convey array shapes,
  units (this codebase mixes ps/µs/nm — say which one explicitly), point/edge
  conventions, or valid input ranges. Don't add a docstring that just repeats
  the function name.
- Validate untrusted input (CLI arguments, file contents) by raising a
  specific exception (`ValueError`, etc.) with a message that says what was
  expected. Reserve bare `assert` for internal invariants that should never
  be false given correct callers — not for anything reachable from a CLI
  argument or file the user points the tool at.
- Don't load data with `pickle` on a path that can be user- or
  CLI-controlled; see [`SECURITY.md`](SECURITY.md). Use `.npz`/`.npy` for
  arrays and JSON for small parameter objects instead — see
  `to_dict()`/`from_dict()` on `WeibullFitter`/`GammaFitter`
  (`processes/distribution_fitter.py`) or `Trajectory.to_npz_arrays()` for
  the existing pattern.
- New standalone scripts under `scripts/` that are meant to be run directly
  (not just imported) should be registered as a `gas-traj-*` console script
  in `pyproject.toml`'s `[project.scripts]`, not use a `sys.path.append`
  hack to find the package.

## Tests

Add or update a test under `tests/` for any behavior change. The suite runs
in seconds and is expected to stay that way — it doesn't touch real
molecular-dynamics data (that lives outside the repository); use small
synthetic fixtures instead, following the existing tests as examples.

## Commit messages and pull requests

Commit subjects in this repository are short, imperative, and describe the
change itself (e.g. "Type-annotate base/reader.py; fix mistyped return
type"), not a changelog-style category prefix. Explain the *why* in the body
when it isn't obvious from the diff. The pull request template
(`.github/PULL_REQUEST_TEMPLATE.md`) has a checklist covering the points
above.

## Reporting bugs and requesting features

Use the issue templates under `.github/ISSUE_TEMPLATE/`. For reproduction
mismatches against a specific paper figure or table, name it explicitly and
say which command from `docs/reproduction.md` you ran.

Security vulnerabilities should **not** be reported through a public issue —
see [`SECURITY.md`](SECURITY.md).

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
