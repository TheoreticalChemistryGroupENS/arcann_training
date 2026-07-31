# Contributions

We warmly welcome contributions to ArcaNN. If you have ideas, code contributions, or suggested optimizations, please feel free to submit them.

## Code layout

The `arcann_training/` package is organized by **step**, mirroring the iterative procedure: `initialization/`, `exploration/`, `labeling/`, `training/`, and `test/` each contain one module per *phase* (`prepare.py`, `launch.py`, `check.py`, ...), plus a `utils.py` for step-specific helpers. Code shared across steps (JSON/XML/YAML handling, machine and Slurm configuration, LAMMPS/i-PI/PLUMED file helpers, logging) lives in `common/`. Unit tests live in `arcann_training/unittests/`.

## Development setup

The project is managed with [`uv`](https://docs.astral.sh/uv/) (the dev/lint/doc tooling is declared as `dependency-groups` in `pyproject.toml`, not as `pip` extras). From a clone of the repository:

```bash
uv sync --locked --dev
pre-commit install
```

This creates a local `.venv` with ArcaNN installed in editable mode plus the `dev` group (`ruff`, `pre-commit`). If you'd rather use a plain `pip`/Conda environment (e.g. the one from [Installation](../getting-started/installation.md)), install it in editable mode and add `ruff` yourself: `pip install -e . && pip install ruff`.

This project uses [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting, and Python's built-in `unittest` for tests; both are wired into `.pre-commit-config.yaml` and run automatically on `git commit`. You can also run them manually:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run python -m unittest discover arcann_training/unittests
```

CI (see `.github/workflows/`) runs this same unit test matrix (Python 3.10–3.13) and rebuilds the documentation site on every push, so make sure both pass locally before opening a pull request.

## Reporting bugs

See [Unexpected Behavior](./unexpected_behavior.md) for how to report issues.
