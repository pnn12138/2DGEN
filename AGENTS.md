# Repository Guidelines

## Project Structure & Module Organization
- Root `pyproject.toml` defines the 2D generation prototype; code sits in `2DGEN/` (dataset scaffolding) with entry script `main.py`. Shared sample data live under `data/` and figures under `test_fig/`.
- `P_TASK/` is a standalone property-prediction subproject (its own `pyproject.toml` and virtualenv) with Hydra configs in `P_TASK/conf/`, code in `P_TASK/src/p_task/`, helper scripts in `P_TASK/scripts/`, and cached datasets in `P_TASK/data/`.
- Keep large raw datasets in the existing `data/` or `P_TASK/data/` trees; avoid adding new top-level folders without discussing first.

## Build, Test, and Development Commands
- Install dependencies (root project): `uv sync` (always use `uv run ...` to execute in this env, not bare `python`)
- Run the basic entry script: `uv run python main.py`
- Install dependencies for the P_TASK subproject: `cd P_TASK && uv sync`
- Download the Jdft2d metadata/splits: `cd P_TASK && uv run python scripts/download_jdft2d.py`
- Launch a quick CGCNN training dry-run: `cd P_TASK && uv run python -m p_task.train_jdft2d trainer.trainer.max_epochs=1`
- Run tests (when added): `uv run pytest` (or inside `P_TASK`, `uv run pytest`)

## Coding Style & Naming Conventions
- Python 3.12+ in root, Python 3.10+ in `P_TASK`; follow PEP 8 with 4-space indentation and descriptive snake_case module/function names.
- Use type hints and brief docstrings as seen in `p_task` modules; keep functions small and favor pure helpers over inline logic.
- Configuration follows Hydra: keep defaults under `P_TASK/conf/`; use capitalized config group names (e.g., `Jdft2d.yaml`) and override via CLI rather than hardcoding.

## Testing Guidelines
- Prefer `pytest` with `test_*.py` files colocated in a `tests/` folder or next to the module under test; use fixtures for small synthetic graphs/structures to avoid large I/O.
- For data loaders, add smoke tests that instantiate `Jdft2dDataModule` with a temporary cache dir; seed via `cfg.task.seed` to keep splits reproducible.
- When introducing training changes, include a minimal run command in the PR description and attach metric snapshots if applicable.

## Commit & Pull Request Guidelines
- Match the existing concise, action-focused commit style (short present-tense summaries; Chinese is fine): e.g., “完善数据加载” or “Add JDfT2d cache guard”.
- For PRs, include: purpose + scope, key commands run (tests/train/download), config overrides used, and any data requirements or output paths. Add screenshots/plots from `test_fig` style when visualizing results.
- Keep diffs focused; prefer separate PRs for data preparation, model changes, and visualization. Document new config options in `README` or inline comments.

## Data & Reproducibility Tips
- Avoid committing large raw datasets or checkpoints; store them under `data/` or `P_TASK/data/` locally and document download scripts/URLs instead.
- Respect deterministic seeds (`task.seed`) and note any nondeterministic steps. If adding new datasets, mirror the existing metadata/split layout so scripts remain compatible.
