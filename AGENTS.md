# Repository Guidelines

## Project Structure & Module Organization

This repository is an early Python robotics simulation prototype. `main.py` registers Gymnasium Robotics environments, opens `AdroitHandDoor-v1` with human rendering, and runs 1,000 random-action steps, resetting completed episodes before closing the environment.

- `pyproject.toml` defines dependencies and requires Python 3.13 or newer.
- `.python-version` selects Python 3.13; `uv.lock` records resolved dependencies, including the MuJoCo 3.3.7 pin.
- `README.md` is currently empty; add user-facing setup instructions there as workflows develop.
- No dedicated source package, tests directory, or repository-owned assets exist yet. Put future automated tests in `tests/`.

## Build, Test, and Development Commands

- `uv sync --locked`: install the dependencies recorded in `uv.lock` into the local virtual environment.
- `uv run main.py`: run the interactive hand simulation. This requires a graphical display and working OpenGL support.
- `uv run python -m py_compile main.py`: check the entry point for syntax errors without launching the viewer.

There is no separate build process or configured automated test command.

## Coding Style & Naming Conventions

Use four-space indentation and follow Python's PEP 8 conventions. Use `snake_case` for functions, variables, and modules, and `PascalCase` for classes. Keep imports at the top of files. Extract reusable simulation logic into functions as the prototype grows, and ensure environments are closed when execution finishes. No formatter or linter is currently configured.

## Testing Guidelines

For simulation changes, run the syntax check and interactive smoke test. Confirm the viewer opens, stepping completes, episode resets work, and shutdown succeeds. No test framework or coverage threshold is established. Name future test files `tests/test_*.py`; document their runner when introducing them.

## Commit & Pull Request Guidelines

The repository has no commits yet, so no historical commit convention exists. Use concise, imperative subjects such as `Add deterministic environment seeding`. Keep changes focused.

Pull requests should describe the change, include validation commands and results, and link relevant issues. Include screenshots for visible rendering changes. Keep dependency changes synchronized with `uv.lock`; exclude local environments and generated logs such as `MUJOCO_LOG.TXT` from commits.
