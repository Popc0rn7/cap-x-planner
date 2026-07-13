# Repository Guidelines

## Project Structure & Module Organization

`capx/` contains the Python package. Environment orchestration lives in `capx/envs/`, robot and perception backends in `capx/integrations/`, model-serving utilities in `capx/serving/`, and reusable agent skills in `capx/skills/`. Task YAML files are grouped by embodiment under `env_configs/`. Tests live in `tests/`, with hardware- or model-dependent coverage under `tests/integrations/`. Documentation is in `docs/`; training helpers are in `scripts/` and `verl_agent_reward/`. The React/Vite interface is isolated in `web-ui/`. Treat `capx/third_party/` as vendored code and avoid editing it unless updating an integration intentionally.

## Build, Test, and Development Commands

- `uv sync --extra dev --extra robosuite` installs the development environment for Robosuite. LIBERO conflicts with Robosuite and should use its separate environment described in `README.md`.
- `uv run pytest tests/test_environments.py -q` runs the core environment tests.
- `uv run pytest tests/test_environments.py::test_franka_pick_place_code_env -q` runs one targeted test.
- `./scripts/regression_test.sh quick` performs the short, 10-trial smoke test.
- `uv run --no-sync --active capx/envs/launch.py --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml` launches a representative evaluation.
- `ruff check .` and `ruff format .` lint and format Python.
- From `web-ui/`, use `npm install`, `npm run dev`, and `npm run build` for frontend work.

## Coding Style & Naming Conventions

Use four-space indentation, type hints for public interfaces, and concise docstrings for non-obvious behavior. Ruff targets Python 3.12 with a 100-character line length; follow its import sorting and modernization rules. Use `snake_case` for Python modules, functions, variables, and YAML filenames; use `PascalCase` for classes and React components. Keep environment configuration names descriptive, for example `franka_libero_spatial_0.yaml`.

## Testing Guidelines

Pytest is the primary framework. Name files `test_*.py` and tests `test_*`. Add focused unit tests near the affected subsystem and mark expensive external-service checks with `@pytest.mark.integration`. Some integration tests require CUDA, checkpoints, simulators, or `HYRL_INTEGRATION_REAL=1`; document these prerequisites and skip cleanly when absent. There is no fixed coverage threshold, but changes should exercise new behavior and relevant reward regressions.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects such as `Fix SAM3 tensor device mismatch`. Keep commits focused and explain observable behavior, not implementation chronology. Pull requests should include a concise problem/solution summary, exact validation commands, simulator and GPU assumptions, and linked issues. Include screenshots for `web-ui/` changes and reward or success-rate evidence for environment changes. Do not commit API keys, downloaded checkpoints, generated outputs, or simulator datasets.
