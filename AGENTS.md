# Repository Guidelines

## Local Experiment Environment

This Windows checkout at `F:\code\cap-x` is the Codex workspace for reading and
editing files. Do not install Python dependencies, run `uv sync`, initialize
large submodules, or launch CaP-X experiments directly from this Windows path.

For runtime work, use the prepared WSL2 Ubuntu environment:

```powershell
wsl -d Ubuntu-22.04
```

Inside WSL, the runnable project copy is:

```bash
cd /home/capx/code/cap-x
```

Use this WSL path for all experiment commands, dependency installation, `uv`
commands, simulator launches, pytest runs, and output inspection. If a command is
issued from Codex/PowerShell, wrap it through WSL, for example:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; <command>'
```

In this Codex desktop environment, WSL distro registration is only visible from
the elevated command context: plain `wsl.exe --list` may show no distributions,
while elevated `wsl.exe --list --all --verbose` shows `Ubuntu-22.04`. Therefore,
run CaP-X experiment, simulator, `uv`, and pytest commands through the elevated
PowerShell/Codex command context when invoking WSL from Codex.

WSL has been configured with:

- Project copy: `/home/capx/code/cap-x`
- Python environment: `/home/capx/code/cap-x/.venv`
- `uv`: `/home/capx/.local/bin/uv`
- Windows proxy from WSL: `http://172.28.192.1:7890`
- Git proxy: `http.proxy` and `https.proxy` set to `http://172.28.192.1:7890`
- Verified GPU in WSL: `torch.cuda.is_available() == True`, CUDA `12.8`

The Robosuite base environment has already been installed in WSL with:

```bash
uv sync --frozen --extra robosuite \
  --no-sources-package nvidia-curobo \
  --no-install-package nvidia-curobo \
  --no-progress
```

This skips local editable `nvidia-curobo` because WSL currently has NVIDIA driver
forwarding but not the full CUDA Toolkit (`CUDA_HOME` is not set). Do not retry a
plain `uv sync --extra robosuite` unless CUDA Toolkit has been installed and
`CUDA_HOME=/usr/local/cuda` is configured.

The following minimal Robosuite privileged oracle smoke test has passed:

```bash
export MUJOCO_GL=egl
uv run --no-sync capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml \
  --use-oracle-code True \
  --total-trials 1 \
  --num-workers 1 \
  --record-video False
```

Observed result:

```text
success rate / average reward / completed: 1.000/1.000/1
Reward: 1.0
Task Completed: True
```

Only use the Windows checkout for source edits and documentation edits. If those
edits need to be used by WSL experiments, copy or sync them into
`/home/capx/code/cap-x` before running.

## Project Structure & Module Organization

`capx/` contains the Python package. Environment launch and runner code lives in `capx/envs/`, robot and perception integrations in `capx/integrations/`, serving utilities in `capx/serving/`, web backend code in `capx/web/`, and helpers in `capx/utils/`. YAML experiment definitions are under `env_configs/`, grouped by task family. Tests live in `tests/`, with external-service or hardware-heavy checks in `tests/integrations/`. Documentation is in `docs/`, scripts in `scripts/`, RL reward code in `verl_agent_reward/`, and the React/Vite interface in `web-ui/`.

## Build, Test, and Development Commands

- `uv sync`: install the base Python environment from `pyproject.toml` and `uv.lock`.
- `uv sync --extra robosuite`: install Robosuite support. Use separate environments for conflicting extras such as LIBERO.
- `uv run pytest tests/test_environments.py -q`: run the main environment unit tests.
- `uv run pytest tests/integrations -q`: run integration tests when required dependencies, models, and GPU access are available.
- `ruff check .` / `ruff format .`: lint and format Python code.
- `cd web-ui && npm install`: install frontend dependencies.
- `cd web-ui && npm run dev`: start the Vite UI locally.
- `cd web-ui && npm run build`: type-check and build the web UI.

## Coding Style & Naming Conventions

Python targets 3.10-3.12, with Ruff configured for Python 3.12, 100-character lines, import sorting, and bugbear/simplification checks. Use 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and descriptive YAML names such as `franka_robosuite_cube_stack_privileged.yaml`. Frontend code uses TypeScript, React function components, `PascalCase` component files, and hooks named `useSomething`.

## Testing Guidelines

Prefer focused `pytest` tests near the behavior being changed. Name tests `test_*.py` and test functions `test_<behavior>`. Keep simulator, model-download, and GPU-dependent coverage in `tests/integrations/` or document required setup in the test. Before merging environment changes, run the relevant regression command from `docs/development.md` and compare expected rewards.

## Commit & Pull Request Guidelines

Recent history uses short, imperative subjects such as `Fix SAM3 tensor device mismatch when using non-default CUDA device` and `Update README with clearer simulator setup instructions`. Keep commits scoped and mention the subsystem when useful. Pull requests should include a concise problem/solution summary, commands run, linked issues, screenshots for `web-ui` changes, and notes for new environment variables, model access, or simulator prerequisites.

## Security & Configuration Tips

Do not commit secrets or local credentials. Files such as `.openrouterkey` are intentionally git-ignored. Large third-party dependencies are vendored through submodules; after cloning, run `git submodule update --init --recursive`.
