#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/cap-x
export MUJOCO_GL=egl JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
root=artifacts/spill_wipe_ac_glm5_qwen7b_g16_20260912_r03/eval_A16
curl --fail --silent http://127.0.0.1:8116/docs >/dev/null
.venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack generate --root "$root" --policy base
sam_pid=
cleanup() { if [[ -n "$sam_pid" ]]; then kill "$sam_pid" || true; fi; }
trap cleanup EXIT
if ! curl --fail --silent http://127.0.0.1:8114/docs >/dev/null; then
    .venv/bin/python -m capx.serving.launch_sam3_server > .codex_spill_base_sam3.log 2>&1 &
    sam_pid=$!
    for ((attempt=0; attempt<120; attempt++)); do
        if curl --fail --silent http://127.0.0.1:8114/docs >/dev/null; then break; fi
        kill -0 "$sam_pid"
        sleep 5
    done
    curl --fail --silent http://127.0.0.1:8114/docs >/dev/null
fi
.venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack evaluate --root "$root" --policy base
.venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack summarize --root "$root"
