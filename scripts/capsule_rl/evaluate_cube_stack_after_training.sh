#!/usr/bin/env bash
# Run on the prepared SeeTaCloud host after the specified training process exits.
set -euo pipefail
train_pid=${1:?training PID required}
training_root=${2:?training artifact directory required}
evaluation_root=${3:?new evaluation artifact directory required}
log_root=${4:?log directory required}
export MUJOCO_GL=egl JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
unset CAPX_CONTROLLER_API_KEY
mkdir -p "$log_root"
while kill -0 "$train_pid" 2>/dev/null; do sleep 30; done
.venv/bin/python - "$training_root" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
protocol = json.loads((root / "protocol.json").read_text())
result = json.loads((root / "training_result.json").read_text())
assert result["completed_group_count"] == protocol["group_count"] == 32
assert result["optimizer_step_delta"] > 0
print("Completed 32 training groups; starting paired evaluation", flush=True)
PY
for entry in 'sam3 8114' 'contact_graspnet 8115' 'pyroki 8116'; do
    read -r service port <<< "$entry"
    if ! curl -fsS --max-time 2 "http://127.0.0.1:$port/docs" >/dev/null; then
        nohup .venv/bin/python -m "capx.serving.launch_${service}_server" \
            > "$log_root/$service.log" 2>&1 < /dev/null &
    fi
    ready=false
    for ((attempt=0; attempt<60; attempt++)); do
        if curl -fsS --max-time 2 "http://127.0.0.1:$port/docs" >/dev/null 2>&1; then
            ready=true
            break
        fi
        sleep 5
    done
    if [[ "$ready" != true ]]; then echo "$service failed to become ready" >&2; exit 1; fi
done
.venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack prepare \
    --root "$evaluation_root" --training-root "$training_root" \
    --seeds 45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64 \
    --samples-per-seed 4 > "$log_root/evaluation_prepare.log" 2>&1
for policy in base trained_lora; do
    .venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack generate \
        --root "$evaluation_root" --policy "$policy" > "$log_root/${policy}_generate.log" 2>&1
    .venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack evaluate \
        --root "$evaluation_root" --policy "$policy" > "$log_root/${policy}_evaluate.log" 2>&1
done
.venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack summarize --root "$evaluation_root"
