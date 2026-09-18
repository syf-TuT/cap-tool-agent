#!/usr/bin/env bash
# Run inside the prepared Linux checkout. No credentials are stored here.
set -euo pipefail
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CAPX_FORCE_STREAMING_CHAT_COMPLETIONS=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN=robosuite_capsule20_qwen7b_nonpriv_s01_20_20260915
OUT="outputs/$RUN"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"
git diff --ignore-space-at-eol > "$OUT/source_changes.patch"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
start() {
  local name=$1; shift
  "$@" > "$OUT/$name.log" 2>&1 &
  pids+=("$!")
}
start model .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model .codex-downloads/models/Qwen2.5-Coder-7B-Instruct \
  --served-model-name Qwen2.5-Coder-7B-Instruct --host 127.0.0.1 --port 8120 \
  --dtype bfloat16 --gpu-memory-utilization 0.45 --max-model-len 32768 \
  --seed 20260915 --enforce-eager
start pyroki .venv/bin/python -m capx.serving.launch_pyroki_server \
  --port 8116 --host 127.0.0.1 --robot panda_description --target-link panda_hand
start sam3 .venv/bin/python -m capx.serving.launch_sam3_server \
  --device cuda --port 8114 --host 127.0.0.1
start graspnet .venv/bin/python -m capx.serving.launch_contact_graspnet_server \
  --device cuda --port 8115 --host 127.0.0.1
for port in 8120 8116 8114 8115; do
  ready=0
  for attempt in $(seq 1 120); do
    if curl -sf --max-time 2 "http://127.0.0.1:$port/docs" >/dev/null; then ready=1; break; fi
    sleep 5
  done
  if [ "$ready" -ne 1 ]; then echo "Service $port not ready"; exit 1; fi
done
tasks=(cube_lift cube_stack cube_restack spill_wipe two_arm_lift two_arm_handover)
configs=(
  env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml
  env_configs/cube_stack/franka_robosuite_cube_stack.yaml
  env_configs/cube_restack/franka_robosuite_cube_restack.yaml
  env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml
  env_configs/two_arm_lift/franka_robosuite_two_arm_lift.yaml
  env_configs/two_arm_handover/two_arm_handover.yaml
)
for i in "${!tasks[@]}"; do
  task=${tasks[$i]}
  config=${configs[$i]}
  cp "$config" "$OUT/${task}_source.yaml"
  .venv/bin/python - "$config" "$OUT/${task}_capsule.yaml" <<'PY'
import sys
import yaml
from pathlib import Path
config = yaml.safe_load(Path(sys.argv[1]).read_text())
assert config['env']['cfg'].get('privileged', False) is False
assert not config['env']['cfg'].get('multi_turn_prompt')
config.update(agent_mode='capsule', max_capsule_steps=20,
              capsule_execution_granularity='semantic_group',
              capsule_max_regions_per_group=20,
              capsule_llm_step_compact_context=True,
              capsule_require_task_success_for_finish=True,
              checkpoint_policy='region', rollback_policy='none',
              capsule_feedback_level='source_region_repair_hint')
Path(sys.argv[2]).write_text(yaml.safe_dump(config, sort_keys=False))
PY
  config="$OUT/${task}_capsule.yaml"
  echo "START $task $(date -u +%FT%TZ)"
  .venv/bin/python capx/envs/launch.py --config-path "$config" \
    --server-url http://127.0.0.1:8120/v1/chat/completions \
    --model Qwen2.5-Coder-7B-Instruct --temperature 1 --max-tokens 20480 \
    --total-trials 20 --num-workers 4 --record-video False \
    --use-visual-feedback False --use-img-differencing False \
    --use-video-differencing False --use-oracle-code False \
    --use-parallel-ensemble False --use-multimodel False \
    --output-dir "$OUT/$task" > "$OUT/$task.log" 2>&1
  echo "END $task $(date -u +%FT%TZ)"
done
echo COMPLETED
