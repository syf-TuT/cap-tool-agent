#!/usr/bin/env bash
# Run from the prepared Linux checkout. Controller credentials stay in the environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CAPX_CONTROLLER_API_KEY:?Set the frozen Controller API key}"
if ! curl -fsS --max-time 3 http://127.0.0.1:8116/docs >/dev/null; then
  echo "Start the Pyroki service on port 8116 before training." >&2
  exit 1
fi
exec .venv/bin/python -m scripts.capsule_rl.run_cube_stack_ac \
  --task spill_wipe --groups 16 \
  --controller-model glm-5 \
  --controller-endpoint https://coding.dashscope.aliyuncs.com/v1 \
  --model "${CAPX_PROGRAM_MODEL:-.codex-downloads/models/Qwen2.5-Coder-7B-Instruct}" \
  --verl "${CAPX_VERL_SOURCE:-.codex-downloads/verl-v0.6.1}" \
  --run-id "${CAPX_RUN_ID:-spill_wipe_ac_glm5_qwen7b_g16_20260912}" "$@"
