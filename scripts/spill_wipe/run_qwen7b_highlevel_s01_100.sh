#!/usr/bin/env bash
# Run from the prepared Linux CaP-X checkout; no credentials are required here.
set -euo pipefail
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CAPX_FORCE_STREAMING_CHAT_COMPLETIONS=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN=spill_wipe_highlevel_qwen7b_s01_100_20260912
OUT="outputs/$RUN"
mkdir -p "$OUT"
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
  --dtype bfloat16 --gpu-memory-utilization 0.4 --max-model-len 8192 \
  --seed 20260912 --enforce-eager
start pyroki .venv/bin/python -m capx.serving.launch_pyroki_server \
  --port 8116 --host 127.0.0.1 --robot panda_description --target-link panda_hand
start sam3 .venv/bin/python -m capx.serving.launch_sam3_server \
  --device cuda --port 8114 --host 127.0.0.1
for port in 8120 8116 8114; do
  ready=0
  for attempt in $(seq 1 120); do
    if curl -sf --max-time 2 "http://127.0.0.1:$port/docs" >/dev/null; then ready=1; break; fi
    sleep 5
  done
  if [ "$ready" -ne 1 ]; then echo "Service $port not ready"; exit 1; fi
done
git rev-parse HEAD > "$OUT/git_commit.txt"
for condition in privileged nonprivileged; do
  suffix=""
  if [ "$condition" = privileged ]; then suffix="_privileged"; fi
  config="env_configs/spill_wipe/franka_robosuite_spill_wipe${suffix}.yaml"
  cp "$config" "$OUT/${condition}_source.yaml"
  .venv/bin/python capx/envs/launch.py --config-path "$config" \
    --server-url http://127.0.0.1:8120/v1/chat/completions \
    --model Qwen2.5-Coder-7B-Instruct --temperature 0.7 --max-tokens 4096 \
    --total-trials 100 --num-workers 1 --record-video True \
    --use-visual-feedback False --use-oracle-code False \
    --output-dir "$OUT/$condition" > "$OUT/$condition.log" 2>&1
done
echo COMPLETED
