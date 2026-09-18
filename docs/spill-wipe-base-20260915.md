# Spill Wipe initial base model evaluation

The untrained Qwen2.5-Coder-7B-Instruct model achieved **8/80 (10%)**:
47 program errors and 25 task failures. No LoRA adapter or controller was used.
Evaluation used the frozen A16 evaluation protocol: seeds 201–220, four
samples per seed, temperature 0.7, top-p 0.8, top-k 20, repetition penalty
1.1 and at most 4096 generated tokens. The evaluator's completed summary
verified all 80 records and paired physical reset evidence against A16.

| Policy | Success |
| --- | ---: |
| Initial base | 8/80 (10%) |
| A16 | 10/80 (12.5%) |
| A32 | 9/80 (11.25%) |
| A48 | 13/80 (16.25%) |
| C16 | 8/80 (10%) |
| C32 | 12/80 (15%) |
| C48 | 23/80 (28.75%) |

C48 exceeds the observed base success rate by 18.75 percentage points;
A48 exceeds it by 6.25 points. These are descriptive comparisons on the same
scenes used during the continuation experiments, not independent holdout results.

Reproduce remotely with `bash scripts/spill_wipe/run_base_s201_220.sh` from
`/root/autodl-tmp/cap-x`. The launcher requires Pyroki on port 8116 and starts
SAM3 if needed. Results, generated code and replay records are under
`artifacts/spill_wipe_ac_glm5_qwen7b_g16_20260912_r03/eval_A16`, in the
`generations/base`, `evaluation/base` and `summary.json` paths.
The run log is `.codex_spill_base_eval.log`. The original frozen protocol and
trained A16 evaluation records were reused. No additional tests were added.
