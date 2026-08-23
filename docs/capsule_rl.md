# Capsule-Critique-GRPO：本地实现与服务器验证

本文档描述 Cube Stack privileged MVP 的服务器验证入口。当前仓库中的本地测试只覆盖
schema、repair lineage、fake clean replay、7+1 组装、loss/trainer mock 和脚本 dry-run；它们
不创建 Robosuite/MuJoCo 环境，不启动 PyRoKi、Controller、Program 模型或 optimizer。
因此，在服务器 gate 1--6 全部通过以前，状态只能写成“本地代码实现完成”，不能写成
“Capsule-RL runtime verified”。

## 服务器前置条件

使用配置模板：

```text
env_configs/cube_stack/capsule_rl/franka_robosuite_cube_stack_capsule_critique_grpo.yaml
```

运行前填写并检查：

- 单独 clone/checkout 的 VeRL 源码路径 `runtime.verl_source_path` 与完整
  `runtime.verl_pinned_sha`；仓库当前不包含可由 fresh clone 自动初始化的 VeRL gitlink，不能
  依赖模板默认目录；
- 项目 checkout 与 VeRL checkout 都必须以各自 Git worktree 顶层作为配置路径，且不能含
  staged、unstaged 或 untracked 文件；每个 gate 会在执行前后复核同一个干净 HEAD；
- `runtime.dataset_path`、`runtime.program_model_path`、`runtime.verl_resolved_config_path` 和
  `runtime.output_dir`；resolved VeRL YAML 必须已存在；
- `trainer_factory` 必须指向项目自有 factory；模板使用
  `capx.rl.capsule.server_factory:create_trainer`；
- Program actor 的 endpoint、model 与 `api_key_env`；
- 正式 `main_ppo` 每次启动使用新的 `runtime.run_id`；同名 checkpoint claim 不会覆盖；
- 独立且冻结的 Controller endpoint、model 与 `CONTROLLER_API_KEY`（或配置指定的变量名）；
- PyRoKi 服务地址，默认 `http://127.0.0.1:8116`；
- 可用 NVIDIA GPU/CUDA，并在运行环境设置 `MUJOCO_GL=egl`；
- privileged Cube Stack 环境、`FrankaControlPrivilegedApi`、EGL 与模型权重均可访问；
- `render: false`、`record_video: false`、Controller `frozen: true`。

Program actor 与 Controller 必须是两个独立服务。训练只更新 Program；Controller 的模型和
凭据不得指向正在更新的 actor。

Program 的 base/revision 采样由 pinned VeRL checkout 中同一个 trainable actor/rollout worker
完成，随后也由该 worker 计算 old log-prob 并更新；不能把一个未同步的外部 Program endpoint
当作训练 rollout 来源。`program_service` 的 endpoint/model/credential 用于服务器 preflight 和
部署身份检查，必须与本次 Program actor 部署一致。Capsule runtime 要求启动前 Ray 尚未初始化，
由本次 session 独占创建并在成功或失败后关闭，以免遗留 actor/ref worker。

推荐目录约定：

```text
outputs/<experiment-slug>/       # simulator/trainer 原始输出（忽略提交）
artifacts/<experiment-slug>/     # gate JSON、审计表和报告（忽略大文件）
remote_results/<experiment-slug>/# 从服务器下载的结果
```

## 安全验证模式

每个入口支持 `--validate-only` 或 `--dry-run`。验证模式只会：

- 解析 YAML 并检查 7+1、2×2 repair、12 turns、8192/2048 与 gamma=0.1；
- 检查数据集、VeRL checkout、Program 模型、Python 和输出父目录；
- 展开并打印命令参数；
- 对运行型 gate 检查 runner executable 与必需占位符。

验证模式不会执行 runner，不会联网探测 endpoint，不会 import torch，不会创建环境，也不会
启动服务、仿真、LLM 或 optimizer。Gate 2--6 的 canonical wrapper 锁定仓内
`scripts.capsule_rl.server_adapter` argv，不接受 `--runner-command` 覆盖。

执行模式拒绝覆盖已有 success、failure 或 log artifact。外部 runner 实际写入同目录唯一
staging 路径；成功退出且对应 gate verifier 在 staging 文件上通过后，wrapper 才以独占方式
发布最终 success 文件。wrapper 同时 capture 子进程输出并独占写入
`<artifact>.stdout.log` 与 `<artifact>.stderr.log`。若 runner、post-Git、typed verifier 或发布失败，
success 路径保持空缺，独立的 `<artifact>.failure.json` 以 `passed: false` 记录异常；Gate 1 也遵守
相同的 success/failure 分离合同，并且仅在全部检查与 typed verifier 通过后发布 success。若目标在执行
期间出现，本次 gate 失败且旧证据不变。

Gate 2--5 的 child failure 先从唯一 staging failure 以 hard link 提升为最终 failure，再清理
staging。Gate 6 由持有 checkpoint claim 的 adapter 直接发布 failure，wrapper 只验证并复用，
不会覆盖。若最终 failure 暂时无法发布，清理逻辑保留唯一 staging failure，供运维恢复。

## Gate artifact 公共合同

Gate 1--6 的 JSON 顶层必须共享 `schema_version: 1`、规范 gate 名、`passed: true`、
`execution_mode: repository_server_adapter_v1`、同一个
非空 `run_id`、同一个小写 `config_sha256`、`dataset_sha256` 与完整 `git_sha`。Gate 7 固定验证
`gate01_preflight.json` 至 `gate06_trainer.json`，逐项调用 typed verifier，并要求六项共享
run/config/dataset/commit identity，以及同一个 `resolved_environment_sha256` 和
`verl_resolved_config_sha256`；随后把每个文件 SHA-256 与前一个 SHA-256 写入 hash
manifest。任何 gate 的 success 旁若同时存在 `.failure.json`，或 artifact 来自非 canonical
execution mode，Gate 7 都会拒绝 `runtime_verified`。

Gate 7 审计并物化正式 bundle 后，再验证项目 trainer 配置：

```bash
python -m capx.rl.capsule.main_ppo \
  --config artifacts/cube_stack_capsule_rl_seed_resolved/capsule_rl.seed_resolved.yaml \
  --validate-only
```

## Dataset/config preparation

源 JSONL 每行至少包含 `task_id` 与 `prompt`。下面只预览 seed 展开，不写文件：

```bash
python -m scripts.capsule_rl.prepare_dataset_config \
  --config "$CAPSULE_CONFIG" \
  --source-dataset data/cube_stack_tasks.jsonl \
  --output-dir artifacts/cube_stack_capsule_rl_prepare \
  --seeds 5,6 \
  --validate-only
```

移除 `--validate-only` 后才会创建 `capsule_rl.dataset.jsonl` 和
`capsule_rl.resolved.yaml`。若目标目录已存在，脚本拒绝覆盖。

上一步只展开 seed，不会在本地或准备阶段伪造 `initial_state_sha256`。服务器完成 Gate 1--7
验证后，如要启动正式 `main_ppo`，需把所有训练 task 的真实 reset hash 物化成新的不可变
dataset/config bundle。先安全预览：

```bash
python -m scripts.capsule_rl.materialize_resolved_dataset \
  --config artifacts/cube_stack_capsule_rl_prepare/capsule_rl.resolved.yaml \
  --gate7-audit artifacts/$RUN_ID/audit_summary.json \
  --output-dir artifacts/cube_stack_capsule_rl_seed_resolved \
  --validate-only
```

移除验证参数后，该 server-only 命令才会按 task seed 创建真实环境并 reset，调用
`resolve_task_instances`，输出完整 `TaskInstanceV1` JSONL、更新了 `runtime.dataset_path` 的 YAML、
Gate 7 audit 副本和 `bundle_manifest.json`。manifest 绑定 source config/dataset SHA、Gate 7
audit SHA/run_id/typed task identities、resolved environment、resolved VeRL YAML，以及 output
dataset/config SHA；新 YAML 同时写入 `runtime.bundle_manifest_path` 与
`runtime.gate7_audit_path`。输出目录必须不存在；bundle 以独占目录发布，`BaseException` 也只会
清理仍由本进程持有身份的 staging/部分输出，绝不覆盖或删除并发替换的目录。正式训练使用新
YAML。Smoke Gate 1--6 仍应全程
使用同一原始配置以保持 artifact `config_sha256` 一致；物化 bundle 服务于随后单独启动的正式
训练，不得在同一 gate audit 中途切换配置。

resolver 输出还必须与 source dataset 按原顺序逐行绑定：`task_id`、`environment_seed`、
`prompt`、`environment`、`api`、`privilege`、`metadata` 等所有 `TaskInstanceV1` 不可变字段必须
完全相同，不能用同 count 的另一批 task 替换或重排。只有 source 缺少
`initial_state_sha256` 或明确使用全零 placeholder 时，才允许补成真实 reset hash；source 已有的
非 placeholder hash 也属于不可变字段。

正式 `main_ppo` 在 `--validate-only` 和训练入口都会重算 manifest、source/output config 与
dataset、Gate 7 audit、environment config 和 resolved VeRL YAML 的 bytes SHA；任何缺失或漂移
都会在加载 trainer、模型或 optimizer 前失败。训练结束后再次验证同一 provenance snapshot，
防止训练期间替换 bundle 输入。

## Staged gates

Gate 2--6 wrapper 默认调用仓内
`python -m scripts.capsule_rl.server_adapter <subcommand>`，不再依赖占位的外部 adapter。
这些 canonical wrapper 禁止自定义 runner；需要调试自有 runner 时必须使用另一个输出目录，且其
artifact 被视为 noncanonical，不能进入 Gate 7。wrapper 从 `--config` 读取 task、dataset、Program actor、
Controller 与输出目录；artifact 父目录名就是传给 adapter 的 `run_id`，所以六个 artifact 必须
放在同一个 `artifacts/$RUN_ID/`。Gate 4 的 P0 由 Program actor 采样的七失败 base batch 按
partial reward/token distance 选出，不接受来源不明的外部 P0。Gate 6 默认读取同目录
`gate05_guided_group.json`，并记录该输入文件的 SHA-256。
每次 adapter 执行还会从同一 config 重新计算 dataset bytes SHA-256，写入公共 gate envelope；
同时在执行前后重算 resolved environment 与 resolved VeRL YAML SHA-256。若任一依赖在单个
gate 执行期间发生变化，则拒绝发布成功 artifact；Gate 6 还要求 Gate 5 输入携带完全相同的
dependency identity。

建议先对每个 wrapper 运行 `--validate-only`，确认打印出的完整 adapter argv 后再移除该参数。

### 1. Preflight

Preflight 确认提交 SHA、VeRL pinned SHA、依赖路径、GPU/CUDA、`MUJOCO_GL=egl`、Program
模型、两个服务凭据/端口、PyRoKi 和 resolved environment SHA-256：

```bash
python -m scripts.capsule_rl.server_preflight \
  --config "$CAPSULE_CONFIG" \
  --artifact artifacts/$RUN_ID/gate01_preflight.json \
  --run-id "$RUN_ID" \
  --validate-only
```

移除验证参数才会进行只读 GPU/端口探测并写 artifact。任何必需检查失败都停止。artifact
同时记录 resolved environment、resolved VeRL config 与 Capsule config 的 SHA-256；此外会直接
按 `runtime.dataset_path` 的落盘 bytes 计算 `dataset_sha256`（JSONL/JSON/Parquet 均不做
规范化），并记录经过 `TaskInstanceV1` 类型检查的 `task_id`/`environment_seed` 摘要。即使
dataset 被 Git 忽略，也不能在后续 gate 中无痕替换内容。

### 2. Seed gate

真实 Cube Stack privileged 环境必须在同一验证流程按 5→6→5 reset；两个 seed 5 的
`initial_state_sha256` 相同，seed 6 不同：

```bash
python -m scripts.capsule_rl.check_seed_determinism \
  --config "$CAPSULE_CONFIG" \
  --artifact artifacts/$RUN_ID/gate02_seed.json \
  --validate-only
```

wrapper 默认展开为仓内命令：

```bash
python -m scripts.capsule_rl.server_adapter \
  --config "$CAPSULE_CONFIG" \
  --artifact artifacts/$RUN_ID/gate02_seed.json \
  --run-id "$RUN_ID" seed --seeds 5,6,5
```

artifact 必须包含 `seeds: [5,6,5]` 与三个规范化初态 SHA-256。
`prepare_dataset_config` 的输出可以暂时没有 `initial_state_sha256`。Gate adapter 的
server-only `resolve_task_instances` 会用真实环境按每行 `environment_seed` reset，在内存中构造
带 hash 的不可变 `TaskInstanceV1`，不修改源 dataset；随后 clean replay 必须再次得到同一 hash。
正式 `python -m capx.rl.capsule.main_ppo` 则使用严格 loader，dataset 每行必须已经包含由前置
seed/replay gate 固化的 `initial_state_sha256`，缺失时直接拒绝启动，不会在训练过程中隐式补齐。

### 3. Oracle replay gate

同一持久 worker 连续进行两次 oracle `reset(seed) -> step(full_program)`；两次都必须
clean-success，且 `worker_id` 相同：

```bash
python -m scripts.capsule_rl.oracle_clean_replay \
  --config "$CAPSULE_CONFIG" \
  --artifact artifacts/$RUN_ID/gate03_oracle.json \
  --validate-only
```

其默认 adapter subcommand 是 `oracle --seed 5 --replays 2`。task 来自
`runtime.dataset_path`；oracle source 来自配置指定 Cube Stack 环境的 `oracle_code`。

该 gate 同时审计 RNG、IK warm-start、执行 namespace 与 watchdog；不得经过 Capsule
Controller。每条 replay 必须内嵌完整 `ProgramReplayResultV1`，由 typed schema 重新检查
reward、task-completed、truncated、fatal error、源码 hash、初态 hash 与 seed。artifact 还必须
显式记录 `direct_replay: true`、`controller_used: false`，并为两次 replay 记录相同
`worker_id`/`reset_seed` 以及 `namespace_fresh`、`api_state_cleared`、`watchdog_active` 证据。
其中 namespace/API 两项必须读取 replay diagnostics 中真实的
`reset_info.capsule_reset_evidence`，禁止硬编码成功。

### 4. Collector gate

选择一个已知失败 P0/seed，对 2 个 P0 各采集 2 条 repair，最多 12 Controller turns；
中间 edit 不进入 simulator replay。此 gate 允许没有成功 guided candidate，但必须保存四条
完整 RepairTrace/Audit，并证明 Controller frozen。`base_results` 保存两个 typed、semantic
failure 的 `ProgramReplayResultV1`；infra/evaluator error 不得伪装成 reward 0。
`repair_traces` 每条记录包含 `p0_rank`、`trajectory_index` 和可由 `RepairTraceV1` 精确重建
的 `trace`；四条记录覆盖完整 2×2，trace identity/hash 必须与对应 P0 一致，且
`intermediate_replay_count` 必须为 0：

```bash
python -m scripts.capsule_rl.controller_collector_smoke \
  --config "$CAPSULE_CONFIG" \
  --artifact artifacts/$RUN_ID/gate04_collector.json \
  --validate-only
```

默认 adapter subcommand 是
`collector --p0-count 2 --trajectories 2 --max-turns 12`。

### 5. Guided gate

继续采样，直到获得至少一个 PT 与重新生成 P_hat 均 clean-success 的真实组。最终学习组
必须恰好 8 个 member、7 个 base reward 0、1 个 guided reward 1；训练文本只含 original
prompt + P_hat。artifact 保存完整 `LearningGroupV1`、七个 typed base replay result、固定顺序
的完整四条 2×2 `repair_attempts`，以及包含 `RepairTraceV1`、P0/PT/P_hat 三个
`ProgramReplayResultV1` 的 `selected_repair`。verifier 复用 trainer provenance contract，核对
同 task/seed/初态、确定性 P0 选择、四条 rank/index/trajectory identity、P0→trace→PT、
P_hat→guided member，并证明 selected attempt 是固定顺序中的首个成功者；同时要求
`training_input_contains_critique: false`：

```bash
python -m scripts.capsule_rl.build_verified_group \
  --config "$CAPSULE_CONFIG" \
  --artifact artifacts/$RUN_ID/gate05_guided_group.json \
  --validate-only
```

默认 adapter subcommand 是
`guided --group-size 8 --base-count 7 --guided-count 1 --max-group-attempts 20`。

### 6. Trainer gate

仅使用已经验证的 `[0,0,0,0,0,0,0,1]` `LearningGroupV1` 做一个 optimizer step。artifact
必须记录恰好一个 step、有限且非零的 gradient norm、指标和 checkpoint：

```bash
python -m scripts.capsule_rl.one_step_trainer_smoke \
  --config "$CAPSULE_CONFIG" \
  --artifact artifacts/$RUN_ID/gate06_trainer.json \
  --validate-only
```

默认 adapter subcommand 是
`trainer --optimizer-steps 1 --group-rewards 0,0,0,0,0,0,0,1 --guided-artifact
artifacts/$RUN_ID/gate05_guided_group.json`。Gate 6 从该文件重建完整四条 2×2
`repair_attempts`，不会只保留 selected attempt；checkpoint 路径包含经过清理并附 hash 后缀的
`run_id`，artifact 同时记录 `guided_artifact_sha256`。Gate 6 在启动 worker 前排他 claim 本次
run 目录，先把 checkpoint 写入同一文件系统的唯一 staging，确认至少一个文件后再发布到
final 路径，并生成带 file count、tree SHA-256 和 optimizer step before/after 的 manifest；
现存 run 目录一律拒绝覆盖。

这一步必须保持 rollout importance sampling 关闭、group advantage 不做 std 归一化、base
使用 clipped GRPO/reference KL、guided token 使用 gamma=0.1 shaping。artifact 还必须证明
`guided_token_mask_present: true`、`rollout_is: false`、
`norm_adv_by_std_in_grpo: false`，记录正数 `guided_token_count`、
`guided_mask_response_only: true`、`actor_update_skipped: false`，包含非空有限指标，并指向
服务器上实际存在的绝对 checkpoint 路径。verifier 还要求 artifact 明确记录并匹配
`loss_mode: capsule_critique`、`capsule_gamma: 0.1` 与 `reference_kl_enabled: true`。
一次 `update_actor` RPC 只记录为 `actor_update_rpcs: 1`，不能单独当作 optimizer step 证据。
actor worker 会从 AdamW optimizer state 读取各 rank 的 before/after step，要求所有 rank 一致
且 delta 恰为 1；AMP 或其他原因跳步会令 Gate 6 失败。还必须记录并验证
`rollout_mode: sync`、`ppo_epochs: 1`、`ppo_mini_batch_size: 8`、
`ulysses_sequence_parallel_size: 1`、可整除 8 的 `data_parallel_world_size` 和正的
`reference_kl_coef`；actor optimizer 与 VeRL trainer 的 `total_training_steps` 由
`dataset row count × total_epochs` 统一设置。

### 7. Result audit

Gate 1--6 结束后汇总 base/repair/PT/P_hat、retry、infra failure、guided shaping 与 optimizer
证据：

```bash
python -m scripts.capsule_rl.analyze_artifacts \
  --input-dir artifacts/$RUN_ID \
  --output-json artifacts/$RUN_ID/audit_summary.json \
  --output-report artifacts/$RUN_ID/audit_report.md \
  --validate-only
```

缺失 gate、identity 不一致、同 gate success/failure 并存、noncanonical 来源或 typed evidence
无效都会令命令非零退出，且不会生成成功报告。
Gate 7 还会重算 Gate 5 文件 SHA-256、checkpoint tree SHA-256 与 manifest，拒绝只在 JSON 中
声明但未由落盘字节支持的依赖或 checkpoint。
六个 gate 的 `dataset_sha256` 必须完全一致；Oracle、Collector 与 Guided 的 typed
`(task_id, environment_seed, initial_state_sha256)` 也必须形成同一个 seed-5 task identity。
六个 gate 的 resolved environment 与 resolved VeRL YAML SHA-256 也必须逐项一致；Gate 7
把 seed-5 typed identity、两项 dependency hash、run/config/Git/dataset identity 一并写入
`audit_summary.json`，供正式 bundle materializer 绑定。
只有 gate 1--6 全部通过时，报告才写入 `runtime_verified: true`、六项状态和 artifact hash
chain，才能把状态升级为“Capsule-RL runtime verified”；正式多-seed 训练、成功率比较和
消融不属于这些 smoke gate。JSON 与 Markdown 报告先分别写入 staging，再作为一对发布；第二个
文件发布失败时会回滚本次已发布的第一个文件，避免留下半套审计结果。

## 失败恢复

任何 gate 失败都立即停止，不跨 gate 继续：

1. 保留该 gate 已生成的 success JSON 或 `<artifact>.failure.json`、配套 stdout/stderr logs、
   提交 SHA 和 resolved config；failure JSON 固定包含 `schema_version: 1`、规范 gate、
   `passed: false`、`run_id`、可取得的 config/Git/dataset SHA，以及 exception type/message/stage；
   success 与 failure 若因外部事故同时出现，该目录不能通过 Gate 7，必须保留取证并换新 run 目录；
2. 若为 infra/evaluator failure，确认最多两次 retry 后整组丢弃，不能改写成 reward 0；
3. 若 worker timeout/crash，确认 poisoned worker 已重建，再从失败 gate 重新开始；
4. 若 seed hash 不稳定，先修复 RNG/placement sampler/IK/namespace，禁止继续 collector；
5. 若 PT 成功但 P_hat 失败，保留 repair evidence，但不得注入 guided member；
6. 若 trainer 梯度为 0、NaN/Inf 或未保存 checkpoint，Gate 6 失败；
7. Gate 6 的 post-check/verifier/publication 失败仍先 rollback 当前 checkpoint transaction，
   再写 failure JSON；若 rollback 自身也失败，原始异常仍是主 `exception`，并追加
   `rollback_exception`。正常保存失败会自动清理当前进程持有的 claim；若进程崩溃留下
   `.capsule_checkpoint_claim`，先核对 run 目录与 owner marker，再按服务器运维流程隔离该目录，
   不得直接覆盖半成品；
8. 若所有 scheduled group 都因未知 replay/infra 原因丢弃，完整
   `discarded_groups.json` 会先落盘，run 随后非零退出且不写 checkpoint；
9. 修复后使用同一提交、配置和输出新目录重跑失败 gate；已有 success、failure 和 log 路径都
   是 immutable evidence，不得删除或覆盖后原地重试。

不得把本地 mock 通过、单个 collector artifact、PT-only 成功或 dry-run 成功表述为 runtime
verified。
