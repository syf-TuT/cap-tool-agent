# Capsule 实验档案审计

## 1. 审计范围与口径

本文件只记录仓库中能够定位到具体文件的实验事实。实验目录包含多轮开发期运行，代码提交、模型调用模式、超时、重试和结果格式并不完全一致。因此，下文区分三类证据：

- **可直接合并**：任务、方法、API、模型、代码提交、超时和结果口径一致，且种子集合互不重叠；
- **有条件参考**：主要条件接近，但存在流式模式、重试、结果选择或代码版本差异；
- **不可直接比较**：任务、接口、提交版本、结果协议或基础设施状态存在实质差异。

除非结果文件明确给出，`sandbox_rc != 0` 不自动等同于任务失败；`task_completed` 是任务完成的主要判据。缺失、超时、无效结果和服务失败不能静默计入任务失败率。

## 2. 可复核的主要运行

| 编号 | 任务与方法 | 核心设置 | 结果 | 证据位置 | 可用性 |
|---|---|---|---|---|---|
| E01 | Cube Stack，Capsule 首轮 20 种子 | DeepSeek V4 Flash；FrankaControlApi；非特权；450 s；30 Capsule 步；流式调用；Git 提交未记录 | 9/20 完成；平均最终奖励 0.4512 | `remote_results/r1_20/.codex_cube_stack_capsule_frankacontrolapi_packy_seeds1_20_summary.json` | 可作为早期首轮开发结果，不能代表当前实现 |
| E02 | Cube Stack，Capsule 按每个种子选择最新运行 | DeepSeek V4 Flash；FrankaControlApi；非特权；450 s；30 步；主要提交 `7058f82...`；包含重试运行 | 17/20 完成；平均奖励 0.8614；2 个超时；选择规则为 `latest run per seed` | `remote_results/cube_stack/20260714T0619Z_capsule_dsv4flash_s16-20/aggregate_seeds_01_20.json` | 重试包容结果；不是首次尝试成功率 |
| E03 | Cube Stack，多轮重新生成基线，种子 11--20 | DeepSeek V4 Flash；FrankaControlApi；非特权；最多 5 次重新生成；4096 tokens；非流式 | 6/10 完成；平均奖励 0.6052；平均运行时间 51.407 s | `remote_results/cube_stack_original_mt_20260708_seed11_20_results.tgz` 内 `results.tsv` | 与 E01/E02 条件并非完全一致，仅作条件化参考 |
| E04 | Cube Lifting，多轮重新生成，种子 1--10 | DeepSeek V4 Flash；FrankaControlApi；非特权；450 s；最多 5 次重新生成；4096 tokens；非流式 | 8/10 完成；平均奖励 0.8998 | `remote_results/cube_lift_mt_s1_10_20260708/.codex_runs/.../summary.json` | 可与同协议的种子 11--20 合并 |
| E05 | Cube Lifting，多轮重新生成，种子 11--20 | 与 E04 同类设置 | 10/10 完成；平均奖励 1.0 | `remote_results/legacy/rr/clmt_1120/.codex_runs/.../summary.json` | 与 E04 合并后为 18/20；平均奖励待统一复算 |
| E06 | Cube Lifting，Capsule，种子 1--20 | DeepSeek V4 Flash；FrankaControlApi；非特权；开发期 Capsule | 13 个完成、3 个未完成、4 个缺失/无有效结果；有效结果上的完成率为 13/16 | `remote_results/legacy/cl20/*contentonly_summary.json` | 缺失结果不能计作算法失败；与 E04/E05 需条件化比较 |
| E07 | Cube Restack，Capsule，种子 1--20 | DeepSeek V4 Flash；FrankaControlApi；非特权；450 s；30 步；提交 `7058f82...` | 4/20 完成；平均奖励 0.277433；11 个视频 | `remote_results/cube_restack/20260714T0641Z_capsule_ds4flash_s01-20/run.json` | 可作为该版本的完整任务结果 |
| E08 | Two-arm Handover，Capsule，种子 1--20 | DeepSeek V4 Flash；FrankaHandoverApi；450 s；30 步；提交 `bfe6d02...` | 0/20 完成；3 个正常结束，17 个试验预算耗尽；平均奖励 0.024035 | `remote_results/two_arm_handover/20260711T1209Z_capsule_ds4flash_s01-20/run.json` | 负结果；必须区分预算耗尽与正常结束 |
| E09 | Two-arm Handover，Capsule，种子 1--10 | DeepSeek V4 Flash；FrankaHandoverApi；750 s；30 步；提交 `7058f82...` | 0/10 完成；平均奖励 0.169420 | `remote_results/two_arm_handover/20260714T1012Z_capsule_ds4flash_s01-10/data/.codex_two_arm_handover_*_summary.json` | 与 E08 版本和超时不同，不合并 |
| E10 | Two-arm Lift，Capsule，种子 1--5 | DeepSeek V4 Flash；FrankaTwoArmLiftApi；非特权；700 s；30 步；提交 `687b7f5...` | 1/5 完成；平均奖励 0.201896 | `remote_results/two_arm_lift/20260713T0756Z_capsule_deepseek-v4-flash_s01-05/data/.codex_experiments/.../summary.json` | 可与 E11 合并 |
| E11 | Two-arm Lift，Capsule，种子 6--20 | 与 E10 相同提交、API、超时和步数 | 10/15 完成；平均奖励 0.484623 | `remote_results/two_arm_lift/20260713T0905Z_capsule_deepseek-v4-flash_s06-20/data/.codex_experiments/.../summary.json` | 与 E10 合并后为 11/20，合并平均奖励约 0.41394 |

## 3. 可比性判断

### 3.1 可以直接合并

- E04 与 E05：均为 Cube Lifting 的原始多轮重新生成方法，种子集合互补。完成数可合并为 18/20；平均奖励应从 20 条原始记录统一复算后再用于终稿。
- E10 与 E11：任务、方法、API、模型、提交、超时和步预算一致，种子集合互补。合并后 Capsule 在 Two-arm Lift 上完成 11/20，按两个子集样本数加权的平均奖励约为 0.41394。

### 3.2 只能有条件参考

- E01 与 E03：任务和 API 接近，但 Capsule 使用流式调用且 Git 提交未知，基线使用非流式调用；不能把差异归因于恢复机制。
- E02 与 E03：E02 按每个种子选择最新运行并包含重试，E03 是单轮基线集合；可以展示“重试后可达到的结果”，不能写成公平的首次尝试优势。
- E04/E05 与 E06：任务、模型和接口接近，但 Capsule 档案有 4 个缺失结果，且结果 schema 与基线不同；只能分别报告。

### 3.3 不可直接比较或合并

- E08 与 E09：代码提交、超时和结果集合不同；只可用于说明增加超时并未在该批次产生任务完成，同时平均奖励有所变化。
- Cube Stack、Cube Restack、Two-arm Handover 与 Two-arm Lift 之间：任务奖励尺度、API、物体与控制难度不同，成功率和平均奖励不组成统一排行榜。
- 5 篇外部论文中的成功率与本仓库结果：数据集、试次、重试、模型和平均方式不同，禁止直接排名。

## 4. 结果分类规则

| 类别 | 判定规则 | 论文写法 |
|---|---|---|
| 任务成功 | `run_outcome=finished` 且 `task_completed=true`，或旧档案明确记录 `task_completed=1` | 计入任务完成数 |
| 算法失败 | 正常完成执行但 `task_completed=false`，或明确的执行失败 | 与预算和基础设施失败分列 |
| 试验预算耗尽 | `trial_budget_exhausted` 或 Capsule 步预算耗尽 | 不写成模型服务故障 |
| 提供商失败 | HTTP、限流、服务超时或响应错误 | 不计入纯算法失败率，单独报告 |
| 实验基础设施失败 | 父进程守卫、结果缺失、无效 schema、服务未就绪等 | 单列并保留重跑信息 |
| 旧档案未知 | 只能从目录名或不完整摘要推断 | 只作补充材料，不形成主要结论 |

当前 `capx/utils/experiment_results.py` 已实现重试感知聚合和互斥失败桶，但多数历史档案早于该 schema。终稿前应使用当前聚合器统一复算原始试次。

## 5. 允许进入初稿的有限结论

1. Capsule 在多个 Robosuite 任务上能够完成程序分段执行并产生可回放的代码、追踪、反馈和视频档案。
2. 在 Cube Stack 的“每种子选取最新运行”集合中，17/20 个种子完成任务；该结果包含重试，不能代表首次尝试成功率。
3. 不同任务上的结果差异较大：Cube Restack 为 4/20，Two-arm Handover 的两个主要集合均为 0 完成，Two-arm Lift 的同版本合并集合为 11/20。
4. 这些结果说明当前系统可运行，但不足以单独证明效应边界前向恢复相对多轮重新生成提高了成功率；缺少统一提交、统一调用模式和严格配对的消融实验。

## 6. 待统一复算

- 使用当前结果 schema 重建所有任务的首次尝试成功率、按重试预算成功率和总尝试数；
- 统一计算每任务的平均奖励、置信区间、LLM logical calls/attempts、token、LLM 时间、试验时间和机器人原语调用数；
- 在相同提交、API、模型参数、超时、种子和服务模式下重跑 Capsule 逐动作循环与多轮重新生成基线；
- 从追踪档案统计 `append_recovery` 触发次数、执行次数、有效次数及失败原因；
- 核对 E06 的 4 个缺失结果属于提供商、超时还是实验基础设施失败。
