# Cube Lift 训练采样协议排查（2026-09-05）

**结论：已确认正式训练没有复现历史特权基线的完整提示词和解码参数。**
普通评测和 clean replay 对正常程序采用相同的最终物理状态判定；本次未发现训练
通过更严格的判分规则大量扣掉已经完成任务的普通采样。六个案例的配对回放支持这一结论。
未重新进行大批量模型生成，因此不能量化提示词、top-p、top-k 等因素各自造成多少百分点下降。

本次只增加诊断脚本与报告。工作区原有的八个文件修改早于本次排查。

审计对象均位于远端 `/root/autodl-tmp/cap-x`：

- 历史特权基线：`remote_results/cube_lift_base_vs_lora100_heldout_t07_s25_124_20260828_gate4fix_r02`。
  100 个种子中 59 次成功、3 次 sandbox 错误。
- 正式训练配置：`artifacts/cube_lift_capsule_rl_train16_direct_nogates_seeds5_20_20260904_r01/direct_runtime_r02.yaml`。
  对应 r02 输出目前保存 8 组、64 次普通采样：3 成功、34 程序错误、27 任务失败；5 组全零。
  这些普通采样包含训练期间数据，不能全部当作 step 0 基础模型结果。
- 无更新初始 probe：`artifacts/cube_lift_capsule_rl_prompt_homepose_t07_base_group_probe_seed5_s21_20260904_r01/base_sanity.json`。
  固定 seed 5，共 21 次：2 成功、3 程序错误、16 任务失败。
  先前 8 次 sanity 与此 probe 的前 8 个源码哈希完全相同，不能合并成 29 个独立样本。

**提示词差异及其来源**

| 内容 | 历史特权基线 | 正式训练 r02 | 无更新 probe |
| --- | --- | --- | --- |
| user prompt 字符数 | 2417 | 743 | 2534 |
| 完整 chat template token 数 | 590 | 236 | 662 |
| API 说明 | 完整函数签名、返回值形状和文档 | 五函数简写 | 完整说明并新增 home_pose |
| system prompt | 简短通用 Python 任务提示 | 新的严格机器人程序提示 | 与训练相同 |

基线 system prompt 原文：
`You are a helpful assistant that generates Python code to directly solve the task.`

正式训练省略了“API 已导入”说明、明确的 tuple 返回结构、数组形状及抓取四元数说明。
实际错误包括 `too many values to unpack`、把 tuple 切片当坐标、导入不存在的机器人模块。
这些错误与接口信息减少相符，但这种对应关系本身不是提示词因果效应的独立估计。

训练的 `load_task_instances()` 从 `runtime.dataset_path` 指向的 JSONL 加载 prompt；
生成器直接使用 `task.prompt`，不会在采样时重新调用环境的 `_get_complete_prompt()`。
正式 r02 仍指向旧的 `dataset.seed_resolved.jsonl`。当前源文件已经补全，并不改变旧 JSONL。
`prompt_homepose` probe 的 user prompt 相对历史基线只新增了 `home_pose()` 文档，
但是 system prompt 仍不同；它不是严格的完整提示词对齐实验。

相关实现：`capx/rl/capsule/server_factory.py:258`、`:386`、`:491`，
`capx/envs/tasks/base.py:116`、`:142`。

**解码参数差异及其来源**

| 参数 | 历史特权基线 | 正式训练实际值 | 温度 0.7 的 probe |
| --- | --- | --- | --- |
| temperature | 0.7 | 1.0 | 0.7 |
| top_p | 0.8 | 1.0 | 1.0 |
| top_k | 20 | -1（不截断） | -1 |
| repetition_penalty | 1.1 | 1.0 | 1.0（VeRL 默认） |
| 最大输出 token | 4096 | 2048 | 2048 |

普通推理继承模型 `generation_config.json` 的采样默认值；历史 sampling 审计也记录了
top-p、top-k 和 repetition penalty。VeRL 的 `vllm_rollout_spmd.py:255-270` 从训练
rollout 配置重新构造 `SamplingParams`。正式训练日志明确记录了 1.0/1.0/-1/1.0。

正式配置的 `runtime.verl_resolved_config_path` 仍指向：
`artifacts/capsule_single_a800/fsdp_base_bf16_vllm_util_045-dd461dc500b2-controller-seed-1/resolved/verl.yaml`。
当前仓库配置改成 temperature=0.7，并不更新这个已生成快照。run ID 中的 `t07` 也不会设置参数。
`controller_service.temperature` 控制修复 Controller，不控制普通 Program actor 采样。

`_load_resolved_verl_config()` 读取上述快照，并把输出长度设为
`capsule.revision_response_max_tokens`。本次实际调用配置合成函数确认：formal=2048、
len2048 probe=2048、len4096 配置=4096；没有发现 len4096 被覆盖成 2048 的问题。
只是 len4096 目录没有对应的已完成 `base_sanity.json` 可供比较。

历史 100 个基础模型回答最长 336 token，初始 probe 最长 344 token，均远低于 2048。
所以长度上限有差异，但不能解释这批已保存的短程序为何失败。

相关实现：`capx/rl/capsule/server_factory.py:1289`，
远端 `.codex-downloads/verl-v0.6.1/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:255`。

**任务完成语义与配对回放**

两套环境 YAML 的 `env` 配置相同，均为 `FrankaControlPrivilegedApi`。
普通评测与 clean replay 都使用同一源码规范化函数，执行同一 `CodeExecutionEnvBase.step()`。
程序完整执行后才查询最终 reward 和 task_completed，未采用“中间曾成功就永久记成功”的规则。
底层 Lift 成功条件是方块中心高度高于桌面 0.04 m，并不单独检测是否仍夹持。

clean replay 另外要求没有程序异常、没有步数截断且 raw reward >= 1。
这是与普通评测直接报告物理 task_completed 的边界差异。

在当前远端运行了六例 × 两条执行路径，全部具有相同初始状态哈希、物理完成状态和 reward：

| 案例 | 普通评测物理完成 | clean replay | 观察 |
| --- | --- | --- | --- |
| 历史 baseline seed 25 成功程序 | true | success | 复现历史结果 |
| 历史 baseline seed 102 失败程序 | false | task_failure | 举起后松手，两边都失败 |
| 初始 probe 的成功程序 | true | success | 围栏规范化正常 |
| 初始 probe 的举起后松手程序 | false | task_failure | 举起时 reward=1，松手后约 0.523 |
| 上一程序仅省略最后一次 open_gripper | true | success | 其余动作相同，最终 reward=1 |
| 成功程序后人为追加 ValueError | true | program_error | 显式展示异常优先规则 |

21 次 probe 中的 16 个任务失败源码均在 close_gripper 后再次 open_gripper。
本次对其中一例做了单因素干预，确认松手直接造成该例失败；没有逐个干预全部 16 例。
历史基线也有相同松手失败，说明最终状态语义没有在训练中突然改变。

审计 64 次训练普通采样和 21 次 probe，`observed_task_completed=true` 却被扣为零的数量均为 0。
历史 59 个正样本也都 sandbox_rc=0、reward=1。历史简要结果未保存 truncated，
因此不能只用历史 JSON 完整重算每一例的 clean 分类。

另有 4 次修复流程的 PT 回放成功，但后续 actor revision 因 Markdown 围栏被拒绝，未执行回放。
它们不能直接视为 4 个可用 revision 正样本；这会阻断补救流程，但不是初始成功率下降的原因。
相关边界：`capx/rl/capsule/revision.py:168`。

**后续应修正的顺序**

1. 从历史基线恢复完整 system + user messages，并生成新的 seed-resolved dataset；
   核对采样实际 prompt token IDs 和哈希，避免只改源模板。
2. 在实际引用的 resolved rollout 配置显式设置全部解码参数，并核对 vLLM 生效日志。
   复现阶段保持基线提示词不变；之后再将“举起后保持、不要松手”作为单独变量验证。
3. 保留最终状态语义；它没有解释本批普通采样的正样本损失。
   修复 revision 围栏处理时，应同时保留原始 token/源码身份和规范化执行来源，不能简单替换
   训练序列的原始内容。
4. 冻结模型后在多种子上做 prompt × decoding 的对照，再决定各因素的效应。
   本次六例回放验证的是执行/判分一致性，不是成功率恢复到 59% 的实验。

**验证与复现**

诊断脚本：`scripts/capsule_rl/audit_cube_lift_sampling_protocol.py`。
在远端项目根目录，使用已有依赖并预启动本地 Pyroki 8116；脚本不会加载 LLM 或训练。

```bash
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -m scripts.capsule_rl.audit_cube_lift_sampling_protocol \
  --output artifacts/cube_lift_sampling_protocol_audit_s05_s25_20260905_r02 \
  --tokenizer-audit --replay
```

输出目录必须使用新的名字。配对种子实际为 5、25、102。

本次证据在 `artifacts/cube_lift_sampling_protocol_audit_s05_s25_20260905_r01/`，
其中 `archive_audit.json`、`config_probe.json`、`paired_replay.json` 已复制到 Windows 同名目录。
脚本重新生成前两份 JSON，与首次结果逐项相同；六例配对检查全部通过。

运行 `.venv/bin/python -m pytest tests/test_program_source.py tests/test_capsule_evaluator.py -q`：
**43 passed**。诊断脚本通过远端 `py_compile`。本次启动的 Pyroki 服务已按记录的 PID 停止。
