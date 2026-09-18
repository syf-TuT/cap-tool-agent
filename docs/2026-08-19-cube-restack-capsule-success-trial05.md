# Cube Restack Capsule Success Trace: Trial 05

日期：2026-08-19

本文整理一次通过 capsule 机制最终成功的 `cube_restack` 示例。成功 trial 是 seed/trial `05`。

## 数据来源和边界

远端实验输出目录：

```text
/root/autodl-tmp/cap-x/outputs/MiniMax-M2.5/franka_robosuite_cube_restack_capsule_llm_step_minimax_m25_20260819_0929_timeout295
```

关键原始文件：

```text
capsule_code_trial_05.py
capsule_step_metrics_trial_05.jsonl
capsule_prompts_trial_05.json
capsule_trace_trial_05.json
trial_5_result.json
trial_05_sandboxrc_0_reward_1.000_taskcompleted_1/video_1.000_capsule.mp4
```

写本文档时远端 SSH 端口返回 `Connection refused`，所以没有把完整
`capsule_prompts_trial_05.json` 和 `capsule_trace_trial_05.json` 原文复制进本地。
下面记录的是本次实验已抓取并核对过的字段、最终代码、step metrics 摘要和
agent 可见的 history feedback 摘要。

## 实验配置

```yaml
task: cube_restack
api: FrankaControlApi
privileged: false
agent_mode: capsule
capsule_control_mode: llm_step
capsule_prompt_state_level: proprioceptive
model: MiniMax-M2.5
base_url: https://api.minimaxi.com/v1
thinking: false
streaming: false
sam3: enabled, local cache
contact_graspnet: enabled
pyroki: enabled
vdm: false
visual_feedback: false
record_video: true
timeout_per_run_s: 300
max_capsule_steps: 20
seeds: 1..20
```

本次 20 个 seed 的汇总：

```text
finished: 7
trial_budget_exhausted: 13
task_completed: 1
reported success rate / average reward / completed: 0.143 / 0.152 / 1
average reward over all 20 seeds if timeout rewards count as 0: 0.05322775525897955
```

成功 trial 的结果文件：

```json
{
  "elapsed_seconds": 225.67224533110857,
  "failure_kind": null,
  "failure_message": null,
  "failure_stage": null,
  "finished_at": "2026-08-19T01:44:33.119989Z",
  "llm": {
    "attempt_count": 6,
    "call_count": 6,
    "elapsed_seconds": 210.9922065809369,
    "last_call_index": 6,
    "retry_count": 0,
    "token_count": 0
  },
  "reward": 1.0,
  "run_outcome": "finished",
  "sandbox_rc": 0,
  "schema_version": 1,
  "started_at": "2026-08-19T01:40:47.447756Z",
  "task_completed": true,
  "trial": 5
}
```

## Agent 是否在每个 capsule step 得到真实物体位姿

结论：没有直接在每个 step 的 agent 可见 feedback 中得到真实物体位姿。

本次解析 `capsule_prompts_trial_05.json` 时确认：

```json
{
  "prompt_file_size": 84174,
  "num_prompts": 5,
  "contains_object_poses": false,
  "contains_cubeA": false
}
```

也就是说，LLM 的 step-level action prompt 中没有出现 `object_poses` 或 `cubeA`
这类真实物体位姿字段。它看到的 history feedback 主要是
action/status/message/reward/task_completed/primitive_calls；如果 prompt 中包含
latest post-action observation，在 `proprioceptive` 级别也不是完整 object pose，
而是 reward/task/本体状态一类信息。

但这不等于代码不能访问物体位姿。生成的 Python 代码可以主动调用 API 原语：

```python
get_object_pose("red cube", return_bbox_extent=True)
get_object_pose("green cube", return_bbox_extent=True)
```

这些调用发生在 sandbox 执行过程中，返回值进入 Python 变量，用于后续控制逻辑。
区别是：

- `get_object_pose()` 是代码运行时通过公开 API 主动查询到的位姿。
- capsule step 的 agent-visible feedback 没有把真实位姿数字直接塞回 LLM prompt。
- `capsule_prompt_state_level: proprioceptive` 下，metrics snapshot 倾向记录 reward、
  task_completed、末端/关节等本体状态；`object_poses` 只在 full state snapshot 中进入日志。

## 初始生成代码

LLM 初始生成的是一次直接 pick-and-place：读取红块和绿块位姿，计算红块应该放在绿块顶部的位置，采样红块抓取姿态，抓取、移动、释放、回 home。

```python
import numpy as np

# Get pose and size of the red cube
red_pos, red_quat, red_ext = get_object_pose("red cube", return_bbox_extent=True)

# Get pose and size of the green cube
green_pos, green_quat, green_ext = get_object_pose("green cube", return_bbox_extent=True)

# Compute the position where the red cube should be placed on top of the green cube
dz = green_ext[2] / 2 + red_ext[2] / 2
place_pos = green_pos + np.array([0.0, 0.0, dz])

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red cube")

# Approach and grasp the red cube
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

# Move to the placement position with a downward orientation
down_quat = np.array([0.0, 0.0, 1.0, 0.0])  # gripper pointing down
goto_pose(place_pos, down_quat, z_approach=0.1)

# Release the red cube
open_gripper()

# Return to home pose
home_pose()
```

## Capsule step 总览

| step | action | target | code change | reward before | reward after | task completed after | feedback status | 说明 |
|---:|---|---|---|---:|---:|---|---|---|
| 1 | `run_group` | `group_1` | 无 | 0.003105861317909381 | 0.06216947461113606 | false | success | 执行取位姿、算目标、采样 grasp、接近并闭合夹爪 |
| 2 | `run_group` | `group_2` | 无 | 0.06216947461113606 | 0.0010210361287053826 | false | warning | 执行移动到放置点、开夹爪、回 home，但 reward 下降 |
| 3 | `inspect_variables` | - | 无 | 0.0010210361287053826 | 0.0010210361287053826 | false | success | 检查当前 Python 变量；没有产生机器人动作 |
| 4 | `append_recovery` | - | 追加 recovery 代码 | 0.0010210361287053826 | 0.0010210361287053826 | false | success | 在旧代码末尾追加基于 fresh observation 的恢复逻辑 |
| 5 | `run_group` | `group_3` | 无 | 0.0010210361287053826 | 1.0 | true | success | 执行 recovery group，重新抓取/放置，任务完成 |
| 6 | `finish` | - | 无 | 1.0 | 1.0 | true | success | LLM 看到成功后结束 |

## 每一步代码变化

### Step 0：初始 source revision

初始代码只有上面的直接 pick-and-place。它被 capsule normalizer 切成至少两个
effect-bounded group：

- `group_1`：source lines 1-20，含 `get_object_pose`、`sample_grasp_pose`、
  `goto_pose`、`close_gripper`。
- `group_2`：source lines 21-28，含 `goto_pose`、`open_gripper`、`home_pose`。

### Step 1：执行 `group_1`

动作：

```json
{"action": "run_group", "args": {"group_id": "group_1"}}
```

代码变化：无。

agent 在下一次 prompt history 中看到的 feedback 摘要：

```json
{
  "step_id": 1,
  "action": "run_group",
  "unit_id": "group_1",
  "status": "success",
  "event_status": "success",
  "feedback_status": "success",
  "message": "run_group for group_1 at source lines 1-20 completed with status success.",
  "reward_before": 0.003105861317909381,
  "reward_after": 0.06216947461113606,
  "task_completed_before": false,
  "task_completed_after": false,
  "primitive_calls": [
    "get_object_pose",
    "get_object_pose",
    "sample_grasp_pose",
    "goto_pose",
    "close_gripper"
  ]
}
```

metrics 中已确认的字段：

```json
{
  "step_id": 1,
  "action": "run_group",
  "region_id": "group_1",
  "event_action": "run_group",
  "event_status": "success",
  "trace_event_count": 5,
  "executed_regions_so_far": 8,
  "reward_before": 0.003105861317909381,
  "reward_after": 0.06216947461113606,
  "reward_delta": 0.05906361329322668,
  "best_reward_so_far": 0.06216947461113606,
  "task_completed_before": false,
  "task_completed_after": false,
  "append_recovery_source_appended": false,
  "recovery_execution_attempt": false,
  "recovery_execution_effective": false
}
```

### Step 2：执行 `group_2`

动作：

```json
{"action": "run_group", "args": {"group_id": "group_2"}}
```

代码变化：无。

agent 在下一次 prompt history 中看到的 feedback 摘要：

```json
{
  "step_id": 2,
  "action": "run_group",
  "unit_id": "group_2",
  "status": "warning",
  "event_status": "success",
  "feedback_status": "warning",
  "message": "group_2 executed at source lines 21-28, but no local task progress was observed.",
  "reward_before": 0.06216947461113606,
  "reward_after": 0.0010210361287053826,
  "task_completed_before": false,
  "task_completed_after": false,
  "primitive_calls": [
    "goto_pose",
    "open_gripper",
    "home_pose"
  ]
}
```

这一步是关键失败信号：底层动作执行成功，但任务没有完成，reward 还从
`0.06216947461113606` 降到 `0.0010210361287053826`。因此 capsule feedback
把它标为 `warning`，而不是 `success`。

metrics 中已确认的字段：

```json
{
  "step_id": 2,
  "action": "run_group",
  "region_id": "group_2",
  "event_action": "run_group",
  "event_status": "success",
  "trace_event_count": 3,
  "executed_regions_so_far": 12,
  "reward_before": 0.06216947461113606,
  "reward_after": 0.0010210361287053826,
  "reward_delta": -0.061148438482430674,
  "best_reward_so_far": 0.06216947461113606,
  "reward_drop_from_best": 0.061148438482430674,
  "task_completed_before": false,
  "task_completed_after": false,
  "append_recovery_source_appended": false,
  "recovery_execution_attempt": false,
  "recovery_execution_effective": false
}
```

### Step 3：`inspect_variables`

动作：

```json
{"action": "inspect_variables", "args": {"names": "<captured in raw trace>"}}
```

代码变化：无。

已抓取到的 prompt history feedback 摘要：

```json
{
  "step_id": 3,
  "action": "inspect_variables",
  "status": "success",
  "event_status": "success",
  "feedback_status": "success",
  "message": "inspect_variables completed with status success.",
  "reward_before": 0.0010210361287053826,
  "reward_after": 0.0010210361287053826,
  "task_completed_before": false,
  "task_completed_after": false
}
```

metrics 中已确认的字段：

```json
{
  "step_id": 3,
  "action": "inspect_variables",
  "event_action": "inspect_variables",
  "event_status": "success",
  "trace_event_count": 0,
  "executed_regions_so_far": 12,
  "reward_before": 0.0010210361287053826,
  "reward_after": 0.0010210361287053826,
  "reward_delta": 0.0,
  "best_reward_so_far": 0.06216947461113606,
  "reward_drop_from_best": 0.061148438482430674,
  "task_completed_before": false,
  "task_completed_after": false,
  "append_recovery_source_appended": false,
  "recovery_execution_attempt": false,
  "recovery_execution_effective": false
}
```

### Step 4：追加 recovery 代码

动作：

```json
{
  "action": "append_recovery",
  "args": {
    "source": "<the recovery source shown below>"
  }
}
```

代码变化：在初始代码末尾追加以下恢复逻辑。

```python
state = get_observation()
# Check current state of objects
red_pos, red_quat, red_ext = get_object_pose("red cube", return_bbox_extent=True)
green_pos, green_quat, green_ext = get_object_pose("green cube", return_bbox_extent=True)

# Calculate where the red cube should be on top of green cube
dz = green_ext[2] / 2 + red_ext[2] / 2
target_pos = green_pos + np.array([0.0, 0.0, dz])

# Check if red cube is already properly placed
distance = np.linalg.norm(red_pos - target_pos)
print(f"Red cube position: {red_pos}")
print(f"Target position: {target_pos}")
print(f"Distance to target: {distance}")

if distance > 0.05:
    # Need to re-grasp and place the red cube
    grasp_pos, grasp_quat = sample_grasp_pose("red cube")
    goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
    close_gripper()

    # Move to placement position with down orientation
    down_quat = np.array([0.0, 0.0, 1.0, 0.0])
    goto_pose(target_pos, down_quat, z_approach=0.1)

    # Open gripper to release
    open_gripper()

# Return to home pose
home_pose()
```

已抓取到的 prompt history feedback 摘要：

```json
{
  "step_id": 4,
  "action": "append_recovery",
  "status": "success",
  "event_status": "success",
  "feedback_status": "success",
  "message": "append_recovery completed with status success.",
  "reward_before": 0.0010210361287053826,
  "reward_after": 0.0010210361287053826,
  "task_completed_before": false,
  "task_completed_after": false
}
```

metrics 中已确认的字段：

```json
{
  "step_id": 4,
  "action": "append_recovery",
  "event_action": "append_recovery",
  "event_status": "success",
  "trace_event_count": 0,
  "executed_regions_so_far": 12,
  "reward_before": 0.0010210361287053826,
  "reward_after": 0.0010210361287053826,
  "reward_delta": 0.0,
  "best_reward_so_far": 0.06216947461113606,
  "reward_drop_from_best": 0.061148438482430674,
  "task_completed_before": false,
  "task_completed_after": false,
  "append_recovery_source_appended": true,
  "recovery_execution_attempt": false,
  "recovery_execution_effective": false
}
```

### Step 5：执行 recovery `group_3`

Step 5 不是一次新的 LLM prompt 决策；`append_recovery` 后系统产生待执行的
recovery group，因此 metrics 里 `action_prompt_chars` 为 `0`，`action_prompt_char_budget`
为 `null`。

动作：

```json
{"action": "run_group", "args": {"group_id": "group_3"}}
```

代码变化：无，执行的是 Step 4 追加的 recovery block。

这一步重新读取当前状态，重新获取红块/绿块位姿，重新计算目标位置。如果红块距离目标
超过 `0.05`，就重新 sample grasp、抓取红块、移动到目标位置、释放并回 home。

执行后 reward 从 `0.0010210361287053826` 直接变为 `1.0`，`task_completed_after`
变为 `true`。

metrics 中已确认的字段：

```json
{
  "step_id": 5,
  "action": "run_group",
  "region_id": "group_3",
  "event_action": "run_group",
  "event_status": "success",
  "trace_event_count": 9,
  "executed_regions_so_far": 23,
  "reward_before": 0.0010210361287053826,
  "reward_after": 1.0,
  "reward_delta": 0.9989789638712946,
  "best_reward_so_far": 1.0,
  "reward_drop_from_best": 0.0,
  "task_completed_before": false,
  "task_completed_after": true,
  "append_recovery_source_appended": false,
  "recovery_execution_attempt": true,
  "recovery_execution_reward_improved": true,
  "recovery_execution_trace_improved": true,
  "recovery_execution_improved": true,
  "recovery_execution_effective": true
}
```

根据 recovery 代码和 `trace_event_count: 9`，这一步对应的 primitive call 序列为：

```text
get_observation
get_object_pose
get_object_pose
sample_grasp_pose
goto_pose
close_gripper
goto_pose
open_gripper
home_pose
```

### Step 6：结束

动作：

```json
{"action": "finish", "args": {}}
```

代码变化：无。

Step 6 的 prompt 是新的 LLM 决策，metrics 中 `action_prompt_chars` 为 `22719`。
LLM 看到 `reward_after: 1.0` 和 `task_completed_after: true` 后选择 `finish`。

metrics 中已确认的字段：

```json
{
  "step_id": 6,
  "action": "finish",
  "event_action": "finish",
  "event_status": "success",
  "reward_before": 1.0,
  "reward_after": 1.0,
  "reward_delta": 0.0,
  "best_reward_so_far": 1.0,
  "reward_drop_from_best": 0.0,
  "task_completed_before": true,
  "task_completed_after": true,
  "append_recovery_source_appended": false,
  "recovery_execution_attempt": false,
  "recovery_execution_effective": false
}
```

## 最终代码

`capsule_code_trial_05.py` 的最终内容如下。它等于“初始直接执行代码”加上
Step 4 的 recovery 追加代码。

```python
import numpy as np

# Get pose and size of the red cube
red_pos, red_quat, red_ext = get_object_pose("red cube", return_bbox_extent=True)

# Get pose and size of the green cube
green_pos, green_quat, green_ext = get_object_pose("green cube", return_bbox_extent=True)

# Compute the position where the red cube should be placed on top of the green cube
dz = green_ext[2] / 2 + red_ext[2] / 2
place_pos = green_pos + np.array([0.0, 0.0, dz])

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red cube")

# Approach and grasp the red cube
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

# Move to the placement position with a downward orientation
down_quat = np.array([0.0, 0.0, 1.0, 0.0])  # gripper pointing down
goto_pose(place_pos, down_quat, z_approach=0.1)

# Release the red cube
open_gripper()

# Return to home pose
home_pose()

state = get_observation()
# Check current state of objects
red_pos, red_quat, red_ext = get_object_pose("red cube", return_bbox_extent=True)
green_pos, green_quat, green_ext = get_object_pose("green cube", return_bbox_extent=True)

# Calculate where the red cube should be on top of green cube
dz = green_ext[2] / 2 + red_ext[2] / 2
target_pos = green_pos + np.array([0.0, 0.0, dz])

# Check if red cube is already properly placed
distance = np.linalg.norm(red_pos - target_pos)
print(f"Red cube position: {red_pos}")
print(f"Target position: {target_pos}")
print(f"Distance to target: {distance}")

if distance > 0.05:
    # Need to re-grasp and place the red cube
    grasp_pos, grasp_quat = sample_grasp_pose("red cube")
    goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
    close_gripper()

    # Move to placement position with down orientation
    down_quat = np.array([0.0, 0.0, 1.0, 0.0])
    goto_pose(target_pos, down_quat, z_approach=0.1)

    # Open gripper to release
    open_gripper()

# Return to home pose
home_pose()
```

## 为什么这次 capsule 成功

初始代码不是一次就成功。它在 Step 2 执行完放置和释放后没有完成任务，reward 反而明显下降。
普通单次代码生成到这里就失败了。

capsule 的关键作用是把执行过程拆成可观察的 step：

- Step 1 后 reward 上升，说明抓取/接近阶段有局部进展。
- Step 2 后 reward 下降，feedback 标成 `warning`，提示“动作执行了，但没有本地任务进展”。
- Step 3 允许 agent 检查 Python 变量，但没有改变环境。
- Step 4 使用 `append_recovery`，没有试图回滚已经执行过的 side-effect group，而是在当前场景状态后追加新代码。
- Step 5 执行 recovery 代码，重新读取当前状态并再次抓取/放置，使 reward 达到 `1.0`。
- Step 6 看到任务已完成后 `finish`。

这个例子的核心不是视觉反馈，也不是 privileged object-state feedback；它成功的原因是：

1. 代码能通过公开 `FrankaControlApi` 原语主动查询对象位姿和采样 grasp。
2. capsule runtime 能在 side-effect 执行后把 reward/task_completed 反馈给 LLM。
3. 当已执行动作不可回滚时，agent 用 `append_recovery` 追加基于 fresh state 的恢复逻辑。
4. recovery group 被单独执行并验证，最终完成任务。
