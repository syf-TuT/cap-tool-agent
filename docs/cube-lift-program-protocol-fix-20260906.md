# Cube Lift 训练采样协议修复

实现提交：`76dc641`、`d1d18e6`。采样记录脚本修复：`6b926c3`。

Cube Lift 的 Program 采样恢复到已归档的特权基线协议：

| 项目 | 修复后的值 |
| --- | --- |
| system prompt | `You are a helpful assistant that generates Python code to directly solve the task.` |
| user prompt | 原始完整五个 API 文档，2417 个字符 |
| 完整 chat 输入 | Qwen tokenizer 编码后 590 tokens |
| temperature / top_p / top_k | 0.7 / 0.8 / 20 |
| repetition_penalty | 1.1 |
| max_tokens | 4096 |

用户提示词 SHA-256 为
`3c1b29055ddaf99a5729fc9757cfa28590d1f9a537b0c513af3d80c32ceaf460`。
system prompt SHA-256 为
`b7dadbc1bd88bf7b26a0b11ed1431aea6ada50e5b7f9dc2e663a813bb887e8bc`。

模板在 `program_service` 中声明提示词哈希和完整解码参数。数据准备和任务加载会校验
提示词，防止新运行配置继续使用旧的简化提示词数据集。Program 的生成和训练编码上限
统一为 4096，VeRL 模型和批次的 token 容量随之调整。

固定使用的 VeRL v0.6.1 的 `RolloutConfig` 不接受 `repetition_penalty` 字段。因此
最终配置只写入它支持的字段，完整 `SamplingParams` 在真实 vLLM worker 初始化后设置，
并由 worker 回读验证。初始化日志可能先打印 repetition_penalty=1.0；随后
`Capsule Program sampling parameters` 行及 `protocol.json.worker_sampling` 才是
协议应用后的有效参数。

任务成功仍以程序执行结束时的 clean replay 结果为准。抬起后再次松手导致方块落下仍为
失败；程序异常也不会计作正样本。此前的成对实跑已确认正常程序的基线与 clean replay
判定一致。

## 真实采样验证

远端根目录为 `/root/autodl-tmp/cap-x`。
运行包：`artifacts/cube_lift_program_protocol_base_s05_24_20260906_r01/`。
数据准备覆盖 seed 5–24；本轮在基础权重、optimizer step 0 下验证 seed 5–8，
每个 seed 8 次普通 Program 采样。

```bash
cd /root/autodl-tmp/cap-x
export MUJOCO_GL=egl JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
.venv/bin/python -m scripts.capsule_rl.probe_program_sampling \
  --config artifacts/cube_lift_program_protocol_base_s05_24_20260906_r01/runtime.yaml \
  --output-dir artifacts/cube_lift_program_protocol_base_s05_24_20260906_r01/probe_r03 \
  --seeds 5,6,7,8 --samples 8
```

执行前需要启动 `capx.serving.launch_pyroki_server`，并等待 `127.0.0.1:8116/docs`
可访问。对 Pyroki 同样设置上述 JAX CPU 环境变量，避免其预分配 GPU 显存。
重跑时应选择新的输出目录。

实跑正常退出（exit code 0），32 条中成功 **16 条，成功率 50%**。

| 环境 seed | 成功 / 采样数 | 程序错误 |
| --- | --- | --- |
| 5 | 4 / 8 | 0 |
| 6 | 3 / 8 | 0 |
| 7 | 5 / 8 | 0 |
| 8 | 4 / 8 | 0 |

4 个组均有正样本，优化器 step 在采样前后均为 0，worker 回读的五项采样参数与声明
完全一致。9 个修改文件的远端内容与本地提交逐文件 SHA-256 一致，证据在
`code_manifest.json`。

16 条失败均为任务失败。15 条源码在抓取后再次调用 `open_gripper()`；另外一条为
seed 8 的 sample 4，最终 `goto_pose` 的位置仍是原方块位置，只调整了 `z_approach`，
没有设置真正升高的最终目标位置。没有物理完成却被 clean replay 拒绝的样本。

此前同 seed 5 的基础模型 probe 为 2/21；本轮为 4/8，初始环境 SHA-256 完全相同。
本轮说明初始正样本供给已恢复到合理水平。32 条、4 个环境 seed 的采样不能替代历史
100 个 held-out seed 的评估，也不能据此宣称总体成功率已经精确恢复到历史的 59%。

结果已下载到本地
`remote_results/cube_lift_program_protocol_base_s05_24_20260906_r01/`，
主要证据为 `probe_r03/summary.json`、`probe_r03/protocol.json`、逐样本 JSON 和
`run_r03.log`。下载压缩包 SHA-256：
`cdebf364cc446c2519fa42c4aebd37f867b0f1f18d275c597fc64c19bc9794c3`。
本次启动的 Ray 和 Pyroki 服务已经退出。

后续训练配置为该运行包下的 `training_runtime.yaml`，其 `dataset_path` 指向已经真实
reset 解析的 `dataset.seed_resolved.jsonl`，包含 20 条任务。历史 runtime 文件不会
自动迁移，后续训练应切换到此配置或按更新后的仓库模板重新准备。

按用户要求，本轮以真实运行验证，没有新增测试用例。
