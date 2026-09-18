# Cube Stack 最终 LoRA 与提示词排查，2026-09-08

对已完成的 r03 训练和非特权高层评估进行核验，未发现错误 adapter、权重漏载、
LoRA 未启用、基础模型不匹配或 chat template 不一致。训练与评估的 user prompt
有 API 文档差异；现有证据不足以将 8/80 到 7/80 的变化归因于该差异。

## 直接核验结果

| 检查 | 结果 |
| --- | --- |
| 评估实际绑定的训练结果 | r03，15 次 optimizer update |
| Actor 基础模型 | Qwen2.5-Coder-7B-Instruct；Controller qwen3.7-plus 不参与评估 |
| Adapter 文件与原评估 identity 中的 SHA-256 | 一致 |
| 导出 adapter 与完整 FSDP checkpoint | 392/392 个张量完全一致 |
| 完整 checkpoint 的基础模型权重与原模型 | 全部一致，未发现基础权重被额外修改 |
| 实际加载到 PEFT 的 adapter | 392/392 个张量匹配，共 40,370,176 个参数 |
| Adapter 状态 | active=`default`，禁用层数 0，rank=16，alpha=32，缩放 2 |
| 非有限 adapter 张量、全零 B 张量 | 均为 0 |
| 推理状态 | `model.training=False`，实际 `inference_mode=True` |
| 关闭 LoRA 后与原模型的首 token logits 差异 | 两套提示词下均为 0 |
| 开启 LoRA 后与关闭时的首 token logits 最大绝对差 | 训练提示词 0.125；评估提示词 0.1875 |
| 基础模型与 checkpoint tokenizer 的提示词 token IDs / chat template | 一致 |
| 原评估生成重现 | ordinal 0、30、62，base 和 trained 各 3 条，6/6 逐字一致 |

Adapter SHA-256：
`edb83aec3fdbb7b38e8ef2c10fd064c0c3c6a15fbb796e20a6f8da1df5d5eab7`。

保存的 `adapter_config.json` 包含 `inference_mode=false`；评估代码通过
`PeftModel.from_pretrained(..., is_trainable=False)` 加载，实测推理状态正确。
PEFT 关于忽略 `runtime_config` 的 warning 未导致 adapter 权重被忽略。

实际训练 worker 与评估使用 temperature=0.7、top_p=0.8、top_k=20、
repetition_penalty=1.1、max_new_tokens=4096。训练初始化日志曾先打印 repetition
penalty=1.0，随后 worker 明确回读了安装后的 1.1；不能只依据初始化行判断。
训练使用 VeRL/vLLM，评估使用 Transformers/PEFT；数值参数一致不代表跨引擎逐 token
数值一致。上述 6 条复现使用原评估引擎。

## 提示词差异与更新效果

system prompt、任务目标、六条动作规则完全相同。包含 chat template 的输入长度为
训练 759 tokens、评估 796 tokens。差异只有：

- 评估增加 `return_bbox_extent` 参数说明。
- 物体尺寸返回值说明从世界坐标轴尺寸改为 OBB 尺寸的描述。
- 评估多出 `get_observation()` 文档。原 160 条评估程序没有调用它。

没有发现错误任务提示词、遗漏动作规则、提示词截断或缺少 assistant generation prefix。
评估两种 policy 使用同一份冻结提示词；修复历史未混入最终评估。

进一步对全部 128 条已记录训练程序进行 teacher forcing，比较开启/关闭 adapter 时
原始模型分布的平均 token log probability（含 EOS，不应用采样过滤）：

| 样本集合 | 训练提示词下均值变化 / 上升条数 | 评估提示词下均值变化 / 上升条数 |
| --- | --- | --- |
| 成功 28 条 | +0.001221 / 20 条 | +0.000975 / 22 条 |
| 失败 100 条 | -0.001213 / 25 条 | -0.001106 / 32 条 |
| 其中修复成功 3 条 | +0.000498 / 1 条 | +0.000404 / 2 条 |

这些是已见程序的条件概率诊断，不能当作新增的环境成功率。它们说明更新整体略微偏向
成功程序，换成评估提示词后平均方向仍保留；并非所有成功程序的概率都提高。
尤其三条修复成功程序在原训练提示词下只有一条概率上升，成功样本数量不能直接代表
修复行为已经稳定学会。该现象本身不证明 loss 符号错误或过拟合。

## 已确认的环境与 API 差异

`control_privileged.py:get_object_pose` 总是返回固定尺寸 `[0.05, 0.05, 0.05]`，
即使没有指定 `return_bbox_extent=True`。非特权 `control.py:get_object_pose`
在默认参数下返回 `None`，显式请求时返回感知点云 OBB 的尺寸。
同时，特权抓取使用真实中心和固定姿态，非特权抓取使用 ContactGraspNet 候选姿态。
这些是接口行为和感知条件差异，修改几行提示词不能消除。

现有评估也未固定感知服务的每次请求随机状态。此前记录中，13 对调用抓取的完全相同
程序仍拿到不同抓取姿态；这些相同程序对的成功/失败结果一致，不能据此声称感知随机性
造成了净少一次成功，但它确实限制了对微小差异的归因。

优先级建议：统一共同 primitive 的 API 约定及对应文档，再固定感知请求随机状态，
用同场景的 base/LoRA 比较检查特权与非特权两种条件。提示词差异的任务成功率效应
还需要独立对照；本轮排查没有重跑环境成功率或改写原有评估结果。

## 复现与证据

在 SeeTaCloud `/root/autodl-tmp/cap-x` 执行：

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m scripts.capsule_rl.audit_cube_stack_lora_evaluation \
  --training-root artifacts/cube_stack_capsule_rl_s05_20_20260908_r03 \
  --evaluation-root artifacts/cube_stack_nonprivileged_base_vs_trained_s25_44_20260908_r01 \
  --output artifacts/cube_stack_lora_prompt_audit_<new-run-id>
```

该脚本需要已有模型、checkpoint 和 GPU，仅运行权重、生成重现及已记录程序的概率核验。
本次结果在服务器 `artifacts/cube_stack_lora_prompt_audit_20260908_r02/`，其 `audit.json`
SHA-256 为 `d00aa631236c35c42f2e1c687ecdd6abe3639c1b1ca55c6c5e24509f18560a16`。
本地记录在 `remote_results/cube_stack/20260908T_lora_prompt_audit/`。

初次诊断脚本对模块的 `scaling` 属性类型判断不充分，元数据检查后退出；修正为只读取
LoRA 的字典属性后，r02 全部完成。该错误发生在诊断脚本，不在原训练或原评估中。
