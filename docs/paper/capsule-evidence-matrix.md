# Capsule 论文证据矩阵

## 1. 来源注册

### 1.1 外部文献

| 编号 | 来源 | 用途 |
|---|---|---|
| S01 | *ASPIRE: Agentic Skills Discovery for Robotics*，2026，arXiv:2607.00272v1 | 细粒度执行反馈、失败修复、技能沉淀与迁移 |
| S02 | *Bridging Values and Behavior: A Hierarchical Framework for Proactive Embodied Agents*，2026，arXiv:2604.27699v1 | 主动性分类与多维评价的旁支背景 |
| S03 | *ENPIRE: Agentic Robot Policy Self-Improvement in the Real World*，2026，arXiv:2606.19980v1 | 真实机器人闭环、安全重置、资源与成本 |
| S04 | *GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness for Variational Automation Tasks*，2026，arXiv:2607.05369v1 | 结构化策略、技能契约、仿真排练与吞吐量 |
| S05 | *Playful Agentic Robot Learning*，2026，arXiv:2606.19419v1 | 代码技能、验证诊断循环、失败记忆和负迁移 |

上述元数据与页码位置沿用用户指定 skill 的 `references/evidence-map.md`，不扩展到其他文献。

### 1.2 仓库材料

| 编号 | 来源 | 性质 |
|---|---|---|
| R01 | `docs/plans/2026-07-03-runtime-control-capsules-design.md` | Capsule 总体设计与边界 |
| R02 | `docs/plans/2026-07-03-capsule-trace-feedback-design.md` | 追踪与源代码绑定反馈设计 |
| R03 | `docs/plans/2026-07-05-capsule-forward-recovery-design.md` | 禁止重放与追加恢复设计 |
| R04 | `docs/plans/2026-07-06-capsule-group-normalization-design.md` | 元数据式分组与源码保持不变量 |
| R05 | `docs/plans/2026-07-13-capsule-initial-syntax-recovery-design.md` | 初始语法错误和补丁验证 |
| R06 | `docs/plans/2026-07-13-capsule-recovery-observability-design.md` | 新观测契约与有界数值可观测性 |
| R07 | `docs/plans/2026-07-14-robosuite-nonprivileged-observation-design.md` | 非特权观测边界 |
| R08 | `docs/plans/2026-07-17-effect-bounded-forward-only-recovery-design.md` | 核心贡献命名、自动前向模式与报告协议 |
| I01 | `capx/runtime_control/segmenter.py:38`、`:89` | 原子区域分段与区域分析实现 |
| I02 | `capx/runtime_control/normalizer.py:32`、`:54` | 效应边界分组实现 |
| I03 | `capx/runtime_control/executor.py:13`、`:23` | 持久命名空间内的区域执行 |
| I04 | `capx/runtime_control/trace.py:11`、`:24` | 原语追踪与有界摘要 |
| I05 | `capx/runtime_control/feedback.py:15` | 运行时反馈构建 |
| I06 | `capx/envs/trial.py:726`--`:1323` | Capsule 自动前向试验循环 |
| I07 | `capx/envs/trial.py:1866`--`:1904` | `append_recovery` 语法与新观测验证 |
| I08 | `capx/envs/trial.py:1942`--`:2205` | 历史副作用单元禁止重放与账本更新 |
| I09 | `capx/integrations/base_api.py:106` | API 新观测能力声明 |
| I10 | `capx/utils/experiment_results.py:65` | 重试感知聚合与失败分类 |
| E01--E11 | `docs/paper/capsule-experiment-audit.md` | 经审计的实验档案 |

## 2. 论证链证据映射

| 结论编号 | 结论 | 支持来源与位置 | 类型 | 强度 | 边界与相反证据 | 推荐表述 |
|---|---|---|---|---|---|---|
| C01 | 多种智能体式机器人方法以执行—反馈—修正循环进行任务内自改进 | S01 第 2--7 页；S03 第 1--6 页；S04 第 5--6 页；S05 第 4--6 页 | 多文献综合判断 | 强 | 更新对象分别为程序、训练策略、计算图和技能记忆 | “这些方法虽更新对象不同，但均以可重复的执行—反馈—修正循环作为自改进接口。” |
| C02 | 步骤级轨迹和验证信号为局部故障归因提供依据 | S01 第 3--6 页；S05 第 2、5 页 | 多文献综合判断 | 强 | 无统一基准证明反馈粒度是唯一增益来源 | “在 ASPIRE 与 RATS 的设计语境下，步骤级证据支持更局部的诊断与修复。” |
| C03 | 真实机器人自主迭代依赖安全、验证与重置接口 | S03 第 3--6 页；S01 第 11--12 页 | 外部事实与综合判断 | 强 | 基础设施仍需人工初始化和维护 | “物理闭环首先需要可重复、安全且可验证的实验接口。” |
| C04 | 结构化 API 提供可控性，同时把能力上界绑定到已有原语 | S01 第 11--12 页；S02 第 9 页；S05 第 9 页 | 多文献综合判断 | 强 | GaP 展示可组合技能库，但仍受任务范围限制 | “结构化接口降低生成与执行不确定性，但限制可表达行为范围。” |
| C05 | 技能或记忆复用可能产生总体收益，也可能出现局部退化 | S01 第 12 页及附录第 21 页；S05 第 8--9 页 | 外部事实与综合判断 | 冲突 | RATS 在双臂交接上出现负迁移 | “复用收益具有任务依赖性，检索与再验证仍是关键问题。” |
| C06 | 机器人副作用单元执行后，源代码补丁不会恢复物理前置状态 | R03“Problem”；R08“Core Claim” | 用户设计事实 | 中 | 这是设计动机和系统约束，尚缺独立对照实验验证其性能影响 | “对已改变物理状态的动作，文本修补不能等价恢复其执行前状态。” |
| C07 | Capsule 不把机器人原语暴露为规划器工具，而是控制代码执行 | R01“Decision”“Data Flow”；R02“Goal”；I03 | 用户方法事实 | 强 | 运行时仍调用环境 API，只是不改变生成代码可见名称 | “Capsule 工具化的是执行控制，而非机器人动作原语。” |
| C08 | Capsule 使用 AST 原子区域和元数据式效应分组，且不重写可执行源码 | R04“Metadata-Only Invariants”；I01；I02 | 用户方法事实 | 强 | 分组是启发式效应边界，不代表完整任务语义 | “执行单元由源代码结构、依赖和 API 声明的副作用共同确定。” |
| C09 | Capsule 在持久命名空间中按源序执行，并记录区域局部原语证据 | R01“Architecture”；R02“Trace Integration”；I03；I04；I06 | 用户方法事实 | 强 | 自动前向仅支持效应边界单元模式 | “运行时保留变量状态，并将原语调用证据绑定到当前源码单元。” |
| C10 | 历史副作用单元不能重放或补丁后再执行 | R03“No-Rerun Guard”；R08“Core Claim”；I08 | 用户方法事实 | 强 | 无副作用计算单元仍可在逐步模式中检查或修补 | “副作用账本把已执行物理动作转为只读历史。” |
| C11 | 恢复代码必须调用 API 声明的新观测函数后从当前状态继续 | R03“Runtime Actions”；R06“Recovery observation contract”；I07；I09 | 用户方法事实 | 强 | 没有声明新观测能力的 API 不允许追加恢复 | “前向恢复先刷新当前状态，再产生新的机器人副作用。” |
| C12 | Capsule 能恢复初始语法错误，并拒绝破坏完整程序语法的补丁 | R05；I06；I07 | 用户方法事实 | 强 | 语法修复不代表物理执行失败已被恢复 | “候选补丁在发布前经过整程序语法验证。” |
| C13 | 当前实验显示 Capsule 的任务结果高度依赖任务和运行协议 | E01、E02、E07--E11 | 用户实验事实 | 强 | 不同任务、版本与重试规则不可合并 | “在所审计任务中，完成率从 0/20 到重试包容集合的 17/20 不等，数值只能在各自协议内解释。” |
| C14 | 现有证据不足以证明 Capsule 相对多轮重新生成提高成功率 | E01--E06 | 综合判断 | 强 | 缺少同提交、同调用模式和统一失败 schema 的配对实验 | “当前基线结果用于界定可行性，尚不能形成因果性性能结论。” |
| C15 | 重试和服务异常需要与算法结果分开报告 | R08“Experiment Reporting”；I10；E02、E08 | 用户方法与实验事实 | 强 | 历史档案并非全部包含新 schema | “报告首次尝试、重试预算和基础设施失败，避免选择性结果掩盖真实成本。” |

## 3. 方法组件—困难—验证对应关系

| 困难 | Capsule 组件 | 输入 | 输出 | 当前验证 | 证据边界 |
|---|---|---|---|---|---|
| 单条语句执行过碎 | 效应边界分组 | AST 区域、定义—使用关系、API 副作用声明 | 有序执行单元 | 单元测试与实验步日志 | 尚未证明 3--8 个单元是最优粒度 |
| 失败难以定位 | 原语追踪与源代码绑定反馈 | 活跃单元、调用事件、异常、前后状态 | 局部反馈与证据 | 追踪 JSON、反馈测试 | 奖励不作为可靠局部后置条件 |
| 物理动作不可回滚 | 副作用账本与禁止重放 | 已执行单元、实际调用轨迹 | 历史副作用集合、非法动作事件 | 运行时守卫测试 | 不提供 MuJoCo 或真实机器人任意回滚 |
| 当前状态与原程序假设不一致 | `append_recovery` | 失败证据、新观测函数集合、恢复源码 | 追加的新执行单元 | 恢复动作和轨迹档案 | 恢复有效率待统一复算 |
| 初始源码无法解析 | 整源码临时单元与补丁验证 | 语法错误源码 | 可修补临时组或非法补丁事件 | 语法恢复测试 | 仅处理可由模型修复的源码问题 |
| 模型服务和重试混淆算法结果 | 结构化试次结果与失败桶 | 每次试验、LLM 调用与进程状态 | 重试感知聚合 | 当前聚合器测试 | 历史档案需要迁移 |

## 4. 禁止或待核实的主张

- “Capsule 首次解决机器人代码恢复”：来源待核实，当前 5 篇文献不足以支持“首次”。
- “效应边界前向恢复显著提高任务成功率”：待统一配对实验和统计检验。
- “Capsule 可安全部署到真实机器人”：来源待核实；当前仓库实验主要是 Robosuite。
- “执行单元具有任务语义或可靠后置条件”：与 R08 的非目标冲突，不应声称。
- “17/20 是 Capsule 的首次尝试成功率”：与 E02 的选择规则冲突，禁止使用。
- “所有超时均属于算法失败”：与失败分类规则冲突，禁止使用。
- “跨任务平均成功率”：任务和协议不同，禁止合并。

## 5. 术语约定

| 英文 | 推荐中文 | 首次形式 | 后续形式 | 边界 |
|---|---|---|---|---|
| Code-as-Policy | 代码即策略 | 代码即策略（Code-as-Policy，CaP） | CaP / 代码策略 | 不等于把原语直接作为工具调用 |
| runtime-control Capsule | 运行时控制 Capsule | 运行时控制 Capsule（runtime-control Capsule） | Capsule | 系统名，不译为物理碰撞胶囊 |
| effect-bounded execution unit | 效应边界执行单元 | 效应边界执行单元（effect-bounded execution unit） | 执行单元 | 不是完整任务语义单元 |
| effect-bounded forward-only recovery | 效应边界前向恢复 | 效应边界前向恢复（effect-bounded forward-only recovery） | 前向恢复 | 强调禁止历史副作用重放 |
| side-effect ledger | 副作用账本 | 副作用账本（side-effect ledger） | 副作用账本 | 记录已执行机器人副作用单元 |
| source-bound feedback | 源代码绑定反馈 | 源代码绑定反馈（source-bound feedback） | 局部反馈 | 绑定区域/行号，不直接选择机器人动作 |
| append recovery | 追加恢复 | 追加恢复动作（`append_recovery`） | 追加恢复 | 必须调用新观测函数 |
| retry-aware reporting | 重试感知报告 | 重试感知报告（retry-aware reporting） | 重试感知报告 | 区分首次尝试与重试后结果 |
