# Capsule

面向机器人代码策略的效应边界前向恢复运行时，以及在该运行时上训练程序策略的 Capsule Critique RL。

语言模型仍生成调用原有机器人函数的 Python 程序。Capsule 加在程序和环境之间：按效应边界把程序分成执行单元，在同一次试次的持久命名空间里逐单元执行，并把失败定位到具体源码。已经产生机器人副作用的单元写入账本，不能再次运行，也不能改完后当作历史片段重放。恢复代码只能追加到程序末尾，并且必须先调用环境声明的新观测函数，再从当前物理状态继续。

默认决策循环向模型暴露这些运行时动作：`run_group`、`patch_group`、`inspect_variables`、`append_recovery`、`finish`。这些动作本身不直接移动机器人。主机每轮至多执行一个动作，保存有界反馈，再交给下一轮决策。

仓库分成两部分：

| 部分 | 做什么 | 代码 |
| --- | --- | --- |
| Capsule 运行时 | 执行、定位失败、禁止重放、追加恢复 | `capx/runtime_control/`，入口 `capx/envs/launch.py` |
| Capsule Critique RL | 用环境奖励和修复轨迹训练生成程序的模型 | `capx/rl/capsule/`，脚本 `scripts/capsule_rl/` |

Python 包的 import 路径是 `capx`。

## 安装

需要 Python 3.10–3.12、一块支持 CUDA 的 GPU，以及 [uv](https://docs.astral.sh/uv/)。

```bash
git clone --recurse-submodules <repo-url> && cd <repo>
git submodule update --init --recursive

curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10 && uv venv -p 3.10
uv sync
```

Robosuite 和 LIBERO 依赖的 Robosuite 版本不同，各自使用一个虚拟环境。

### Robosuite

```bash
uv sync --extra robosuite
```

### LIBERO

```bash
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --active --extra libero --extra contactgraspnet
```

任务选择、Molmo 外置服务和非特权程序约束见 [docs/libero-tasks.md](docs/libero-tasks.md)。

可选组件：

```bash
uv sync --extra verl              # Capsule Critique RL，依赖 VeRL
uv sync --extra contactgraspnet   # Contact-GraspNet 抓取规划
uv sync --extra curobo            # cuRobo 运动规划，需要本机 CUDA
```

## Capsule 运行时

实现在 `capx/runtime_control/`：分段、效应边界、原语追踪、副作用账本和 `append_recovery`。`capx/envs/launch.py` 读取 `env_configs/` 里 `agent_mode: capsule` 的 YAML，在试次循环中调用这套运行时。

感知服务由任务 YAML 在评测开始时拉起。模型请求走本机的 OpenAI 兼容接口，默认地址是 `http://127.0.0.1:8110/chat/completions`。把密钥放在被 git 忽略的 `.openrouterkey` 里：

```bash
echo "sk-or-v1-your-key-here" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

提供商、CLI 和 YAML 字段见 [docs/configuration.md](docs/configuration.md)。

Robosuite Cube Stack，完整动作集（修补和追加恢复都打开）：

```bash
export MUJOCO_GL=egl
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_capsule_llm_step_a4_full.yaml \
    --total-trials 1 \
    --num-workers 1
```

LIBERO Object 任务 0：

```bash
source .venv-libero/bin/activate
export MUJOCO_GL=egl
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml \
    --total-trials 1 \
    --num-workers 1
```

同一任务族的消融配置：

| 配置 | 行为 |
| --- | --- |
| `a1` | 一次生成 |
| `a2` | Capsule，关闭修补 |
| `a3` | Capsule，关闭追加恢复 |
| `a4` | Capsule，修补和追加恢复都打开 |

## Capsule Critique RL

在 Capsule 运行时收集的程序和修复轨迹上训练。默认目标是 Capsule-Critique-GRPO（命令行里的 `cc-grpo`）。

代码位置：

- `capx/rl/capsule/`：训练器、策略损失、VeRL 配置。静态检查入口是 `python -m capx.rl.capsule.main_ppo`。
- `scripts/capsule_rl/`：准备数据、训练和评测。Cube Stack 用 `train_cube_stack.py`，训练后评测用 `evaluate_cube_stack.py`。
- `env_configs/cube_stack/capsule_rl/`：Cube Stack 的训练 YAML。

先做不启动仿真器、Ray 或优化器的静态检查。模板里的 `/path/to` 在正式训练前要换成本机路径；`--validate-only` 只检查配置是否能解析。

```bash
uv run --extra verl --no-sync python -m capx.rl.capsule.main_ppo \
    --config env_configs/cube_stack/capsule_rl/franka_robosuite_cube_stack_capsule_critique_grpo.yaml \
    --validate-only
```

`prepare` 需要程序模型路径，以及一份单独检出的 VeRL。`train` 只读取 `prepare` 写出的 `--root`。

```bash
uv run --extra verl --no-sync python scripts/capsule_rl/train_cube_stack.py prepare \
    --root outputs/cube_stack_capsule_rl \
    --model /path/to/program-model \
    --verl /path/to/verl

uv run --extra verl --no-sync python scripts/capsule_rl/train_cube_stack.py train \
    --root outputs/cube_stack_capsule_rl
```

评测脚本是 `scripts/capsule_rl/evaluate_cube_stack.py`，阶段为 `prepare`、`generate`、`evaluate`、`summarize`。`prepare` 需要 `--training-root` 指向一次训练目录。

## 许可证

本仓库以 [MIT License](LICENSE) 发布。
