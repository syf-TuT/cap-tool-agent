# CaP-X 用户运行手册

本文面向第一次运行本项目的用户，说明如何安装依赖、启动模型服务、运行评测任务、打开 Web UI，以及如何做基本测试和排障。

## 1. 项目用途与运行入口

CaP-X 是一个用于评测和改进机器人操作类 Code-as-Policy 智能体的框架。常用运行入口如下：

| 入口                                | 用途                                  |
| ----------------------------------- | ------------------------------------- |
| `capx/envs/launch.py`               | 运行机器人任务评测或启动交互式 Web UI |
| `capx/serving/openrouter_server.py` | 启动 OpenAI 兼容的本地 LLM 代理       |
| `capx/serving/launch_servers.py`    | 预启动感知和控制 API 服务             |
| `scripts/regression_test.sh`        | 运行回归和烟雾测试                    |
| `web-ui/`                           | React/Vite 前端源码                   |

主要任务配置放在 `env_configs/`，例如：

- `env_configs/cube_stack/franka_robosuite_cube_stack.yaml`
- `env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml`
- `env_configs/libero/franka_libero_spatial_0.yaml`
- `env_configs/r1pro/r1pro_pick_up_radio.yaml`

## 2. 运行环境要求

推荐使用 Linux 或 WSL2。项目中的 `uv` 环境配置面向 `linux x86_64`，并且 Robosuite、LIBERO、BEHAVIOR/Isaac Sim 等机器人仿真依赖在 Linux 环境下最稳定。原生 Windows 可能需要额外处理 MuJoCo、CUDA、图形驱动和脚本兼容问题。

基础要求：

- Python 3.10 到 3.12，Robosuite 和 BEHAVIOR 推荐 Python 3.10。
- CUDA 可用的 NVIDIA GPU。运行 SAM3、ContactGraspNet、Isaac Sim 或大规模并行评测时基本需要 GPU。
- `git`、`uv`、可用的 C/C++ 编译工具链。
- 如果运行 Web UI，需要 Node.js/npm；项目也会在启动 Web UI 时尝试自动准备 Node 环境。

## 3. 获取代码和子模块

首次克隆推荐直接拉取子模块：

```bash
git clone --recurse-submodules https://github.com/capgym/cap-x
cd cap-x
```

如果已经克隆但没有拉取子模块，在项目根目录执行：

```bash
git submodule update --init --recursive
```

子模块很重要，因为 `pyproject.toml` 中有多个本地依赖来自 `capx/third_party/`，例如 Robosuite、LIBERO-PRO、SAM3、cuRobo 和 ContactGraspNet。

## 4. 安装基础 Python 环境

安装 `uv` 后，在项目根目录创建并同步环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv venv -p 3.10
source .venv/bin/activate
uv sync
```

如果只做代码阅读、单元测试或轻量开发，可以先安装基础环境。真正运行仿真任务时，还需要根据模拟器选择额外依赖。

## 5. 选择并安装模拟器依赖

### 5.1 Robosuite 任务

Robosuite 是最适合先跑通的任务族。安装命令：

```bash
uv sync --extra robosuite
```

常用配置示例：

```bash
env_configs/cube_stack/franka_robosuite_cube_stack.yaml
env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml
env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml
env_configs/nut_assembly/franka_robosuite_nut_assembly.yaml
env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml
```

### 5.2 LIBERO-PRO 任务

LIBERO 依赖自己的 Robosuite 版本，不能和普通 Robosuite extra 混装在同一个虚拟环境中。请创建单独环境：

```bash
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --active --extra libero --extra contactgraspnet
```

示例运行配置：

```bash
env_configs/libero/franka_libero_spatial_0.yaml
env_configs/libero/franka_libero_goal_1.yaml
```

更多 LIBERO 任务说明见 `docs/libero-tasks.md`。

### 5.3 BEHAVIOR / Isaac Sim 任务

BEHAVIOR 任务依赖 NVIDIA Isaac Sim 和 OmniGibson，环境最重，建议在确认 Robosuite 已跑通后再配置：

```bash
cd capx/third_party/b1k
./uv_install.sh --dataset --accept-dataset-tos
cd ../../..
```

安装完成后，激活 b1k 环境并复制 cuRobo JIT 头文件：

```bash
source capx/third_party/b1k/.venv/bin/activate
cp capx/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/
```

无显示器服务器还需要：

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1
```

运行前设置：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
```

更多 BEHAVIOR 任务说明见 `docs/behavior-tasks.md`。

## 6. 配置 LLM 代理

CaP-X 默认通过本地 OpenAI 兼容接口请求模型，默认地址是：

```text
http://127.0.0.1:8110/chat/completions
```

最简单的方式是使用 OpenRouter：

```bash
echo "sk-or-v1-your-key-here" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

注意：

- `.openrouterkey` 已被 git 忽略，不要提交真实 key。
- 运行普通模型评测时需要保持这个代理进程运行。
- 如果只用环境自带 oracle code 做自检，可以在部分任务中加 `--use-oracle-code True`，不需要让模型生成代码。

也可以使用 vLLM 或自定义 OpenAI 兼容服务，详见 `docs/configuration.md`。

## 7. 感知和控制 API 服务

多数 YAML 配置会自动启动所需服务，例如：

- SAM3：默认端口 `8114`
- ContactGraspNet：默认端口 `8115`
- PyRoKi：默认端口 `8116`

正常情况下直接运行评测即可，不需要手动启动这些服务。若要多个评测共享同一组服务，可以预启动：

```bash
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

可用 profile：

| profile   | 服务                          | 适用场景                  |
| --------- | ----------------------------- | ------------------------- |
| `minimal` | PyRoKi                        | oracle 或 privileged 自检 |
| `default` | SAM3、ContactGraspNet、PyRoKi | 常规 Robosuite 视觉任务   |
| `full`    | default 加 OWL-ViT、SAM2      | 更完整的视觉服务          |

SAM3 权重需要 HuggingFace 访问权限。首次使用前需要申请访问并登录：

```bash
huggingface-cli login
```

## 8. 快速自检

### 8.1 运行单元测试

```bash
uv run pytest tests/test_environments.py -q
```

如果只想跑某个环境测试：

```bash
uv run pytest tests/test_environments.py::test_franka_pick_place_code_env -q
```

### 8.2 使用 oracle code 跑一个小评测

这适合检查仿真、输出目录和任务配置是否能工作：

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml \
    --use-oracle-code True \
    --total-trials 3 \
    --num-workers 1
```

如果配置启用了 `record_video`，需要提供输出目录：

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml \
    --use-oracle-code True \
    --total-trials 3 \
    --num-workers 1 \
    --record-video True \
    --output-dir ./outputs/manual_smoke
```

## 9. 运行 Robosuite 评测

先确保已经安装：

```bash
uv sync --extra robosuite
```

启动 LLM 代理：

```bash
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

在另一个终端运行单轮评测：

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model "google/gemini-3.1-pro-preview" \
    --total-trials 10 \
    --num-workers 2
```

运行多轮视觉差分评测：

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml \
    --model "google/gemini-3.1-pro-preview" \
    --total-trials 10 \
    --num-workers 2
```

常用参数：

| 参数                | 含义                                                        |
| ------------------- | ----------------------------------------------------------- |
| `--config-path`     | 必填，任务 YAML 路径                                        |
| `--model`           | 模型名称，传给本地 LLM 代理                                 |
| `--server-url`      | LLM 代理地址，默认 `http://127.0.0.1:8110/chat/completions` |
| `--temperature`     | 采样温度                                                    |
| `--total-trials`    | 覆盖 YAML 中的试验次数                                      |
| `--num-workers`     | 并行 worker 数                                              |
| `--record-video`    | 是否保存视频                                                |
| `--output-dir`      | 输出目录                                                    |
| `--use-oracle-code` | 使用环境内置参考代码，不请求模型生成                        |
| `--web-ui`          | 启动交互式 Web UI                                           |

## 10. 运行 LIBERO-PRO 评测

激活 LIBERO 专用环境：

```bash
source .venv-libero/bin/activate
```

启动 LLM 代理：

```bash
python capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

在另一个终端运行：

```bash
python capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --model "google/gemini-3.1-pro-preview" \
    --total-trials 10 \
    --num-workers 1
```

如果在服务器或 CI 中运行，可能需要创建 `~/.libero/config.yaml`，具体路径模板见 `docs/libero-tasks.md`。

## 11. 运行 BEHAVIOR 评测

激活 BEHAVIOR 环境并设置变量：

```bash
source capx/third_party/b1k/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
```

运行示例任务：

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml \
    --model "google/gemini-3.1-pro-preview" \
    --total-trials 3 \
    --num-workers 1
```

多 GPU 机器上，Isaac Sim 使用 `OMNIGIBSON_GPU_ID` 选择 GPU，而不是只看 `CUDA_VISIBLE_DEVICES`：

```bash
OMNIGIBSON_GPU_ID=0 OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml
```

首次运行时 cuRobo 和 Isaac Sim 可能会编译内核和 shader，启动慢是正常现象。

## 12. 启动交互式 Web UI

后端会在 `--web-ui True` 时启动 Web UI 服务，默认端口是 `8200`：

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --web-ui True
```

浏览器打开：

```text
http://localhost:8200
```

如果要单独开发前端：

```bash
cd web-ui
npm install
npm run dev
```

正式构建前端：

```bash
cd web-ui
npm run build
```

## 13. 回归测试和开发检查

运行主环境单元测试：

```bash
uv run pytest tests/test_environments.py -q
```

运行集成测试时，需要满足外部服务、模型、GPU、仿真器等条件：

```bash
uv run pytest tests/integrations -q
```

运行回归脚本：

```bash
./scripts/regression_test.sh quick
./scripts/regression_test.sh test1
```

代码格式和 lint：

```bash
ruff check .
ruff format .
```

Web UI 构建检查：

```bash
cd web-ui
npm run build
```

## 14. 输出文件在哪里

评测输出由 YAML 中的 `output_dir` 或命令行 `--output-dir` 控制。运行器会把模型名插入输出路径，oracle 运行会使用 `oracle` 作为模型名。

通常输出会包含：

- 模型生成或 oracle 使用的 Python 代码。
- 每个 trial 的日志和摘要。
- 如果开启视频，包含环境执行视频。
- 最终成功率、平均 reward、完成数量等统计。

## 15. 常见问题

### 15.1 Robosuite 和 LIBERO 依赖冲突

不要在同一个虚拟环境里同时安装 `--extra robosuite` 和 `--extra libero`。普通 Robosuite 使用默认 `.venv`，LIBERO 使用 `.venv-libero`。

### 15.2 找不到第三方包或本地源码依赖

先确认子模块已初始化：

```bash
git submodule update --init --recursive
```

然后重新同步当前环境：

```bash
uv sync
```

或同步对应 extra。

### 15.3 LLM 连接失败

确认代理服务在运行：

```bash
curl http://127.0.0.1:8110/health
```

如果端口不同，运行评测时显式传入：

```bash
--server-url http://127.0.0.1:YOUR_PORT/chat/completions
```

### 15.4 SAM3 权重下载失败

确认已申请访问 SAM3 权重，并完成 HuggingFace 登录：

```bash
huggingface-cli login
```

### 15.5 服务器无显示器导致渲染失败

Robosuite/LIBERO 常用：

```bash
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
```

BEHAVIOR/Isaac Sim 常用：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
```

### 15.6 `record_video requires --output-dir`

如果开启 `--record-video True`，必须同时传入输出目录：

```bash
--record-video True --output-dir ./outputs/my_run
```

## 16. 推荐首次运行顺序

建议按以下顺序验证：

1. 初始化子模块：`git submodule update --init --recursive`
2. 创建基础环境：`uv python install 3.10 && uv venv -p 3.10 && uv sync`
3. 安装 Robosuite：`uv sync --extra robosuite`
4. 跑单元测试：`uv run pytest tests/test_environments.py -q`
5. 用 oracle code 跑 3 个 trial。
6. 配置 OpenRouter key 并启动 LLM 代理。
7. 跑 `cube_stack` 的 10-trial 小评测。
8. 需要交互观察时再启动 `--web-ui True`。
9. Robosuite 跑通后，再配置 LIBERO 或 BEHAVIOR。
