# WSL Ubuntu E 盘安装记录

本文记录本机已经完成的 WSL Ubuntu 安装步骤和当前状态，便于后续继续配置 CaP-X。

## 已完成事项

1. 确认本机已安装 WSL。

   已检测到 WSL 版本：

   ```text
   WSL version: 2.7.3.0
   Windows: 10.0.22631.5335
   ```

2. 确认 E 盘存在，且目标安装目录最初未被占用。

   目标目录：

   ```text
   E:\WSL\Ubuntu-22.04
   ```

3. 使用 WSL 安装器把 Ubuntu 22.04 安装到 E 盘。

   使用的安装方式等价于：

   ```powershell
   wsl --install Ubuntu-22.04 `
     --location E:\WSL\Ubuntu-22.04 `
     --name Ubuntu-22.04 `
     --no-launch `
     --web-download
   ```

4. 将 `Ubuntu-22.04` 设置为默认 WSL 发行版。

   当前 WSL 列表显示：

   ```text
   NAME             STATE    VERSION
   Ubuntu-22.04     Running  2
   docker-desktop   Stopped  2
   ```

5. 验证 Ubuntu 22.04 可以正常启动。

   验证结果：

   ```text
   WSL_OK
   Linux ... microsoft-standard-WSL2 ... x86_64
   ```

6. 创建并设置默认 Linux 用户。

   当前默认用户：

   ```text
   USER=capx
   HOME=/home/capx
   uid=1000(capx) gid=1000(capx) groups=1000(capx),27(sudo)
   ```

7. 验证 WSL 内可以看到 NVIDIA GPU。

   检测结果：

   ```text
   NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB
   ```

   说明 WSL2 已经能访问 Windows 本机 NVIDIA 显卡。当前 WSL 中可见显存约为 6GB。

## 当前状态

```text
WSL 发行版：Ubuntu-22.04
安装位置：E:\WSL\Ubuntu-22.04
WSL 版本：2
默认用户：capx
Linux Home：/home/capx
GPU：NVIDIA GeForce RTX 3060 Laptop GPU
WSL 可见显存：6144 MiB
```

## 后续进入 Ubuntu

在 Windows PowerShell 中执行：

```powershell
wsl -d Ubuntu-22.04
```

进入 Ubuntu 后检查 GPU：

```bash
nvidia-smi
```

## 后续建议

后续配置 CaP-X 时，建议把项目放在 WSL 的 Linux 文件系统内部，而不是直接放在 Windows 挂载路径里。

推荐路径：

```bash
mkdir -p ~/code
cd ~/code
git clone --recurse-submodules https://github.com/capgym/cap-x
cd cap-x
```

不推荐长期在下面这类路径中安装 Python 依赖或运行仿真：

```text
/mnt/c/...
/mnt/e/...
```

原因是 Windows 挂载盘上的大量小文件读写会明显拖慢 Python 包、`node_modules`、仿真资源和构建过程。

## 已观察到的提示

启动 WSL 时曾出现过一条 localhost/NAT 相关警告。Ubuntu 已能正常启动，GPU 也能识别，当前暂不影响继续使用。

## 后续配置记录

### 1. 复制项目到 WSL 内部文件系统

已将 Windows 侧项目：

```text
F:\code\cap-x
```

复制到 WSL 内部：

```text
/home/capx/code/cap-x
```

后续安装和运行均建议在 WSL 内部路径执行：

```bash
cd /home/capx/code/cap-x
```

### 2. 安装 uv

已在 WSL 用户环境安装 `uv`：

```text
uv 0.11.13
```

路径：

```text
/home/capx/.local/bin/uv
```

### 3. 配置 WSL 代理

Windows 代理端口：

```text
7890
7897
```

WSL 默认网关：

```text
172.28.192.1
```

已验证 WSL 通过以下代理可访问 GitHub：

```bash
http://172.28.192.1:7890
http://172.28.192.1:7897
```

已设置 Git 全局代理：

```bash
git config --global http.proxy http://172.28.192.1:7890
git config --global https.proxy http://172.28.192.1:7890
```

### 4. 初始化关键子模块

已初始化 Robosuite 路线当前需要的关键子模块：

```text
capx/third_party/sam3
capx/third_party/robosuite
capx/third_party/contact_graspnet_pytorch
capx/third_party/curobo
```

当前状态：

```text
sam3                      6fe87d64...
robosuite                 97292732...
contact_graspnet_pytorch  da3dcfb2...
curobo                    d64c4b00...
```

### 5. 创建虚拟环境

已创建项目虚拟环境：

```text
/home/capx/code/cap-x/.venv
```

Python 版本：

```text
Python 3.10.12
```

### 6. 安装构建工具链

`uv sync` 编译 `pyliblzfse` 时需要 C 编译器。已安装：

```bash
sudo apt-get install -y build-essential python3.10-dev
```

期间遇到过一次 `Hash Sum mismatch`，通过清理 apt 缓存并重新更新索引解决：

```bash
sudo apt-get clean
sudo apt-get update -o Acquire::http::No-Cache=True -o Acquire::https::No-Cache=True
```

### 7. 同步 Robosuite Python 环境

完整 `uv sync --extra robosuite` 会尝试解析本地 `nvidia-curobo`，而 WSL 当前没有 CUDA Toolkit，因此会报：

```text
CUDA_HOME environment variable is not set
```

为先跑通 Robosuite 基础环境，已使用以下命令绕开本地 cuRobo 源：

```bash
uv sync --frozen --extra robosuite \
  --no-sources-package nvidia-curobo \
  --no-install-package nvidia-curobo \
  --no-progress
```

同步结果：

```text
Installed 203 packages
```

当前大小：

```text
.venv              约 9.2G
/home/capx/.cache/uv 约 6.9G
```

### 8. 导入验证

已验证以下 Python 包可以导入：

```text
capx ok
robosuite 1.5.1
torch 2.9.1+cu128
cuda_available True
cuda_version 12.8
```

Robosuite 导入时出现 private macro file、可选 robot models、mink IK 相关 warning；这些是可选配置提示，不是导入失败。

### 9. 当前注意事项

当前环境适合继续尝试 Robosuite 轻量任务和 oracle/privileged 自检。

尚未完成：

```text
完整 cuRobo 本地 editable 安装
完整 LIBERO 子模块和 LIBERO 环境
完整 BEHAVIOR / Isaac Sim 环境
```

如果后续需要 cuRobo 或 BEHAVIOR，需要在 WSL 内安装匹配的 CUDA Toolkit，并设置：

```bash
export CUDA_HOME=/usr/local/cuda
```
