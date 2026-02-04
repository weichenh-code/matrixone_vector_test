# Docker GPU 测试指南

本文档说明如何使用 Docker 在 GPU 机器上运行测试，无需在宿主机安装 CUDA、Conda 等工具。

## 前置条件

### 宿主机要求

- ✅ **NVIDIA 驱动已安装**（您已完成）
  ```bash
  nvidia-smi  # 应该能正常显示
  ```

- ✅ **nvidia-container-toolkit 已安装**（您已有）
  ```bash
  dpkg -l | grep nvidia-container-toolkit  # 应该有输出
  ```

- ✅ **Docker 已安装**
  ```bash
  docker --version
  ```

---

## 方案 1: 使用预构建的 RAPIDS 镜像（最简单）

直接使用 Rapids AI 官方镜像，已包含所有依赖。

### 1.1 拉取镜像

```bash
# 拉取 Rapids AI 官方镜像（CUDA 11.8）
docker pull rapidsai/rapidsai:24.02-cuda11.8-runtime-ubuntu22.04-py3.10
```

### 1.2 运行容器并挂载代码目录

```bash
# 进入项目目录
cd /path/to/matrixone_vector_test

# 启动容器（挂载当前目录到容器的 /workspace）
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  rapidsai/rapidsai:24.02-cuda11.8-runtime-ubuntu22.04-py3.10 \
  bash
```

### 1.3 在容器中验证环境

```bash
# 检查 GPU
nvidia-smi

# 检查 Python 和依赖
python --version
python -c "from cuvs.neighbors import ivf_flat; import cupy; print('✓ 环境正常')"
```

### 1.4 运行测试

```bash
# 在容器中运行测试
python test_gpu.py
python test_issue.py
```

---

## 方案 2: 构建自定义镜像（推荐，可定制）

使用提供的 Dockerfile 构建自己的镜像。

### 2.1 构建镜像

```bash
# 在项目目录下
cd /path/to/matrixone_vector_test

# 构建镜像（约需 5-10 分钟）
docker build -t gpu-kmeans-test:latest .

# 查看镜像
docker images | grep gpu-kmeans-test
```

### 2.2 运行容器

```bash
# 启动容器
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  gpu-kmeans-test:latest
```

容器启动后会自动激活 `gpu_test` conda 环境。

### 2.3 运行测试

```bash
# 已经在容器内部，conda 环境已激活
python test_gpu.py
python test_issue.py
```

---

## 方案 3: 使用 docker-compose（最方便）

### 3.1 创建 docker-compose.yml

已为您创建 `docker-compose.yml` 文件（见下方）。

### 3.2 启动并运行

```bash
# 启动容器
docker-compose run --rm gpu-test

# 容器内自动挂载代码，直接运行
python test_gpu.py
python test_issue.py
```

---

## Docker 命令参数说明

### `--gpus all`
让容器可以访问宿主机的所有 GPU。

### `-v $(pwd):/workspace`
挂载当前目录到容器的 `/workspace`，这样：
- 容器可以访问您的代码文件
- 修改代码无需重新构建镜像
- 测试输出可以保存在宿主机

### `--rm`
容器退出后自动删除，避免占用空间。

### `-it`
交互式终端，可以在容器内执行命令。

---

## 验证 GPU 在容器中可用

```bash
# 在容器内运行
nvidia-smi

# 应该看到与宿主机相同的 GPU 信息
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA...  |
# | GPU  Name                 ...                                              |
# |   0  NVIDIA GeForce RTX 3090  ...                                         |
# +-----------------------------------------------------------------------------+
```

---

## 常见问题

### 问题 1: docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

**原因**: nvidia-container-toolkit 未正确安装或 Docker 未重启

**解决方案**:
```bash
# 重启 Docker 服务
sudo systemctl restart docker

# 验证 GPU 支持
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 问题 2: 容器内无法访问 GPU

**解决方案**:
```bash
# 确保使用 --gpus all 参数
docker run --rm -it --gpus all [镜像名] bash

# 在容器内检查
nvidia-smi
```

### 问题 3: 修改代码后需要重新构建镜像吗？

**不需要！** 因为使用了 `-v $(pwd):/workspace` 挂载，代码修改会实时同步到容器。

只有修改了 **Dockerfile** 或需要安装新的**系统级依赖**时，才需要重新构建镜像。

---

## 完整运行示例

### 快速开始（使用预构建镜像）

```bash
# 1. 进入项目目录
cd /path/to/matrixone_vector_test

# 2. 运行容器
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  rapidsai/rapidsai:24.02-cuda11.8-runtime-ubuntu22.04-py3.10 \
  bash

# 3. 在容器内运行测试
python test_gpu.py
python test_issue.py

# 4. 退出容器
exit
```

### 监控 GPU 使用（在宿主机另一个终端）

```bash
# 持续监控
watch -n 1 nvidia-smi

# 查看 GPU 进程
nvidia-smi pmon
```

---

## Docker vs 宿主机安装对比

| 特性 | Docker 方案 | 宿主机安装 |
|-----|-----------|----------|
| 隔离性 | ✅ 完全隔离 | ❌ 共享环境 |
| 影响其他用户 | ✅ 不影响 | ⚠️ 可能冲突 |
| 安装复杂度 | ✅ 简单 | ⚠️ 较复杂 |
| 版本管理 | ✅ 镜像版本化 | ⚠️ 手动管理 |
| GPU 访问 | ✅ 通过 nvidia-docker | ✅ 直接访问 |
| 性能 | ✅ 接近原生 | ✅ 原生性能 |
| 清理卸载 | ✅ 删除容器即可 | ⚠️ 需手动清理 |

---

## 推荐方案

**对于您的情况，推荐使用 方案 1（预构建镜像）或 方案 2（自定义镜像）**：

1. ✅ 不影响宿主机和其他用户
2. ✅ 环境隔离，可复现
3. ✅ 随时可以删除容器
4. ✅ 可以同时运行多个不同版本的测试环境

---

## 下一步

1. 选择一个方案（推荐方案 1 最简单）
2. 运行容器并验证 GPU 可用
3. 运行测试
4. 如有问题，查看下面的故障排除

---

**创建时间**: 2026-02-04  
**文档版本**: 1.0
