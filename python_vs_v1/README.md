# Python VS V1 - GPU KMeans 测试代码

本目录包含用于测试 GPU KMeans 功能的 Python 代码和相关文档。

## 📁 目录结构

```
python_vs_v1/
├── README.md                    # 本文件
│
├── 核心代码 (Python)
│   ├── gpu_kmeans.py           # GPU KMeans 实现（使用 cuVS）
│   ├── test_gpu.py             # 简单快速测试
│   └── test_issue.py           # 完整性能测试
│
├── 运行指南 (Markdown)
│   ├── RUN_TESTS_GUIDE.md      # 在 Linux GPU 机器上运行测试的完整指南
│   └── DOCKER_GUIDE.md         # Docker 环境运行指南
│
├── Docker 配置
│   ├── Dockerfile              # 自定义 Docker 镜像
│   └── docker-compose.yml      # Docker Compose 配置
│
└── 技术文档 (Markdown)
    ├── CUVS_USAGE.md           # cuVS 库使用说明
    ├── GPU_KMEANS_README.md    # GPU KMeans 详细说明
    ├── CODE_MAPPING.md         # Go 与 Python 代码映射关系
    └── IMPLEMENTATION_SUMMARY.md # 实现总结
```

## 🚀 快速开始

### 在有 GPU 的 Linux 机器上运行

1. **环境准备**（需要 NVIDIA GPU 和驱动）
   ```bash
   # 检查 GPU
   nvidia-smi
   
   # 创建 conda 环境
   conda create -n gpu_test python=3.10 -y
   conda activate gpu_test
   
   # 安装依赖（CUDA 12.x）
   pip install cuvs-cu12 cupy-cuda12x numpy
   ```

2. **运行测试**
   ```bash
   cd python_vs_v1
   
   # 快速测试（约 30 秒）
   python test_gpu.py
   
   # 完整测试（约 2-10 分钟）
   python test_issue.py
   ```

详细步骤请查看 `RUN_TESTS_GUIDE.md`。

### 使用 Docker 运行

```bash
cd python_vs_v1

# 构建镜像
docker build -t gpu-kmeans-test:latest .

# 运行测试
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  gpu-kmeans-test:latest
```

详细步骤请查看 `DOCKER_GUIDE.md`。

## 📝 文件说明

### 核心代码

- **`gpu_kmeans.py`** (508 行)
  - GPU KMeans 实现，使用 NVIDIA cuVS 库
  - 对应 Go 代码：`pkg/vectorindex/ivfflat/kmeans/device/gpu.go`
  - 主要类：`GPUKMeansClusterer`

- **`test_gpu.py`** (229 行)
  - 简单快速测试
  - 数据规模：1,024 个向量，128 维
  - 运行时间：约 5-30 秒

- **`test_issue.py`** (398 行)
  - 完整性能和并发测试
  - 数据规模：100,000 个向量，128 维
  - 并发搜索：4 线程 × 1000 次迭代
  - 运行时间：约 2-10 分钟

### 文档

- **`RUN_TESTS_GUIDE.md`** - 完整的测试运行指南
- **`DOCKER_GUIDE.md`** - Docker 环境配置和使用
- **`CUVS_USAGE.md`** - cuVS 库 API 说明
- **`GPU_KMEANS_README.md`** - GPU KMeans 详细文档
- **`CODE_MAPPING.md`** - Go ↔ Python 代码对应关系
- **`IMPLEMENTATION_SUMMARY.md`** - 实现总结和设计决策

## ⚠️ 注意事项

### 关于 .gitignore

**重要**：项目根目录的 `.gitignore` 文件包含 `*.py` 规则，会忽略所有 Python 文件。

这就是为什么这些文件被放在 `python_vs_v1` 文件夹中。如果需要提交到 git：

```bash
# 方案 1: 强制添加（推荐）
git add -f python_vs_v1/*.py

# 方案 2: 修改 .gitignore，移除 *.py 规则（需谨慎）
```

### 环境要求

- **GPU**: NVIDIA GPU（计算能力 >= 6.0）
- **显存**: 至少 8GB（推荐 16GB+）
- **驱动**: NVIDIA 驱动 >= 470.x
- **CUDA**: 11.8 或 12.x
- **Python**: 3.9-3.11

## 🔗 代码对应关系

| Go 代码 | Python 代码 | 说明 |
|--------|------------|------|
| `pkg/vectorindex/ivfflat/kmeans/device/gpu.go` | `gpu_kmeans.py` | GPU KMeans 实现 |
| `pkg/vectorindex/ivfflat/kmeans/device/issue_test.go` | `test_issue.py` | 完整测试 |
| `TestIvfAndBruteForceForIssue()` | `TestIvfAndBruteForceForIssue()` | 测试函数 |

详细映射请参考 `CODE_MAPPING.md`。

## 📊 性能参考

### NVIDIA RTX 3090 (24GB)

| 测试 | 数据规模 | 运行时间 | 吞吐量 |
|-----|---------|---------|--------|
| test_gpu.py | 1K × 128 | ~0.15秒 | ~7K 向量/秒 |
| test_issue.py (聚类) | 100K × 128 | ~3秒 | ~33K 向量/秒 |
| test_issue.py (搜索) | 4000次 | ~20秒 | ~200 搜索/秒 |

## 🆘 故障排除

### 问题：找不到 GPU

```bash
nvidia-smi  # 应该能看到 GPU 信息
```

### 问题：cuVS 导入失败

```bash
pip list | grep cuvs
pip install --upgrade cuvs-cu12 cupy-cuda12x
```

### 问题：内存不足

```bash
# 监控 GPU 显存
watch -n 1 nvidia-smi

# 减小测试数据规模（修改 test_issue.py 中的 dsize）
```

更多问题请参考 `RUN_TESTS_GUIDE.md` 的常见问题部分。

---

**创建日期**: 2026-02-04  
**版本**: 1.0  
**作者**: AI Assistant
