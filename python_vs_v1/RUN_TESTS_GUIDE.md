# GPU测试运行指南

本文档详细说明如何在有GPU卡的Linux机器上运行Python GPU KMeans测试。

## 前置要求

### 硬件要求
- ✅ NVIDIA GPU（支持CUDA，计算能力 >= 6.0）
- ✅ 至少 8GB GPU 显存（推荐 16GB+）
- ✅ 至少 16GB 系统内存

### 软件要求
- ✅ Linux 操作系统（Ubuntu 20.04/22.04 或 CentOS 7/8）
- ✅ NVIDIA 驱动程序（版本 >= 470.x）
- ✅ CUDA Toolkit（11.x 或 12.x）
- ✅ Python 3.9-3.11

---

## 步骤1: 检查GPU环境

### 1.1 检查NVIDIA驱动

```bash
# 检查NVIDIA驱动是否安装
nvidia-smi

# 预期输出示例：
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA A100-SXM...  Off  | 00000000:00:04.0 Off |                    0 |
# | N/A   30C    P0    50W / 400W |      0MiB / 40960MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

**如果命令不存在**，需要先安装NVIDIA驱动：
```bash
# Ubuntu
sudo apt-get update
sudo apt-get install nvidia-driver-525

# 安装后重启
sudo reboot
```

### 1.2 检查CUDA版本

```bash
# 检查CUDA版本
nvcc --version

# 预期输出示例：
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 11.8, V11.8.89
```

**记住你的CUDA版本**（11.x 或 12.x），后续安装cuVS时会用到。

---

## 步骤2: 安装Python和依赖

### 2.1 检查Python版本

```bash
# 检查Python版本（需要3.9-3.11）
python3 --version

# 如果版本不对，安装正确版本
# Ubuntu
sudo apt-get install python3.10 python3.10-venv python3.10-dev

# 或者使用conda
conda create -n gpu_test python=3.10
conda activate gpu_test
```

### 2.2 创建虚拟环境（推荐）

```bash
# 进入项目目录
cd /path/to/matrixone_vector_test

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 如果使用conda
# conda create -n gpu_test python=3.10
# conda activate gpu_test
```

### 2.3 升级pip

```bash
pip install --upgrade pip setuptools wheel
```

---

## 步骤3: 安装cuVS和依赖

### 3.1 确定CUDA版本对应的包

| 你的CUDA版本 | 安装命令 |
|------------|---------|
| CUDA 11.x | `pip install cuvs-cu11 cupy-cuda11x` |
| CUDA 12.x | `pip install cuvs-cu12 cupy-cuda12x` |

### 3.2 安装cuVS和CuPy

**方法1: 使用pip（推荐）**

```bash
# CUDA 11.x
pip install cuvs-cu11 cupy-cuda11x numpy

# CUDA 12.x
pip install cuvs-cu12 cupy-cuda12x numpy
```

**方法2: 使用conda**

```bash
# 激活conda环境
conda activate gpu_test

# 安装cuVS和依赖
conda install -c rapidsai -c conda-forge -c nvidia \
    cuvs python=3.10 cuda-version=11.8

# 如果是CUDA 12.x
conda install -c rapidsai -c conda-forge -c nvidia \
    cuvs python=3.10 cuda-version=12.0
```

### 3.3 验证安装

```bash
# 测试cuVS安装
python3 << EOF
try:
    from cuvs.neighbors import ivf_flat, brute_force
    import cupy as cp
    print("✓ cuVS安装成功！")
    print(f"  cuVS版本: {import_module('cuvs').__version__ if hasattr(import_module('cuvs'), '__version__') else 'OK'}")
    print(f"  CuPy版本: {cp.__version__}")
    
    # 测试GPU
    gpu_available = cp.cuda.runtime.getDeviceCount() > 0
    if gpu_available:
        print(f"  GPU数量: {cp.cuda.runtime.getDeviceCount()}")
        print("✓ GPU可用！")
    else:
        print("✗ 未检测到GPU")
except ImportError as e:
    print(f"✗ 安装失败: {e}")
    exit(1)
EOF
```

**预期输出**：
```
✓ cuVS安装成功！
  cuVS版本: OK
  CuPy版本: 12.3.0
  GPU数量: 1
✓ GPU可用！
```

---

## 步骤4: 准备测试文件

### 4.1 确认文件存在

```bash
cd /path/to/matrixone_vector_test

# 检查必需的文件
ls -lh gpu_kmeans.py test_gpu.py test_issue.py

# 预期输出：
# -rw-r--r-- 1 user group  15K gpu_kmeans.py
# -rw-r--r-- 1 user group   8K test_gpu.py
# -rw-r--r-- 1 user group  14K test_issue.py
```

### 4.2 检查文件权限

```bash
# 添加执行权限
chmod +x test_gpu.py test_issue.py
```

---

## 步骤5: 运行测试

### 5.1 运行 test_gpu.py（简单测试）

这是一个基础测试，数据规模较小（1024个向量），适合快速验证环境。

```bash
# 激活虚拟环境（如果还没激活）
source venv/bin/activate  # 或 conda activate gpu_test

# 运行测试
python3 test_gpu.py
```

**预期运行时间**: 约 5-30 秒

**预期输出**:
```
======================================================================
TestGpu - GPU KMeans聚类测试
======================================================================

测试参数:
  dim = 128
  dsize = 1024
  nlist = 128

生成随机向量...
  向量形状: (1024, 128)
  数据类型: float32
  ⏱ 数据生成耗时: 0.0023秒

创建KMeans聚类器...
  ✓ 聚类器创建成功
  ⏱ 创建耗时: 0.0156秒

执行GPU KMeans聚类...
  ✓ 聚类执行成功
  ⏱ KMeans聚类耗时: 0.0891秒 (核心计算时间)

验证结果...
  ✓ 类型检查通过: numpy.ndarray
  ✓ 数据类型检查通过: float32
  ✓ 形状检查通过: (128, 128)
  ✓ 数据有效性检查通过（无NaN）
  ✓ 数据有效性检查通过（无Inf）
  ⏱ 结果验证耗时: 0.0012秒

...

======================================================================
⏱  性能统计
======================================================================
  数据生成耗时:         0.0023秒
  创建聚类器耗时:       0.0156秒
  KMeans聚类耗时:       0.0891秒 ⭐ (核心)
  结果验证耗时:         0.0012秒
  ------------------------------------------------------------------
  总耗时:               0.1082秒

  KMeans占比:           82.3%
  吞吐量:               11491 向量/秒
======================================================================
✓ TestGpu 测试通过！
======================================================================
```

### 5.2 运行 test_issue.py（完整测试）

这是一个完整的压力测试，包含大数据集（100,000个向量）和并发搜索。

```bash
# 运行测试
python3 test_issue.py

# 或者重定向输出到文件
python3 test_issue.py 2>&1 | tee test_issue_output.log
```

**预期运行时间**: 约 2-10 分钟（取决于GPU性能）

**预期输出**:
```
======================================================================
TestIvfAndBruteForceForIssue - IVF-Flat + Brute Force 测试
======================================================================

测试参数:
  dimension = 128
  dsize = 100000
  nlist = 128
  limit = 1
  threads = 4
  iterations_per_thread = 1000

生成随机向量...
  向量形状: (100000, 128)
  数据类型: float32
  ⏱ 数据生成耗时: 0.3456秒

查询向量:
  查询数量: 8192
  查询形状: (8192, 128)

执行 IVF-Flat KMeans 聚类...
  ✓ 聚类成功
  ⏱ KMeans 聚类耗时: 2.1234秒
  聚类中心形状: (128, 128)

✓ 聚类结果验证通过

======================================================================
开始并发暴力搜索测试
======================================================================

启动 4 个并发线程...
  线程0: 完成 100/1000 (10.0%) 耗时 1.23秒
  线程1: 完成 100/1000 (10.0%) 耗时 1.24秒
  线程2: 完成 100/1000 (10.0%) 耗时 1.25秒
  线程3: 完成 100/1000 (10.0%) 耗时 1.26秒
  ...
  线程0: 完成所有 1000 次搜索, 总耗时 12.3456秒
  线程1: 完成所有 1000 次搜索, 总耗时 12.4567秒
  线程2: 完成所有 1000 次搜索, 总耗时 12.5678秒
  线程3: 完成所有 1000 次搜索, 总耗时 12.6789秒

======================================================================
✓ 所有搜索测试通过!
======================================================================

======================================================================
⏱  性能统计
======================================================================
  数据生成耗时:         0.3456秒
  KMeans聚类耗时:       2.1234秒
  并发搜索总耗时:       15.6789秒
  ------------------------------------------------------------------
  总耗时:               18.1479秒

搜索性能:
  总搜索次数:           4000
  平均线程耗时:         12.5000秒
  最快线程耗时:         12.3456秒
  最慢线程耗时:         12.6789秒
  平均单次搜索耗时:     3.9197毫秒
  有效吞吐量:           255.08 次搜索/秒
  查询向量吞吐量:       2089062 向量/秒
======================================================================
✓ TestIvfAndBruteForceForIssue 测试通过！
======================================================================
```

---

## 步骤6: 监控GPU使用情况

在运行测试的同时，可以在另一个终端监控GPU状态：

```bash
# 持续监控GPU状态（每1秒刷新）
watch -n 1 nvidia-smi

# 或者使用更详细的监控
nvidia-smi dmon -i 0 -s pucvmet

# 查看GPU进程
nvidia-smi pmon -i 0
```

---

## 常见问题排查

### 问题1: ImportError: No module named 'cuvs'

**原因**: cuVS未安装或安装失败

**解决方案**:
```bash
# 重新安装cuVS
pip uninstall cuvs cuvs-cu11 cuvs-cu12 -y
pip install cuvs-cu11  # 或 cuvs-cu12

# 验证安装
python3 -c "from cuvs.neighbors import ivf_flat; print('OK')"
```

### 问题2: RuntimeError: CUDA error

**原因**: GPU不可用或驱动问题

**解决方案**:
```bash
# 检查GPU状态
nvidia-smi

# 检查CUDA可用性
python3 << EOF
import cupy as cp
print(f"CUDA可用: {cp.cuda.is_available()}")
print(f"GPU数量: {cp.cuda.runtime.getDeviceCount()}")
EOF

# 如果显示不可用，重启服务器
sudo reboot
```

### 问题3: Out of Memory (OOM)

**原因**: GPU显存不足

**解决方案**:
```bash
# 检查显存使用
nvidia-smi

# 减小测试数据规模（修改test_issue.py）
# 将 dsize = 100000 改为 dsize = 10000
# 将 iterations_per_thread = 1000 改为 100

# 或者清理GPU缓存
python3 << EOF
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()
EOF
```

### 问题4: 运行很慢

**原因**: 可能在CPU上运行而不是GPU

**检查方法**:
```bash
# 运行测试时，在另一个终端查看GPU使用率
watch -n 1 nvidia-smi

# 如果 "GPU-Util" 接近 0%，说明没有使用GPU
```

**解决方案**:
```bash
# 确认cuVS正确安装
pip list | grep cuvs

# 确认CuPy正确安装
python3 -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0))"
```

### 问题5: ModuleNotFoundError: No module named 'gpu_kmeans'

**原因**: Python找不到gpu_kmeans.py文件

**解决方案**:
```bash
# 确保在正确的目录
pwd
ls gpu_kmeans.py  # 确认文件存在

# 如果文件在其他目录，添加到Python路径
export PYTHONPATH=$PYTHONPATH:/path/to/matrixone_vector_test

# 或者直接在包含gpu_kmeans.py的目录运行
cd /path/to/matrixone_vector_test
python3 test_gpu.py
```

---

## 性能基准参考

### NVIDIA A100 (40GB)

| 测试 | 数据规模 | 聚类数 | 耗时 | 吞吐量 |
|-----|---------|--------|------|--------|
| test_gpu.py | 1K × 128 | 128 | ~0.1秒 | ~10K 向量/秒 |
| test_issue.py (聚类) | 100K × 128 | 128 | ~2秒 | ~50K 向量/秒 |
| test_issue.py (搜索) | 4000次 | - | ~15秒 | ~250 搜索/秒 |

### NVIDIA RTX 3090 (24GB)

| 测试 | 数据规模 | 聚类数 | 耗时 | 吞吐量 |
|-----|---------|--------|------|--------|
| test_gpu.py | 1K × 128 | 128 | ~0.15秒 | ~7K 向量/秒 |
| test_issue.py (聚类) | 100K × 128 | 128 | ~3秒 | ~33K 向量/秒 |
| test_issue.py (搜索) | 4000次 | - | ~20秒 | ~200 搜索/秒 |

### NVIDIA T4 (16GB)

| 测试 | 数据规模 | 聚类数 | 耗时 | 吞吐量 |
|-----|---------|--------|------|--------|
| test_gpu.py | 1K × 128 | 128 | ~0.2秒 | ~5K 向量/秒 |
| test_issue.py (聚类) | 100K × 128 | 128 | ~5秒 | ~20K 向量/秒 |
| test_issue.py (搜索) | 4000次 | - | ~30秒 | ~130 搜索/秒 |

---

## 快速运行脚本

创建一个一键运行脚本：

```bash
# 创建运行脚本
cat > run_all_tests.sh << 'EOF'
#!/bin/bash
set -e

echo "=========================================="
echo "GPU KMeans 测试套件"
echo "=========================================="
echo

# 检查GPU
echo "检查GPU状态..."
nvidia-smi || { echo "错误: 未检测到GPU"; exit 1; }
echo

# 激活虚拟环境
if [ -d "venv" ]; then
    echo "激活虚拟环境..."
    source venv/bin/activate
fi

# 检查依赖
echo "检查依赖..."
python3 -c "from cuvs.neighbors import ivf_flat; import cupy" || {
    echo "错误: cuVS或CuPy未安装"
    exit 1
}
echo "✓ 依赖检查通过"
echo

# 运行test_gpu.py
echo "=========================================="
echo "运行 test_gpu.py..."
echo "=========================================="
python3 test_gpu.py || { echo "test_gpu.py 失败"; exit 1; }
echo

# 运行test_issue.py
echo "=========================================="
echo "运行 test_issue.py..."
echo "=========================================="
python3 test_issue.py || { echo "test_issue.py 失败"; exit 1; }
echo

echo "=========================================="
echo "✓ 所有测试通过！"
echo "=========================================="
EOF

# 添加执行权限
chmod +x run_all_tests.sh

# 运行
./run_all_tests.sh
```

---

## 总结

### ✅ 运行测试的完整命令序列

```bash
# 1. 检查环境
nvidia-smi
nvcc --version
python3 --version

# 2. 安装依赖
pip install cuvs-cu11 cupy-cuda11x numpy  # 或 cu12

# 3. 验证安装
python3 -c "from cuvs.neighbors import ivf_flat; print('OK')"

# 4. 运行测试
cd /path/to/matrixone_vector_test
python3 test_gpu.py      # 快速测试
python3 test_issue.py    # 完整测试
```

### 📊 测试文件对比

| 文件 | 数据规模 | 运行时间 | 测试目的 |
|-----|---------|---------|---------|
| `test_gpu.py` | 1K 向量 | ~10秒 | 快速验证环境 |
| `test_issue.py` | 100K 向量 | ~2-10分钟 | 完整性能测试 |

### 🎯 建议运行顺序

1. **先运行 test_gpu.py** - 快速验证环境是否正确
2. **再运行 test_issue.py** - 完整的性能和稳定性测试

---

**最后更新**: 2026-02-03  
**文档版本**: 1.0
