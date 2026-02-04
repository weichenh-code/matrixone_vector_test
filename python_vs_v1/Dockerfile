# GPU KMeans 测试 Docker 镜像
# 基于 NVIDIA CUDA 官方镜像，包含完整的开发环境

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

# 安装基础工具
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -all -y

# 创建 conda 环境并安装依赖
RUN conda create -n gpu_test python=3.10 -y && \
    echo "source activate gpu_test" >> ~/.bashrc

# 激活环境并安装 Python 包
SHELL ["/bin/bash", "-c"]
RUN source activate gpu_test && \
    pip install --upgrade pip && \
    pip install \
    cuvs-cu11 \
    cupy-cuda11x \
    numpy \
    && conda clean -all -y

# 设置工作目录
WORKDIR /workspace

# 验证安装
RUN source activate gpu_test && \
    python -c "from cuvs.neighbors import ivf_flat; import cupy as cp; print('✓ cuVS and CuPy installed successfully')"

# 默认启动 bash
CMD ["/bin/bash", "-c", "source activate gpu_test && exec bash"]
