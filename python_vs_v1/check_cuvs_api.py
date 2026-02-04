#!/usr/bin/env python3
"""
检查 cuVS API 版本和正确的访问方式
"""

import numpy as np
import cupy as cp
from cuvs.neighbors import ivf_flat

print("=" * 70)
print("cuVS API 检查")
print("=" * 70)

# 创建简单测试数据
n_samples = 100
n_features = 32
n_lists = 10

vecs = np.random.rand(n_samples, n_features).astype(np.float32)
dataset = cp.asarray(vecs, dtype=cp.float32)

# 创建索引参数
index_params = ivf_flat.IndexParams(
    n_lists=n_lists,
    metric="sqeuclidean",
    kmeans_n_iters=10,
    kmeans_trainset_fraction=1.0,
    add_data_on_build=True
)

print(f"\n构建索引...")
print(f"  数据: {n_samples} 个 {n_features} 维向量")
print(f"  聚类数: {n_lists}")

# 构建索引
index = ivf_flat.build(index_params, dataset)
cp.cuda.Stream.null.synchronize()

print(f"\n索引类型: {type(index)}")
print(f"索引类: {index.__class__.__name__}")

# 列出所有属性和方法
print(f"\n索引的所有属性和方法:")
attrs = [attr for attr in dir(index) if not attr.startswith('_')]
for i, attr in enumerate(attrs, 1):
    attr_type = type(getattr(index, attr)).__name__
    print(f"  {i:2d}. {attr:30s} ({attr_type})")

# 尝试不同的方式访问聚类中心
print(f"\n尝试访问聚类中心:")

attempts = [
    ("index.centers", lambda: index.centers),
    ("index.centers_", lambda: index.centers_),
    ("index.get_centers()", lambda: index.get_centers()),
    ("index.centers()", lambda: index.centers()),
    ("index.list_centers()", lambda: index.list_centers()),
    ("index.cluster_centers_", lambda: index.cluster_centers_),
]

for name, func in attempts:
    try:
        result = func()
        print(f"  ✓ {name:30s} 成功! 类型: {type(result)}, 形状: {getattr(result, 'shape', 'N/A')}")
        
        # 如果成功，尝试转换为 numpy
        try:
            if hasattr(result, '__array__'):
                arr = np.asarray(result)
            elif hasattr(result, 'get'):  # CuPy array
                arr = cp.asnumpy(result)
            else:
                arr = cp.asnumpy(cp.asarray(result))
            print(f"      转换为 numpy: {arr.shape}, dtype: {arr.dtype}")
        except Exception as e:
            print(f"      转换失败: {e}")
            
    except AttributeError as e:
        print(f"  ✗ {name:30s} AttributeError: {e}")
    except TypeError as e:
        print(f"  ✗ {name:30s} TypeError: {e}")
    except Exception as e:
        print(f"  ✗ {name:30s} {type(e).__name__}: {e}")

print("\n" + "=" * 70)
