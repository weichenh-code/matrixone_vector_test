#!/usr/bin/env python3
"""
检查 cuVS 中的 KMeans 相关功能
"""

import numpy as np
import cupy as cp

print("=" * 70)
print("cuVS 模块检查")
print("=" * 70)

# 检查 cuvs 的所有模块
try:
    import cuvs
    print("\n✓ cuvs 导入成功")
    print(f"  cuvs 模块内容:")
    for item in dir(cuvs):
        if not item.startswith('_'):
            print(f"    - {item}")
except Exception as e:
    print(f"\n✗ cuvs 导入失败: {e}")

# 检查 cuvs.cluster
print("\n" + "-" * 70)
print("检查 cuvs.cluster 模块")
print("-" * 70)
try:
    from cuvs import cluster
    print("✓ cuvs.cluster 存在")
    print(f"  cluster 模块内容:")
    for item in dir(cluster):
        if not item.startswith('_'):
            print(f"    - {item}")
except Exception as e:
    print(f"✗ cuvs.cluster 不存在: {e}")

# 检查 cuvs.cluster.kmeans
print("\n" + "-" * 70)
print("检查 cuvs.cluster.kmeans")
print("-" * 70)
try:
    from cuvs.cluster import kmeans
    print("✓ cuvs.cluster.kmeans 存在")
    print(f"  kmeans 模块内容:")
    for item in dir(kmeans):
        if not item.startswith('_'):
            print(f"    - {item}")
            
    # 尝试使用 KMeans
    print(f"\n尝试使用 KMeans:")
    n_samples, n_features, n_clusters = 100, 32, 10
    vecs = np.random.rand(n_samples, n_features).astype(np.float32)
    dataset = cp.asarray(vecs, dtype=cp.float32)
    
    # 尝试不同的 KMeans API
    attempts = [
        ("kmeans.fit", lambda: kmeans.fit),
        ("kmeans.predict", lambda: kmeans.predict),
        ("kmeans.KMeans", lambda: kmeans.KMeans),
        ("kmeans.fit_predict", lambda: kmeans.fit_predict),
    ]
    
    for name, func in attempts:
        try:
            obj = func()
            print(f"  ✓ {name:30s} 存在, 类型: {type(obj).__name__}")
        except Exception as e:
            print(f"  ✗ {name:30s} {type(e).__name__}")
            
except Exception as e:
    print(f"✗ cuvs.cluster.kmeans 不存在: {e}")

# 检查 cuML (可能作为备选)
print("\n" + "=" * 70)
print("检查 cuML (备选方案)")
print("=" * 70)
try:
    from cuml.cluster import KMeans as cuMLKMeans
    print("✓ cuML.cluster.KMeans 可用")
    
    # 测试 cuML KMeans
    n_samples, n_features, n_clusters = 100, 32, 10
    vecs = np.random.rand(n_samples, n_features).astype(np.float32)
    
    kmeans_model = cuMLKMeans(n_clusters=n_clusters, max_iter=10)
    kmeans_model.fit(vecs)
    centers = kmeans_model.cluster_centers_
    
    print(f"  测试成功!")
    print(f"    输入: {vecs.shape}")
    print(f"    聚类中心: {centers.shape}")
    print(f"    聚类中心类型: {type(centers)}")
    
except ImportError as e:
    print(f"✗ cuML 不可用: {e}")
except Exception as e:
    print(f"✗ cuML KMeans 测试失败: {e}")

print("\n" + "=" * 70)
