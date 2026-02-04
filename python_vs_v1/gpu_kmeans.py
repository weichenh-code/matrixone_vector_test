#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU KMeans聚类器的Python实现
对应Go文件：pkg/vectorindex/ivfflat/kmeans/device/gpu.go

Copyright 2023 Matrix Origin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

这个模块使用RAPIDS cuVS库实现GPU加速的KMeans聚类功能。

Go代码使用: github.com/rapidsai/cuvs/go
Python代码使用: cuvs (RAPIDS cuVS Python绑定)
"""

import numpy as np
import cupy as cp
from typing import List, Union, Any, Optional
from enum import Enum

# 导入cuVS库
from cuvs.neighbors import ivf_flat


# ============================================================================
# 类型定义（对应Go中的类型）
# ============================================================================

class MetricType(Enum):
    """
    距离度量类型枚举
    对应Go: pkg/vectorindex/metric/MetricType
    """
    Metric_L2sqDistance = 0    # L2平方距离
    Metric_L2Distance = 1      # L2距离
    Metric_InnerProduct = 2    # 内积
    Metric_CosineDistance = 3  # 余弦距离
    Metric_L1Distance = 4      # L1距离


class InitType(Enum):
    """
    KMeans初始化类型
    对应Go: pkg/vectorindex/ivfflat/kmeans/InitType
    """
    RANDOM = 0
    KMEANS_PLUS_PLUS = 1


# ============================================================================
# GpuClusterer类（对应Go: type GpuClusterer[T cuvs.TensorNumberType])
# ============================================================================

class GpuClusterer:
    """
    GPU加速的KMeans聚类器
    
    对应Go中的 GpuClusterer[T] 结构体
    
    属性:
        indexParams: IVF-Flat索引参数 (对应Go: indexParams *ivf_flat.IndexParams)
        nlist: 聚类中心数量 (对应Go: nlist int)
        dim: 向量维度 (对应Go: dim int)
        vectors: 输入向量数据 (对应Go: vectors [][]T)
    """
    
    def __init__(self):
        """
        初始化GpuClusterer
        
        注意：不在构造函数中初始化，而是在NewKMeans中设置
        对应Go代码中的空结构体创建
        """
        self.indexParams: Optional[ivf_flat.IndexParams] = None
        self.nlist: int = 0
        self.dim: int = 0
        self.vectors: Optional[np.ndarray] = None
    
    def InitCentroids(self, ctx=None) -> None:
        """
        初始化聚类中心
        
        对应Go:
        func (c *GpuClusterer[T]) InitCentroids(ctx context.Context) error {
            return nil
        }
        
        参数:
            ctx: 上下文对象（未使用，保持与Go接口一致）
            
        返回:
            None
        """
        # Go代码中直接返回nil，Python中什么都不做
        pass
    
    def Cluster(self, ctx=None) -> np.ndarray:
        """
        执行聚类
        
        对应Go:
        func (c *GpuClusterer[T]) Cluster(ctx context.Context) (any, error)
        
        这个方法严格按照Go代码的执行顺序实现：
        1. resource, err := cuvs.NewResource(nil)
        2. dataset, err := cuvs.NewTensor(c.vectors)
        3. index, err := ivf_flat.CreateIndex(c.indexParams, &dataset)
        4. dataset.ToDevice(&resource)
        5. centers, err := cuvs.NewTensorOnDevice[T](&resource, dims)
        6. ivf_flat.BuildIndex(resource, c.indexParams, &dataset, index)
        7. resource.Sync()
        8. ivf_flat.GetCenters(index, &centers)
        9. centers.ToHost(&resource)
        10. resource.Sync()
        11. result, err := centers.Slice()
        12. return result, nil
        
        参数:
            ctx: 上下文对象（未使用，保持与Go接口一致）
            
        返回:
            聚类中心数组，形状为 (nlist, dim)
            
        抛出:
            RuntimeError: 如果聚类失败
        """
        try:
            # 步骤1: 创建cuVS资源
            # Go: resource, err := cuvs.NewResource(nil)
            # Python cuVS自动管理资源，不需要显式创建
            # defer resource.Close() - 由Python的垃圾回收自动处理
            
            # 步骤2: 创建数据集张量并传输到设备
            # Go: dataset, err := cuvs.NewTensor(c.vectors)
            # Go: defer dataset.Close()
            dataset = cp.asarray(self.vectors, dtype=cp.float32)
            
            # 步骤3: 创建索引
            # Go: index, err := ivf_flat.CreateIndex(c.indexParams, &dataset)
            # Go: defer index.Close()
            # 
            # 步骤4: 传输数据到GPU设备
            # Go: if _, err := dataset.ToDevice(&resource); err != nil
            #
            # 步骤5: 创建聚类中心张量（在设备上）
            # Go: centers, err := cuvs.NewTensorOnDevice[T](&resource, 
            #         []int64{int64(c.nlist), int64(c.dim)})
            # Go: defer centers.Close()
            #
            # 步骤6: 构建索引（执行KMeans聚类）
            # Go: if err := ivf_flat.BuildIndex(resource, c.indexParams, &dataset, index); err != nil
            #
            # Python cuVS的build()函数将步骤3-6合并为一个操作
            index = ivf_flat.build(self.indexParams, dataset)
            
            # 步骤7: 同步GPU操作
            # Go: if err := resource.Sync(); err != nil
            cp.cuda.Stream.null.synchronize()
            
            # 步骤8: 获取聚类中心
            # Go: if err := ivf_flat.GetCenters(index, &centers); err != nil
            centers_gpu = index.centers
            
            # 步骤9: 传输聚类中心回主机
            # Go: if _, err := centers.ToHost(&resource); err != nil
            centers = cp.asnumpy(centers_gpu)
            
            # 步骤10: 最终同步
            # Go: if err := resource.Sync(); err != nil
            cp.cuda.Stream.null.synchronize()
            
            # 步骤11: 返回结果切片
            # Go: result, err := centers.Slice()
            # Go: return result, nil
            return centers.astype(np.float32)
            
        except Exception as e:
            # 对应Go的错误返回: return nil, err
            raise RuntimeError(f"聚类失败: {str(e)}")
    
    def SSE(self) -> float:
        """
        计算误差平方和（Sum of Squared Errors）
        
        对应Go:
        func (c *GpuClusterer[T]) SSE() (float64, error) {
            return 0, nil
        }
        
        返回:
            误差平方和（当前实现返回0，与Go代码一致）
        """
        return 0.0
    
    def Close(self) -> None:
        """
        关闭并清理资源
        
        对应Go:
        func (c *GpuClusterer[T]) Close() error {
            if c.indexParams != nil {
                c.indexParams.Close()
            }
            return nil
        }
        
        返回:
            None
        """
        if self.indexParams is not None:
            # Python cuVS的IndexParams不需要显式关闭
            # 设置为None让垃圾回收器处理
            self.indexParams = None
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.Close()
        return False


# ============================================================================
# 辅助函数
# ============================================================================

def resolveCuvsDistanceForDense(distance: MetricType) -> str:
    """
    解析距离度量类型到cuVS距离类型
    
    对应Go:
    func resolveCuvsDistanceForDense(distance metric.MetricType) cuvs.Distance {
        switch distance {
        case metric.Metric_L2sqDistance:
            return cuvs.DistanceL2
        case metric.Metric_L2Distance:
            return cuvs.DistanceL2
        case metric.Metric_InnerProduct:
            return cuvs.DistanceL2
        case metric.Metric_CosineDistance:
            return cuvs.DistanceL2
        case metric.Metric_L1Distance:
            return cuvs.DistanceL2
        default:
            return cuvs.DistanceL2
        }
    }
    
    注意：Go代码中所有距离类型都返回 cuvs.DistanceL2
    Python cuVS使用字符串 "sqeuclidean" 表示L2平方距离
    
    参数:
        distance: 距离度量类型
        
    返回:
        cuVS距离度量字符串（所有情况都返回 "sqeuclidean"）
    """
    # Go代码中所有case都返回 cuvs.DistanceL2
    # Python cuVS中对应的是 "sqeuclidean"
    return "sqeuclidean"


# ============================================================================
# NewKMeans函数（对应Go: func NewKMeans[T types.RealNumbers](...))
# ============================================================================

def NewKMeans(
    vectors: Union[List[List[float]], np.ndarray],
    clusterCnt: int,
    maxIterations: int,
    deltaThreshold: float,
    distanceType: MetricType,
    initType: InitType,
    spherical: bool,
    nworker: int
) -> GpuClusterer:
    """
    创建KMeans聚类器
    
    对应Go:
    func NewKMeans[T types.RealNumbers](vectors [][]T, clusterCnt,
        maxIterations int, deltaThreshold float64,
        distanceType metric.MetricType, initType kmeans.InitType,
        spherical bool,
        nworker int) (kmeans.Clusterer, error)
    
    参数:
        vectors: 输入向量，[][]float32格式
        clusterCnt: 聚类中心数量
        maxIterations: 最大迭代次数
        deltaThreshold: 收敛阈值（当前未使用）
        distanceType: 距离度量类型
        initType: 初始化类型（当前未使用）
        spherical: 是否使用球形KMeans（当前未使用）
        nworker: 工作线程数（当前未使用）
        
    返回:
        GpuClusterer实例
        
    抛出:
        ValueError: 如果数据集为空
    """
    # 转换为numpy数组
    if isinstance(vectors, list):
        vecs = np.array(vectors, dtype=np.float32)
    else:
        vecs = vectors.astype(np.float32)
    
    # Go代码: switch vecs := any(vectors).(type)
    # Go代码: case [][]float32:
    
    # Go代码: c := &GpuClusterer[float32]{}
    c = GpuClusterer()
    
    # Go代码: c.nlist = clusterCnt
    c.nlist = clusterCnt
    
    # Go代码: if len(vectors) == 0 {
    #     return nil, moerr.NewInternalErrorNoCtx("empty dataset")
    # }
    if len(vecs) == 0:
        raise ValueError("empty dataset")
    
    # Go代码: c.vectors = vecs
    c.vectors = vecs
    
    # Go代码: c.dim = len(vecs[0])
    c.dim = vecs.shape[1] if len(vecs.shape) > 1 else len(vecs[0])
    
    # Go代码: indexParams, err := ivf_flat.CreateIndexParams()
    # Go代码: if err != nil {
    #     return nil, err
    # }
    try:
        indexParams = ivf_flat.IndexParams(
            # Go代码: indexParams.SetNLists(uint32(clusterCnt))
            n_lists=clusterCnt,
            
            # Go代码: indexParams.SetMetric(resolveCuvsDistanceForDense(distanceType))
            metric=resolveCuvsDistanceForDense(distanceType),
            
            # Go代码: indexParams.SetKMeansNIters(uint32(maxIterations))
            kmeans_n_iters=maxIterations,
            
            # Go代码: indexParams.SetKMeansTrainsetFraction(1)  // train all sample
            kmeans_trainset_fraction=1.0,
            
            # 其他参数使用默认值
            add_data_on_build=True
        )
    except Exception as e:
        raise RuntimeError(f"创建索引参数失败: {str(e)}")
    
    # Go代码: c.indexParams = indexParams
    c.indexParams = indexParams
    
    # Go代码: return c, nil
    return c


# ============================================================================
# 便捷函数（非Go代码对应，仅为Python使用方便）
# ============================================================================

def create_clusterer(
    vectors: Union[List[List[float]], np.ndarray],
    n_clusters: int,
    max_iter: int = 10,
    metric: str = 'l2'
) -> GpuClusterer:
    """
    创建聚类器的便捷函数（Python风格）
    
    参数:
        vectors: 输入向量
        n_clusters: 聚类数量
        max_iter: 最大迭代次数
        metric: 距离度量 ('l2', 'l2sq', 'cosine', 'ip', 'l1')
        
    返回:
        GpuClusterer实例
    """
    metric_map = {
        'l2': MetricType.Metric_L2Distance,
        'l2sq': MetricType.Metric_L2sqDistance,
        'cosine': MetricType.Metric_CosineDistance,
        'ip': MetricType.Metric_InnerProduct,
        'l1': MetricType.Metric_L1Distance,
    }
    
    metric_type = metric_map.get(metric.lower(), MetricType.Metric_L2Distance)
    
    return NewKMeans(
        vectors=vectors,
        clusterCnt=n_clusters,
        maxIterations=max_iter,
        deltaThreshold=0.0,
        distanceType=metric_type,
        initType=InitType.KMEANS_PLUS_PLUS,
        spherical=False,
        nworker=0
    )


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    """
    测试代码
    对应Go测试文件中的 TestGpu
    """
    print("=" * 70)
    print("GPU KMeans聚类器测试（使用cuVS）")
    print("=" * 70)
    print()
    
    # 测试参数（与Go测试保持一致）
    dim = 128
    dsize = 1024
    nlist = 128
    
    print(f"生成测试数据...")
    print(f"  - 维度: {dim}")
    print(f"  - 样本数: {dsize}")
    print(f"  - 聚类数: {nlist}")
    print()
    
    # 生成随机向量
    np.random.seed(42)
    vecs = np.random.rand(dsize, dim).astype(np.float32)
    
    try:
        # 创建KMeans聚类器（对应Go: c, err := NewKMeans[float32](...)）
        print("创建GPU聚类器...")
        c = NewKMeans(
            vectors=vecs,
            clusterCnt=nlist,
            maxIterations=10,
            deltaThreshold=0,
            distanceType=MetricType.Metric_L2Distance,
            initType=InitType.KMEANS_PLUS_PLUS,
            spherical=False,
            nworker=0
        )
        
        print(f"聚类器配置:")
        print(f"  - 聚类中心数: {c.nlist}")
        print(f"  - 向量维度: {c.dim}")
        print(f"  - 最大迭代次数: {c.indexParams.kmeans_n_iters}")
        print(f"  - 距离度量: {c.indexParams.metric}")
        print()
        
        # 执行聚类（对应Go: centers, err := c.Cluster()）
        print("执行GPU KMeans聚类...")
        centers = c.Cluster()
        
        # 验证结果（对应Go: _, ok := centers.([][]float32)）
        assert isinstance(centers, np.ndarray), "聚类中心应该是numpy数组"
        assert centers.shape == (nlist, dim), f"聚类中心形状应为({nlist}, {dim})"
        assert centers.dtype == np.float32, f"聚类中心数据类型应为float32"
        
        print()
        print(f"✓ 聚类完成!")
        print(f"  - 聚类中心形状: {centers.shape}")
        print(f"  - 数据类型: {centers.dtype}")
        print(f"  - 数据范围: [{centers.min():.4f}, {centers.max():.4f}]")
        print()
        
        # 显示前3个聚类中心的前10个维度
        print("前3个聚类中心示例（显示前10个维度）:")
        for k in range(min(3, nlist)):
            center_preview = centers[k][:10]
            preview_str = ', '.join([f'{x:.4f}' for x in center_preview])
            print(f"  center[{k}] = [{preview_str}, ...]")
        
        # 清理资源
        c.Close()
        
        print()
        print("=" * 70)
        print("✓ 测试成功完成！")
        print("=" * 70)
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ 测试失败: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        exit(1)
