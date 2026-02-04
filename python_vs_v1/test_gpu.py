#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU KMeans测试
对应Go文件：pkg/vectorindex/ivfflat/kmeans/device/gpu_test.go

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
"""

import numpy as np
import time
from gpu_kmeans import NewKMeans, MetricType, InitType


def TestGpu():
    """
    测试GPU KMeans聚类
    
    对应Go测试:
    func TestGpu(t *testing.T) {
        dim := 128
        dsize := 1024
        nlist := 128
        vecs := make([][]float32, dsize)
        for i := range vecs {
            vecs[i] = make([]float32, dim)
            for j := range vecs[i] {
                vecs[i][j] = rand.Float32()
            }
        }
        
        c, err := NewKMeans[float32](vecs, nlist, 10, 0, 
                                     metric.Metric_L2Distance, 0, false, 0)
        require.NoError(t, err)
        
        centers, err := c.Cluster()
        require.NoError(t, err)
        
        _, ok := centers.([][]float32)
        require.True(t, ok)
    }
    """
    print("=" * 70)
    print("TestGpu - GPU KMeans聚类测试")
    print("=" * 70)
    print()
    
    # 记录总测试开始时间
    total_start_time = time.time()
    
    # Go: dim := 128
    dim = 128
    
    # Go: dsize := 1024
    dsize = 1024
    
    # Go: nlist := 128
    nlist = 128
    
    print(f"测试参数:")
    print(f"  dim = {dim}")
    print(f"  dsize = {dsize}")
    print(f"  nlist = {nlist}")
    print()
    
    # Go: vecs := make([][]float32, dsize)
    # Go: for i := range vecs {
    # Go:     vecs[i] = make([]float32, dim)
    # Go:     for j := range vecs[i] {
    # Go:         vecs[i][j] = rand.Float32()
    # Go:     }
    # Go: }
    print("生成随机向量...")
    data_gen_start = time.time()
    vecs = np.random.rand(dsize, dim).astype(np.float32)
    data_gen_time = time.time() - data_gen_start
    print(f"  向量形状: {vecs.shape}")
    print(f"  数据类型: {vecs.dtype}")
    print(f"  ⏱ 数据生成耗时: {data_gen_time:.4f}秒")
    print()
    
    # Go: c, err := NewKMeans[float32](vecs, nlist, 10, 0, 
    #                                  metric.Metric_L2Distance, 0, false, 0)
    # Go: require.NoError(t, err)
    print("创建KMeans聚类器...")
    create_start = time.time()
    try:
        c = NewKMeans(
            vectors=vecs,
            clusterCnt=nlist,
            maxIterations=10,
            deltaThreshold=0,
            distanceType=MetricType.Metric_L2Distance,
            initType=InitType.KMEANS_PLUS_PLUS,  # Go中传入0，对应枚举值
            spherical=False,
            nworker=0
        )
        create_time = time.time() - create_start
        print("  ✓ 聚类器创建成功")
        print(f"  ⏱ 创建耗时: {create_time:.4f}秒")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        raise
    print()
    
    # Go: centers, err := c.Cluster()
    # Go: require.NoError(t, err)
    print("执行GPU KMeans聚类...")
    cluster_start = time.time()
    try:
        centers = c.Cluster()
        cluster_time = time.time() - cluster_start
        print("  ✓ 聚类执行成功")
        print(f"  ⏱ KMeans聚类耗时: {cluster_time:.4f}秒 (核心计算时间)")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        raise
    print()
    
    # Go: _, ok := centers.([][]float32)
    # Go: require.True(t, ok)
    print("验证结果...")
    verify_start = time.time()
    
    # 检查是否为numpy数组（对应Go的类型断言）
    assert isinstance(centers, np.ndarray), \
        f"centers应该是numpy数组，实际类型: {type(centers)}"
    print(f"  ✓ 类型检查通过: numpy.ndarray")
    
    # 检查数据类型是否为float32
    assert centers.dtype == np.float32, \
        f"centers数据类型应为float32，实际类型: {centers.dtype}"
    print(f"  ✓ 数据类型检查通过: {centers.dtype}")
    
    # 检查形状
    expected_shape = (nlist, dim)
    assert centers.shape == expected_shape, \
        f"centers形状应为{expected_shape}，实际形状: {centers.shape}"
    print(f"  ✓ 形状检查通过: {centers.shape}")
    
    # 检查是否有有效数据
    assert not np.isnan(centers).any(), "centers包含NaN值"
    print(f"  ✓ 数据有效性检查通过（无NaN）")
    
    assert not np.isinf(centers).any(), "centers包含Inf值"
    print(f"  ✓ 数据有效性检查通过（无Inf）")
    
    verify_time = time.time() - verify_start
    print(f"  ⏱ 结果验证耗时: {verify_time:.4f}秒")
    print()
    print(f"聚类结果统计:")
    print(f"  - 聚类中心数量: {centers.shape[0]}")
    print(f"  - 向量维度: {centers.shape[1]}")
    print(f"  - 数值范围: [{centers.min():.6f}, {centers.max():.6f}]")
    print(f"  - 平均值: {centers.mean():.6f}")
    print(f"  - 标准差: {centers.std():.6f}")
    print()
    
    # Go代码中有注释掉的打印语句：
    # /*
    #     for k, center := range centroids {
    #         fmt.Printf("center[%d] = %v\n", k, center)
    #     }
    # */
    # 我们可选地打印前几个中心
    print("前3个聚类中心（显示前10个维度）:")
    for k in range(min(3, nlist)):
        center_preview = centers[k][:10]
        preview_str = ', '.join([f'{x:.4f}' for x in center_preview])
        print(f"  center[{k}] = [{preview_str}, ...]")
    print()
    
    # 计算总耗时
    total_time = time.time() - total_start_time
    
    print("=" * 70)
    print("⏱  性能统计")
    print("=" * 70)
    print(f"  数据生成耗时:         {data_gen_time:.4f}秒")
    print(f"  创建聚类器耗时:       {create_time:.4f}秒")
    print(f"  KMeans聚类耗时:       {cluster_time:.4f}秒 ⭐ (核心)")
    print(f"  结果验证耗时:         {verify_time:.4f}秒")
    print(f"  " + "-" * 66)
    print(f"  总耗时:               {total_time:.4f}秒")
    print()
    print(f"  KMeans占比:           {cluster_time/total_time*100:.1f}%")
    print(f"  吞吐量:               {dsize/cluster_time:.0f} 向量/秒")
    print("=" * 70)
    print("✓ TestGpu 测试通过！")
    print("=" * 70)


def main():
    """主函数"""
    try:
        TestGpu()
        return 0
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"✗ 断言失败: {e}")
        print("=" * 70)
        return 1
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ 测试失败: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
