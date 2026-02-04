#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IVF-Flat + Brute Force 测试
对应Go文件：pkg/vectorindex/ivfflat/kmeans/device/issue_test.go

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
import cupy as cp
import time
import threading
from typing import Tuple, List
from cuvs.neighbors import ivf_flat, brute_force


def get_centers(vecs: np.ndarray, 
                dim: int, 
                cluster_cnt: int, 
                distance_type: str = "sqeuclidean",
                max_iterations: int = 10) -> np.ndarray:
    """
    获取KMeans聚类中心
    
    对应Go:
    func getCenters(vecs [][]float32, dim int, clusterCnt int, 
                    distanceType cuvs.Distance, maxIterations int) ([][]float32, error)
    
    使用 IVF-Flat 索引进行 KMeans 聚类
    
    参数:
        vecs: 输入向量数组，形状为 (n_samples, n_features)
        dim: 向量维度
        cluster_cnt: 聚类中心数量
        distance_type: 距离度量类型
        max_iterations: 最大迭代次数
        
    返回:
        聚类中心数组，形状为 (cluster_cnt, dim)
    """
    # Go: resource, err := cuvs.NewResource(nil)
    # Python cuVS 自动管理资源
    
    # Go: indexParams, err := ivf_flat.CreateIndexParams()
    index_params = ivf_flat.IndexParams(
        # Go: indexParams.SetNLists(uint32(clusterCnt))
        n_lists=cluster_cnt,
        
        # Go: indexParams.SetMetric(distanceType)
        metric=distance_type,
        
        # Go: indexParams.SetKMeansNIters(uint32(maxIterations))
        kmeans_n_iters=max_iterations,
        
        # Go: indexParams.SetKMeansTrainsetFraction(1)
        kmeans_trainset_fraction=1.0,
        
        add_data_on_build=True
    )
    
    # Go: dataset, err := cuvs.NewTensor(vecs)
    # Go: dataset.ToDevice(&resource)
    dataset = cp.asarray(vecs, dtype=cp.float32)
    
    # Go: index, _ := ivf_flat.CreateIndex(indexParams, &dataset)
    # Go: ivf_flat.BuildIndex(resource, indexParams, &dataset, index)
    index = ivf_flat.build(index_params, dataset)
    
    # Go: resource.Sync()
    cp.cuda.Stream.null.synchronize()
    
    # Go: ivf_flat.GetCenters(index, &centers)
    centers_gpu = index.centers
    
    # Go: centers.ToHost(&resource)
    centers = cp.asnumpy(centers_gpu)
    
    # Go: resource.Sync()
    cp.cuda.Stream.null.synchronize()
    
    # Go: result, err := centers.Slice()
    # Go: return result, nil
    return centers.astype(np.float32)


def search(dataset_vec: np.ndarray,
          queries_vec: np.ndarray,
          limit: int,
          distance_type: str = "sqeuclidean") -> Tuple[np.ndarray, np.ndarray]:
    """
    使用暴力搜索查找最近邻
    
    对应Go:
    func Search(datasetvec [][]float32, queriesvec [][]float32, limit uint, 
                distanceType cuvs.Distance) (retkeys any, retdistances []float64, err error)
    
    参数:
        dataset_vec: 数据集向量
        queries_vec: 查询向量
        limit: 返回的最近邻数量
        distance_type: 距离度量类型
        
    返回:
        (neighbors, distances) 元组
        - neighbors: 最近邻索引，形状 (n_queries, limit)
        - distances: 距离值，形状 (n_queries, limit)
    """
    # Go: resource, err := cuvs.NewResource(nil)
    # Python cuVS 自动管理资源
    
    # Go: dataset, err := cuvs.NewTensor(datasetvec)
    # Go: dataset.ToDevice(&resource)
    dataset = cp.asarray(dataset_vec, dtype=cp.float32)
    
    # Go: queries, err := cuvs.NewTensor(queriesvec)
    # Go: queries.ToDevice(&resource)
    queries = cp.asarray(queries_vec, dtype=cp.float32)
    
    # Go: index, err := brute_force.CreateIndex()
    # Go: brute_force.BuildIndex(resource, &dataset, distanceType, 2.0, index)
    # Python cuVS 的 brute_force API
    index = brute_force.build(dataset, metric=distance_type)
    
    # Go: resource.Sync()
    cp.cuda.Stream.null.synchronize()
    
    # Go: neighbors, distances 张量创建
    # Go: brute_force.SearchIndex(resource, *index, &queries, &neighbors, &distances)
    # Python API: search(index, queries, k, neighbors=None, distances=None, resources=None)
    distances_gpu, neighbors_gpu = brute_force.search(
        index,
        queries,
        limit
    )
    
    # Go: neighbors.ToHost(&resource)
    # Go: distances.ToHost(&resource)
    neighbors = cp.asnumpy(neighbors_gpu)
    distances = cp.asnumpy(distances_gpu)
    
    # Go: resource.Sync()
    cp.cuda.Stream.null.synchronize()
    
    # Go: 将结果转换为扁平数组
    # retdistances = make([]float64, len(distancesSlice)*int(limit))
    # keys = make([]int64, len(neighborsSlice)*int(limit))
    
    return neighbors.astype(np.int64), distances.astype(np.float64)


def TestIvfAndBruteForceForIssue():
    """
    测试 IVF-Flat KMeans + Brute Force 搜索
    
    对应Go:
    func TestIvfAndBruteForceForIssue(t *testing.T)
    
    测试流程:
    1. 生成 100,000 个 128 维随机向量
    2. 使用 IVF-Flat 进行 KMeans 聚类，得到 128 个聚类中心
    3. 启动 4 个线程，每个线程循环 1000 次
    4. 每次在聚类中心上执行暴力搜索
    """
    print("=" * 70)
    print("TestIvfAndBruteForceForIssue - IVF-Flat + Brute Force 测试")
    print("=" * 70)
    print()
    
    # 记录总测试开始时间
    total_start_time = time.time()
    
    # Go: dimension := uint(128)
    dimension = 128
    
    # Go: limit := uint(1)
    limit = 1
    
    # Go: dsize := 100000
    dsize = 100000
    
    # Go: nlist := 128
    nlist = 128
    
    print(f"测试参数:")
    print(f"  dimension = {dimension}")
    print(f"  dsize = {dsize}")
    print(f"  nlist = {nlist}")
    print(f"  limit = {limit}")
    print(f"  threads = 4")
    print(f"  iterations_per_thread = 1000")
    print()
    
    # Go: vecs := make([][]float32, dsize)
    # Go: for i := range vecs {
    # Go:     vecs[i] = make([]float32, dimension)
    # Go:     for j := range vecs[i] {
    # Go:         vecs[i][j] = rand.Float32()
    # Go:     }
    # Go: }
    print("生成随机向量...")
    data_gen_start = time.time()
    np.random.seed(42)  # 设置随机种子
    vecs = np.random.rand(dsize, dimension).astype(np.float32)
    data_gen_time = time.time() - data_gen_start
    print(f"  向量形状: {vecs.shape}")
    print(f"  数据类型: {vecs.dtype}")
    print(f"  ⏱ 数据生成耗时: {data_gen_time:.4f}秒")
    print()
    
    # Go: queries := vecs[:8192]
    queries = vecs[:8192]
    print(f"查询向量:")
    print(f"  查询数量: {len(queries)}")
    print(f"  查询形状: {queries.shape}")
    print()
    
    # Go: centers, err := getCenters(vecs, int(dimension), nlist, cuvs.DistanceL2, 10)
    # Go: require.NoError(t, err)
    print("执行 IVF-Flat KMeans 聚类...")
    kmeans_start = time.time()
    try:
        centers = get_centers(vecs, dimension, nlist, "sqeuclidean", 10)
        kmeans_time = time.time() - kmeans_start
        print(f"  ✓ 聚类成功")
        print(f"  ⏱ KMeans 聚类耗时: {kmeans_time:.4f}秒")
        print(f"  聚类中心形状: {centers.shape}")
        print(f"  聚类中心类型: {centers.dtype}")
    except Exception as e:
        print(f"  ✗ 聚类失败: {e}")
        raise
    print()
    
    # 验证聚类结果
    assert centers.shape == (nlist, dimension), f"聚类中心形状错误: {centers.shape}"
    assert centers.dtype == np.float32, f"聚类中心类型错误: {centers.dtype}"
    print(f"✓ 聚类结果验证通过")
    print()
    
    # Go: var wg sync.WaitGroup
    # Go: for n := 0; n < 4; n++ {
    # Go:     wg.Add(1)
    # Go:     go func() {
    # Go:         defer wg.Done()
    # Go:         for i := 0; i < 1000; i++ {
    # Go:             _, _, err := Search(centers, queries, limit, cuvs.DistanceL2)
    # Go:             require.NoError(t, err)
    # Go:         }
    # Go:     }()
    # Go: }
    # Go: wg.Wait()
    
    print("=" * 70)
    print("开始并发暴力搜索测试")
    print("=" * 70)
    print()
    
    num_threads = 4
    iterations_per_thread = 1000
    total_searches = num_threads * iterations_per_thread
    
    errors = []
    thread_times = []
    
    def worker_thread(thread_id: int):
        """工作线程函数"""
        thread_start = time.time()
        thread_errors = []
        
        try:
            for i in range(iterations_per_thread):
                try:
                    # Go: _, _, err := Search(centers, queries, limit, cuvs.DistanceL2)
                    neighbors, distances = search(centers, queries, limit, "sqeuclidean")
                    
                    # 验证结果
                    assert neighbors.shape == (len(queries), limit), \
                        f"neighbors形状错误: {neighbors.shape}"
                    assert distances.shape == (len(queries), limit), \
                        f"distances形状错误: {distances.shape}"
                    
                    # 每100次打印一次进度
                    if (i + 1) % 100 == 0:
                        elapsed = time.time() - thread_start
                        print(f"  线程{thread_id}: 完成 {i+1}/{iterations_per_thread} "
                              f"({(i+1)/iterations_per_thread*100:.1f}%) "
                              f"耗时 {elapsed:.2f}秒")
                        
                except Exception as e:
                    thread_errors.append((i, str(e)))
                    
        except Exception as e:
            thread_errors.append((-1, f"线程异常: {str(e)}"))
        
        thread_time = time.time() - thread_start
        thread_times.append(thread_time)
        
        if thread_errors:
            errors.extend([(thread_id, err) for err in thread_errors])
        
        print(f"  线程{thread_id}: 完成所有 {iterations_per_thread} 次搜索, "
              f"总耗时 {thread_time:.4f}秒")
    
    # 启动所有线程
    print(f"启动 {num_threads} 个并发线程...")
    search_start = time.time()
    
    threads = []
    for n in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(n,))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    search_time = time.time() - search_start
    
    print()
    print("=" * 70)
    
    # 检查错误
    if errors:
        print(f"✗ 测试失败，共 {len(errors)} 个错误:")
        for thread_id, (iter_id, error_msg) in errors[:10]:  # 只显示前10个
            print(f"  线程{thread_id}, 迭代{iter_id}: {error_msg}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")
        raise RuntimeError(f"测试失败，共 {len(errors)} 个错误")
    
    print("✓ 所有搜索测试通过!")
    print("=" * 70)
    print()
    
    # 计算总耗时
    total_time = time.time() - total_start_time
    
    # 性能统计
    print("=" * 70)
    print("⏱  性能统计")
    print("=" * 70)
    print(f"  数据生成耗时:         {data_gen_time:.4f}秒")
    print(f"  KMeans聚类耗时:       {kmeans_time:.4f}秒")
    print(f"  并发搜索总耗时:       {search_time:.4f}秒")
    print(f"  " + "-" * 66)
    print(f"  总耗时:               {total_time:.4f}秒")
    print()
    print(f"搜索性能:")
    print(f"  总搜索次数:           {total_searches}")
    print(f"  平均线程耗时:         {np.mean(thread_times):.4f}秒")
    print(f"  最快线程耗时:         {np.min(thread_times):.4f}秒")
    print(f"  最慢线程耗时:         {np.max(thread_times):.4f}秒")
    print(f"  平均单次搜索耗时:     {search_time/total_searches*1000:.4f}毫秒")
    print(f"  有效吞吐量:           {total_searches/search_time:.2f} 次搜索/秒")
    print(f"  查询向量吞吐量:       {total_searches*len(queries)/search_time:.0f} 向量/秒")
    print("=" * 70)
    print("✓ TestIvfAndBruteForceForIssue 测试通过！")
    print("=" * 70)


def main():
    """主函数"""
    try:
        TestIvfAndBruteForceForIssue()
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
