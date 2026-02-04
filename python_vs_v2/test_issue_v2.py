#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IVF-Flat + Brute Force Test
Corresponds to Go file: pkg/vectorindex/ivfflat/kmeans/device/issue_test.go

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
    Get KMeans cluster centers
    
    Corresponds to Go:
    func getCenters(vecs [][]float32, dim int, clusterCnt int, 
                    distanceType cuvs.Distance, maxIterations int) ([][]float32, error)
    
    Uses IVF-Flat index for KMeans clustering
    
    Args:
        vecs: Input vector array, shape (n_samples, n_features)
        dim: Vector dimension
        cluster_cnt: Number of cluster centers
        distance_type: Distance metric type
        max_iterations: Maximum number of iterations
        
    Returns:
        Cluster centers array, shape (cluster_cnt, dim)
    """
    # Go: resource, err := cuvs.NewResource(nil)
    # Python cuVS automatically manages resources
    
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
    Use brute force search to find nearest neighbors
    
    Corresponds to Go:
    func Search(datasetvec [][]float32, queriesvec [][]float32, limit uint, 
                distanceType cuvs.Distance) (retkeys any, retdistances []float64, err error)
    
    Args:
        dataset_vec: Dataset vectors
        queries_vec: Query vectors
        limit: Number of nearest neighbors to return
        distance_type: Distance metric type
        
    Returns:
        (neighbors, distances) tuple
        - neighbors: Nearest neighbor indices, shape (n_queries, limit)
        - distances: Distance values, shape (n_queries, limit)
    """
    # Go: resource, err := cuvs.NewResource(nil)
    # Python cuVS automatically manages resources
    
    # Go: dataset, err := cuvs.NewTensor(datasetvec)
    # Go: dataset.ToDevice(&resource)
    dataset = cp.asarray(dataset_vec, dtype=cp.float32)
    
    # Go: queries, err := cuvs.NewTensor(queriesvec)
    # Go: queries.ToDevice(&resource)
    queries = cp.asarray(queries_vec, dtype=cp.float32)
    
    # Go: index, err := brute_force.CreateIndex()
    # Go: brute_force.BuildIndex(resource, &dataset, distanceType, 2.0, index)
    # Python cuVS brute_force API
    index = brute_force.build(dataset, metric=distance_type)
    
    # Go: resource.Sync()
    cp.cuda.Stream.null.synchronize()
    
    # Go: neighbors, distances tensor creation
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
    
    # Go: Convert results to flat arrays
    # retdistances = make([]float64, len(distancesSlice)*int(limit))
    # keys = make([]int64, len(neighborsSlice)*int(limit))
    
    return neighbors.astype(np.int64), distances.astype(np.float64)


def TestIvfAndBruteForceForIssue():
    """
    Test IVF-Flat KMeans + Brute Force search
    
    Corresponds to Go:
    func TestIvfAndBruteForceForIssue(t *testing.T)
    
    Test workflow:
    1. Generate 100,000 random vectors of dimension 128
    2. Use IVF-Flat for KMeans clustering to get 128 cluster centers
    3. Launch 4 threads, each running 1000 iterations
    4. Each iteration performs brute force search on cluster centers
    """
    print("=" * 70)
    print("TestIvfAndBruteForceForIssue - IVF-Flat + Brute Force Test")
    print("=" * 70)
    print()
    
    # Record total test start time
    total_start_time = time.time()
    
    # Go: dimension := uint(128)
    dimension = 128
    
    # Go: limit := uint(1)
    limit = 1
    
    # Go: dsize := 100000
    dsize = 100000
    
    # Go: nlist := 128
    nlist = 128
    
    print(f"Test Parameters:")
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
    print("Generating random vectors...")
    data_gen_start = time.time()
    np.random.seed(42)  # Set random seed for reproducibility
    vecs = np.random.rand(dsize, dimension).astype(np.float32)
    data_gen_time = time.time() - data_gen_start
    print(f"  Vector shape: {vecs.shape}")
    print(f"  Data type: {vecs.dtype}")
    print(f"  ⏱ Data generation time: {data_gen_time:.4f}s")
    print()
    
    # Go: queries := vecs[:8192]
    queries = vecs[:8192]
    print(f"Query vectors:")
    print(f"  Query count: {len(queries)}")
    print(f"  Query shape: {queries.shape}")
    print()
    
    # Go: centers, err := getCenters(vecs, int(dimension), nlist, cuvs.DistanceL2, 10)
    # Go: require.NoError(t, err)
    print("Performing IVF-Flat KMeans clustering...")
    kmeans_start = time.time()
    try:
        centers = get_centers(vecs, dimension, nlist, "sqeuclidean", 10)
        kmeans_time = time.time() - kmeans_start
        print(f"  ✓ Clustering succeeded")
        print(f"  ⏱ KMeans clustering time: {kmeans_time:.4f}s")
        print(f"  Cluster centers shape: {centers.shape}")
        print(f"  Cluster centers type: {centers.dtype}")
    except Exception as e:
        print(f"  ✗ Clustering failed: {e}")
        raise
    print()
    
    # Validate clustering results
    assert centers.shape == (nlist, dimension), f"Incorrect cluster centers shape: {centers.shape}"
    assert centers.dtype == np.float32, f"Incorrect cluster centers type: {centers.dtype}"
    print(f"✓ Clustering results validation passed")
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
    print("Starting Concurrent Brute Force Search Test")
    print("=" * 70)
    print()
    
    num_threads = 4
    iterations_per_thread = 1000
    total_searches = num_threads * iterations_per_thread
    
    errors = []
    thread_times = []
    
    def worker_thread(thread_id: int):
        """Worker thread function"""
        thread_start = time.time()
        thread_errors = []
        
        try:
            for i in range(iterations_per_thread):
                try:
                    # Go: _, _, err := Search(centers, queries, limit, cuvs.DistanceL2)
                    neighbors, distances = search(centers, queries, limit, "sqeuclidean")
                    
                    # Validate results
                    assert neighbors.shape == (len(queries), limit), \
                        f"Incorrect neighbors shape: {neighbors.shape}"
                    assert distances.shape == (len(queries), limit), \
                        f"Incorrect distances shape: {distances.shape}"
                    
                    # Print progress every 100 iterations
                    if (i + 1) % 100 == 0:
                        elapsed = time.time() - thread_start
                        print(f"  Thread {thread_id}: Completed {i+1}/{iterations_per_thread} "
                              f"({(i+1)/iterations_per_thread*100:.1f}%) "
                              f"elapsed {elapsed:.2f}s")
                        
                except Exception as e:
                    thread_errors.append((i, str(e)))
                    
        except Exception as e:
            thread_errors.append((-1, f"Thread exception: {str(e)}"))
        
        thread_time = time.time() - thread_start
        thread_times.append(thread_time)
        
        if thread_errors:
            errors.extend([(thread_id, err) for err in thread_errors])
        
        print(f"  Thread {thread_id}: Completed all {iterations_per_thread} searches, "
              f"total time {thread_time:.4f}s")
    
    # Launch all threads
    print(f"Launching {num_threads} concurrent threads...")
    search_start = time.time()
    
    threads = []
    for n in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(n,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    search_time = time.time() - search_start
    
    print()
    print("=" * 70)
    
    # Check for errors
    if errors:
        print(f"✗ Test failed with {len(errors)} error(s):")
        for thread_id, (iter_id, error_msg) in errors[:10]:  # Show first 10 only
            print(f"  Thread {thread_id}, iteration {iter_id}: {error_msg}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more error(s)")
        raise RuntimeError(f"Test failed with {len(errors)} error(s)")
    
    print("✓ All search tests passed!")
    print("=" * 70)
    print()
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Performance statistics
    print("=" * 70)
    print("⏱  Performance Statistics")
    print("=" * 70)
    print(f"  Data generation time:     {data_gen_time:.4f}s")
    print(f"  KMeans clustering time:   {kmeans_time:.4f}s")
    print(f"  Concurrent search time:   {search_time:.4f}s")
    print(f"  " + "-" * 66)
    print(f"  Total time:               {total_time:.4f}s")
    print()
    print(f"Search Performance:")
    print(f"  Total searches:           {total_searches}")
    print(f"  Average thread time:      {np.mean(thread_times):.4f}s")
    print(f"  Fastest thread time:      {np.min(thread_times):.4f}s")
    print(f"  Slowest thread time:      {np.max(thread_times):.4f}s")
    print(f"  Average search time:      {search_time/total_searches*1000:.4f}ms")
    print(f"  Effective throughput:     {total_searches/search_time:.2f} searches/s")
    print(f"  Query vector throughput:  {total_searches*len(queries)/search_time:.0f} vectors/s")
    print("=" * 70)
    print("✓ TestIvfAndBruteForceForIssue test passed!")
    print("=" * 70)


def main():
    """Main function"""
    try:
        TestIvfAndBruteForceForIssue()
        return 0
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"✗ Assertion failed: {e}")
        print("=" * 70)
        return 1
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ Test failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
