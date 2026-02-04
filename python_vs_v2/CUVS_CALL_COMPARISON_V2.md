# cuVS API Call Detailed Comparison: Go vs Python

This document provides a detailed comparison of cuVS API calls in `issue_test.go` and `test_issue_v2.py` for the `TestIvfAndBruteForceForIssue` test, listing each method call, corresponding interface, and underlying C API.

---

## Table of Contents

1. [getCenters / get_centers Function Comparison](#1-getcenters--get_centers-function-comparison)
2. [Search / search Function Comparison](#2-search--search-function-comparison)
3. [TestIvfAndBruteForceForIssue Main Workflow Comparison](#3-testivfandbruteforceforissue-main-workflow-comparison)
4. [Key Differences Summary](#4-key-differences-summary)

---

## 1. getCenters / get_centers Function Comparison

### 1.1 Resource Manager Creation

| Go Code (line 35) | Python Code (line 55-56) | Underlying C API |
|-------------------|--------------------------|------------------|
| `resource, err := cuvs.NewResource(nil)` | `# Python cuVS automatically manages resources` | `cuvsResourcesCreate()` |
| `defer resource.Close()` (line 39) | No explicit close needed | `cuvsResourcesDestroy()` |

**Underlying C API**:
```c
// cuvs/c/core.h
cuvsError_t cuvsResourcesCreate(cuvsResources_t* res);
cuvsError_t cuvsResourcesDestroy(cuvsResources_t res);
```

**Difference**:
- ✅ Go: Explicit resource creation and destruction
- ✅ Python: Automatic management, uses default resources or optional `resources` parameter
- 🔍 **Underlying call**: Go explicitly calls C API, Python calls internally or uses default resources

---

### 1.2 IVF-Flat Index Parameters Creation

| Go Code (line 41-50) | Python Code (line 59-73) | Underlying C API |
|----------------------|--------------------------|------------------|
| `indexParams, err := ivf_flat.CreateIndexParams()` | `index_params = ivf_flat.IndexParams(...)` | `cuvsIvfFlatIndexParamsCreate()` |
| `defer indexParams.Close()` (line 45) | No explicit close needed | `cuvsIvfFlatIndexParamsDestroy()` |

**Underlying C API**:
```c
// cuvs/c/neighbors/ivf_flat.h
cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t* params);
cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t params);
```

**Difference**:
- ✅ Go: Create empty parameter object first, then set each parameter
- ✅ Python: Pass all parameters directly to constructor
- 🔍 **Underlying call**: Both call the same C API, but encapsulation differs

---

### 1.3 Setting Index Parameters

#### 1.3.1 Set Number of Clusters (n_lists)

| Go Code (line 47) | Python Code (line 61) | Underlying C API |
|-------------------|----------------------|------------------|
| `indexParams.SetNLists(uint32(clusterCnt))` | `n_lists=cluster_cnt` | `cuvsIvfFlatIndexParamsSetNLists()` |

**Underlying C API**:
```c
cuvsError_t cuvsIvfFlatIndexParamsSetNLists(
    cuvsIvfFlatIndexParams_t params,
    uint32_t n_lists
);
```

#### 1.3.2 Set Distance Metric

| Go Code (line 48) | Python Code (line 64) | Underlying C API |
|-------------------|----------------------|------------------|
| `indexParams.SetMetric(distanceType)` | `metric=distance_type` | `cuvsIvfFlatIndexParamsSetMetric()` |
| Input: `cuvs.DistanceL2` | Input: `"sqeuclidean"` | |

**Underlying C API**:
```c
cuvsError_t cuvsIvfFlatIndexParamsSetMetric(
    cuvsIvfFlatIndexParams_t params,
    cuvsDistanceType metric
);
```

**⚠️ Important Difference**:
- Go: `cuvs.DistanceL2` → Maps to C enum `CUVS_DISTANCE_L2_SQUARED`
- Python: `"sqeuclidean"` → Maps to C enum `CUVS_DISTANCE_L2_SQUARED`
- 🔍 **Underlying same**: Both map to `CUVS_DISTANCE_L2_SQUARED`

#### 1.3.3 Set KMeans Iteration Count

| Go Code (line 49) | Python Code (line 67) | Underlying C API |
|-------------------|----------------------|------------------|
| `indexParams.SetKMeansNIters(uint32(maxIterations))` | `kmeans_n_iters=max_iterations` | `cuvsIvfFlatIndexParamsSetKMeansNIters()` |
| Input: `10` | Input: `10` | |

**Underlying C API**:
```c
cuvsError_t cuvsIvfFlatIndexParamsSetKMeansNIters(
    cuvsIvfFlatIndexParams_t params,
    uint32_t kmeans_n_iters
);
```

#### 1.3.4 Set KMeans Training Set Fraction

| Go Code (line 50) | Python Code (line 70) | Underlying C API |
|-------------------|----------------------|------------------|
| `indexParams.SetKMeansTrainsetFraction(1)` | `kmeans_trainset_fraction=1.0` | `cuvsIvfFlatIndexParamsSetKMeansTrainsetFraction()` |

**Underlying C API**:
```c
cuvsError_t cuvsIvfFlatIndexParamsSetKMeansTrainsetFraction(
    cuvsIvfFlatIndexParams_t params,
    float kmeans_trainset_fraction
);
```

#### 1.3.5 Python-Specific Parameter

| Go Code | Python Code (line 72) | Underlying C API |
|---------|----------------------|------------------|
| No such parameter | `add_data_on_build=True` | No direct correspondence (Python binding feature) |

**Difference**:
- ❌ Go: No such parameter
- ✅ Python: Controls whether to add data when building index
- 🔍 **Does not affect underlying**: This is a Python binding layer convenience feature

---

### 1.4 Dataset Tensor Creation and Transfer

| Go Code (line 52-63) | Python Code (line 77) | Underlying C API |
|----------------------|----------------------|------------------|
| `dataset, err := cuvs.NewTensor(vecs)` | No such step | `cuvsTensorCreate()` (on host) |
| `defer dataset.Close()` (line 56) | No explicit close needed | `cuvsTensorDestroy()` |
| `dataset.ToDevice(&resource)` (line 61) | `dataset = cp.asarray(vecs, dtype=cp.float32)` | `cudaMemcpy()` (internal call) |

**Underlying C API**:
```c
// cuvs/c/core.h
cuvsError_t cuvsTensorCreate(
    cuvsTensor_t* tensor,
    void* data,
    int64_t* shape,
    int32_t rank,
    cuvsDLDataType dtype
);

// CUDA Runtime API
cudaError_t cudaMemcpy(
    void* dst,
    const void* src,
    size_t count,
    cudaMemcpyKind kind
);
```

**Important Difference**:
- Go: **Two steps**
  1. Create tensor on host (`cuvs.NewTensor`)
  2. Explicitly transfer to device (`ToDevice`)
- Python: **One step**
  - Use CuPy to create array directly on GPU (`cp.asarray`)
- 🔍 **Underlying differs**: 
  - Go: Host allocation → GPU copy
  - Python: Direct GPU allocation (if data in NumPy) or GPU copy (if already in CuPy)

---

### 1.5 Index Creation

| Go Code (line 58-59) | Python Code | Underlying C API |
|----------------------|-------------|------------------|
| `index, _ := ivf_flat.CreateIndex(indexParams, &dataset)` | No explicit creation step | `cuvsIvfFlatIndexCreate()` |
| `defer index.Close()` (line 59) | No explicit close needed | `cuvsIvfFlatIndexDestroy()` |

**Underlying C API**:
```c
cuvsError_t cuvsIvfFlatIndexCreate(cuvsIvfFlatIndex_t* index);
cuvsError_t cuvsIvfFlatIndexDestroy(cuvsIvfFlatIndex_t index);
```

**Difference**:
- ✅ Go: Explicitly create empty index object
- ✅ Python: Automatically created inside `build()` function
- 🔍 **Underlying same**: Both call `cuvsIvfFlatIndexCreate()`

---

### 1.6 Cluster Centers Tensor Creation

| Go Code (line 65-68) | Python Code | Underlying C API |
|----------------------|-------------|------------------|
| `centers, err := cuvs.NewTensorOnDevice[float32](&resource, []int64{int64(clusterCnt), int64(dim)})` | No explicit creation (automatic) | `cuvsTensorCreate()` + GPU allocation |

**Underlying C API**:
```c
cuvsError_t cuvsTensorCreate(cuvsTensor_t* tensor, ...);
// + cudaMalloc() for device memory
```

**Difference**:
- ✅ Go: Explicitly pre-allocate GPU memory for storing cluster centers
- ✅ Python: Index object automatically manages cluster centers memory
- 🔍 **Underlying differs**: Go manual memory management, Python automatic

---

### 1.7 Build Index (Execute KMeans)

| Go Code (line 70-72) | Python Code (line 81) | Underlying C API |
|----------------------|----------------------|------------------|
| `ivf_flat.BuildIndex(resource, indexParams, &dataset, index)` | `index = ivf_flat.build(index_params, dataset)` | `cuvsIvfFlatBuild()` |

**Underlying C API**:
```c
// cuvs/c/neighbors/ivf_flat.h
cuvsError_t cuvsIvfFlatBuild(
    cuvsResources_t res,
    cuvsIvfFlatIndexParams_t params,
    cuvsTensor_t* dataset,
    cuvsIvfFlatIndex_t index
);
```

**Difference**:
- Go: Parameter order `(resource, params, dataset, index)`, index as output parameter
- Python: Parameter order `(params, dataset)`, returns index object
- 🔍 **Underlying same**: Both call `cuvsIvfFlatBuild()`, execute same KMeans algorithm

---

### 1.8 Synchronize GPU

| Go Code (line 74-76) | Python Code (line 84) | Underlying C API |
|----------------------|----------------------|------------------|
| `resource.Sync()` | `cp.cuda.Stream.null.synchronize()` | `cuvsResourcesSync()` or `cudaStreamSynchronize()` |

**Underlying C API**:
```c
// cuVS API
cuvsError_t cuvsResourcesSync(cuvsResources_t res);

// CUDA Runtime API
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
```

**Difference**:
- Go: Sync through cuVS resource object
- Python: Directly call CUDA stream sync
- 🔍 **Underlying similar**: Both ensure GPU operations complete

---

### 1.9 Get Cluster Centers

| Go Code (line 78-80) | Python Code (line 87) | Underlying C API |
|----------------------|----------------------|------------------|
| `ivf_flat.GetCenters(index, &centers)` | `centers_gpu = index.centers` | `cuvsIvfFlatIndexGetCenters()` |

**Underlying C API**:
```c
// cuvs/c/neighbors/ivf_flat.h
cuvsError_t cuvsIvfFlatIndexGetCenters(
    cuvsIvfFlatIndex_t index,
    cuvsTensor_t* centers
);
```

**Important Difference**:
- Go: **Function call** `GetCenters(index, &centers)`, requires pre-allocated centers tensor
- Python: **Property access** `index.centers`, returns CuPy array
- 🔍 **Underlying same**: Both call `cuvsIvfFlatIndexGetCenters()`

---

### 1.10 Transfer Cluster Centers Back to Host

| Go Code (line 82-84) | Python Code (line 90) | Underlying C API |
|----------------------|----------------------|------------------|
| `centers.ToHost(&resource)` | `centers = cp.asnumpy(centers_gpu)` | `cudaMemcpy()` (device to host) |

**Underlying C API**:
```c
cudaError_t cudaMemcpy(
    void* dst,        // host pointer
    const void* src,  // device pointer
    size_t count,
    cudaMemcpyKind kind  // cudaMemcpyDeviceToHost
);
```

**Difference**:
- Go: Explicitly call `ToHost()` method
- Python: Use CuPy's `asnumpy()` method
- 🔍 **Underlying same**: Both call `cudaMemcpy(... cudaMemcpyDeviceToHost)`

---

### 1.11 Final Synchronization

| Go Code (line 86-88) | Python Code (line 93) | Underlying C API |
|----------------------|----------------------|------------------|
| `resource.Sync()` | `cp.cuda.Stream.null.synchronize()` | `cuvsResourcesSync()` or `cudaStreamSynchronize()` |

**Difference**: Same as 1.8

---

### 1.12 Return Results

| Go Code (line 90-95) | Python Code (line 97) | Underlying Operation |
|----------------------|----------------------|---------------------|
| `result, err := centers.Slice()` | `return centers.astype(np.float32)` | Data type enforcement |
| `return result, nil` | | |

**Difference**:
- Go: `Slice()` method converts tensor to Go 2D slice `[][]float32`
- Python: Directly return NumPy array, ensure type is `float32`
- 🔍 **No underlying C call**: This is language-level data structure conversion

---

## 2. Search / search Function Comparison

### 2.1 Resource Manager Creation

| Go Code (line 103-107) | Python Code (line 122-123) | Underlying C API |
|------------------------|---------------------------|------------------|
| `resource, err := cuvs.NewResource(nil)` | `# Python cuVS automatically manages resources` | `cuvsResourcesCreate()` |
| `defer resource.Close()` | No explicit close needed | `cuvsResourcesDestroy()` |

**Difference**: Same as 1.1

---

### 2.2 Dataset Tensor Creation and Transfer

| Go Code (line 109-113, 139-141) | Python Code (line 127) | Underlying C API |
|---------------------------------|------------------------|------------------|
| `dataset, err := cuvs.NewTensor(datasetvec)` | No such step | `cuvsTensorCreate()` |
| `defer dataset.Close()` | No explicit close needed | `cuvsTensorDestroy()` |
| `dataset.ToDevice(&resource)` | `dataset = cp.asarray(dataset_vec, dtype=cp.float32)` | `cudaMemcpy()` |

**Difference**: Same as 1.4

---

### 2.3 Query Tensor Creation and Transfer

| Go Code (line 121-125, 159-161) | Python Code (line 131) | Underlying C API |
|---------------------------------|------------------------|------------------|
| `queries, err := cuvs.NewTensor(queriesvec)` | No such step | `cuvsTensorCreate()` |
| `defer queries.Close()` | No explicit close needed | `cuvsTensorDestroy()` |
| `queries.ToDevice(&resource)` | `queries = cp.asarray(queries_vec, dtype=cp.float32)` | `cudaMemcpy()` |

**Difference**: Same as 1.4

---

### 2.4 Brute Force Index Creation

| Go Code (line 115-119) | Python Code | Underlying C API |
|------------------------|-------------|------------------|
| `index, err := brute_force.CreateIndex()` | No explicit creation | `cuvsBruteForceIndexCreate()` |
| `defer index.Close()` | No explicit close needed | `cuvsBruteForceIndexDestroy()` |

**Underlying C API**:
```c
// cuvs/c/neighbors/brute_force.h
cuvsError_t cuvsBruteForceIndexCreate(cuvsBruteForceIndex_t* index);
cuvsError_t cuvsBruteForceIndexDestroy(cuvsBruteForceIndex_t index);
```

**Difference**:
- ✅ Go: Explicitly create empty index object
- ✅ Python: Automatically created inside `build()` function
- 🔍 **Underlying same**: Both call same C API

---

### 2.5 Output Tensor Creation (neighbors and distances)

| Go Code (line 127-137) | Python Code | Underlying C API |
|------------------------|-------------|------------------|
| `neighbors, err := cuvs.NewTensorOnDevice[int64](&resource, []int64{int64(len(queriesvec)), int64(limit)})` | No explicit creation | `cuvsTensorCreate()` + `cudaMalloc()` |
| `defer neighbors.Close()` | No explicit close needed | `cuvsTensorDestroy()` + `cudaFree()` |
| `distances, err := cuvs.NewTensorOnDevice[float32](&resource, []int64{int64(len(queriesvec)), int64(limit)})` | No explicit creation | `cuvsTensorCreate()` + `cudaMalloc()` |
| `defer distances.Close()` | No explicit close needed | `cuvsTensorDestroy()` + `cudaFree()` |

**Underlying C API**:
```c
cuvsError_t cuvsTensorCreate(...);
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
```

**Important Difference**:
- ✅ Go: **Explicitly pre-allocate** GPU memory for output
  - neighbors: `int64` type, shape `(n_queries, k)`
  - distances: `float32` type, shape `(n_queries, k)`
- ✅ Python: **Automatically allocate** output memory (inside `search()`)
- 🔍 **Underlying same**: Both need GPU memory allocation, but timing differs

---

### 2.6 Build Brute Force Index

| Go Code (line 147-152) | Python Code (line 136) | Underlying C API |
|------------------------|------------------------|------------------|
| `brute_force.BuildIndex(resource, &dataset, distanceType, 2.0, index)` | `index = brute_force.build(dataset, metric=distance_type)` | `cuvsBruteForceBuild()` |

**Underlying C API**:
```c
// cuvs/c/neighbors/brute_force.h
cuvsError_t cuvsBruteForceBuild(
    cuvsResources_t res,
    cuvsTensor_t* dataset,
    cuvsDistanceType metric,
    float metric_arg,
    cuvsBruteForceIndex_t index
);
```

**Parameter Comparison**:

| Parameter | Go | Python | Description |
|-----------|-----|--------|-------------|
| resource | `resource` | Automatic | cuVS resource |
| dataset | `&dataset` | `dataset` | Dataset tensor |
| metric | `distanceType` (cuvs.DistanceL2) | `metric=distance_type` ("sqeuclidean") | Distance metric |
| metric_arg | `2.0` | Default `2.0` | Distance parameter |
| index | `index` | Auto-created and returned | Index object |

**Difference**:
- Go: Parameter `metric_arg=2.0` **explicitly passed**
- Python: Parameter `metric_arg` **uses default value** 2.0
- 🔍 **Underlying same**: Both call `cuvsBruteForceBuild()`, metric_arg is 2.0

---

### 2.7 Synchronize GPU

| Go Code (line 154-156) | Python Code (line 139) | Underlying C API |
|------------------------|------------------------|------------------|
| `resource.Sync()` | `cp.cuda.Stream.null.synchronize()` | `cuvsResourcesSync()` or `cudaStreamSynchronize()` |

**Difference**: Same as 1.8

---

### 2.8 Execute Search

| Go Code (line 164-167) | Python Code (line 144-148) | Underlying C API |
|------------------------|---------------------------|------------------|
| `brute_force.SearchIndex(resource, *index, &queries, &neighbors, &distances)` | `distances_gpu, neighbors_gpu = brute_force.search(index, queries, limit)` | `cuvsBruteForceSearch()` |

**Underlying C API**:
```c
// cuvs/c/neighbors/brute_force.h
cuvsError_t cuvsBruteForceSearch(
    cuvsResources_t res,
    cuvsBruteForceIndex_t index,
    cuvsTensor_t* queries,
    cuvsTensor_t* neighbors,
    cuvsTensor_t* distances
);
```

**⭐ Key Differences**:

| Aspect | Go | Python |
|--------|-----|--------|
| Function name | `SearchIndex()` | `search()` |
| Parameter count | 5 | 3 (+ optional) |
| resource | Explicitly pass `resource` | Automatic or optional `resources=` |
| index | Dereference `*index` | Direct pass `index` |
| queries | Pointer `&queries` | CuPy array `queries` |
| k (limit) | Implicit (inferred from neighbors shape) | Explicit `limit` parameter |
| neighbors | Pre-allocated pointer `&neighbors` | Auto-allocated, returned by function |
| distances | Pre-allocated pointer `&distances` | Auto-allocated, returned by function |
| Return value | `error` | `(distances, neighbors)` tuple |

**🔍 Underlying same**: Both call `cuvsBruteForceSearch()`

---

### 2.9 Transfer Results Back to Host

| Go Code (line 170-178) | Python Code (line 152-153) | Underlying C API |
|------------------------|---------------------------|------------------|
| `neighbors.ToHost(&resource)` | `neighbors = cp.asnumpy(neighbors_gpu)` | `cudaMemcpy(... cudaMemcpyDeviceToHost)` |
| `distances.ToHost(&resource)` | `distances = cp.asnumpy(distances_gpu)` | `cudaMemcpy(... cudaMemcpyDeviceToHost)` |

**Difference**: Same as 1.10

---

### 2.10 Final Synchronization

| Go Code (line 180-182) | Python Code (line 156) | Underlying C API |
|------------------------|------------------------|------------------|
| `resource.Sync()` | `cp.cuda.Stream.null.synchronize()` | `cuvsResourcesSync()` or `cudaStreamSynchronize()` |

**Difference**: Same as 1.8

---

### 2.11 Result Conversion

| Go Code (line 185-209) | Python Code (line 162) | Operation |
|------------------------|------------------------|-----------|
| `neighborsSlice, err := neighbors.Slice()` | `return neighbors.astype(np.int64), ...` | Data format conversion |
| `distancesSlice, err := distances.Slice()` | `return ..., distances.astype(np.float64)` | Data format conversion |
| Flatten to 1D array (line 196-208) | Keep 2D array | **Important difference** |

**⭐ Key Difference**:

#### Go Return Format:
```go
// neighbors: []int64 (1D flat array)
// length = n_queries × limit
// access: keys[i*limit + j]

// distances: []float64 (1D flat array)
// length = n_queries × limit
// access: retdistances[i*limit + j]
```

#### Python Return Format:
```python
# neighbors: np.ndarray (2D array)
# shape = (n_queries, limit)
# access: neighbors[i, j]

# distances: np.ndarray (2D array)
# shape = (n_queries, limit)
# access: distances[i, j]
```

**🔍 Data content same, only organization differs**

---

## 3. TestIvfAndBruteForceForIssue Main Workflow Comparison

### 3.1 Test Parameter Setup

| Parameter | Go Code (line 216-224) | Python Code (line 187-196) | Value |
|-----------|----------------------|---------------------------|--------|
| dimension | `dimension := uint(128)` | `dimension = 128` | 128 |
| limit | `limit := uint(1)` | `limit = 1` | 1 |
| dsize | `dsize := 100000` | `dsize = 100000` | 100000 |
| nlist | `nlist := 128` | `nlist = 128` | 128 |

**✅ Completely identical**

---

### 3.2 Random Vector Generation

| Go Code (line 225-231) | Python Code (line 214-221) | Underlying Operation |
|------------------------|---------------------------|---------------------|
| Double for loop, generate one by one | NumPy vectorized operation | Random number generation |
| `rand.Float32()` | `np.random.rand()` | |
| **No random seed** | `np.random.seed(42)` ⚠️ | **Difference** |

**⚠️ Key Difference**:
- Go: **No random seed**, generates different data each run
- Python: **Sets random seed** `np.random.seed(42)`, results reproducible
- 🔍 **Impact**: Both generate **different data**, but this doesn't affect functional testing correctness

---

### 3.3 Query Vector Selection

| Go Code (line 232) | Python Code (line 225) |
|-------------------|----------------------|
| `queries := vecs[:8192]` | `queries = vecs[:8192]` |

**✅ Completely identical**: Both take first 8192 vectors as queries

---

### 3.4 KMeans Clustering Call

| Go Code (line 234-235) | Python Code (line 236) | cuVS Call |
|----------------------|----------------------|-----------|
| `centers, err := getCenters(vecs, int(dimension), nlist, cuvs.DistanceL2, 10)` | `centers = get_centers(vecs, dimension, nlist, "sqeuclidean", 10)` | See Section 1 |
| `require.NoError(t, err)` | `try ... except ...` | Different error handling |

**Parameter Comparison**:

| Parameter | Go | Python |
|-----------|-----|--------|
| vecs | `[][]float32` | `np.ndarray (100000, 128)` |
| dimension | `int(128)` | `128` |
| nlist | `128` | `128` |
| distance_type | `cuvs.DistanceL2` | `"sqeuclidean"` |
| max_iterations | `10` | `10` |

**✅ Parameters equivalent**, distance types both correspond to L2 squared distance

---

### 3.5 Concurrent Search Test

#### 3.5.1 Concurrency Model

| Go Code (line 237-262) | Python Code (line 271-331) | Concurrency Primitive |
|------------------------|---------------------------|---------------------|
| `sync.WaitGroup` | `threading.Thread` | |
| `go func() { ... }()` | `threading.Thread(target=...).start()` | |
| `wg.Wait()` | `thread.join()` | |

**Difference**:
- Go: **Goroutine** (lightweight coroutine)
- Python: **Thread** (OS thread)
- 🔍 **Same concurrency count**: Both use 4 concurrent execution units

#### 3.5.2 Iteration Count

| Go Code (line 244) | Python Code (line 272) |
|-------------------|----------------------|
| `for i := 0; i < 1000; i++` | `iterations_per_thread = 1000` |

**✅ Completely identical**: Each concurrent unit executes 1000 searches

#### 3.5.3 Search Call

| Go Code (line 245-246) | Python Code (line 287) | cuVS Call |
|----------------------|----------------------|-----------|
| `_, _, err := Search(centers, queries, limit, cuvs.DistanceL2)` | `neighbors, distances = search(centers, queries, limit, "sqeuclidean")` | See Section 2 |
| `require.NoError(t, err)` | `assert neighbors.shape == ...` | Different validation |

**Parameter Comparison**:

| Parameter | Go | Python |
|-----------|-----|--------|
| dataset | `centers` (128, 128) | `centers` (128, 128) |
| queries | `queries` (8192, 128) | `queries` (8192, 128) |
| limit | `uint(1)` | `1` |
| distance_type | `cuvs.DistanceL2` | `"sqeuclidean"` |

**✅ Parameters equivalent**

---

## 4. Key Differences Summary

### 4.1 API Encapsulation Style Differences

| Aspect | Go | Python | Impact |
|--------|-----|--------|--------|
| **Resource Management** | Explicit create/destroy | Automatic | Code verbosity |
| **Memory Allocation** | Explicit pre-allocate output | Automatic allocation | Code flexibility |
| **Error Handling** | Return `error` | Raise exception | Error handling approach |
| **Parameter Passing** | Pointer/reference | Value passing | Language feature |
| **Synchronization** | `resource.Sync()` | `cp.cuda.Stream.null.synchronize()` | API abstraction level |

### 4.2 cuVS Call Differences Summary

| cuVS Operation | Go API | Python API | Underlying C API | Same? |
|---------------|--------|-----------|-----------------|-------|
| Create resource | `cuvs.NewResource()` | Automatic | `cuvsResourcesCreate()` | ✅ |
| Create index params | `ivf_flat.CreateIndexParams()` + Setters | `ivf_flat.IndexParams(...)` | `cuvsIvfFlatIndexParamsCreate()` + Setters | ✅ |
| Create tensor | `cuvs.NewTensor()` + `ToDevice()` | `cp.asarray()` | `cuvsTensorCreate()` + `cudaMemcpy()` | ⚠️ Different approach |
| Build IVF index | `ivf_flat.BuildIndex()` | `ivf_flat.build()` | `cuvsIvfFlatBuild()` | ✅ |
| Get cluster centers | `ivf_flat.GetCenters()` | `index.centers` | `cuvsIvfFlatIndexGetCenters()` | ✅ |
| Build Brute Force | `brute_force.BuildIndex()` | `brute_force.build()` | `cuvsBruteForceBuild()` | ✅ |
| Search | `brute_force.SearchIndex()` | `brute_force.search()` | `cuvsBruteForceSearch()` | ✅ |
| Sync GPU | `resource.Sync()` | `cp.cuda.Stream.null.synchronize()` | `cuvsResourcesSync()` | ✅ |
| Transfer to host | `tensor.ToHost()` | `cp.asnumpy()` | `cudaMemcpy(D2H)` | ✅ |

**Conclusion**: 
- ✅ **Underlying C API calls completely identical**
- ⚠️ **Encapsulation differs**, but doesn't affect functional equivalence

### 4.3 Test Data Differences

| Data | Go | Python | Impact |
|------|-----|--------|--------|
| **Random seed** | None | `np.random.seed(42)` | ⚠️ Different data |
| **Data generation** | Generate one by one | NumPy vectorized | Performance difference |
| **Data size** | 100,000 × 128 | 100,000 × 128 | ✅ Same |
| **Query count** | 8192 | 8192 | ✅ Same |
| **Cluster count** | 128 | 128 | ✅ Same |

**⚠️ Important**: Due to different random seeds, Go and Python generate **different data**, but this doesn't affect functional testing correctness.

### 4.4 Result Format Differences

| Output | Go | Python | Comparable? |
|--------|-----|--------|------------|
| **neighbors** | 1D `[]int64` | 2D `np.ndarray (n_queries, k)` | ⚠️ Different format |
| **distances** | 1D `[]float64` | 2D `np.ndarray (n_queries, k)` | ⚠️ Different format |
| **Data content** | Same (if inputs same) | Same (if inputs same) | ✅ Content equivalent |

### 4.5 Performance Monitoring Differences

| Feature | Go | Python |
|---------|-----|--------|
| Progress output | ❌ None | ✅ Every 100 iterations |
| Performance stats | ❌ None | ✅ Detailed (time, throughput) |
| Error collection | ❌ Fail immediately | ✅ Collect all errors |
| Time measurement | ❌ None | ✅ Phased timing |

---

## 5. Complete cuVS C API Call Chain Summary

### getCenters / get_centers

```
1. cuvsResourcesCreate()              (Go explicit, Python automatic)
2. cuvsIvfFlatIndexParamsCreate()     (Go explicit, Python automatic)
3. cuvsIvfFlatIndexParamsSetNLists()
4. cuvsIvfFlatIndexParamsSetMetric()
5. cuvsIvfFlatIndexParamsSetKMeansNIters()
6. cuvsIvfFlatIndexParamsSetKMeansTrainsetFraction()
7. cuvsTensorCreate() + cudaMemcpy()  (Create and transfer dataset)
8. cuvsIvfFlatIndexCreate()           (Go explicit, Python automatic)
9. cuvsIvfFlatBuild()                 ⭐ Core: Execute KMeans
10. cuvsResourcesSync()
11. cuvsIvfFlatIndexGetCenters()
12. cudaMemcpy(D2H)                   (Transfer cluster centers to host)
13. cuvsResourcesSync()
14. cuvsResourcesDestroy()            (Go explicit, Python automatic)
```

### Search / search

```
1. cuvsResourcesCreate()              (Go explicit, Python automatic)
2. cuvsTensorCreate() + cudaMemcpy()  (Dataset and queries)
3. cuvsBruteForceIndexCreate()        (Go explicit, Python automatic)
4. cuvsTensorCreate() + cudaMalloc()  (Go pre-allocate output, Python automatic)
5. cuvsBruteForceBuild()              ⭐ Core: Build index
6. cuvsResourcesSync()
7. cuvsBruteForceSearch()             ⭐ Core: Search
8. cudaMemcpy(D2H)                    (Transfer results to host)
9. cuvsResourcesSync()
10. cuvsResourcesDestroy()            (Go explicit, Python automatic)
```

---

## 6. Equivalence Verification

### ✅ Core Algorithm Equivalence

1. **KMeans Clustering**: Both call `cuvsIvfFlatBuild()`, execute same IVF-Flat KMeans algorithm
2. **Brute Force Search**: Both call `cuvsBruteForceSearch()`, execute same brute force search algorithm
3. **Distance Metric**: `cuvs.DistanceL2` and `"sqeuclidean"` both map to `CUVS_DISTANCE_L2_SQUARED`
4. **Parameter Settings**: All key parameters (n_lists, iterations, metric) completely identical

### ⚠️ Non-Core Differences

1. **Random Data**: Different (but doesn't affect functional testing)
2. **Encapsulation**: Different (but underlying calls identical)
3. **Result Format**: Different (1D vs 2D, but content equivalent)
4. **Performance Monitoring**: Python more detailed

### 🎯 Conclusion

**Go and Python tests are functionally completely equivalent**, both correctly calling cuVS C API and executing the same algorithm logic. Differences only in:
- Language binding encapsulation
- Resource management abstraction level
- Test data generation method (random seed)
- Result format organization

**Core test logic and cuVS calls completely consistent, test results are comparable.**

---

**Created**: 2026-02-04  
**Version**: 2.0  
**Corresponds to Go file**: `pkg/vectorindex/ivfflat/kmeans/device/issue_test.go`  
**Corresponds to Python file**: `python_vs_v2/test_issue_v2.py`
