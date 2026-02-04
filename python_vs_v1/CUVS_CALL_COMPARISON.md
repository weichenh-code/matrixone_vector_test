# cuVS API 调用详细对比：Go vs Python

本文档详细对比 `issue_test.go` 和 `test_issue.py` 中 `TestIvfAndBruteForceForIssue` 的每个 cuVS API 调用，并列出对应的底层 C API。

---

## 目录

1. [getCenters / get_centers 函数对比](#1-getcenters--get_centers-函数对比)
2. [Search / search 函数对比](#2-search--search-函数对比)
3. [TestIvfAndBruteForceForIssue 主流程对比](#3-testivfandbruteforceforissue-主流程对比)
4. [关键差异总结](#4-关键差异总结)

---

## 1. getCenters / get_centers 函数对比

### 1.1 资源管理器创建

| Go 代码 (行35) | Python 代码 (行55-56) | 底层 C API |
|----------------|----------------------|-----------|
| `resource, err := cuvs.NewResource(nil)` | `# Python cuVS 自动管理资源` | `cuvsResourcesCreate()` |
| `defer resource.Close()` (行39) | 无需显式关闭 | `cuvsResourcesDestroy()` |

**底层 C API**:
```c
// cuvs/c/core.h
cuvsError_t cuvsResourcesCreate(cuvsResources_t* res);
cuvsError_t cuvsResourcesDestroy(cuvsResources_t res);
```

**差异**:
- ✅ Go: 显式创建和销毁资源
- ✅ Python: 自动管理，使用默认资源或可选传入 `resources` 参数
- 🔍 **底层调用**: Go 显式调用 C API，Python 在内部调用或使用默认资源

---

### 1.2 IVF-Flat 索引参数创建

| Go 代码 (行41-50) | Python 代码 (行59-73) | 底层 C API |
|-------------------|----------------------|-----------|
| `indexParams, err := ivf_flat.CreateIndexParams()` | `index_params = ivf_flat.IndexParams(...)` | `cuvsIvfFlatIndexParamsCreate()` |
| `defer indexParams.Close()` (行45) | 无需显式关闭 | `cuvsIvfFlatIndexParamsDestroy()` |

**底层 C API**:
```c
// cuvs/c/neighbors/ivf_flat.h
cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t* params);
cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t params);
```

**差异**:
- ✅ Go: 先创建空参数对象，再逐个设置
- ✅ Python: 直接用构造函数传递所有参数
- 🔍 **底层调用**: 都调用相同的 C API，但封装方式不同

---

### 1.3 设置索引参数

#### 1.3.1 设置聚类数量 (n_lists)

| Go 代码 (行47) | Python 代码 (行61) | 底层 C API |
|----------------|-------------------|-----------|
| `indexParams.SetNLists(uint32(clusterCnt))` | `n_lists=cluster_cnt` | `cuvsIvfFlatIndexParamsSetNLists()` |

**底层 C API**:
```c
cuvsError_t cuvsIvfFlatIndexParamsSetNLists(
    cuvsIvfFlatIndexParams_t params,
    uint32_t n_lists
);
```

#### 1.3.2 设置距离度量 (metric)

| Go 代码 (行48) | Python 代码 (行64) | 底层 C API |
|----------------|-------------------|-----------|
| `indexParams.SetMetric(distanceType)` | `metric=distance_type` | `cuvsIvfFlatIndexParamsSetMetric()` |
| 传入: `cuvs.DistanceL2` | 传入: `"sqeuclidean"` | |

**底层 C API**:
```c
cuvsError_t cuvsIvfFlatIndexParamsSetMetric(
    cuvsIvfFlatIndexParams_t params,
    cuvsDistanceType metric
);
```

**⚠️ 重要差异**:
- Go: `cuvs.DistanceL2` → 对应 C 枚举 `CUVS_DISTANCE_L2_SQUARED`
- Python: `"sqeuclidean"` → 对应 C 枚举 `CUVS_DISTANCE_L2_SQUARED`
- 🔍 **底层相同**: 都映射到 `CUVS_DISTANCE_L2_SQUARED`

#### 1.3.3 设置 KMeans 迭代次数

| Go 代码 (行49) | Python 代码 (行67) | 底层 C API |
|----------------|-------------------|-----------|
| `indexParams.SetKMeansNIters(uint32(maxIterations))` | `kmeans_n_iters=max_iterations` | `cuvsIvfFlatIndexParamsSetKMeansNIters()` |
| 传入: `10` | 传入: `10` | |

**底层 C API**:
```c
cuvsError_t cuvsIvfFlatIndexParamsSetKMeansNIters(
    cuvsIvfFlatIndexParams_t params,
    uint32_t kmeans_n_iters
);
```

#### 1.3.4 设置 KMeans 训练集比例

| Go 代码 (行50) | Python 代码 (行70) | 底层 C API |
|----------------|-------------------|-----------|
| `indexParams.SetKMeansTrainsetFraction(1)` | `kmeans_trainset_fraction=1.0` | `cuvsIvfFlatIndexParamsSetKMeansTrainsetFraction()` |

**底层 C API**:
```c
cuvsError_t cuvsIvfFlatIndexParamsSetKMeansTrainsetFraction(
    cuvsIvfFlatIndexParams_t params,
    float kmeans_trainset_fraction
);
```

#### 1.3.5 Python 独有参数

| Go 代码 | Python 代码 (行72) | 底层 C API |
|---------|-------------------|-----------|
| 无此参数 | `add_data_on_build=True` | 无直接对应（Python 绑定特性） |

**差异**:
- ❌ Go: 无此参数
- ✅ Python: 控制是否在构建索引时添加数据
- 🔍 **不影响底层**: 这是 Python 绑定层的便利特性

---

### 1.4 数据集张量创建和传输

| Go 代码 (行52-63) | Python 代码 (行77) | 底层 C API |
|-------------------|-------------------|-----------|
| `dataset, err := cuvs.NewTensor(vecs)` | 无此步骤 | `cuvsTensorCreate()` (在主机) |
| `defer dataset.Close()` (行56) | 无需显式关闭 | `cuvsTensorDestroy()` |
| `dataset.ToDevice(&resource)` (行61) | `dataset = cp.asarray(vecs, dtype=cp.float32)` | `cudaMemcpy()` (内部调用) |

**底层 C API**:
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

**重要差异**:
- Go: **两步**
  1. 在主机创建张量 (`cuvs.NewTensor`)
  2. 显式传输到设备 (`ToDevice`)
- Python: **一步**
  - 使用 CuPy 直接在 GPU 创建数组 (`cp.asarray`)
- 🔍 **底层不同**: 
  - Go: 主机分配 → GPU 拷贝
  - Python: 直接 GPU 分配（如果数据在 NumPy）或 GPU 拷贝（如果已在 CuPy）

---

### 1.5 索引创建

| Go 代码 (行58-59) | Python 代码 | 底层 C API |
|-------------------|-------------|-----------|
| `index, _ := ivf_flat.CreateIndex(indexParams, &dataset)` | 无显式创建步骤 | `cuvsIvfFlatIndexCreate()` |
| `defer index.Close()` (行59) | 无需显式关闭 | `cuvsIvfFlatIndexDestroy()` |

**底层 C API**:
```c
cuvsError_t cuvsIvfFlatIndexCreate(cuvsIvfFlatIndex_t* index);
cuvsError_t cuvsIvfFlatIndexDestroy(cuvsIvfFlatIndex_t index);
```

**差异**:
- ✅ Go: 显式创建空索引对象
- ✅ Python: 在 `build()` 函数内部自动创建
- 🔍 **底层相同**: 都调用 `cuvsIvfFlatIndexCreate()`

---

### 1.6 聚类中心张量创建

| Go 代码 (行65-68) | Python 代码 | 底层 C API |
|-------------------|-------------|-----------|
| `centers, err := cuvs.NewTensorOnDevice[float32](&resource, []int64{int64(clusterCnt), int64(dim)})` | 无显式创建（自动） | `cuvsTensorCreate()` + GPU 分配 |

**底层 C API**:
```c
cuvsError_t cuvsTensorCreate(cuvsTensor_t* tensor, ...);
// + cudaMalloc() for device memory
```

**差异**:
- ✅ Go: 显式预分配 GPU 内存用于存储聚类中心
- ✅ Python: 索引对象自动管理聚类中心内存
- 🔍 **底层差异**: Go 手动内存管理，Python 自动管理

---

### 1.7 构建索引（执行 KMeans）

| Go 代码 (行70-72) | Python 代码 (行81) | 底层 C API |
|-------------------|-------------------|-----------|
| `ivf_flat.BuildIndex(resource, indexParams, &dataset, index)` | `index = ivf_flat.build(index_params, dataset)` | `cuvsIvfFlatBuild()` |

**底层 C API**:
```c
// cuvs/c/neighbors/ivf_flat.h
cuvsError_t cuvsIvfFlatBuild(
    cuvsResources_t res,
    cuvsIvfFlatIndexParams_t params,
    cuvsTensor_t* dataset,
    cuvsIvfFlatIndex_t index
);
```

**差异**:
- Go: 参数顺序 `(resource, params, dataset, index)`，index 作为输出参数
- Python: 参数顺序 `(params, dataset)`，返回 index 对象
- 🔍 **底层相同**: 都调用 `cuvsIvfFlatBuild()`，执行相同的 KMeans 算法

---

### 1.8 同步 GPU

| Go 代码 (行74-76) | Python 代码 (行84) | 底层 C API |
|-------------------|-------------------|-----------|
| `resource.Sync()` | `cp.cuda.Stream.null.synchronize()` | `cuvsResourcesSync()` or `cudaStreamSynchronize()` |

**底层 C API**:
```c
// cuVS API
cuvsError_t cuvsResourcesSync(cuvsResources_t res);

// CUDA Runtime API
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
```

**差异**:
- Go: 通过 cuVS resource 对象同步
- Python: 直接调用 CUDA stream 同步
- 🔍 **底层类似**: 都确保 GPU 操作完成

---

### 1.9 获取聚类中心

| Go 代码 (行78-80) | Python 代码 (行87) | 底层 C API |
|-------------------|-------------------|-----------|
| `ivf_flat.GetCenters(index, &centers)` | `centers_gpu = index.centers` | `cuvsIvfFlatIndexGetCenters()` |

**底层 C API**:
```c
// cuvs/c/neighbors/ivf_flat.h
cuvsError_t cuvsIvfFlatIndexGetCenters(
    cuvsIvfFlatIndex_t index,
    cuvsTensor_t* centers
);
```

**重要差异**:
- Go: **函数调用** `GetCenters(index, &centers)`，需要预分配 centers 张量
- Python: **属性访问** `index.centers`，返回 CuPy 数组
- 🔍 **底层相同**: 都调用 `cuvsIvfFlatIndexGetCenters()`

---

### 1.10 传输聚类中心回主机

| Go 代码 (行82-84) | Python 代码 (行90) | 底层 C API |
|-------------------|-------------------|-----------|
| `centers.ToHost(&resource)` | `centers = cp.asnumpy(centers_gpu)` | `cudaMemcpy()` (device to host) |

**底层 C API**:
```c
cudaError_t cudaMemcpy(
    void* dst,        // host pointer
    const void* src,  // device pointer
    size_t count,
    cudaMemcpyKind kind  // cudaMemcpyDeviceToHost
);
```

**差异**:
- Go: 显式调用 `ToHost()` 方法
- Python: 使用 CuPy 的 `asnumpy()` 方法
- 🔍 **底层相同**: 都调用 `cudaMemcpy(... cudaMemcpyDeviceToHost)`

---

### 1.11 最终同步

| Go 代码 (行86-88) | Python 代码 (行93) | 底层 C API |
|-------------------|-------------------|-----------|
| `resource.Sync()` | `cp.cuda.Stream.null.synchronize()` | `cuvsResourcesSync()` or `cudaStreamSynchronize()` |

**差异**: 同 1.8

---

### 1.12 返回结果

| Go 代码 (行90-95) | Python 代码 (行97) | 底层操作 |
|-------------------|-------------------|----------|
| `result, err := centers.Slice()` | `return centers.astype(np.float32)` | 数据类型确保 |
| `return result, nil` | | |

**差异**:
- Go: `Slice()` 方法将张量转换为 Go 的二维切片 `[][]float32`
- Python: 直接返回 NumPy 数组，确保类型为 `float32`
- 🔍 **无底层 C 调用**: 这是语言层面的数据结构转换

---

## 2. Search / search 函数对比

### 2.1 资源管理器创建

| Go 代码 (行103-107) | Python 代码 (行122-123) | 底层 C API |
|---------------------|------------------------|-----------|
| `resource, err := cuvs.NewResource(nil)` | `# Python cuVS 自动管理资源` | `cuvsResourcesCreate()` |
| `defer resource.Close()` | 无需显式关闭 | `cuvsResourcesDestroy()` |

**差异**: 同 1.1

---

### 2.2 数据集张量创建和传输

| Go 代码 (行109-113, 139-141) | Python 代码 (行127) | 底层 C API |
|------------------------------|-------------------|-----------|
| `dataset, err := cuvs.NewTensor(datasetvec)` | 无此步骤 | `cuvsTensorCreate()` |
| `defer dataset.Close()` | 无需显式关闭 | `cuvsTensorDestroy()` |
| `dataset.ToDevice(&resource)` | `dataset = cp.asarray(dataset_vec, dtype=cp.float32)` | `cudaMemcpy()` |

**差异**: 同 1.4

---

### 2.3 查询张量创建和传输

| Go 代码 (行121-125, 159-161) | Python 代码 (行131) | 底层 C API |
|------------------------------|-------------------|-----------|
| `queries, err := cuvs.NewTensor(queriesvec)` | 无此步骤 | `cuvsTensorCreate()` |
| `defer queries.Close()` | 无需显式关闭 | `cuvsTensorDestroy()` |
| `queries.ToDevice(&resource)` | `queries = cp.asarray(queries_vec, dtype=cp.float32)` | `cudaMemcpy()` |

**差异**: 同 1.4

---

### 2.4 Brute Force 索引创建

| Go 代码 (行115-119) | Python 代码 | 底层 C API |
|---------------------|-------------|-----------|
| `index, err := brute_force.CreateIndex()` | 无显式创建 | `cuvsBruteForceIndexCreate()` |
| `defer index.Close()` | 无需显式关闭 | `cuvsBruteForceIndexDestroy()` |

**底层 C API**:
```c
// cuvs/c/neighbors/brute_force.h
cuvsError_t cuvsBruteForceIndexCreate(cuvsBruteForceIndex_t* index);
cuvsError_t cuvsBruteForceIndexDestroy(cuvsBruteForceIndex_t index);
```

**差异**:
- ✅ Go: 显式创建空索引对象
- ✅ Python: 在 `build()` 函数内部自动创建
- 🔍 **底层相同**: 都调用相同的 C API

---

### 2.5 输出张量创建（neighbors 和 distances）

| Go 代码 (行127-137) | Python 代码 | 底层 C API |
|---------------------|-------------|-----------|
| `neighbors, err := cuvs.NewTensorOnDevice[int64](&resource, []int64{int64(len(queriesvec)), int64(limit)})` | 无显式创建 | `cuvsTensorCreate()` + `cudaMalloc()` |
| `defer neighbors.Close()` | 无需显式关闭 | `cuvsTensorDestroy()` + `cudaFree()` |
| `distances, err := cuvs.NewTensorOnDevice[float32](&resource, []int64{int64(len(queriesvec)), int64(limit)})` | 无显式创建 | `cuvsTensorCreate()` + `cudaMalloc()` |
| `defer distances.Close()` | 无需显式关闭 | `cuvsTensorDestroy()` + `cudaFree()` |

**底层 C API**:
```c
cuvsError_t cuvsTensorCreate(...);
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
```

**重要差异**:
- ✅ Go: **显式预分配** GPU 内存用于输出
  - neighbors: `int64` 类型，形状 `(n_queries, k)`
  - distances: `float32` 类型，形状 `(n_queries, k)`
- ✅ Python: **自动分配** 输出内存（在 `search()` 内部）
- 🔍 **底层相同**: 都需要分配 GPU 内存，但时机不同

---

### 2.6 构建 Brute Force 索引

| Go 代码 (行147-152) | Python 代码 (行136) | 底层 C API |
|---------------------|-------------------|-----------|
| `brute_force.BuildIndex(resource, &dataset, distanceType, 2.0, index)` | `index = brute_force.build(dataset, metric=distance_type)` | `cuvsBruteForceBuild()` |

**底层 C API**:
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

**参数对比**:

| 参数 | Go | Python | 说明 |
|-----|-----|--------|------|
| resource | `resource` | 自动管理 | cuVS 资源 |
| dataset | `&dataset` | `dataset` | 数据集张量 |
| metric | `distanceType` (cuvs.DistanceL2) | `metric=distance_type` ("sqeuclidean") | 距离度量 |
| metric_arg | `2.0` | 默认 `2.0` | 距离参数 |
| index | `index` | 自动创建和返回 | 索引对象 |

**差异**:
- Go: 参数 `metric_arg=2.0` **显式传递**
- Python: 参数 `metric_arg` **使用默认值** 2.0
- 🔍 **底层相同**: 都调用 `cuvsBruteForceBuild()`，metric_arg 都是 2.0

---

### 2.7 同步 GPU

| Go 代码 (行154-156) | Python 代码 (行139) | 底层 C API |
|---------------------|-------------------|-----------|
| `resource.Sync()` | `cp.cuda.Stream.null.synchronize()` | `cuvsResourcesSync()` or `cudaStreamSynchronize()` |

**差异**: 同 1.8

---

### 2.8 执行搜索

| Go 代码 (行164-167) | Python 代码 (行144-148) | 底层 C API |
|---------------------|------------------------|-----------|
| `brute_force.SearchIndex(resource, *index, &queries, &neighbors, &distances)` | `distances_gpu, neighbors_gpu = brute_force.search(index, queries, limit)` | `cuvsBruteForceSearch()` |

**底层 C API**:
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

**⭐ 关键差异**:

| 方面 | Go | Python |
|-----|-----|--------|
| 函数名 | `SearchIndex()` | `search()` |
| 参数数量 | 5 个 | 3 个（+ 可选） |
| resource | 显式传递 `resource` | 自动管理或可选 `resources=` |
| index | 解引用传递 `*index` | 直接传递 `index` |
| queries | 指针 `&queries` | CuPy 数组 `queries` |
| k (limit) | 隐式（从 neighbors 形状推导） | 显式 `limit` 参数 |
| neighbors | 预分配指针 `&neighbors` | 自动分配，函数返回 |
| distances | 预分配指针 `&distances` | 自动分配，函数返回 |
| 返回值 | `error` | `(distances, neighbors)` 元组 |

**🔍 底层相同**: 都调用 `cuvsBruteForceSearch()`

---

### 2.9 传输结果回主机

| Go 代码 (行170-178) | Python 代码 (行152-153) | 底层 C API |
|---------------------|------------------------|-----------|
| `neighbors.ToHost(&resource)` | `neighbors = cp.asnumpy(neighbors_gpu)` | `cudaMemcpy(... cudaMemcpyDeviceToHost)` |
| `distances.ToHost(&resource)` | `distances = cp.asnumpy(distances_gpu)` | `cudaMemcpy(... cudaMemcpyDeviceToHost)` |

**差异**: 同 1.10

---

### 2.10 最终同步

| Go 代码 (行180-182) | Python 代码 (行156) | 底层 C API |
|---------------------|-------------------|-----------|
| `resource.Sync()` | `cp.cuda.Stream.null.synchronize()` | `cuvsResourcesSync()` or `cudaStreamSynchronize()` |

**差异**: 同 1.8

---

### 2.11 结果转换

| Go 代码 (行185-209) | Python 代码 (行162) | 操作 |
|---------------------|-------------------|------|
| `neighborsSlice, err := neighbors.Slice()` | `return neighbors.astype(np.int64), ...` | 数据格式转换 |
| `distancesSlice, err := distances.Slice()` | `return ..., distances.astype(np.float64)` | 数据格式转换 |
| 扁平化为 1D 数组 (行196-208) | 保持 2D 数组 | **重要差异** |

**⭐ 关键差异**:

#### Go 返回格式:
```go
// neighbors: []int64 (1D 扁平数组)
// 长度 = n_queries × limit
// 访问: keys[i*limit + j]

// distances: []float64 (1D 扁平数组)
// 长度 = n_queries × limit
// 访问: retdistances[i*limit + j]
```

#### Python 返回格式:
```python
# neighbors: np.ndarray (2D 数组)
# 形状 = (n_queries, limit)
# 访问: neighbors[i, j]

# distances: np.ndarray (2D 数组)
# 形状 = (n_queries, limit)
# 访问: distances[i, j]
```

**🔍 数据内容相同，只是组织方式不同**

---

## 3. TestIvfAndBruteForceForIssue 主流程对比

### 3.1 测试参数设置

| 参数 | Go 代码 (行216-224) | Python 代码 (行187-196) | 值 |
|-----|-------------------|----------------------|-----|
| dimension | `dimension := uint(128)` | `dimension = 128` | 128 |
| limit | `limit := uint(1)` | `limit = 1` | 1 |
| dsize | `dsize := 100000` | `dsize = 100000` | 100000 |
| nlist | `nlist := 128` | `nlist = 128` | 128 |

**✅ 完全相同**

---

### 3.2 随机向量生成

| Go 代码 (行225-231) | Python 代码 (行214-221) | 底层操作 |
|-------------------|------------------------|----------|
| 双层 for 循环逐个生成 | NumPy 向量化操作 | 随机数生成 |
| `rand.Float32()` | `np.random.rand()` | |
| **无随机种子** | `np.random.seed(42)` ⚠️ | **差异** |

**⚠️ 关键差异**:
- Go: **无随机种子**，每次运行生成不同数据
- Python: **设置随机种子** `np.random.seed(42)`，结果可复现
- 🔍 **影响**: 两者生成的数据**不同**，但这不影响功能测试的正确性

---

### 3.3 查询向量选择

| Go 代码 (行232) | Python 代码 (行225) |
|----------------|-------------------|
| `queries := vecs[:8192]` | `queries = vecs[:8192]` |

**✅ 完全相同**: 都取前 8192 个向量作为查询

---

### 3.4 KMeans 聚类调用

| Go 代码 (行234-235) | Python 代码 (行236) | cuVS 调用 |
|-------------------|-------------------|----------|
| `centers, err := getCenters(vecs, int(dimension), nlist, cuvs.DistanceL2, 10)` | `centers = get_centers(vecs, dimension, nlist, "sqeuclidean", 10)` | 详见第 1 节 |
| `require.NoError(t, err)` | `try ... except ...` | 错误处理方式不同 |

**参数对比**:

| 参数 | Go | Python |
|-----|-----|--------|
| vecs | `[][]float32` | `np.ndarray (100000, 128)` |
| dimension | `int(128)` | `128` |
| nlist | `128` | `128` |
| distance_type | `cuvs.DistanceL2` | `"sqeuclidean"` |
| max_iterations | `10` | `10` |

**✅ 参数等价**，距离类型都对应 L2 平方距离

---

### 3.5 并发搜索测试

#### 3.5.1 并发模型

| Go 代码 (行237-262) | Python 代码 (行271-331) | 并发原语 |
|-------------------|------------------------|----------|
| `sync.WaitGroup` | `threading.Thread` | |
| `go func() { ... }()` | `threading.Thread(target=...).start()` | |
| `wg.Wait()` | `thread.join()` | |

**差异**:
- Go: **Goroutine** (轻量级协程)
- Python: **Thread** (操作系统线程)
- 🔍 **并发数相同**: 都是 4 个并发执行单元

#### 3.5.2 迭代次数

| Go 代码 (行244) | Python 代码 (行272) |
|----------------|-------------------|
| `for i := 0; i < 1000; i++` | `iterations_per_thread = 1000` |

**✅ 完全相同**: 每个并发单元执行 1000 次搜索

#### 3.5.3 搜索调用

| Go 代码 (行245-246) | Python 代码 (行287) | cuVS 调用 |
|-------------------|-------------------|----------|
| `_, _, err := Search(centers, queries, limit, cuvs.DistanceL2)` | `neighbors, distances = search(centers, queries, limit, "sqeuclidean")` | 详见第 2 节 |
| `require.NoError(t, err)` | `assert neighbors.shape == ...` | 验证方式不同 |

**参数对比**:

| 参数 | Go | Python |
|-----|-----|--------|
| dataset | `centers` (128, 128) | `centers` (128, 128) |
| queries | `queries` (8192, 128) | `queries` (8192, 128) |
| limit | `uint(1)` | `1` |
| distance_type | `cuvs.DistanceL2` | `"sqeuclidean"` |

**✅ 参数等价**

---

## 4. 关键差异总结

### 4.1 API 封装风格差异

| 方面 | Go | Python | 影响 |
|-----|-----|--------|------|
| **资源管理** | 显式创建/销毁 | 自动管理 | 代码冗长度 |
| **内存分配** | 显式预分配输出 | 自动分配 | 代码灵活性 |
| **错误处理** | 返回 `error` | 抛出异常 | 错误处理方式 |
| **参数传递** | 指针/引用 | 值传递 | 语言特性 |
| **同步方式** | `resource.Sync()` | `cp.cuda.Stream.null.synchronize()` | API 抽象层次 |

### 4.2 cuVS 调用差异汇总

| cuVS 操作 | Go API | Python API | 底层 C API | 是否相同 |
|----------|--------|-----------|-----------|---------|
| 创建资源 | `cuvs.NewResource()` | 自动管理 | `cuvsResourcesCreate()` | ✅ |
| 创建索引参数 | `ivf_flat.CreateIndexParams()` + Setters | `ivf_flat.IndexParams(...)` | `cuvsIvfFlatIndexParamsCreate()` + Setters | ✅ |
| 创建张量 | `cuvs.NewTensor()` + `ToDevice()` | `cp.asarray()` | `cuvsTensorCreate()` + `cudaMemcpy()` | ⚠️ 方式不同 |
| 构建 IVF 索引 | `ivf_flat.BuildIndex()` | `ivf_flat.build()` | `cuvsIvfFlatBuild()` | ✅ |
| 获取聚类中心 | `ivf_flat.GetCenters()` | `index.centers` | `cuvsIvfFlatIndexGetCenters()` | ✅ |
| 构建 Brute Force | `brute_force.BuildIndex()` | `brute_force.build()` | `cuvsBruteForceBuild()` | ✅ |
| 搜索 | `brute_force.SearchIndex()` | `brute_force.search()` | `cuvsBruteForceSearch()` | ✅ |
| 同步 GPU | `resource.Sync()` | `cp.cuda.Stream.null.synchronize()` | `cuvsResourcesSync()` | ✅ |
| 传输回主机 | `tensor.ToHost()` | `cp.asnumpy()` | `cudaMemcpy(D2H)` | ✅ |

**结论**: 
- ✅ **底层 C API 调用完全相同**
- ⚠️ **封装方式有差异**，但不影响功能等价性

### 4.3 测试数据差异

| 数据 | Go | Python | 影响 |
|-----|-----|--------|------|
| **随机种子** | 无 | `np.random.seed(42)` | ⚠️ 数据不同 |
| **数据生成** | 逐个生成 | NumPy 向量化 | 性能差异 |
| **数据规模** | 100,000 × 128 | 100,000 × 128 | ✅ 相同 |
| **查询数量** | 8192 | 8192 | ✅ 相同 |
| **聚类数量** | 128 | 128 | ✅ 相同 |

**⚠️ 重要**: 由于随机种子不同，Go 和 Python 生成的数据**不同**，但这不影响测试的功能正确性。

### 4.4 结果格式差异

| 输出 | Go | Python | 可比性 |
|-----|-----|--------|-------|
| **neighbors** | 1D `[]int64` | 2D `np.ndarray (n_queries, k)` | ⚠️ 格式不同 |
| **distances** | 1D `[]float64` | 2D `np.ndarray (n_queries, k)` | ⚠️ 格式不同 |
| **数据内容** | 相同（如果输入相同） | 相同（如果输入相同） | ✅ 内容等价 |

### 4.5 性能监控差异

| 功能 | Go | Python |
|-----|-----|--------|
| 进度输出 | ❌ 无 | ✅ 每 100 次迭代 |
| 性能统计 | ❌ 无 | ✅ 详细统计（时间、吞吐量） |
| 错误收集 | ❌ 立即失败 | ✅ 收集所有错误 |
| 时间测量 | ❌ 无 | ✅ 分阶段计时 |

---

## 5. 完整 cuVS C API 调用链总结

### getCenters / get_centers

```
1. cuvsResourcesCreate()              (Go 显式, Python 自动)
2. cuvsIvfFlatIndexParamsCreate()     (Go 显式, Python 自动)
3. cuvsIvfFlatIndexParamsSetNLists()
4. cuvsIvfFlatIndexParamsSetMetric()
5. cuvsIvfFlatIndexParamsSetKMeansNIters()
6. cuvsIvfFlatIndexParamsSetKMeansTrainsetFraction()
7. cuvsTensorCreate() + cudaMemcpy()  (创建和传输数据集)
8. cuvsIvfFlatIndexCreate()           (Go 显式, Python 自动)
9. cuvsIvfFlatBuild()                 ⭐ 核心：执行 KMeans
10. cuvsResourcesSync()
11. cuvsIvfFlatIndexGetCenters()
12. cudaMemcpy(D2H)                   (传输聚类中心回主机)
13. cuvsResourcesSync()
14. cuvsResourcesDestroy()            (Go 显式, Python 自动)
```

### Search / search

```
1. cuvsResourcesCreate()              (Go 显式, Python 自动)
2. cuvsTensorCreate() + cudaMemcpy()  (数据集和查询)
3. cuvsBruteForceIndexCreate()        (Go 显式, Python 自动)
4. cuvsTensorCreate() + cudaMalloc()  (Go 预分配输出, Python 自动)
5. cuvsBruteForceBuild()              ⭐ 核心：构建索引
6. cuvsResourcesSync()
7. cuvsBruteForceSearch()             ⭐ 核心：搜索
8. cudaMemcpy(D2H)                    (传输结果回主机)
9. cuvsResourcesSync()
10. cuvsResourcesDestroy()            (Go 显式, Python 自动)
```

---

## 6. 等价性验证

### ✅ 核心算法等价

1. **KMeans 聚类**: 都调用 `cuvsIvfFlatBuild()`，执行相同的 IVF-Flat KMeans 算法
2. **Brute Force 搜索**: 都调用 `cuvsBruteForceSearch()`，执行相同的暴力搜索算法
3. **距离度量**: `cuvs.DistanceL2` 和 `"sqeuclidean"` 都映射到 `CUVS_DISTANCE_L2_SQUARED`
4. **参数设置**: 所有关键参数（n_lists, iterations, metric）完全相同

### ⚠️ 非核心差异

1. **随机数据**: 不同（但不影响功能测试）
2. **封装方式**: 不同（但底层调用相同）
3. **结果格式**: 不同（1D vs 2D，但内容等价）
4. **性能监控**: Python 更详细

### 🎯 结论

**Go 和 Python 测试在功能上完全等价**，都正确调用了 cuVS C API，执行相同的算法逻辑。差异仅在于：
- 语言绑定的封装方式
- 资源管理的抽象层次
- 测试数据的生成方式（随机种子）
- 结果格式的组织方式

**核心测试逻辑和 cuVS 调用完全一致，测试结果具有可比性。**

---

**创建时间**: 2026-02-04  
**文档版本**: 1.0  
**对应 Go 文件**: `pkg/vectorindex/ivfflat/kmeans/device/issue_test.go`  
**对应 Python 文件**: `python_vs_v1/test_issue.py`
