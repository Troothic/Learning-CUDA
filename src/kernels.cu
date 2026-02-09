#include <vector>
#include <cuda_fp16.h>
#include <cmath>

#include "../tester/utils.h"

// ============================================================================
// 第一题：trace 矩阵迹计算（性能优化版）
// ============================================================================

/**
 * @brief Warp 级别规约求和（使用 shuffle 指令，无需共享内存同步）
 * 
 * 比共享内存规约更快，因为 warp 内线程隐式同步
 */
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    // 使用 __shfl_down_sync 进行 warp 内规约
    // 0xffffffff 表示所有 32 个线程都参与
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Block 级别规约求和（结合 warp shuffle 和共享内存）
 * 
 * 先在每个 warp 内用 shuffle 规约，再用共享内存汇总各 warp 结果
 */
template <typename T>
__device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];  // 最多 32 个 warp
    
    int lane = threadIdx.x % 32;     // warp 内索引 (0-31)
    int wid = threadIdx.x / 32;      // warp 索引
    
    // 第一步：warp 内规约
    val = warpReduceSum(val);
    
    // 每个 warp 的 lane 0 将结果写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // 只有第一个 warp 读取并规约所有 warp 的结果
    int numWarps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < numWarps) ? shared[lane] : T(0);
    
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

/**
 * @brief 高性能 trace kernel（使用 warp shuffle 优化）
 * 
 * 相比基础版本的优化点：
 * 1. 使用 warp shuffle 替代部分共享内存同步
 * 2. 每个线程可以处理多个对角线元素（提高数据并行度）
 */
template <typename T>
__global__ void traceKernelOptimized(const T* __restrict__ d_input, T* d_output, 
                                      size_t diag_len, size_t cols) {
    T sum = T(0);
    
    // 每个线程处理多个对角线元素（grid-stride loop）
    // 这样可以处理任意大小的矩阵，且提高了数据并行度
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < diag_len; 
         idx += blockDim.x * gridDim.x) {
        // 对角线元素索引：row i, col i -> 线性索引 i * cols + i
        sum += d_input[idx * cols + idx];
    }
    
    // Block 级别规约
    sum = blockReduceSum(sum);
    
    // Block 内第一个线程将结果原子加到输出
    if (threadIdx.x == 0) {
        atomicAdd(d_output, sum);
    }
}

// int 类型特化（处理类型兼容性）
template <>
__global__ void traceKernelOptimized<int>(const int* __restrict__ d_input, int* d_output, 
                                           size_t diag_len, size_t cols) {
    int sum = 0;
    
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < diag_len; 
         idx += blockDim.x * gridDim.x) {
        sum += d_input[idx * cols + idx];
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(d_output, sum);
    }
}

/**
 * @brief 计算矩阵的迹（对角线元素之和）- 优化版本
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0 || h_input.empty()) {
        return T(0);
    }
    
    size_t diag_len = (rows < cols) ? rows : cols;
    size_t input_size = rows * cols;
    
    // 分配 GPU 内存
    T* d_input;
    T* d_output;
    RUNTIME_CHECK(cudaMalloc(&d_input, input_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_output, sizeof(T)));
    
    // 拷贝数据到 GPU
    RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(T), cudaMemcpyHostToDevice));
    
    // 初始化输出为 0
    T zero = T(0);
    RUNTIME_CHECK(cudaMemcpy(d_output, &zero, sizeof(T), cudaMemcpyHostToDevice));
    
    // 优化的启动配置
    // 使用较少的 block 数量 + grid-stride loop 提高效率
    const int blockSize = 256;
    const int numBlocks = min(32, (int)((diag_len + blockSize - 1) / blockSize));
    
    traceKernelOptimized<T><<<numBlocks, blockSize>>>(d_input, d_output, diag_len, cols);
    RUNTIME_CHECK(cudaDeviceSynchronize());
    
    // 拷贝结果回主机
    T result;
    RUNTIME_CHECK(cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost));
    
    // 释放 GPU 内存
    RUNTIME_CHECK(cudaFree(d_input));
    RUNTIME_CHECK(cudaFree(d_output));
    
    return result;
}

// ============================================================================
// 第二题：Flash Attention 实现（性能优化版）
// ============================================================================

// 块大小配置
constexpr int BLOCK_SIZE_Q = 32;    // 每个 block 处理的 query 数量
constexpr int BLOCK_SIZE_KV = 32;   // 每次加载的 K/V 块大小

/**
 * @brief 辅助函数：float 到模板类型 T 的转换
 */
template <typename T>
__device__ __forceinline__ T float_to_T(float val) {
    return static_cast<T>(val);
}

template <>
__device__ __forceinline__ half float_to_T<half>(float val) {
    return __float2half(val);
}

/**
 * @brief 辅助函数：模板类型 T 到 float 的转换
 */
template <typename T>
__device__ __forceinline__ float T_to_float(T val) {
    return static_cast<float>(val);
}

template <>
__device__ __forceinline__ float T_to_float<half>(half val) {
    return __half2float(val);
}

/**
 * @brief 优化版 Flash Attention Kernel
 * 
 * 优化点：
 * 1. 使用共享内存缓存 K/V 块，减少全局内存访问
 * 2. 使用 warp shuffle 进行快速规约
 * 3. 循环展开提高指令级并行
 */
template <typename T>
__global__ void flashAttentionKernelOptimized(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size,
    int tgt_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal,
    float scale
) {
    // Grid: (batch_size * query_heads, tgt_seq_len)
    int batch_head_idx = blockIdx.x;
    int query_pos = blockIdx.y;
    int tid = threadIdx.x;
    
    int batch_idx = batch_head_idx / query_heads;
    int head_idx = batch_head_idx % query_heads;
    
    // GQA 映射
    int heads_per_group = query_heads / kv_heads;
    int kv_head_idx = head_idx / heads_per_group;
    
    // 计算指针偏移
    const T* q_ptr = Q + (batch_idx * tgt_seq_len * query_heads + query_pos * query_heads + head_idx) * head_dim;
    const T* k_base = K + (batch_idx * src_seq_len * kv_heads + kv_head_idx) * head_dim;
    const T* v_base = V + (batch_idx * src_seq_len * kv_heads + kv_head_idx) * head_dim;
    T* o_ptr = O + (batch_idx * tgt_seq_len * query_heads + query_pos * query_heads + head_idx) * head_dim;
    
    // 共享内存布局：Q向量 + K块 + V块 + warp规约临时空间
    extern __shared__ char smem[];
    float* s_q = reinterpret_cast<float*>(smem);
    float* s_k = s_q + head_dim;      // K 块缓存
    float* s_v = s_k + BLOCK_SIZE_KV * head_dim;  // V 块缓存
    float* warp_buf = s_v + BLOCK_SIZE_KV * head_dim;  // warp 规约缓存
    
    // 加载 Q 到共享内存
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = T_to_float(q_ptr[d]);
    }
    __syncthreads();
    
    // 在线 softmax 状态
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    // 输出累加器（每个线程负责部分维度）
    float o_acc[16];  // 假设 head_dim / blockDim.x <= 16
    int dims_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o_acc[i] = 0.0f;
    }
    
    // 分块处理 K/V
    int kv_end = is_causal ? min(query_pos + 1, src_seq_len) : src_seq_len;
    
    for (int kv_block_start = 0; kv_block_start < kv_end; kv_block_start += BLOCK_SIZE_KV) {
        int kv_block_end = min(kv_block_start + BLOCK_SIZE_KV, kv_end);
        int block_len = kv_block_end - kv_block_start;
        
        // 协作加载 K 块到共享内存
        for (int i = tid; i < block_len * head_dim; i += blockDim.x) {
            int kv_local = i / head_dim;
            int d = i % head_dim;
            int kv_pos = kv_block_start + kv_local;
            s_k[kv_local * head_dim + d] = T_to_float(k_base[kv_pos * kv_heads * head_dim + d]);
        }
        
        // 协作加载 V 块到共享内存
        for (int i = tid; i < block_len * head_dim; i += blockDim.x) {
            int kv_local = i / head_dim;
            int d = i % head_dim;
            int kv_pos = kv_block_start + kv_local;
            s_v[kv_local * head_dim + d] = T_to_float(v_base[kv_pos * kv_heads * head_dim + d]);
        }
        __syncthreads();
        
        // 处理当前块中的每个 K/V 位置
        for (int kv_local = 0; kv_local < block_len; kv_local++) {
            // 计算 Q @ K^T
            float partial_dot = 0.0f;
            #pragma unroll 4
            for (int d = tid; d < head_dim; d += blockDim.x) {
                partial_dot += s_q[d] * s_k[kv_local * head_dim + d];
            }
            
            // Warp 内规约
            for (int offset = 16; offset > 0; offset >>= 1) {
                partial_dot += __shfl_down_sync(0xffffffff, partial_dot, offset);
            }
            
            // 跨 warp 规约
            int lane = tid % 32;
            int warp_id = tid / 32;
            
            if (lane == 0) {
                warp_buf[warp_id] = partial_dot;
            }
            __syncthreads();
            
            float score = 0.0f;
            if (warp_id == 0) {
                int num_warps = (blockDim.x + 31) / 32;
                float val = (lane < num_warps) ? warp_buf[lane] : 0.0f;
                for (int offset = 16; offset > 0; offset >>= 1) {
                    val += __shfl_down_sync(0xffffffff, val, offset);
                }
                if (lane == 0) {
                    warp_buf[0] = val * scale;
                }
            }
            __syncthreads();
            score = warp_buf[0];
            
            // 在线 softmax 更新
            float m_new = fmaxf(m_prev, score);
            float l_new = l_prev * expf(m_prev - m_new) + expf(score - m_new);
            
            float correction = expf(m_prev - m_new);
            float weight = expf(score - m_new);
            
            // 更新输出累加器
            #pragma unroll
            for (int i = 0; i < dims_per_thread; i++) {
                int d = tid + i * blockDim.x;
                if (d < head_dim) {
                    o_acc[i] = o_acc[i] * correction + weight * s_v[kv_local * head_dim + d];
                }
            }
            
            m_prev = m_new;
            l_prev = l_new;
        }
        __syncthreads();
    }
    
    // 归一化并写入输出
    #pragma unroll
    for (int i = 0; i < dims_per_thread; i++) {
        int d = tid + i * blockDim.x;
        if (d < head_dim) {
            float out_val = (l_prev > 0.0f) ? (o_acc[i] / l_prev) : 0.0f;
            o_ptr[d] = float_to_T<T>(out_val);
        }
    }
}

/**
 * @brief Flash Attention 入口函数（优化版）
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = q_size;
    
    h_o.resize(o_size);
    
    // 分配 GPU 内存
    T *d_q, *d_k, *d_v, *d_o;
    RUNTIME_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_k, kv_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_v, kv_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));
    
    // 拷贝数据到 GPU
    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemset(d_o, 0, o_size * sizeof(T)));
    
    // 缩放因子
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // Kernel 启动配置
    int threads_per_block = min(256, max(32, head_dim));
    dim3 grid(batch_size * query_heads, target_seq_len);
    dim3 block(threads_per_block);
    
    // 共享内存大小：Q + K块 + V块 + warp缓存
    size_t smem_size = head_dim * sizeof(float) +                    // Q
                       BLOCK_SIZE_KV * head_dim * sizeof(float) +    // K 块
                       BLOCK_SIZE_KV * head_dim * sizeof(float) +    // V 块
                       32 * sizeof(float);                           // warp 缓存
    
    flashAttentionKernelOptimized<T><<<grid, block, smem_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal, scale
    );
    RUNTIME_CHECK(cudaDeviceSynchronize());
    
    // 拷贝结果回主机
    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));
    
    // 释放 GPU 内存
    RUNTIME_CHECK(cudaFree(d_q));
    RUNTIME_CHECK(cudaFree(d_k));
    RUNTIME_CHECK(cudaFree(d_v));
    RUNTIME_CHECK(cudaFree(d_o));
}

// ============================================================================
// 显式模板实例化（链接必需，请勿修改）
// ============================================================================
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
