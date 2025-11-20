// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh", "wgmma-interface.cuh", "kernel.cu"]}
// TL+ {"compile_flags": ["-lcuda", "-lcublas"]}

#include <algorithm>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <vector>
#include "tma-interface.cuh"
#include "wgmma-interface.cuh"

typedef __nv_bfloat16 bf16;

////////////////////////////////////////////////////////////////////////////////
// Part 1: Matrix Multiplication for M = 8192, N = 8192, K = 8192
////////////////////////////////////////////////////////////////////////////////

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define TILE_M 128
#define TILE_N 256
#define TILE_K 64
#define WARP_GROUP_THREADS 128
#define WGMMA_M 64
#define WGMMA_K 16
#define WGMMA_N 256

// needed for wgmma
#define THREADS_X 32
#define THREADS_Y 4 // 4 for consumer, 1 for producer
// #define PRODUCER 1
#define WARPGROUPS 3
#define CONSUMERS (WARPGROUPS - 1)

#define CONSUME_M ((TILE_M) / (CONSUMERS))

#define BUFFERS 4

#define K_ITERS ((TILE_K) / (WGMMA_K))
#define M_ITERS ((TILE_M / CONSUMERS) / (WGMMA_M))
#define N_ITERS ((TILE_N) / (WGMMA_N))

__global__ void
    __launch_bounds__(384, 1, 1)
        h100_matmul(
            int M,
            int N,
            int K,
            bf16 *A,
            bf16 *B,
            bf16 *C,
            __grid_constant__ const CUtensorMap A_map,
            __grid_constant__ const CUtensorMap B_map)
{
    extern __shared__ __align__(128) unsigned char shmem_raw[];
    bf16 *shmem_ptr = reinterpret_cast<bf16 *>(shmem_raw);

    constexpr int A_TILE_ELEMS = TILE_M * TILE_K;
    constexpr int B_TILE_ELEMS = TILE_N * TILE_K;

    // TODO: refactor to use buffers once working
    bf16 *a_shmem[2];
    for (int i = 0; i < 2; i++)
    {
        a_shmem[i] = shmem_ptr + i * A_TILE_ELEMS;
    }

    bf16 *b_shmem[2];
    bf16 *b_base = shmem_ptr + 2 * A_TILE_ELEMS;
    for (int i = 0; i < 2; i++)
    {

        b_shmem[i] = b_base + i * B_TILE_ELEMS;
    }

    __shared__ __align__(8) uint64_t pbar[2];
    __shared__ __align__(8) uint64_t cbar[2];

    __shared__ uint64_t *produced[4];
    __shared__ uint64_t *consumed[4];

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;
    int thread_id = warp_id * 32 + lane_id + threadIdx.z * WARP_GROUP_THREADS;
    int wg_id = threadIdx.z; // warpgroup id

    int a_row_base = blockIdx.x * TILE_M;
    int b_row_base = blockIdx.y * TILE_N;

    bool is_producer = wg_id == 0;

    if (thread_id == 0)
    {
        // only need two barriers, one for each parity of k tile
        for (int i = 0; i < 2; i++)
        {
            produced[i] = &pbar[i];
            consumed[i] = &cbar[i];

            init_barrier(produced[i], 1);
            init_barrier(consumed[i], CONSUMERS * WARP_GROUP_THREADS); // 2 wg * 128 threads each
        }
        async_proxy_fence();
    }

    __syncthreads();
    // start queueing wgmma operations for the next iteration before the first one is done

    // loop along K dimension over whole array
    int num_tiles = CEIL_DIV(K, TILE_K);
    if (is_producer && lane_id == 0 && warp_id == 0)
    {
        warpgroup_reg_dealloc<32>();
        for (int k_tile = 0; k_tile < num_tiles; k_tile++) // k
        {
            int m_tile = lane_id;
            // tma load
            int buffer_id = k_tile % 2; // odd k tiles to buf1, even to buf2
            bf16 *a_buf = a_shmem[buffer_id];
            bf16 *b_buf = b_shmem[buffer_id];

            int phase = (k_tile / 2) % 2; // consumer increments every 2
            int global_k = k_tile * TILE_K;

            if (k_tile > 0) // beyond the first k tile, have to wait for consumer
            {
                wait(consumed[buffer_id], !phase);
            }

            //
            cp_async_bulk_tensor_2d_global_to_shared(a_buf, &A_map, global_k, a_row_base, produced[buffer_id]);
            cp_async_bulk_tensor_2d_global_to_shared(b_buf, &B_map, global_k, b_row_base, produced[buffer_id]);

            expect_bytes_and_arrive(produced[buffer_id], (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(bf16));
        }
    }

    if (!is_producer)
    {
        warpgroup_reg_alloc<160>();
        // initialize d
        float d[M_ITERS][N_ITERS][16][8];

#pragma unroll
        for (int row = 0; row < M_ITERS; row++)
        {
            for (int col = 0; col < N_ITERS; col++)
            {
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        d[row][col][i][j] = 0.0f;
                    }
                }
            }
        }

        // wg 1 processes m_tile 0, wg 2 processes m_tile 1
        int m_tile = wg_id - 1;
        for (int k_tile = 0; k_tile < num_tiles; k_tile++) // k
        {
            int buffer_id = k_tile % 2;                                     // odd k tiles to buf1, even to buf2
            bf16 *a_buf = a_shmem[buffer_id] + m_tile * CONSUME_M * TILE_K; // offset to correct part of the 128 M block in A
            bf16 *b_buf = b_shmem[buffer_id];

            int phase = (k_tile / 2) % 2;
            int bar_id = m_tile * CONSUMERS + buffer_id;

            wait(produced[buffer_id], phase);
            // for each wgmma core tile in the larger tile
            warpgroup_arrive();
            for (int row = 0; row < M_ITERS; row++)
            {
                for (int col = 0; col < N_ITERS; col++)
                {
                    // get the correct row of shmem
                    bf16 *a_tile = a_buf + TILE_K * row * WGMMA_M;
                    bf16 *b_tile = b_buf + TILE_K * col * WGMMA_N;

                    // #pragma unroll
                    for (uint i = 0; i < K_ITERS; i++) // k_iters
                    {
                        // 2 bytes per elt, TILE_K elts across, 8 elts down in block
                        int sbo = sizeof(bf16) * TILE_K * 8;

                        uint64_t desc_a = make_smem_desc<SWIZZLE_128B>(a_tile + i * WGMMA_K, 1, sbo);
                        uint64_t desc_b = make_smem_desc<SWIZZLE_128B>(b_tile + i * WGMMA_K, 1, sbo);

                        wgmma_n256<1, 1, 1, 0, 0>(desc_a, desc_b, d[row][col]);
                    }
                }
            }
            wgmma_commit();
            wgmma_wait<0>();
            warpgroup_arrive();
            arrive(consumed[buffer_id], 1);
        }

        int a_row = a_row_base + m_tile * (TILE_M / CONSUMERS);
        int b_row = b_row_base;
        // writeback results to global memory
        for (int row = 0; row < M_ITERS; row++)
        {
            for (int col = 0; col < N_ITERS; col++)
            {
                for (int i = 0; i < 128; i++)
                {
                    int m = 16 * warp_id + ((i / 2) & 1) * 8 + (lane_id / 4);
                    int n = (lane_id % 4) * 2 + (i / 4) * 8 + (i % 2);

                    int glob_n = b_row + col * WGMMA_N + n;
                    int glob_m = a_row + row * WGMMA_M + m;

                    C[glob_n * M + glob_m] = d[row][col][i / 8][i % 8];
                }
            }
        }
    }
}

void launch_h100_matmul(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C)
{
    CUtensorMap A_map;
    const cuuint64_t global_dim_a[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(M)};
    const cuuint64_t global_strides_a[1] = {static_cast<cuuint64_t>(K * sizeof(bf16))};
    const cuuint32_t box_dim_a[2] = {TILE_K, TILE_M};
    const cuuint32_t element_strides_a[2] = {1, 1};
    CUDA_CHECK(
        cuTensorMapEncodeTiled(
            &A_map,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            A,
            global_dim_a,
            global_strides_a,
            box_dim_a,
            element_strides_a,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CUtensorMap B_map;
    const cuuint64_t global_dim_b[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(N)};
    const cuuint64_t global_strides_b[1] = {static_cast<cuuint64_t>(K * sizeof(bf16))};
    const cuuint32_t box_dim_b[2] = {TILE_K, TILE_N};
    const cuuint32_t element_strides_b[2] = {1, 1};
    CUDA_CHECK(
        cuTensorMapEncodeTiled(
            &B_map,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            B,
            global_dim_b,
            global_strides_b,
            box_dim_b,
            element_strides_b,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    size_t shmem_size_bytes = (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(bf16) * 2 + sizeof(uint64_t) * 4;
    CUDA_CHECK(cudaFuncSetAttribute(
        h100_matmul,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size_bytes));
    // printf("shmem_size_bytes=%zu\n", shmem_size_bytes);

    dim3 gridDim = dim3(CEIL_DIV(M, TILE_M), CEIL_DIV(N, TILE_N));
    dim3 blockDim = dim3(THREADS_X, THREADS_Y, WARPGROUPS); // for wgmma, TODO hardcoded fix
    h100_matmul<<<gridDim, blockDim, shmem_size_bytes>>>(M, N, K, A, B, C, A_map, B_map);
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

static constexpr size_t kNumOfWarmupIterations = 2;
static constexpr size_t kNumOfOuterIterations = 1;
static constexpr size_t kNumOfInnerIterations = 10;

#define BENCHPRESS(func, flops, ...)                                      \
    do                                                                    \
    {                                                                     \
        std::cout << "Running " << #func << " ...\n";                     \
        for (size_t i = 0; i < kNumOfWarmupIterations; ++i)               \
        {                                                                 \
            func(__VA_ARGS__);                                            \
        }                                                                 \
        cudaDeviceSynchronize();                                          \
        std::vector<float> times(kNumOfOuterIterations);                  \
        cudaEvent_t start, stop;                                          \
        cudaEventCreate(&start);                                          \
        cudaEventCreate(&stop);                                           \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i)                \
        {                                                                 \
            cudaEventRecord(start);                                       \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j)            \
            {                                                             \
                func(__VA_ARGS__);                                        \
            }                                                             \
            cudaEventRecord(stop);                                        \
            cudaEventSynchronize(stop);                                   \
            float elapsed_time;                                           \
            cudaEventElapsedTime(&elapsed_time, start, stop);             \
            times[i] = elapsed_time / kNumOfInnerIterations;              \
        }                                                                 \
        cudaEventDestroy(start);                                          \
        cudaEventDestroy(stop);                                           \
        std::sort(times.begin(), times.end());                            \
        float best_time_ms = times[0];                                    \
        float tflops = (flops * 1e-9) / best_time_ms;                     \
        std::cout << "  Runtime: " << best_time_ms << " ms" << std::endl; \
        std::cout << "  TFLOP/s: " << tflops << std::endl;                \
    } while (0)

void runCublasRef(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C)
{
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float alpha = 1, beta = 0;
    cublasStatus_t status =
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                     A, CUDA_R_16BF, K, B, CUDA_R_16BF, N, &beta, C,
                     CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS error: " << status << std::endl;
        exit(1);
    }
}

void init_matrix(bf16 *mat, int N)
{
    std::default_random_engine generator(0);
    std::normal_distribution<float> distribution(0, 1);
    for (int i = 0; i < N; i++)
    {
        mat[i] = distribution(generator);
    }
}

bool check_correctness(bf16 *ref, bf16 *test, int N, float tolerance = 0.1f)
{
    int mismatches = 0;
    int total = N;
    for (int i = 0; i < N; i++)
    {
        float ref_val = __bfloat162float(ref[i]);
        float test_val = __bfloat162float(test[i]);
        float diff = std::abs(ref_val - test_val);
        if (diff > tolerance)
        {
            if (mismatches < 10)
            { // Print first 10 mismatches
                std::cout << "  Mismatch at index " << i << ": ref=" << ref_val
                          << ", test=" << test_val << ", diff=" << diff
                          << std::endl;
            }
            mismatches++;
        }
    }
    std::cout << "Total mismatches: " << mismatches << " / " << total << " ("
              << (100.0 * mismatches / total) << "%)" << std::endl;
    return mismatches == 0;
}

int main()
{

    const int M = 8192, N = 8192, K = 8192;

    bf16 *A = (bf16 *)malloc(sizeof(bf16) * M * K);
    bf16 *B = (bf16 *)malloc(sizeof(bf16) * K * N);
    bf16 *C = (bf16 *)malloc(sizeof(bf16) * M * N);

    init_matrix(A, M * K);
    init_matrix(B, K * N);
    memset(C, 0, sizeof(bf16) * M * N);

    bf16 *dA;
    bf16 *dB;
    bf16 *dC;
    bf16 *dCublas;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(bf16) * M * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(bf16) * K * N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(bf16) * M * N));
    CUDA_CHECK(cudaMalloc(&dCublas, sizeof(bf16) * M * N));

    CUDA_CHECK(cudaMemcpy(dA, A, sizeof(bf16) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sizeof(bf16) * K * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(dCublas, C, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));

    std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

    bf16 *hCublas = (bf16 *)malloc(sizeof(bf16) * M * N);
    bf16 *hOurs = (bf16 *)malloc(sizeof(bf16) * M * N);

    runCublasRef(M, N, K, dA, dB, dCublas);
    launch_h100_matmul(M, N, K, dA, dB, dC);

    CUDA_CHECK(cudaMemcpy(hCublas, dCublas, sizeof(bf16) * M * N,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(hOurs, dC, sizeof(bf16) * M * N, cudaMemcpyDeviceToHost));

    bool correct = check_correctness(hCublas, hOurs, M * N, 0.01f);
    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    long flops = 2LL * M * N * K;
    BENCHPRESS(runCublasRef, flops, M, N, K, dA, dB, dCublas);

    BENCHPRESS(launch_h100_matmul, flops, M, N, K, dA, dB, dC);

    free(hCublas);
    free(hOurs);

    return 0;
}
