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

#define TILE_M 64
#define TILE_N 64
#define TILE_K 64
#define WARP_GROUP_THREADS 128
#define WGMMA_M 64
#define WGMMA_K 16
#define WGMMA_N 8

// needed for wgmma
#define THREADS_X 32
#define THREADS_Y 4

#define K_ITERS ((TILE_K) / (WGMMA_K))
#define M_ITERS ((TILE_M) / (WGMMA_M))
#define N_ITERS ((TILE_N) / (WGMMA_N))

__global__ void h100_matmul(
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
    bf16 *a_shmem = reinterpret_cast<bf16 *>(shmem_raw);
    bf16 *b_shmem = a_shmem + (TILE_M * TILE_K);
    size_t tile_bytes = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(bf16);
    size_t mbar_off = (tile_bytes + 16) & ~size_t(16);
    uint64_t *mbar = reinterpret_cast<uint64_t *>(shmem_raw + mbar_off);
    // uint64_t *bbar = abar + 1;

    // __shared__ uint64_t *mbar;

    // int thread_id = threadIdx.x;

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;
    // int lane_id = thread_id % 32;
    // int warp_id = thread_id / 32;

    int a_row = blockIdx.x * TILE_M;
    int b_row = blockIdx.y * TILE_N;

    // // initialize C to 0
    // for (int i = threadIdx.x; i < TILE_M; i += blockDim.x)
    // {
    //     for (int j = threadIdx.y; j < TILE_N; j += blockDim.y)
    //     {
    //         C[(a_row + i) * N + (b_row + j)] = 0.0f;
    //     }
    // }

    // __syncthreads();

    // put in shared temporarily to see if pthe problem was register spillage 
    __shared__ float d[M_ITERS][N_ITERS][4];

    // #pragma unroll
    // for (int row = 0; row < M_ITERS; row++)
    // {
    //     for (int col = 0; col < N_ITERS; col++)
    //     {
    //         for (int i = 0; i < 4; i++)
    //         {
    //             d[row][col][i] = 0.0f;
    //         }
    //     }
    // }

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("K: %d\n", K);
    // }

    for (int global_k = 0; global_k < K; global_k += TILE_K) // k
    {

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            // if (blockIdx.x == 0 && blockIdx.y == 0)
            // {
            //     printf("iter %d, d: %f\n", global_k, d[0][0][0]);
            // }
            init_barrier(mbar, 1);
            // init_barrier(bbar, 1);
            async_proxy_fence();

            cp_async_bulk_tensor_2d_global_to_shared(a_shmem, &A_map, a_row, global_k, mbar);
            expect_bytes_and_arrive(mbar, TILE_M * TILE_K * sizeof(bf16));

            wait(mbar, 0);
            cp_async_bulk_tensor_2d_global_to_shared(b_shmem, &B_map, b_row, global_k, mbar);
            expect_bytes_and_arrive(mbar, TILE_N * TILE_K * sizeof(bf16));

            wait(mbar, 1);
        }
        __syncthreads();

        for (int row = 0; row < M_ITERS; row++) // m_iters
        {
            for (int col = 0; col < N_ITERS; col++) // n_iters
            {

                // uint64_t desc_as[K_ITERS];
                // uint64_t desc_bs[K_ITERS];

                // gloabl ids
                // int row_id = row * WGMMA_M + a_row;
                // int col_id = col * WGMMA_N + b_row;

                //
                bf16 *a_tile = a_shmem + TILE_K * row * WGMMA_M;
                bf16 *b_tile = b_shmem + TILE_K * col * WGMMA_N;

                // #pragma unroll
                warpgroup_arrive();
                for (uint i = 0; i < K_ITERS; i++) // k_iters
                {
                    // 2 bytes per elt, TILE_K elts across, 8 elts down in block
                    int sbo = sizeof(bf16) * TILE_K * 8;
                    uint64_t desc_a = make_smem_desc<SWIZZLE_128B>(a_tile + i * WGMMA_K, 1, sbo);
                    uint64_t desc_b = make_smem_desc<SWIZZLE_128B>(b_tile + i * WGMMA_K, 1, sbo);
                    wgmma_n8<1, 1, 1, 0, 0>(desc_a, desc_b, d[row][col]);
                    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && row == 0 && col == 0)
                    // {
                    //     printf("iter %d right after k loop, d: %f\n", global_k, d[0][0][0]);
                    // }
                }
                wgmma_commit();
                wgmma_wait<0>();
            }
        }
        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        // {
        //     printf("iter %d, d: %f\n", global_k, d[0][0][0]);
        // }

        __syncthreads();
    }

    // __syncthreads();

    //
    for (int row = 0; row < M_ITERS; row++)
    {
        for (int col = 0; col < N_ITERS; col++)
        {
            for (int i = 0; i < 2; i++)
            {
                int m = 16 * warp_id + (i * 8) + (lane_id / 4);
                for (int j = 0; j < 2; j++)
                {
                    int n = ((lane_id % 4) * 2 + j);
                    int row_id = row * WGMMA_M + a_row;
                    int col_id = col * WGMMA_N + b_row;

                    C[(n + col_id) * M + m + row_id] = d[row][col][i * 2 + j];
                }
            }
        }
    }
}

void launch_h100_matmul(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C)
{
    CUtensorMap A_map;
    const cuuint64_t global_dim_a[2] = {TILE_K, TILE_M};
    const cuuint64_t global_strides_a[1] = {TILE_K * sizeof(bf16)};
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
    const cuuint64_t global_dim_b[2] = {TILE_K, TILE_N};
    const cuuint64_t global_strides_b[1] = {TILE_K * sizeof(bf16)};
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

    size_t shmem_size_bytes = (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(bf16) + sizeof(uint64_t);
    CUDA_CHECK(cudaFuncSetAttribute(
        h100_matmul,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size_bytes));
    printf("shmem_size_bytes=%zu\n", shmem_size_bytes);

    dim3 gridDim = dim3(CEIL_DIV(M, TILE_M), CEIL_DIV(N, TILE_N));
    dim3 blockDim = dim3(THREADS_X, THREADS_Y); // for wgmma, TODO hardcoded fix
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
        mat[i] = 1;
        // distribution(generator);
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
