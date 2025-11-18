// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh", "wgmma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdio.h>

#include "tma-interface.cuh"
#include "wgmma-interface.cuh"

typedef __nv_bfloat16 bf16;

////////////////////////////////////////////////////////////////////////////////
// Convenience setup to test different WGMMA variants
////////////////////////////////////////////////////////////////////////////////

/// <--- your code here --->
#define WARP_GROUP_THREADS 128

// Set up M, N and K to set up the test
const int M = 64;
const int N = 64;
const int K = 64;

// const int swizzle = 128;
// const int box_dim = 64;

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void wgmma_test(bf16 *a, bf16 *b, float *c,
                           __grid_constant__ const CUtensorMap a_map, __grid_constant__ const CUtensorMap b_map)
{
    extern __shared__ __align__(128) unsigned char shmem_raw[];
    bf16 *a_shmem = reinterpret_cast<bf16 *>(shmem_raw);
    bf16 *b_shmem = a_shmem + (TILE_M * TILE_K);

    size_t tile_bytes = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(bf16);
    size_t mbar_off = (tile_bytes + 16) & ~size_t(16);
    uint64_t *mbar = reinterpret_cast<uint64_t *>(shmem_raw + mbar_off);

    int thread_id = threadIdx.x;
    int lane_id = thread_id % 32;
    int warp_id = thread_id / 32;

    if (thread_id == 0)
    {
        init_barrier(mbar, 1);
        async_proxy_fence();

        cp_async_bulk_tensor_2d_global_to_shared(a_shmem, &a_map, 0, 0, mbar);
        expect_bytes_and_arrive(mbar, TILE_M * TILE_K * sizeof(bf16));

        wait(mbar, 0);
        cp_async_bulk_tensor_2d_global_to_shared(b_shmem, &b_map, 0, 0, mbar);
        expect_bytes_and_arrive(mbar, TILE_N * TILE_K * sizeof(bf16));
        wait(mbar, 1);
    }

    float d[8][4];

    warpgroup_arrive();

    for (int k0 = 0; k0 < K; k0 += 16)
    {
        for (int n0 = 0; n0 < N; n0 += 8)
        {
            bf16 *a_ptr = a_shmem + k0;
            bf16 *b_ptr = b_shmem + n0 * TILE_K + k0;
            uint64_t desc_a = make_smem_desc<SWIZZLE_128B>(a_ptr, 1, 1024);
            uint64_t desc_b = make_smem_desc<SWIZZLE_128B>(b_ptr, 1, 1024);
            wgmma_n8<1, 1, 1, 0, 0>(desc_a, desc_b, d[n0 / 8]);
        }
    }

    // wgmma_n8<1, 1, 1, 0, 0>(desc_a0, desc_b0, d[0]);
    // wgmma_n8<1, 1, 1, 0, 0>(desc_a1, desc_b1, d[0]);
    wgmma_commit();
    wgmma_wait<0>();

    __syncthreads();

    for (int glob_n = 0; glob_n < N; glob_n += 8)
    {
        for (int i = 0; i < 2; i++)
        {
            int m = 16 * warp_id + (i * 8) + (lane_id / 4);
            for (int j = 0; j < 2; j++)
            {
                int n = ((lane_id % 4) * 2 + j) + glob_n;
                c[n * TILE_M + m] = d[glob_n / 8][i * 2 + j];
            }
        }
    }
}

template <int TILE_M, int TILE_N, int TILE_K>
void launch_wgmma_test(bf16 *a, bf16 *b, float *c)
{

    CUtensorMap a_map;
    const cuuint64_t global_dim[2] = {TILE_K, TILE_M};
    const cuuint64_t global_strides[1] = {TILE_K * sizeof(bf16)};
    const cuuint32_t box_dim[2] = {TILE_K, TILE_M};
    const cuuint32_t element_strides[2] = {1, 1};
    CUDA_CHECK(
        cuTensorMapEncodeTiled(
            &a_map,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            a,
            global_dim,
            global_strides,
            box_dim,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CUtensorMap b_map;
    const cuuint64_t global_dim_b[2] = {TILE_K, TILE_N};
    const cuuint64_t global_strides_b[1] = {TILE_K * sizeof(bf16)};
    const cuuint32_t box_dim_b[2] = {TILE_K, TILE_N};
    const cuuint32_t element_strides_b[2] = {1, 1};
    CUDA_CHECK(
        cuTensorMapEncodeTiled(
            &b_map,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            b,
            global_dim_b,
            global_strides_b,
            box_dim_b,
            element_strides_b,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    size_t shmem_size_bytes = (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(bf16) + sizeof(uint64_t) * 2;
    wgmma_test<TILE_M, TILE_N, TILE_K><<<1, WARP_GROUP_THREADS, shmem_size_bytes>>>(a, b, c, a_map, b_map);
}

/// <--- your code here --->

int main()
{
    // Initialize source matrix on host
    bf16 *a = (bf16 *)malloc(M * K * sizeof(bf16));
    bf16 *b = (bf16 *)malloc(N * K * sizeof(bf16));

    // Feel free to change the initialization logic if you need to.
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            a[i * K + j] = (i + j) / 10.0f;
        }
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++)
        {
            b[j * N + i] = i + j / 10.0f;
        }
    }

    float *d_c;
    bf16 *d_a, *d_b;
    cudaMalloc(&d_a, M * K * sizeof(bf16));
    cudaMalloc(&d_b, N * K * sizeof(bf16));
    cudaMalloc(&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, a, M * K * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * K * sizeof(bf16), cudaMemcpyHostToDevice);

    // Compute CPU reference
    float *cpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float temp = 0.0f;
            for (int k = 0; k < K; k++)
            {
                float a_row = (float)a[i * K + k];
                float a_col = (float)b[k + j * K];
                temp += a_row * a_col;
            }
            cpu_output[j * M + i] = temp;
        }
    }

    float *gpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++)
    {
        gpu_output[i] = 0;
    }
    cudaMemcpy(d_c, gpu_output, M * N * sizeof(float), cudaMemcpyHostToDevice);

    printf("\n\nRunning WGMMA test M=%d, N=%d, K=%d...\n\n", M, N, K);
    launch_wgmma_test<M, N, K>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(gpu_output, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // check results
    bool correct = true;
    for (int idx = 0; idx < M * N; idx++)
    {
        if (fabs(cpu_output[idx] - gpu_output[idx]) > 0.01f)
        {
            correct = false;
            int j = idx / M;
            int i = idx % M;
            printf(
                "\nFirst mismatch at (%d, %d): CPU=%.0f, GPU=%.0f\n",
                i,
                j,
                cpu_output[idx],
                gpu_output[idx]);
            break;
        }
    }

    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);

    return 0;
}