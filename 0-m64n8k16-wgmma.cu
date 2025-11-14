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

/// <--- your code here --->
#define WARP_GROUP_THREADS 128
////////////////////////////////////////////////////////////////////////////////
// Part 0: No Swizzle WGGMA load for M = 64, N = 8, K = 16
////////////////////////////////////////////////////////////////////////////////

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void wgmma_m64n8k16(bf16 *a, bf16 *b, float *c)
{
    extern __shared__ bf16 shmem[];
    bf16 *a_shmem = shmem;
    bf16 *b_shmem = shmem + TILE_M * TILE_K;

    // int lane_id = threadIdx.x % 32;
    // int warp_id = threadIdx.x / 32;
    int thread_id = threadIdx.x;

    const int core_rows = 8;
    const int core_k = 16 / sizeof(bf16);

    // int num_core_rows = TILE_M / core_rows;
    int num_core_k = TILE_K / core_k;

    for (int a_idx = thread_id; a_idx < TILE_M * TILE_K; a_idx += WARP_GROUP_THREADS)
    {
        int m = a_idx / TILE_K;
        int k = a_idx % TILE_K;
        int core_rows_id = (m / core_rows);
        int core_k_id = (k / core_k);

        int core_id = core_rows_id * num_core_k + core_k_id;
        int core_offset = core_id * (core_rows * core_k);

        int core_rows_offset = m % core_rows;
        int core_k_offset = k % core_k;

        int shmem_idx = core_offset + core_rows_offset * core_k + core_k_offset;
        a_shmem[shmem_idx] = a[m * TILE_K + k];
    }

    for (int b_idx = thread_id; b_idx < TILE_N * TILE_K; b_idx += WARP_GROUP_THREADS)
    {
        int n = b_idx / TILE_K;
        int k = b_idx % TILE_K;

        int core_rows_id = (n / core_rows);
        int core_k_id = (k / core_k);

        int core_id = core_rows_id * num_core_k + core_k_id;
        int core_offset = core_id * (core_rows * core_k);

        int core_rows_offset = n % core_rows;
        int core_k_offset = k % core_k;

        int shmem_idx = core_offset + core_rows_offset * core_k + core_k_offset;
        b_shmem[shmem_idx] = b[n * TILE_K + k];
    }

    __syncthreads();

    uint64_t desc_a = make_smem_desc<NO_SWIZZLE>(a_shmem, 128, 128 * 2);
    uint64_t desc_b = make_smem_desc<NO_SWIZZLE>(b_shmem, 128, 128 * 2);

    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    warpgroup_arrive();

    wgmma_n8<1, 1, 1, 0, 0>(desc_a, desc_b, d);
    wgmma_commit();
    wgmma_wait<0>();

    for (int i = 0; i < 2; i++)
    {
        int m = 16 * warp_id + (i * 8) + (lane_id / 4);
        for (int j = 0; j < 2; j++)
        {
            int n = ((lane_id % 4) * 2 + j);
            c[n * TILE_M + m] = d[i * 2 + j];
        }
    }
}

template <int TILE_M, int TILE_N, int TILE_K>
void launch_wgmma_m64n8k16(bf16 *a, bf16 *b, float *c)
{
    size_t shmem_size_bytes = (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(bf16);

    wgmma_m64n8k16<TILE_M, TILE_N, TILE_K><<<1, WARP_GROUP_THREADS, shmem_size_bytes>>>(a, b, c);
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main()
{
    const int M = 64;
    const int N = 8;
    const int K = 16;

    // Initialize source matrix on host
    bf16 *a = (bf16 *)malloc(M * K * sizeof(bf16));
    bf16 *b = (bf16 *)malloc(N * K * sizeof(bf16));
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            a[i * K + j] = i + j;
        }
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++)
        {
            b[j * N + i] = i + j;
            // printf("b[%d, %d] = %f\n", j, i, float(b[j * N + i]) );
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

    printf("\n\nRunning No Swizzle WGMMA M=%d, N=%d, K=%d...\n\n", M, N, K);
    launch_wgmma_m64n8k16<M, N, K>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(gpu_output, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // check results
    bool correct = true;
    int num_incorrect = 0;
    for (int idx = 0; idx < M * N; idx++)
    {
        if (fabs(cpu_output[idx] - gpu_output[idx]) > 0.01f)
        {
            correct = false;
            int j = idx / M;
            int i = idx % M;
            printf(
                "\n mismatch at (%d, %d): CPU=%.0f, GPU=%.0f\n",
                i,
                j,
                cpu_output[idx],
                gpu_output[idx]);
            num_incorrect++;
            if (num_incorrect > 10)
            {
                break;
            }
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