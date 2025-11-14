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

////////////////////////////////////////////////////////////////////////////////
// Part 0: No Swizzle WGGMA load for M = 64, N = 8, K = 16
////////////////////////////////////////////////////////////////////////////////
#define CORE 8
template <int TILE_M, int TILE_N, int TILE_K>
__global__ void wgmma_m64n8k16(bf16 *a, bf16 *b, float *c)
{

    extern __shared__ bf16 shmem[];
    bf16 *a_vals = shmem;
    bf16 *b_vals = a_vals + TILE_M * TILE_K;

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = idx; i < TILE_M * TILE_K; i += blockDim.x * blockDim.y)
    {
        int row = i / TILE_K;
        int col = i % TILE_K;

        int chunk_row = row / CORE;
        int chunk_col = col / CORE;

        int in_row = row % CORE;
        int in_col = col % CORE;

        int chunk_size = CORE * CORE;
        int chunks_k = TILE_K / CORE;

        int shmem_idx = (chunk_row * chunks_k + chunk_col) * chunk_size + in_row * CORE + in_col;

        a_vals[shmem_idx] = a[i];
    }

    for (int i = idx; i < TILE_K * TILE_N; i += blockDim.x * blockDim.y)
    {
        int row = i / TILE_K;
        int col = i % TILE_K;

        int chunk_row = row / CORE;
        int chunk_col = col / CORE;

        int in_row = row % CORE;
        int in_col = col % CORE;

        int chunk_size = CORE * CORE;
        int chunks_k = TILE_K / CORE;

        int shmem_idx = (chunk_row * chunks_k + chunk_col) * chunk_size + in_row * CORE + in_col;

        b_vals[shmem_idx] = b[i];
    }

    __syncthreads();

    uint64_t a_desc = make_smem_desc<NO_SWIZZLE>(a_vals, 128, 256);
    uint64_t b_desc = make_smem_desc<NO_SWIZZLE>(b_vals, 128, 256);

    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    warpgroup_arrive();

    wgmma_n8<1, 1, 1, 0, 0>(a_desc, b_desc, d);
    wgmma_commit();
    wgmma_wait<0>();

    int x = threadIdx.x;
    int y = threadIdx.y;

    for (int i = 0; i < 2; i++)
    {
        int m = 16 * y + (8 * i) + x / 4;
        for (int j = 0; j < 2; j++)
        {
            int n = (x % 4) * 2 + j;
            c[n * TILE_M + m] = d[i * 2 + j];
        }
    }
}

template <int TILE_M, int TILE_N, int TILE_K>
void launch_wgmma_m64n8k16(bf16 *a, bf16 *b, float *c)
{

    // <--- your code here --->
    dim3 threads = dim3(32, 4); // one group
    int shmem_size = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(bf16);

    wgmma_m64n8k16<TILE_M, TILE_N, TILE_K><<<1, threads, shmem_size>>>(a, b, c);
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
