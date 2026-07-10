// seeds/tiledMatmul.cu
// SATO SWARM Seed 2: Tiled Matmul — Compute + shared memory
// Self-contained tiled matrix multiply (1024x1024 default).
// Demonstrates shared memory tiling, compute intensity, launch config sensitivity.
// Target on MI300X: high % of FP32 / FP16 peak TFLOPS depending on precision.
//
// After successful port, the pipeline profiles with amd-smi during the hot kernel
// and computes achieved TFLOPS + efficiency.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define TILE_SIZE 16

// Tiled matrix multiply: C = A * B
// Each block computes a TILE_SIZE x TILE_SIZE tile of C using shared memory.
__global__ void matmulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int N) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load tiles into shared memory
    if (row < N && (t * TILE_SIZE + threadIdx.x) < N) {
      As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (col < N && (t * TILE_SIZE + threadIdx.y) < N) {
      Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot product
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < N && col < N) {
    C[row * N + col] = sum;
  }
}

int main(int argc, char** argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 1024;
  if (N % TILE_SIZE != 0) {
    // For simplicity in the seed we prefer nice multiples of TILE
    N = ((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
  }

  size_t bytes = (size_t)N * N * sizeof(float);
  printf("SATO SWARM tiledMatmul seed\n");
  printf("Matrix size: %dx%d (%.2f MB per matrix)\n", N, N, bytes / (1024.0 * 1024.0));
  printf("FLOPs (2*M*N*K): %.2f GFLOPs\n", 2.0 * N * N * N / 1e9);

  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;

  h_A = (float*)malloc(bytes);
  h_B = (float*)malloc(bytes);
  h_C = (float*)malloc(bytes);

  // Simple initialization (identity-like for easy verification)
  for (int i = 0; i < N * N; i++) {
    h_A[i] = (float)((i % 17) + 1) * 0.1f;
    h_B[i] = (float)((i % 13) + 1) * 0.1f;
    h_C[i] = 0.0f;
  }

  CHECK_CUDA(cudaMalloc(&d_A, bytes));
  CHECK_CUDA(cudaMalloc(&d_B, bytes));
  CHECK_CUDA(cudaMalloc(&d_C, bytes));

  CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

  // Warmup
  matmulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed kernel
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  matmulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // Quick sanity check on a known element (C[0][0] approx)
  float expected_c00 = 0.0f;
  for (int k = 0; k < N; k++) {
    expected_c00 += h_A[0 * N + k] * h_B[k * N + 0];
  }
  printf("Sanity C[0,0] = %.4f (expected ~%.4f)\n", h_C[0], expected_c00);

  double seconds = ms / 1000.0;
  double flops = 2.0 * (double)N * N * N;
  double achieved_tflops = flops / seconds / 1e12;

  printf("\n=== Tiled Matmul Timing ===\n");
  printf("Kernel time: %.3f ms\n", ms);
  printf("Achieved: %.2f TFLOPS\n", achieved_tflops);
  printf("Theoretical FP32 peak reference (MI300X): ~163 TFLOPS\n");
  printf("Efficiency (FP32 approx): %.1f%%\n", (achieved_tflops / 163.0) * 100.0);

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  printf("tiledMatmul seed completed successfully.\n");
  return 0;
}
