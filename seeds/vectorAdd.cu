// seeds/vectorAdd.cu
// SATO SWARM Seed 1: vectorAdd — Memory bandwidth hero
// Self-contained CUDA example for CUDA → HIP autonomous porting.
// Target: High % of the actual detected GPU's memory bandwidth peak —
// see src/tools/execution.py's detect_gpu_theoretical_peaks(), which
// computes the real peak live from rocminfo/amd-smi; no specific GPU or
// bandwidth number is assumed here.
//
// This file + minimal host driver is the complete input for the pipeline.
// After hipify + hipcc + run on real hardware, the baseline captures
// amd-smi metrics and computes achieved BW from real measurements.

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

// Simple vector add kernel — classic memory-bound workload
__global__ void vectorAdd(const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ c,
                          size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char** argv) {
  // Default: ~256M elements → ~1 GB read + 0.5 GB write = ~1.5 GB moved per run
  size_t n = (argc > 1) ? strtoull(argv[1], NULL, 10) : (256 * 1024 * 1024ULL);
  size_t bytes = n * sizeof(float);

  printf("SATO SWARM vectorAdd seed\n");
  printf("Elements: %zu (%.2f MB per vector)\n", n, bytes / (1024.0 * 1024.0));
  printf("Total data moved (read+write): %.2f GB\n", (bytes * 2.0) / 1e9);

  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;

  h_a = (float*)malloc(bytes);
  h_b = (float*)malloc(bytes);
  h_c = (float*)malloc(bytes);

  if (!h_a || !h_b || !h_c) {
    fprintf(stderr, "Host allocation failed\n");
    return 1;
  }

  // Initialize with simple pattern (no need for high entropy for BW test)
  for (size_t i = 0; i < n; i++) {
    h_a[i] = (float)(i & 0xFF) * 0.01f;
    h_b[i] = (float)((i + 17) & 0xFF) * 0.01f;
    h_c[i] = 0.0f;
  }

  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMalloc(&d_c, bytes));

  CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  // Warmup
  int blockSize = 256;
  int gridSize = (int)((n + blockSize - 1) / blockSize);
  vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed run — use events for accurate kernel time (hipEvent after port)
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  // Copy back result (for validator correctness check)
  CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

  // Simple self-check (first + last elements)
  float expected_first = h_a[0] + h_b[0];
  float expected_last = h_a[n-1] + h_b[n-1];
  printf("Result check: c[0]=%.4f (exp %.4f), c[n-1]=%.4f (exp %.4f)\n",
         h_c[0], expected_first, h_c[n-1], expected_last);

  // Report timing + derived BW (baseline pipeline will also compute this + amd-smi)
  double seconds = ms / 1000.0;
  double bytes_moved = (double)bytes * 2.0;  // read A + B, write C
  double achieved_gbs = bytes_moved / seconds / 1e9;

  printf("\n=== vectorAdd Timing ===\n");
  printf("Kernel time: %.3f ms\n", ms);
  printf("Achieved bandwidth: %.2f GB/s\n", achieved_gbs);
  // No theoretical-peak / efficiency-% line here on purpose: this seed
  // doesn't know which GPU it's running on, so it can't correctly compute
  // that without hardcoding an assumption. The pipeline computes
  // efficiency downstream, from a live rocminfo/amd-smi query of the
  // actual GPU present — see src/tools/execution.py's
  // detect_gpu_theoretical_peaks().

  // Cleanup
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  printf("vectorAdd seed completed successfully.\n");
  return 0;
}
