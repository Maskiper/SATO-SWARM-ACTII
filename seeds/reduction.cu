// seeds/reduction.cu
// SATO SWARM Seed 3: Parallel Reduction — Control flow, synchronization & atomics
// Tree-style reduction with multiple passes or final atomic.
// Good test for hipify handling of __syncthreads, volatile, atomics, and launch bounds.
// Target: Demonstrate solid occupancy on the detected target GPU architecture.

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

#define BLOCK_SIZE 256

// Simple parallel reduction (sum) using shared memory + atomicAdd at block level
__global__ void reduceSum(const float* __restrict__ input,
                          float* __restrict__ output,
                          size_t n) {
  __shared__ float sdata[BLOCK_SIZE];

  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * (blockDim.x * 2) + tid;
  size_t gridSize = blockDim.x * 2 * gridDim.x;

  float sum = 0.0f;

  // Grid-stride loop
  while (i < n) {
    sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    i += gridSize;
  }

  sdata[tid] = sum;
  __syncthreads();

  // Intra-block reduction
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Atomic add to global result (one atomic per block)
  if (tid == 0) {
    atomicAdd(output, sdata[0]);
  }
}

int main(int argc, char** argv) {
  size_t n = (argc > 1) ? strtoull(argv[1], NULL, 10) : (256 * 1024 * 1024ULL);

  printf("SATO SWARM reduction seed\n");
  printf("Elements: %zu\n", n);

  float *h_in, *h_out;
  float *d_in, *d_out;

  h_in = (float*)malloc(n * sizeof(float));
  h_out = (float*)calloc(1, sizeof(float));  // host result

  if (!h_in || !h_out) {
    fprintf(stderr, "Host alloc failed\n");
    return 1;
  }

  // Fill with 1.0f so final sum == n (easy verification)
  for (size_t i = 0; i < n; i++) h_in[i] = 1.0f;

  CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));

  int blockSize = BLOCK_SIZE;
  int numBlocks = (int)min((size_t)1024, (n + blockSize * 2 - 1) / (blockSize * 2));

  // Warmup
  reduceSum<<<numBlocks, blockSize>>>(d_in, d_out, n);
  CHECK_CUDA(cudaDeviceSynchronize());

  // d_out accumulates via atomicAdd (unlike vectorAdd/tiledMatmul, where
  // every thread writes its own deterministic output index and a second
  // launch is harmless) -- it MUST be reset before the timed run, or the
  // warmup's contribution is still sitting there and gets added to again,
  // silently doubling the result. This was a real bug: without this
  // reset, the timed run reports 2*n instead of n.
  CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  reduceSum<<<numBlocks, blockSize>>>(d_in, d_out, n);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

  printf("Reduction result: %.0f (expected %.0f)\n", h_out[0], (double)n);
  printf("Kernel time: %.3f ms\n", ms);

  // Very rough "effective bandwidth" for a reduction (read only once)
  double seconds = ms / 1000.0;
  double bytes_read = (double)n * sizeof(float);
  double bw_gbs = bytes_read / seconds / 1e9;
  printf("Effective read BW: %.2f GB/s\n", bw_gbs);

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  free(h_in);
  free(h_out);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  // Note: small numerical drift is acceptable for large reductions; validator will use tolerance
  printf("reduction seed completed.\n");
  return 0;
}
