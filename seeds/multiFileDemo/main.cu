// seeds/multiFileDemo/main.cu
// SATO SWARM Seed 5: multiFileDemo — synthetic multi-file project fixture
//
// THIS IS NOT A BENCHMARK SEED. Its entire purpose is testing Part A's
// multi-file support: seeds/multiFileDemo/ is a DIRECTORY (not a single
// .cu file, unlike vectorAdd/tiledMatmul/reduction/repairDemo) containing
// three files -- this one, helper.cuh, and helper.cu. main() here calls
// launchScale(), a host wrapper around a __global__ kernel, both DEFINED
// in helper.cu and DECLARED in helper.cuh -- proving hipify translates
// every file in the directory (not just one), hipcc compiles + LINKS
// multiple resulting .hip.cpp translation units into a single binary,
// and a local #include ("helper.cuh") still resolves after hipify runs
// (see src/tools/execution.py's run_hipify() docstring for why headers
// keep their original filename instead of being renamed to .hip.cpp).
//
// Otherwise deliberately simple: one scale-by-constant kernel, and the
// exact same "Result check: c[0]=... (exp ...), c[n-1]=... (exp ...)" /
// "Kernel time: X ms" output format vectorAdd.cu already uses, so this
// seed's stdout is parsed by the EXISTING regex in
// src/tools/execution.py's parse_binary_output_for_metrics() -- no new
// parser needed -- and the report shows a real hipEvent-timed kernel run
// plus real correctness validation, not just "compiled and didn't
// crash." No achieved-bandwidth/TFLOPS number is computed for this seed
// (src/baseline/pipeline.py's _compute_derived_metrics() only has a
// branch for vectorAdd/tiledMatmul/reduction) -- correctly "Not
// applicable" rather than a fabricated one, since demonstrating a
// bandwidth/TFLOPS formula was never this seed's job.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper.cuh"

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                      \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

int main(void) {
  printf("SATO SWARM multiFileDemo seed\n");

  const int n = 1 << 20;  // ~1M elements
  const float factor = 2.5f;
  const size_t bytes = (size_t)n * sizeof(float);

  float* h_in = (float*)malloc(bytes);
  float* h_out = (float*)malloc(bytes);
  for (int i = 0; i < n; i++) {
    h_in[i] = (float)(i % 100) / 100.0f;
  }

  float *d_in, *d_out;
  CHECK_CUDA(cudaMalloc(&d_in, bytes));
  CHECK_CUDA(cudaMalloc(&d_out, bytes));
  CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  launchScale(d_in, d_out, factor, n);  // defined in helper.cu
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  float expected0 = h_in[0] * factor;
  float expectedLast = h_in[n - 1] * factor;
  printf("Result check: c[0]=%.4f (exp %.4f), c[n-1]=%.4f (exp %.4f)\n",
         h_out[0], expected0, h_out[n - 1], expectedLast);

  printf("\n=== multiFileDemo Timing ===\n");
  printf("Kernel time: %.3f ms\n", ms);

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  free(h_in);
  free(h_out);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  printf("multiFileDemo seed completed successfully.\n");
  return 0;
}
