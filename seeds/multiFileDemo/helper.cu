// seeds/multiFileDemo/helper.cu
// Kernel + host launch wrapper declared in helper.cuh, called from
// main.cu — see main.cu's header comment for what this seed tests.

#include "helper.cuh"

__global__ void scaleKernel(const float* in, float* out, float factor, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * factor;
  }
}

void launchScale(const float* d_in, float* d_out, float factor, int n) {
  const int blockSize = 256;
  const int gridSize = (n + blockSize - 1) / blockSize;
  scaleKernel<<<gridSize, blockSize>>>(d_in, d_out, factor, n);
}
