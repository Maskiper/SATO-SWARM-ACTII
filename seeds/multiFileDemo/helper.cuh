// seeds/multiFileDemo/helper.cuh
// Declares the kernel + host launch wrapper defined in helper.cu.
//
// Kept as a genuinely separate header (not folded into main.cu) so this
// seed exercises a real local #include across translation units — see
// main.cu's header comment for what this seed as a whole is testing.

#ifndef SATO_SWARM_MULTIFILEDEMO_HELPER_CUH
#define SATO_SWARM_MULTIFILEDEMO_HELPER_CUH

__global__ void scaleKernel(const float* in, float* out, float factor, int n);

// Host-side wrapper: computes launch geometry and issues the kernel
// launch. Defined in helper.cu, called from main.cu — the actual
// cross-translation-unit link this seed is testing.
void launchScale(const float* d_in, float* d_out, float factor, int n);

#endif  // SATO_SWARM_MULTIFILEDEMO_HELPER_CUH
