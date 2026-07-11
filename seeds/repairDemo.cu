// seeds/repairDemo.cu
// SATO SWARM Seed 4: repairDemo — Intentional repair-loop test case
//
// THIS IS NOT A HIDDEN TRICK. This file is deliberately constructed to
// fail at one specific, well-understood, VERIFIED-real step, for testing
// the automated repair loop (src/baseline/pipeline.py's
// _attempt_hipcc_repair(), built and tested — see scripts/test_repair_loop.py).
// It uses src/agents/tools.py's apply_search_replace to apply the fix
// recorded in memory/porting_patterns.jsonl's
// "gap_cudaCtxResetPersistingL2Cache" entry's auto_fix field. Nothing
// else in this file is unusual on purpose.
//
// THE GAP (verified against real, live source — not invented):
//
//   cudaCtxResetPersistingL2Cache() is a real CUDA Runtime API function:
//   `__host__ cudaError_t cudaCtxResetPersistingL2Cache(void)` — no
//   arguments, resets all persisting L2 cache lines to normal status,
//   part of the API since CUDA 11.3. Documented at:
//   https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
//
//   hipify-perl (github.com/ROCm/HIPIFY, amd-develop branch,
//   bin/hipify-perl) explicitly knows HIP has no equivalent for it:
//     - Listed in %hash_HipOnlyUnsupportedFunctions (hash declared at
//       bin/hipify-perl:3114): the exact line
//       `'cudaCtxResetPersistingL2Cache' => 1,` is at
//       bin/hipify-perl:5584.
//     - Confirmed ABSENT from %map_core, the actual substitution table
//       (bin/hipify-perl:10286-16314) — grepped directly, zero matches.
//     - When hipify-perl finds this identifier as a bare word token in
//       source, it prints `warning: unsupported HIP identifier:
//       cudaCtxResetPersistingL2Cache` to STDERR
//       (bin/hipify-perl:18186-18190) but this is PURELY informational:
//       there is no exit() call anywhere in the ~18,300-line script tied
//       to warnings (confirmed by search — zero matches). hipify-perl
//       exits 0 regardless, and leaves the identifier byte-for-byte
//       unchanged in the translated output (since it's absent from the
//       substitution table, nothing rewrites it).
//
//   HIP genuinely has no equivalent, under any name: ROCm/HIP's real
//   public header (github.com/ROCm/HIP, develop branch,
//   include/hip/hip_runtime_api.h, 10,524 lines) was fetched and
//   searched directly — no hipCtxResetPersistingL2Cache or equivalent
//   reset/clear function exists anywhere in it. HIP DOES model the
//   *setup* side of this same feature area (hipAccessPolicyWindow,
//   hipAccessPropertyPersisting, hipDeviceAttributePersistingL2CacheMaxSize
//   all exist) — which is exactly what makes this gap easy to miss in
//   practice: an engineer who already ported code using the
//   access-policy-window APIs (which DO translate cleanly) would
//   reasonably assume the matching reset call has a HIP equivalent too.
//   It doesn't.
//
// EXPECTED PIPELINE BEHAVIOR:
//   1. hipify (Porting phase): SUCCEEDS, exit 0 — src/tools/execution.py's
//      run_hipify() reports ok=True (it only checks rc, not stderr
//      content). The unsupported-identifier warning is real but silent
//      from the pipeline's point of view — it lands in logs/hipify.log,
//      not in job.error.
//   2. hipcc (Porting phase, compile step): FAILS — the untranslated
//      cudaCtxResetPersistingL2Cache() call is not a valid symbol in any
//      ROCm/HIP header, producing a genuine "use of undeclared
//      identifier" compile error. This is the exact FAILED state
//      src/baseline/pipeline.py's run_baseline() already handles (see
//      its compile_success branch) — nothing pipeline-side needs to
//      change for this seed to produce a real, diagnosable failure.
//
// Everything else here is deliberately as simple as possible — a single
// no-op kernel launch — so the one intentional gap isn't buried in
// unrelated complexity.

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

__global__ void trivialKernel(int* flag) {
  if (threadIdx.x == 0) {
    *flag = 1;
  }
}

int main(void) {
  printf("SATO SWARM repairDemo seed\n");

  int* d_flag;
  CHECK_CUDA(cudaMalloc(&d_flag, sizeof(int)));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  trivialKernel<<<1, 32>>>(d_flag);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  int h_flag = 0;
  CHECK_CUDA(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
  printf("Flag check: %d (expected %d)\n", h_flag, 1);

  printf("\n=== repairDemo Timing ===\n");
  printf("Kernel time: %.3f ms\n", ms);

  // THE INTENTIONAL GAP -- see the file header comment above for the
  // full verified evidence. hipify-perl leaves this line completely
  // untouched; hipcc will fail to compile it.
  CHECK_CUDA(cudaCtxResetPersistingL2Cache());

  CHECK_CUDA(cudaFree(d_flag));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  printf("repairDemo seed completed successfully.\n");
  return 0;
}
