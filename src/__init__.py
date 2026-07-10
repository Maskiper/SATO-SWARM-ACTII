"""SATO SWARM — Autonomous CUDA -> ROCm migration pipeline for real AMD GPU hardware.

Sequential pipeline: hipify -> hipcc -> validate -> benchmark + amd-smi -> report.
Everything that matters executes natively on the AMD GPU via ROCm. The
target GPU architecture (MI300X, RDNA3, whatever's actually present) is
auto-detected at compile time — see src/tools/execution.py's
detect_gpu_arch() — never assumed.
"""
__version__ = "0.1.0"
