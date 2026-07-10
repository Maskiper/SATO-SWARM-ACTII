"""SATO SWARM — Autonomous CUDA -> ROCm migration pipeline for real AMD MI300X hardware.

Sequential pipeline: hipify -> hipcc -> validate -> benchmark + amd-smi -> report.
Everything that matters executes natively on AMD Instinct MI300X via ROCm.
"""
__version__ = "0.1.0"
