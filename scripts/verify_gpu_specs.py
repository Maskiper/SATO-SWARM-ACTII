#!/usr/bin/env python3
"""GPU theoretical-peak spec verification — standalone, real-hardware only.

Run this directly on the real pod (no full pipeline job needed) to see
exactly what rocminfo + amd-smi report, what
detect_gpu_theoretical_peaks() (src/tools/execution.py) computes from
that, and to sanity-check the result against the card's published spec
sheet BEFORE trusting a real pipeline run's efficiency percentages.

This exists specifically to check one known open risk: GDDR6 cards
(RDNA, gfx1xxx) are sometimes monitored with memory clock reported at
roughly 1/4 of the true pin-rate base clock -- a legacy display
convention some tools inherited from GDDR5 (e.g. a real 16 Gbps GDDR6
chip, as shipped on the RX 6800 XT, is widely reported by monitoring
tools as "2000 MHz", not 8000). If amd-smi follows that same convention
for this card, the computed bandwidth will undershoot the real spec by
roughly 2x. HBM/CDNA cards (MI300X etc.) are not believed to be affected
by this specific risk -- HBM's JEDEC-clock-to-per-pin-rate relationship
is a clean x2. See _query_amd_smi_mem_bus_specs()'s docstring in
src/tools/execution.py for the full reasoning.

Usage (on the pod; SATOSWARM_MOCK must be unset or 0 -- this script
refuses to run in mock mode, since the whole point is a real query):
    python scripts/verify_gpu_specs.py

Prints a human-readable report and also saves it to
preflight_logs/gpu_specs_verify.log.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.execution import MOCK, detect_gpu_theoretical_peaks  # noqa: E402


def _save(lines: list[str]) -> Path:
    out_dir = Path("preflight_logs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "gpu_specs_verify.log"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 70)
    emit("SATO SWARM -- GPU Theoretical Peak Spec Verification")
    emit("=" * 70)
    emit(f"Python: {sys.version.split()[0]}")
    emit(f"Working dir: {Path.cwd()}")
    emit()

    if MOCK:
        emit("ABORTING: SATOSWARM_MOCK=1 is set.")
        emit("This script queries real hardware via rocminfo/amd-smi. With mock")
        emit("mode on, detect_gpu_theoretical_peaks() returns the fixed MI300X")
        emit("mock calibration instead (5300 GB/s / 163.4 TFLOPS) -- it would")
        emit("print output that LOOKS like a real query but isn't.")
        emit()
        emit("Run:  unset SATOSWARM_MOCK   (bash)  or  Remove-Item Env:SATOSWARM_MOCK  (PowerShell)")
        emit("...and try again.")
        _save(lines)
        sys.exit(1)

    emit("Querying rocminfo + amd-smi (real mode, no mock) -- a few seconds...")
    emit()
    peaks = detect_gpu_theoretical_peaks()

    emit("-" * 70)
    emit("PARSED VALUES")
    emit("-" * 70)
    emit(f"Detected architecture (gfx code):   {peaks.gpu_arch!r}")
    emit(f"Marketing name (rocminfo):          {peaks.marketing_name!r}")
    emit(f"Architecture family (classified):   {peaks.arch_family!r}")
    emit()
    emit(f"Compute units (rocminfo):           {peaks.compute_units!r}")
    emit(f"Max engine clock, MHz (rocminfo):   {peaks.engine_clock_mhz!r}")
    emit(f"FLOPs/clock/CU (family constant):   {peaks.flops_per_clock_per_cu!r}")
    emit()
    emit(f"Max memory clock, MHz (amd-smi):    {peaks.mem_clock_mhz!r}")
    emit(f"Memory bus width, bits (amd-smi):   {peaks.bus_width_bits!r}")
    emit()

    emit("-" * 70)
    emit("COMPUTED RESULT")
    emit("-" * 70)
    emit(f"Bandwidth:   {peaks.bandwidth_gbs!r} GB/s   (source: {peaks.bandwidth_source})")
    emit(f"  {peaks.bandwidth_formula_str()}")
    emit()
    emit(f"FP32 TFLOPS: {peaks.fp32_tflops!r} TFLOPS   (source: {peaks.tflops_source})")
    emit(f"  {peaks.tflops_formula_str()}")
    emit()

    emit("-" * 70)
    emit("SANITY CHECK -- do this before trusting a real pipeline run's efficiency%")
    emit("-" * 70)
    card = peaks.marketing_name or peaks.gpu_arch or "this card"
    if peaks.bandwidth_source == "runtime":
        emit(f"Bandwidth was computed LIVE (not the fallback table). Look up {card}'s")
        emit(f"published spec sheet bandwidth and compare it to {peaks.bandwidth_gbs:g} GB/s above:")
        emit("  - Roughly MATCHES  -> amd-smi's memory-clock convention agrees with")
        emit("    the x2-DDR formula here. Trust it.")
        emit("  - Roughly HALF the spec sheet number -> amd-smi is very likely")
        emit("    reporting memory clock using the lower 'quarter-rate' display")
        emit("    convention some GDDR6 monitoring tools use (see the RX 6800 XT")
        emit("    example in _query_amd_smi_mem_bus_specs()'s docstring, in")
        emit("    src/tools/execution.py). If so, mem_clock_mhz above needs an")
        emit("    extra x2 applied (x4 total, not x2) specifically when reading")
        emit("    this field -- fix it in that function, not by hand-adjusting")
        emit("    individual runs.")
        emit("  - Wildly different (not ~1x or ~0.5x) -> one of the amd-smi/")
        emit("    rocminfo key paths parsed here is probably wrong for this ROCm")
        emit("    version -- check the raw output in the full log below against")
        emit("    the key paths tried in _query_amd_smi_mem_bus_specs().")
    elif peaks.bandwidth_source == "fallback_table":
        emit("Bandwidth came from the FALLBACK TABLE, not a live query -- amd-smi")
        emit("didn't return a usable memory clock and/or bus width on this ROCm")
        emit("version. Check the raw amd-smi output in the full log below against")
        emit("the key paths tried in _query_amd_smi_mem_bus_specs()")
        emit("(src/tools/execution.py), and add/fix the key names there.")
    else:
        emit("Bandwidth is UNAVAILABLE -- neither the live query nor the fallback")
        emit("table produced a number for this architecture. Check the raw")
        emit("rocminfo/amd-smi output in the full log below to see what's missing.")
    emit()
    if peaks.tflops_source == "runtime":
        emit(f"FP32 TFLOPS was computed LIVE. Compare {peaks.fp32_tflops:g} TFLOPS against {card}'s")
        emit("published SINGLE-ISSUE FP32 spec specifically -- not any 'dual-issue'")
        emit("marketing figure some RDNA3 cards also quote (roughly 2x higher),")
        emit("which this deliberately doesn't use. See the comment on")
        emit("_CU_FLOPS_PER_CLOCK_BY_FAMILY in src/tools/execution.py for why.")
    elif peaks.tflops_source == "fallback_table":
        emit("FP32 TFLOPS came from the FALLBACK TABLE -- rocminfo didn't return a")
        emit("usable compute-unit count / max clock, or the architecture family")
        emit("wasn't recognized. Check the raw rocminfo output below.")
    else:
        emit("FP32 TFLOPS is UNAVAILABLE -- see the raw rocminfo output below.")
    emit()

    emit("-" * 70)
    emit("FULL RAW QUERY LOG (rocminfo + amd-smi, verbatim)")
    emit("-" * 70)
    emit(peaks.calculation_log)
    emit()

    out_path = _save(lines)
    print(f"Saved to {out_path}")

    if peaks.bandwidth_source == "unavailable" and peaks.tflops_source == "unavailable":
        print()
        print("Both bandwidth and TFLOPS came back unavailable -- rocminfo and")
        print("amd-smi may not be working at all here. Run scripts/preflight.sh")
        print("first to confirm the base toolchain is actually present.")
        sys.exit(2)


if __name__ == "__main__":
    main()
