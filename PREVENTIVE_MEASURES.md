# Preventive Measures (Implemented + Operational)

Last synchronized with source on **February 16, 2026**.

## Documentation Maintenance Rule
Update this file whenever reliability, safety, or performance safeguards change.

## 1. Numerical and Data Safety
- Quantum-number validity enforced in spherical harmonics (`|m| <= l`).
- Coordinate conversions and wavefunction math include epsilon guards around singular regions.
- Intensity normalization clamps values into render-safe range.
- Non-finite intensity handling is guarded before UI/debug reporting.

## 2. Runtime Safety
- Monotonic simulation clock from app startup (`simulation_start`).
- Stale-frame age is bounded by scheduler policy (`max_stale_frames`).
- GPU queue pressure can force CPU refresh path to avoid unbounded stale accumulation.

## 3. Resource Safety
- Shared/staging buffer reuse to reduce allocation churn.
- Explicit buffer bounds checks on uploads/readback copies.
- Deterministic point-count resize path with full buffer revalidation.

## 4. Performance Safeguards
- Adaptive CPU+GPU scheduler with EWMA throughput estimation.
- Dynamic block-size retuning for CPU worker utilization.
- Adaptive quality/fallback policy under sustained frame-budget misses.
- GPU fine culling + draw-indirect compaction to reduce unnecessary draw work.
- Configurable effective/native caps via env vars for controlled load escalation.

## 5. Operational Safeguards
- Runtime exits early on unsupported OS instead of running in undefined backend state.
- Bench suites provide quick, stress, and staged soak profiling.
- Per-stage soak diagnostics help isolate regression hotspots.
- Scientific guardrail suites enforce quantitative CPU invariants, CPU/GPU parity, and docs-claim consistency.
- Scientific CI failures emit structured reports and auto-open/update remediation issues.

## 6. Open Risk Areas
- Large point-count behavior remains hardware sensitive.
- Benchmark variance exists due asynchronous GPU completion timing.
- 100M exact-per-frame target is not currently guaranteed.

## 7. Required Maintenance
When adding/changing safeguards:
1. Update this file with safeguard and protected failure mode.
2. Update `/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/PROGRESS.md`.
3. Update user-facing docs if behavior changed.
