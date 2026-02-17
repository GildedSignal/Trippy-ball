# Wavefunction Visualization

Real-time 3D quantum-wavefunction visualization in Rust with a native Metal render path and egui controls.

## Status (Current)
Last reviewed against source code on **February 17, 2026**.

Implemented and active:
- Native Metal presentation path (`CAMetalLayer` + drawable present)
- Native Metal point render + compute culling + draw-indirect compaction
- Scheduler-owned CPU/GPU wavefunction compute with single-source intensities for presentation
- Runtime no-readback GPU scheduling path (dispatch completion metadata only)
- Native presentation consumes evaluator-owned GPU intensity buffer directly
- Coarse visibility-aware block prioritization for scheduled compute ranges
- Native render path with dirty-driven position uploads and per-frame intensity updates
- Point-cloud debug intensity metrics now use sampled stats (no full-frame normalization copy)
- Native Metal egui painter (`egui` + local winit input bridge, no `egui-wgpu` / `egui-winit`)
- Automatic CPU+GPU cooperative scheduler with stale-frame tolerance
- Effective runtime compute count is now coupled to the native drawable cap to avoid scheduling points that cannot be rendered
- Scheduler approximation recovery hysteresis to reduce exact/approx flapping under load
- Readback queue retained for parity tests/benchmark modes (not required in runtime frame loop)
- Heavy-load worker participation test coverage for the CPU parallel path
- Extended CPU/GPU parity coverage with representative spherical-grid samples (`r/theta/phi`) across `l<=5`
- Metal compute kernel now guards undefined `phi` at origin/pole samples to avoid `NaN` intensity outputs
- Monotonic simulation time semantics
- Adaptive effective-point cap for responsiveness under extreme requested counts

## Platform
This build is currently **macOS-only** at runtime while migration is finalized.

## Documentation Maintenance Policy (Mandatory)
Whenever behavior, architecture, controls, performance logic, or file structure changes, update in the same task:
1. `/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/README.md`
2. `/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/PROGRESS.md`
3. `/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/DOCUMENTATION.md`
4. `/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/SYSTEM_ARCHITECTURE.md`
5. `/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/TECHNICAL_DESIGN.md`

## Quick Start

### Requirements
- Rust stable toolchain
- macOS with Metal-capable GPU

### Run
```bash
cargo run --release
```

### Validation
```bash
cargo check
cargo test --lib -- --skip regression_large_position_upload_no_overrun
TRIPPY_BALL_SCI_SAMPLE_LEVEL=pr cargo test --test scientific_cpu_invariants --no-default-features
TRIPPY_BALL_SCI_SAMPLE_LEVEL=pr cargo test --test scientific_docs_consistency --no-default-features
TRIPPY_BALL_SCI_SAMPLE_LEVEL=pr cargo test --test scientific_gpu_parity_pr
cargo run --release -- --benchmark
cargo run --release -- --benchmark-stress
cargo run --release -- --benchmark-soak
cargo run --release -- --benchmark-soak-30m
cargo run --release -- --benchmark-cap-sweep
cargo run --release -- --benchmark-cap-pair-sweep
```

### CI
GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
- `cargo check --features app`
- `cargo check --no-default-features`
- `cargo test --lib -- --skip regression_large_position_upload_no_overrun`
- `cargo test --test scientific_cpu_invariants --no-default-features`
- `cargo test --test scientific_docs_consistency --no-default-features`
- `cargo test --test scientific_gpu_parity_pr` (macOS scientific PR parity gate)
- `cargo run --release -- --benchmark` (macOS benchmark smoke job)

Nightly scientific deep validation workflow:
- `.github/workflows/scientific-nightly.yml`
- runs expanded CPU invariants/docs and GPU parity sweep (`scientific_gpu_parity_nightly`)

Workflow file: [`/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/.github/workflows/ci.yml`](/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/.github/workflows/ci.yml)

Badge template (replace `<owner>/<repo>` once remote is configured):
```md
![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)
```

## Runtime Controls
- State 1 quantum numbers: `n` (`1..=6`), `l` (`0..=n-1`), `m` (`-l..=l`)
- State 2 quantum numbers: `n2` (`1..=6`), `l2` (`0..=n2-1`), `m2` (`-l2..=l2`)
- Coherent two-state superposition controls: `mix`, `relative_phase` (relative phase offset between states)
- Hydrogen-like model controls: `z` (nuclear charge `Z`), `time_factor` (time scale in `fs/s`)
- Color scheme and color-map tuning (gamma/contrast/brightness/log scale)
- Camera orbit/zoom/auto-rotation
- Point count request with log slider + shorthand input (`K`, `M`, `B`)

Runtime policy is fully automatic; manual CPU/GPU mode toggles are removed.

## Scientific Model
- Angular term: spherical harmonics `Y_l^m(theta, phi)` (Condon-Shortley convention)
- Radial term: hydrogenic radial functions `R_nl(r; Z)` in Bohr-radius units
- Time evolution: hydrogen-like energy levels with coherent phase evolution
- Density output: `|psi(r, t)|^2` for a coherent two-state superposition

Canonical source of scientific claims: `/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/docs/scientific_contract.md`

## Runtime Tuning Environment Variables
- `TRIPPY_BALL_MAX_RENDER_POINTS`
  - default: `5000000`
  - clamp: `1000..20000000`
  - controls effective simulation/render scheduling cap
- `TRIPPY_BALL_NATIVE_SHADOW_POINTS`
  - default: `5000000`
  - clamp: `1000..5000000`
  - controls native Metal drawable submission cap before truncation
  - runtime compute effective count is coupled to this cap when lower than the simulation cap

## Benchmark Tuning Environment Variables
- `TRIPPY_BALL_SOAK_DURATION_SECS`
  - used by `--benchmark-soak-30m`
  - default: `1800` (30 minutes)
  - clamp: `10..43200`
- `TRIPPY_BALL_CAP_SWEEP_CAPS`
  - used by `--benchmark-cap-sweep`
  - format: comma-separated integer caps (example: `2000000,5000000,10000000`)
- `TRIPPY_BALL_CAP_PAIR_SWEEP_SIM_CAPS`
  - used by `--benchmark-cap-pair-sweep`
  - format: comma-separated simulation caps (default: `2000000,5000000,10000000`)
- `TRIPPY_BALL_CAP_PAIR_SWEEP_SHADOW_CAPS`
  - used by `--benchmark-cap-pair-sweep`
  - format: comma-separated shadow caps (default: `200000,1000000,5000000`)

## Scientific Validation Environment Variables
- `TRIPPY_BALL_SCI_SAMPLE_LEVEL`
  - values: `pr` or `nightly`
  - default: `pr`
  - controls scientific test sample cardinality and tolerance profile
- `TRIPPY_BALL_SCI_OWNER`
  - optional GitHub username used by CI remediation automation for issue assignment

## Project Structure
```text
src/
  main.rs                    # App/event loop + monotonic simulation clock
  lib.rs                     # Library exports
  benchmark.rs               # Quick/stress/soak/cap sweep benchmark modes
  telemetry.rs               # Lightweight runtime logging macros
  memory/
    mod.rs
    pod.rs                   # Local POD trait/helpers (bytemuck replacement)
  math/
    mod.rs
    spherical.rs
    kernel_tables.rs
    radial.rs
    presets.rs
  sim/
    scheduler.rs             # Adaptive CPU+GPU scheduler + runtime stats
    data_layout.rs           # Deterministic point generation + point-count cap
  render/
    mod.rs                   # Scheduler orchestration + CPU/GPU intensity updates
    gpu_wavefunction.rs      # Metal compute evaluator (runtime no-readback + optional readback paths)
    points.rs                # Point data + visibility stats
    camera.rs
    color.rs
    buffer_pool.rs
  render_metal/
    mod.rs                   # App-facing renderer bridge
    context.rs               # Native Metal context/pipelines/indirect draw flow
    surface.rs               # CAMetalLayer binding to window
    metal_painter.rs         # Native egui painter
    ring_buffer.rs           # Shared storage ring slots
    shaders/
      point.metal
      wavefunction.metal
      cull.metal
      egui.metal
  ui/
    mod.rs
    controls.rs
    input_bridge.rs          # Local winit -> egui event bridge
```

## Related Docs
- `/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/DOCUMENTATION.md`
- `/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/SYSTEM_ARCHITECTURE.md`
- `/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/TECHNICAL_DESIGN.md`
- `/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/PREVENTIVE_MEASURES.md`
- `/Users/julien_laurent/Projets cursor/Small projects/Trippy ball 5/PROGRESS.md`
