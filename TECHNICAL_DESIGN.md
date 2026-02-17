# Technical Design (Current Code)

Last synchronized with source on **February 17, 2026**.

## Documentation Maintenance Rule
Update this file whenever module APIs, wiring, or shader/pipeline contracts change.

## 1. Crate Layout
- `src/main.rs`: app bootstrap + event loop
- `src/lib.rs`: module exports
- `src/math/`: math kernels and presets
- `src/sim/`: scheduler and data-layout utilities
- `src/render/`: scheduler-facing runtime compute state
- `src/render_metal/`: native Metal presentation stack
- `src/ui/`: egui state and controls
- `src/memory/`: local POD abstractions used by Metal buffer uploads
- `src/telemetry.rs`: local logging macros (no external logger dependency)
- `src/benchmark.rs`: quick/stress/soak/timed-soak/cap-sweep/cap-pair-sweep benchmark modes

## 2. Core Runtime Types
### Math
`WavefunctionParams` (`src/math/mod.rs`):
- `n: usize`
- `l: usize`
- `m: isize`
- `n2: usize`
- `l2: usize`
- `m2: isize`
- `mix: f64`
- `relative_phase: f64`
- `z: f64`
- `time_factor: f64` (`fs/s`)

### Scheduler
`RuntimePolicy` (`src/sim/scheduler.rs`):
- `target_fps`
- `max_stale_frames`
- `block_size`
- `cpu_threads`

`SimulationRuntimeStats` includes:
- frame/gpu/cpu timings
- queue depth + stale frames
- approx mode flag
- worker utilization
- scheduler block size + quality

`AdaptiveScheduler` behavior highlights:
- enters approximation after sustained misses/severe overload
- exits approximation only after a stable recovery streak (hysteresis)

### Render
`render::Renderer` owns:
- `Camera`
- `PointCloud`
- `ColorMap`
- `GpuWavefunctionEvaluator`
- `AdaptiveScheduler`
- merged intensity state

`render_metal::MetalRenderer` owns:
- `render::Renderer` inner state
- native `MetalContext`
- native `MetalSurface`
- native egui `MetalPainter`

## 3. Compute Contracts
### CPU
`evaluate_wavefunction_batch(positions, params, time, intensities)`:
- equal-length input/output slices required
- writes normalized-ready intensity values (after downstream normalization)

`PointCloud::update_intensities(intensities)`:
- validates slice length against effective point count
- computes sampled normalized intensity diagnostics (min/max/avg/zero estimate)
- avoids full-frame normalized intensity buffer copies in hot path

`PointCloud::prioritized_compute_ranges(block_size, scheduled_points)`:
- emits block-aligned compute ranges for the frame budget
- falls back to sequential ranges when culling validity is unavailable
- prefers blocks with higher coarse visibility score when frustum culling data is valid

`PointCloud::set_effective_cap_override(cap)`:
- applies a runtime effective-count cap override (used by Metal renderer to couple compute with drawable cap)
- reinitializes point storage when override changes effective count

### GPU
`GpuWavefunctionEvaluator::enqueue_compute_range(params, time, start, count) -> bool`:
- queues Metal compute for the requested range
- returns false when queue is full or range invalid
- compute bindings:
  - `buffer(0)`: position buffer (`packed_float3`)
  - `buffer(1)`: intensity buffer (`f32`)
  - `buffer(2)`: per-dispatch params
  - `buffer(3)`: shared normalization table (`l<=5`) sourced from `math/kernel_tables`
- kernel angular edge semantics now force `phi=0` when `x/y` are near zero (origin/poles), preventing `NaN` propagation

`GpuWavefunctionEvaluator::poll_completed() -> Vec<CompletedReadback>`:
- returns completed range chunks without blocking frame loop
- readback staging copies only intensity ranges (no position payload)

`GpuWavefunctionEvaluator::enqueue_compute_range_no_readback(...)` + `poll_completed_no_readback()`:
- runtime scheduler hot path (no frame-loop GPU->CPU readback dependency)
- returns completion metadata (points/ms) used by scheduler feedback and stale-frame control

`GpuWavefunctionEvaluator::upload_intensity_range(start, cpu_values)`:
- patches CPU-computed fallback ranges directly into evaluator GPU intensity buffer for presentation parity

`GpuWavefunctionEvaluator::copy_intensity_sample(sample_count, out)`:
- reads a strided sample from evaluator intensity storage for low-overhead runtime diagnostics

Parity test coverage in `src/render/gpu_wavefunction.rs` now includes:
- point-cloud samples
- full `l,m` sweep for `l<=5`
- spherical-coordinate grid samples (`r/theta/phi`) across multiple time/parameter regimes

## 4. Native Metal Pipeline Contracts
Defined in `src/render_metal/context.rs` and `src/render_metal/shaders/*`:
- compute cull: `cull_points`
- compute finalize indirect args: `finalize_cull`
- render: `point_vs` + `point_fs`
- UI: `egui_vs` + `egui_fs`

Frame encode order:
1. cull compute
2. finalize indirect args
3. point draw indirect
4. egui draw
5. present drawable

Wavefunction compute kernel (`evaluate_wavefunction`) remains part of the scheduler GPU evaluator path in `src/render/gpu_wavefunction.rs` and is not dispatched by the presentation path.

Native presentation buffer layout:
- position buffer (static-ish, uploaded on point-count/capacity changes)
- intensity source prefers evaluator-owned shared GPU intensity buffer (fallback local upload path remains)
- visible-index compact buffer (written by cull pass)

## 5. Point-Cap Controls
- Effective simulation cap: `TRIPPY_BALL_MAX_RENDER_POINTS` (`1000..20000000`, default `5000000`)
- Native drawable cap: `TRIPPY_BALL_NATIVE_SHADOW_POINTS` (`1000..5000000`, default `5000000`)
- Runtime compute effective cap is coupled to `min(simulation cap, native drawable cap)`.

## 6. Known Constraints
- Runtime is macOS-only.
- Full exact 100M-per-frame path is not implemented; scheduler fallback prioritizes responsiveness.
- `egui` paint callbacks (`Primitive::Callback`) are intentionally skipped in current native painter.
- Clipboard and URL-open platform outputs from egui are currently traced/ignored by the local input bridge.

## 7. Validation Notes
- CPU heavy-load participation is covered by `math::tests::heavy_parallel_workload_engages_cpu_workers` using an 8-thread Rayon pool.
- Local integration layers have focused tests:
  - `ui::input_bridge::tests::*`
  - `memory::pod::tests::*`
  - `telemetry::tests::*`
- CI pipeline is defined in `.github/workflows/ci.yml` and runs the primary check/test gates plus macOS benchmark smoke.
- Scientific validation adds explicit gates:
  - `tests/scientific_cpu_invariants.rs`
  - `tests/scientific_docs_consistency.rs`
  - `tests/scientific_gpu_parity_pr.rs`
  - `tests/scientific_gpu_parity_nightly.rs` (nightly workflow)
- Scientific contract baseline is maintained in `docs/scientific_contract.md`.
