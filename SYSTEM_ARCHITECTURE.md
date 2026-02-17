# System Architecture (As Implemented)

Last synchronized with source on **February 17, 2026**.

## Documentation Maintenance Rule
Update this document whenever frame flow, module boundaries, or runtime policy changes.

## 1. High-Level Components
1. App Loop (`src/main.rs`)
- Owns window, UI state, and `render_metal::MetalRenderer`
- Drives update + render every frame
- Exposes CLI benchmark entrypoints (`--benchmark`, `--benchmark-stress`, `--benchmark-soak`, `--benchmark-soak-30m`, `--benchmark-cap-sweep`, `--benchmark-cap-pair-sweep`)
- Uses local telemetry macros (`src/telemetry.rs`) for runtime logging

2. Math Engine (`src/math/*`)
- Wavefunction model and CPU evaluators
- Spherical harmonics and radial terms
- SIMD and parallel CPU paths

3. Simulation Runtime (`src/sim/*`)
- Deterministic point generation/data layout
- Effective point-count cap
- Adaptive CPU+GPU scheduler
- Runtime effective count may be additionally capped by the native drawable cap

4. Compute Runtime (`src/render/*`)
- Camera state + frustum extraction
- Point-cloud state + CPU visibility stats + sampled intensity diagnostics
- Hybrid update path (CPU + Metal GPU evaluator)
- Runtime no-readback dispatch queue + shared GPU intensity buffer patching + scheduler diagnostics
- GPU kernel angular edge guards prevent `NaN` propagation at origin/pole samples

5. Native Metal Presentation (`src/render_metal/*`)
- Native context/pipelines/shaders
- CAMetalLayer surface management
- GPU fine culling + draw-indirect
- Native egui painter

6. UI (`src/ui/*`)
- egui controls/state
- local winit input bridge (`src/ui/input_bridge.rs`) replacing `egui-winit`
- Runtime diagnostics panel

7. Verification / CI (`.github/workflows/ci.yml`)
- Automated build and test gates on push/PR
- macOS benchmark smoke execution for runtime regression detection
- Scientific PR gates for CPU invariants/docs consistency and reduced GPU parity
- Automated scientific failure reporting + remediation issue updates

## 2. Frame Data Flow
Per frame:
1. Input/events update UI and camera.
2. UI parameters drive simulation state.
3. `render::Renderer::update_points` runs scheduler-driven CPU/GPU update.
4. `render_metal::context` encodes cull/indirect/draw/egui passes.
5. Native drawable is presented.

Stale-frame behavior:
- Completed GPU dispatches are tracked via completion metadata.
- If stale age reaches policy threshold, scheduler suppresses new GPU submissions and refreshes via CPU path.

## 3. Scheduling and Performance Control
- Scheduler target: 60 FPS
- Max stale frame age: 2
- Dynamic block sizing under CPU underutilization
- Adaptive quality scaling and exact->approx fallback under sustained budget misses
- Approximation recovery hysteresis (requires stable streak before exact-mode re-entry)
- Coarse visibility-guided block priority for scheduled compute ranges
- Separate caps for effective simulation count and native drawable shadow count

## 4. Platform Contract
- Runtime currently supports macOS only.
- Non-macOS startup exits early.
- Metal is the only presentation backend.

## 5. Maintenance Contract
When changing these, update this file:
- frame/update/render sequence
- module ownership boundaries
- stale-frame policy
- scheduler control strategy
- platform/backend assumptions
- scientific verification pipeline and remediation flow
