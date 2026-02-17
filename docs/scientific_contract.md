# Scientific Contract

Last synchronized with implementation on **February 17, 2026**.

## Purpose
This contract defines the scientific model implemented by the application, the accepted approximations, and the claims that must stay consistent across code, UI text, and documentation.

## Implemented Model (Normative)
The rendered density is the squared magnitude of a coherent two-state superposition:

\[
\rho(\mathbf{r}, t) = |\psi(\mathbf{r}, t)|^2
\]

\[
\psi(\mathbf{r}, t) =
\sqrt{1-\text{mix}}\,R_{n l}(r; Z)\,Y_l^m(\theta,\phi)\,e^{-iE_n t_\mathrm{phys}/\hbar}
+ \sqrt{\text{mix}}\,R_{n_2 l_2}(r; Z)\,Y_{l_2}^{m_2}(\theta,\phi)\,e^{-iE_{n_2} t_\mathrm{phys}/\hbar + i\phi_\mathrm{rel}}
\]

with hydrogen-like energy levels:

\[
E_n = -\text{Ry}\,\frac{Z^2}{n^2}
\]

and hydrogenic radial terms \(R_{nl}(r;Z)\), spherical harmonics \(Y_l^m\), and Condon-Shortley phase convention.

## Parameter Semantics (Normative)
- `n`, `l`, `m`: quantum numbers for state 1, with `n >= 1`, `0 <= l < n`, `|m| <= l`.
- `n2`, `l2`, `m2`: quantum numbers for state 2 with the same validity constraints.
- `mix`: population weight for state 2 in `[0, 1]` (amplitudes use square-root mixing).
- `relative_phase`: relative phase offset between states (radians).
- `z`: nuclear charge Z for hydrogen-like ions (`Z >= 1` in UI; numerically clamped to a small positive lower bound in kernels).
- `time_factor`: simulation time scale (fs/s). Physical time is `t_phys = sim_time * time_factor * 1e-15`.
- `r`: treated in Bohr-radius units (`a0`) throughout the radial implementation.

## Numerical Conventions and Guards
- Origin guard: when `r` is near zero, `theta` is forced to `0`.
- Pole guard: when `x` and `y` are near zero, `phi` is forced to `0` to avoid undefined-angle NaN propagation.
- Quantum-number validation: invalid `(n,l,m)` combinations evaluate to zero density.
- Density safety: expected output density is finite and non-negative up to floating-point tolerance.

## Explicitly Unsupported Claims
The current implementation must not be described as:
- a Gaussian radial model for runtime density computation,
- a sinusoid-only time modulation model,
- a complete many-body quantum simulation.

## Verification Requirements
Scientific checks must enforce:
1. Angular normalization and orthogonality behavior.
2. Radial normalization behavior for representative states.
3. CPU/reference consistency and CPU/GPU parity under fixed tolerances.
4. Documentation consistency with this contract.
