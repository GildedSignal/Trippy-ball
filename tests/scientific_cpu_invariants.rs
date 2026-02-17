#[path = "scientific/mod.rs"]
mod scientific;

use num_complex::Complex64;
use scientific::fixtures;
use scientific::reference;
use scientific::tolerances::{current_tolerances, sample_level};
use wavefunction_visualization::math::{evaluate_wavefunction, WavefunctionParams};

#[test]
fn angular_normalization_is_close_to_one() {
    let tolerances = current_tolerances();
    let pairs = fixtures::representative_lm_pairs(sample_level());
    let grid = fixtures::integration_theta_phi_grid(tolerances.theta_samples, tolerances.phi_samples);

    for (l, m) in pairs {
        let mut integral = 0.0f64;
        for (theta, phi, weight) in &grid {
            let y = reference::spherical_harmonic(l, m, *theta, *phi);
            integral += y.norm_sqr() * *weight;
        }

        let err = (integral - 1.0).abs();
        assert!(
            err <= tolerances.angular_normalization_abs,
            "angular normalization failed for (l={}, m={}): integral={} err={} tol={}",
            l,
            m,
            integral,
            err,
            tolerances.angular_normalization_abs
        );
    }
}

#[test]
fn angular_orthogonality_is_respected() {
    let tolerances = current_tolerances();
    let grid = fixtures::integration_theta_phi_grid(tolerances.theta_samples, tolerances.phi_samples);
    let mismatched_pairs = [
        ((1usize, 0isize), (1usize, 1isize)),
        ((2usize, -1isize), (3usize, -1isize)),
        ((3usize, 2isize), (3usize, -2isize)),
        ((5usize, 0isize), (4usize, 0isize)),
    ];

    for ((l1, m1), (l2, m2)) in mismatched_pairs {
        let mut integral = Complex64::new(0.0, 0.0);
        for (theta, phi, weight) in &grid {
            let y1 = reference::spherical_harmonic(l1, m1, *theta, *phi);
            let y2 = reference::spherical_harmonic(l2, m2, *theta, *phi);
            integral += y1 * y2.conj() * *weight;
        }

        let magnitude = integral.norm();
        assert!(
            magnitude <= tolerances.angular_orthogonality_abs,
            "angular orthogonality failed for ({}, {}) vs ({}, {}): integral={} tol={}",
            l1,
            m1,
            l2,
            m2,
            magnitude,
            tolerances.angular_orthogonality_abs
        );
    }
}

#[test]
fn radial_probability_integrates_to_one() {
    let tolerances = current_tolerances();
    let states = [(1usize, 0usize, 1.0f64), (2, 1, 1.0), (4, 2, 2.0), (5, 4, 1.0)];

    for (n, l, z) in states {
        let r_max = (60.0 * (n * n) as f64 / z).max(40.0);
        let integral = reference::integrate_radial_probability(n, l, z, r_max, tolerances.radial_steps);
        let err = (integral - 1.0).abs();
        assert!(
            err <= tolerances.radial_normalization_abs,
            "radial normalization failed for (n={}, l={}, z={}): integral={} err={} tol={} (r_max={}, steps={})",
            n,
            l,
            z,
            integral,
            err,
            tolerances.radial_normalization_abs,
            r_max,
            tolerances.radial_steps
        );
    }
}

#[test]
fn production_density_is_finite_non_negative_and_reference_consistent() {
    let tolerances = current_tolerances();
    let level = sample_level();
    let params_set = fixtures::representative_param_sets(level);
    let times = fixtures::representative_times(level);
    let positions = fixtures::representative_positions(level);

    for params in params_set {
        for &time in &times {
            for &position in &positions {
                let density = evaluate_wavefunction(position, &params, time);
                assert!(
                    density.is_finite(),
                    "density is not finite for params={:?} time={} position={:?}",
                    params,
                    time,
                    position
                );
                assert!(
                    density >= -1e-12,
                    "density is negative for params={:?} time={} position={:?} density={}",
                    params,
                    time,
                    position,
                    density
                );

                let reference = reference::evaluate_density(position, &params, time);
                let abs_err = (density - reference).abs();
                let rel_err = abs_err / reference.abs().max(1e-8);
                assert!(
                    rel_err <= tolerances.density_reference_rel || abs_err <= tolerances.density_reference_abs,
                    "reference mismatch params={:?} time={} position={:?} expected={} actual={} rel_err={} abs_err={} rel_tol={} abs_tol={}",
                    params,
                    time,
                    position,
                    reference,
                    density,
                    rel_err,
                    abs_err,
                    tolerances.density_reference_rel,
                    tolerances.density_reference_abs
                );
            }
        }
    }
}

#[test]
fn pure_state_density_is_stationary_in_time() {
    let tolerances = current_tolerances();
    let level = sample_level();
    let positions = fixtures::representative_positions(level);
    let time_samples = fixtures::representative_times(level);
    let seeds = fixtures::representative_param_sets(level);

    for seed in seeds {
        for mix in [0.0f64, 1.0f64] {
            let params = WavefunctionParams { mix, ..seed };
            for &position in &positions {
                let baseline = evaluate_wavefunction(position, &params, time_samples[0]);
                for &time in &time_samples[1..] {
                    let value = evaluate_wavefunction(position, &params, time);
                    let abs_err = (value - baseline).abs();
                    let rel_err = abs_err / baseline.abs().max(1e-8);
                    assert!(
                        rel_err <= tolerances.stationary_rel || abs_err <= tolerances.density_reference_abs,
                        "stationarity failed params={:?} mix={} position={:?} baseline={} value={} time={} rel_err={} abs_err={} rel_tol={}",
                        params,
                        mix,
                        position,
                        baseline,
                        value,
                        time,
                        rel_err,
                        abs_err,
                        tolerances.stationary_rel
                    );
                }
            }
        }
    }
}
