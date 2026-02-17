#![cfg(target_os = "macos")]

#[path = "scientific/mod.rs"]
mod scientific;

use scientific::fixtures;
use scientific::tolerances::{current_tolerances, SampleLevel};
use wavefunction_visualization::math::{evaluate_wavefunction_batch, WavefunctionParams};
use wavefunction_visualization::render::GpuWavefunctionEvaluator;

#[test]
fn gpu_matches_cpu_reference_on_nightly_sweep() {
    let tolerances = current_tolerances();
    let positions = fixtures::representative_positions(SampleLevel::Nightly);
    let point_count = positions.len();

    let mut evaluator =
        GpuWavefunctionEvaluator::new_with_buffer_pool(&WavefunctionParams::default(), point_count, None);
    evaluator.update_positions(&positions);

    let regimes = [(1.0f64, 0.7f64), (2.0f64, 1.25f64), (4.0f64, 60.0f64)];
    let times = [0.0f64, 0.17, 0.43, 0.89];

    for &(z, time_factor) in &regimes {
        for &time in &times {
            for l in 0..=5usize {
                for m in -(l as isize)..=(l as isize) {
                    let params = WavefunctionParams {
                        n: (l + 1).max(1),
                        l,
                        m,
                        n2: (l + 2).max(2),
                        l2: (l + 1).min(5),
                        m2: m.clamp(-((l + 1).min(5) as isize), (l + 1).min(5) as isize),
                        mix: 0.35,
                        relative_phase: 0.3,
                        z,
                        time_factor,
                    };

                    evaluator.compute(&params, time);
                    let gpu = evaluator.read_intensities();

                    let mut cpu = vec![0.0f32; point_count];
                    evaluate_wavefunction_batch(&positions, &params, time, &mut cpu);

                    for i in 0..point_count {
                        let expected = cpu[i];
                        let actual = gpu[i];
                        assert!(
                            actual.is_finite(),
                            "nightly non-finite gpu value z={} time_factor={} time={} l={} m={} idx={} value={}",
                            z,
                            time_factor,
                            time,
                            l,
                            m,
                            i,
                            actual
                        );
                        assert!(
                            actual >= -1e-6,
                            "nightly negative gpu value z={} time_factor={} time={} l={} m={} idx={} value={}",
                            z,
                            time_factor,
                            time,
                            l,
                            m,
                            i,
                            actual
                        );

                        let abs_err = (expected - actual).abs();
                        let rel_err = abs_err / expected.abs().max(1e-6);
                        assert!(
                            rel_err <= tolerances.cpu_gpu_rel || abs_err <= tolerances.cpu_gpu_abs,
                            "nightly mismatch z={} time_factor={} time={} l={} m={} idx={} expected={} actual={} rel_err={} abs_err={} rel_tol={} abs_tol={}",
                            z,
                            time_factor,
                            time,
                            l,
                            m,
                            i,
                            expected,
                            actual,
                            rel_err,
                            abs_err,
                            tolerances.cpu_gpu_rel,
                            tolerances.cpu_gpu_abs
                        );
                    }
                }
            }
        }
    }
}
