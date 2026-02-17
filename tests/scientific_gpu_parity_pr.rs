#![cfg(target_os = "macos")]

#[path = "scientific/mod.rs"]
mod scientific;

use scientific::fixtures;
use scientific::tolerances::{current_tolerances, sample_level};
use wavefunction_visualization::math::evaluate_wavefunction_batch;
use wavefunction_visualization::render::GpuWavefunctionEvaluator;

#[test]
fn gpu_matches_cpu_reference_on_pr_grid() {
    let tolerances = current_tolerances();
    let level = sample_level();
    let positions = fixtures::representative_positions(level);
    let params_set = fixtures::representative_param_sets(level);
    let times = fixtures::representative_times(level);

    let point_count = positions.len();
    let mut evaluator = GpuWavefunctionEvaluator::new_with_buffer_pool(
        &wavefunction_visualization::math::WavefunctionParams::default(),
        point_count,
        None,
    );
    evaluator.update_positions(&positions);

    for params in params_set {
        for &time in &times {
            evaluator.compute(&params, time);
            let gpu = evaluator.read_intensities();

            let mut cpu = vec![0.0f32; point_count];
            evaluate_wavefunction_batch(&positions, &params, time, &mut cpu);

            for i in 0..point_count {
                let expected = cpu[i];
                let actual = gpu[i];
                assert!(
                    actual.is_finite(),
                    "non-finite gpu intensity for params={:?} time={} idx={} value={}",
                    params,
                    time,
                    i,
                    actual
                );
                assert!(
                    actual >= -1e-6,
                    "negative gpu intensity for params={:?} time={} idx={} value={}",
                    params,
                    time,
                    i,
                    actual
                );

                let abs_err = (expected - actual).abs();
                let rel_err = abs_err / expected.abs().max(1e-6);
                assert!(
                    rel_err <= tolerances.cpu_gpu_rel || abs_err <= tolerances.cpu_gpu_abs,
                    "gpu/cpu mismatch params={:?} time={} idx={} expected={} actual={} rel_err={} abs_err={} rel_tol={} abs_tol={}",
                    params,
                    time,
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
