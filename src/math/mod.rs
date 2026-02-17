// Math Module
//
// This module contains the mathematical functions needed for the wavefunction simulation.
// It includes implementations of spherical harmonics, radial functions, and coordinate transformations.
// The module is optimized for performance using SIMD instructions where possible.

mod kernel_tables;
pub mod presets;
mod radial;
mod spherical;

use glam::Vec3;
use rayon::prelude::*;

pub const GPU_NORMALIZATION_MAX_L: usize = kernel_tables::MAX_L_PRECOMPUTE;
pub const GPU_NORMALIZATION_TABLE_LEN: usize = kernel_tables::NORMALIZATION_TABLE_FLAT_LEN;

pub fn gpu_normalization_table_f32_flat() -> &'static [f32; GPU_NORMALIZATION_TABLE_LEN] {
    kernel_tables::normalization_table_f32_flat()
}

// Parameters for the wavefunction
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavefunctionParams {
    pub n: usize,            // Principal quantum number for state 1
    pub l: usize,            // Orbital quantum number for state 1
    pub m: isize,            // Magnetic quantum number for state 1
    pub n2: usize,           // Principal quantum number for state 2
    pub l2: usize,           // Orbital quantum number for state 2
    pub m2: isize,           // Magnetic quantum number for state 2
    pub mix: f64,            // Population weight of state 2 in [0,1]
    pub relative_phase: f64, // Relative phase offset between states (rad)
    pub z: f64,              // Nuclear charge (hydrogen-like ion)
    pub time_factor: f64,    // Physical time scale in femtoseconds per real second
}

impl Default for WavefunctionParams {
    fn default() -> Self {
        Self {
            n: 2,
            l: 1,
            m: 0,
            n2: 3,
            l2: 2,
            m2: 0,
            mix: 0.35,
            relative_phase: 0.0,
            z: 1.0,
            time_factor: 2.0,
        }
    }
}

// Convert spherical coordinates to Cartesian
#[allow(dead_code)]
pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> Vec3 {
    let x = r * theta.sin() * phi.cos();
    let y = r * theta.sin() * phi.sin();
    let z = r * theta.cos();
    Vec3::new(x as f32, y as f32, z as f32)
}

// Convert Cartesian coordinates to spherical
pub fn cartesian_to_spherical(pos: Vec3) -> (f64, f64, f64) {
    let x = pos.x as f64;
    let y = pos.y as f64;
    let z = pos.z as f64;

    let r = (x * x + y * y + z * z).sqrt();
    let theta = if r < 1e-10 { 0.0 } else { (z / r).acos() };
    let phi = y.atan2(x);

    (r, theta, phi)
}

// Evaluate the wavefunction at a point
pub fn evaluate_wavefunction(pos: Vec3, params: &WavefunctionParams, time: f64) -> f64 {
    const RYDBERG_EV: f64 = 13.605_693_122_994;
    const EV_TO_J: f64 = 1.602_176_634e-19;
    const HBAR_SI: f64 = 1.054_571_817e-34;

    let valid_state = |n: usize, l: usize, m: isize| n >= 1 && l < n && m.unsigned_abs() <= l;
    if !valid_state(params.n, params.l, params.m) || !valid_state(params.n2, params.l2, params.m2)
    {
        return 0.0;
    }

    let (r, theta, phi) = cartesian_to_spherical(pos);
    let z = params.z.max(1e-6);
    let y_lm_1 = spherical::spherical_harmonic(params.l, params.m, theta, phi);
    let y_lm_2 = spherical::spherical_harmonic(params.l2, params.m2, theta, phi);
    let radial_1 = radial::hydrogenic_radial(params.n, params.l, r, z);
    let radial_2 = radial::hydrogenic_radial(params.n2, params.l2, r, z);

    let t_phys_s = time * params.time_factor.max(0.0) * 1e-15;
    let e1_j = -RYDBERG_EV * z * z / (params.n * params.n) as f64 * EV_TO_J;
    let e2_j = -RYDBERG_EV * z * z / (params.n2 * params.n2) as f64 * EV_TO_J;
    let phase1_arg = -e1_j * t_phys_s / HBAR_SI;
    let phase2_arg = -e2_j * t_phys_s / HBAR_SI + params.relative_phase;
    let phase1_re = phase1_arg.cos();
    let phase1_im = phase1_arg.sin();
    let phase2_re = phase2_arg.cos();
    let phase2_im = phase2_arg.sin();

    let mix = params.mix.clamp(0.0, 1.0);
    let amp1 = (1.0 - mix).sqrt();
    let amp2 = mix.sqrt();
    let scale1 = radial_1 * amp1;
    let scale2 = radial_2 * amp2;

    let y1_re = y_lm_1.re * scale1;
    let y1_im = y_lm_1.im * scale1;
    let y2_re = y_lm_2.re * scale2;
    let y2_im = y_lm_2.im * scale2;

    let psi1_re = y1_re * phase1_re - y1_im * phase1_im;
    let psi1_im = y1_re * phase1_im + y1_im * phase1_re;
    let psi2_re = y2_re * phase2_re - y2_im * phase2_im;
    let psi2_im = y2_re * phase2_im + y2_im * phase2_re;

    let psi_re = psi1_re + psi2_re;
    let psi_im = psi1_im + psi2_im;
    psi_re * psi_re + psi_im * psi_im
}

// Batch evaluation of wavefunction for multiple points
// This function processes multiple points at once, with SIMD optimization on supported platforms
pub fn evaluate_wavefunction_batch(
    positions: &[Vec3],
    params: &WavefunctionParams,
    time: f64,
    intensities: &mut [f32],
) {
    // Ensure the output array is the right size
    assert_eq!(
        positions.len(),
        intensities.len(),
        "Input and output arrays must be the same length"
    );

    positions
        .par_iter()
        .zip(intensities.par_iter_mut())
        .for_each(|(&pos, intensity)| {
            *intensity = evaluate_wavefunction(pos, params, time) as f32;
        });
}

// Unit tests for the math module
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_spherical_to_cartesian() {
        // Test conversion from spherical to Cartesian coordinates
        let r = 2.0;
        let theta = PI / 2.0; // 90 degrees
        let phi = 0.0; // 0 degrees

        let result = spherical_to_cartesian(r, theta, phi);

        // Should be (2, 0, 0) for r=2, theta=90°, phi=0°
        assert!((result.x - 2.0).abs() < 1e-5);
        assert!(result.y.abs() < 1e-5);
        assert!(result.z.abs() < 1e-5);
    }

    #[test]
    fn test_cartesian_to_spherical() {
        // Test conversion from Cartesian to spherical coordinates
        let pos = Vec3::new(0.0, 0.0, 3.0);

        let (r, theta, phi) = cartesian_to_spherical(pos);

        // Should be (3, 0, 0) for (0, 0, 3)
        assert!((r - 3.0).abs() < 1e-5);
        assert!(theta.abs() < 1e-5);
        // Phi is undefined when x and y are both 0, but we return 0.0
        assert!(phi.abs() < 1e-5);
    }

    #[test]
    fn test_coordinate_conversion_roundtrip() {
        // Test roundtrip conversion
        let original_r = 2.5;
        let original_theta = PI / 4.0; // 45 degrees
        let original_phi = PI / 3.0; // 60 degrees

        let cartesian = spherical_to_cartesian(original_r, original_theta, original_phi);
        let (r, theta, phi) = cartesian_to_spherical(cartesian);

        // Check if we get back the original values
        assert!((r - original_r).abs() < 1e-5);
        assert!((theta - original_theta).abs() < 1e-5);
        assert!((phi - original_phi).abs() < 1e-5);
    }

    #[test]
    fn test_wavefunction_evaluation() {
        // Test basic wavefunction evaluation
        let params = WavefunctionParams::default();
        let pos = Vec3::new(1.0, 0.0, 0.0);
        let time = 0.0;

        let result = evaluate_wavefunction(pos, &params, time);

        // Result should be positive for a valid wavefunction
        assert!(result > 0.0);
    }

    #[test]
    fn test_batch_evaluation() {
        // Test batch evaluation against individual evaluation
        let params = WavefunctionParams::default();
        let positions = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let time = 0.0;

        let mut batch_results = vec![0.0f32; positions.len()];
        evaluate_wavefunction_batch(&positions, &params, time, &mut batch_results);

        // Compare with individual evaluation
        for (i, &pos) in positions.iter().enumerate() {
            let individual = evaluate_wavefunction(pos, &params, time) as f32;
            assert!((batch_results[i] - individual).abs() < 1e-5);
        }
    }

    #[test]
    fn test_spherical_coordinates_batch() {
        // Test batched cartesian->spherical conversion behavior
        let positions = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];

        let batch: Vec<(f64, f64, f64)> = positions
            .iter()
            .copied()
            .map(cartesian_to_spherical)
            .collect();

        assert_eq!(batch.len(), positions.len());

        for (i, &pos) in positions.iter().enumerate() {
            let (r, theta, phi) = cartesian_to_spherical(pos);
            assert!((batch[i].0 - r).abs() < 1e-5);
            assert!((batch[i].1 - theta).abs() < 1e-5);
            assert!((batch[i].2 - phi).abs() < 1e-5);
        }
    }

    #[test]
    fn test_simd_vs_non_simd() {
        // Test that SIMD and non-SIMD implementations produce the same results
        let params = WavefunctionParams {
            n: 4,
            l: 3,
            m: 2,
            n2: 5,
            l2: 4,
            m2: 2,
            mix: 0.35,
            relative_phase: 0.0,
            z: 1.0,
            time_factor: 2.5,
        };
        let time = 0.0;

        // Generate test points
        let num_points = 1000;
        let mut positions = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let x = (i % 10) as f32 / 5.0 - 1.0;
            let y = ((i / 10) % 10) as f32 / 5.0 - 1.0;
            let z = (i / 100) as f32 / 5.0 - 1.0;
            positions.push(Vec3::new(x, y, z));
        }

        // Calculate using non-SIMD implementation
        let mut non_simd_results = vec![0.0f32; num_points];
        for (i, &pos) in positions.iter().enumerate() {
            non_simd_results[i] = evaluate_wavefunction(pos, &params, time) as f32;
        }

        // Calculate using SIMD implementation
        let mut simd_results = vec![0.0f32; num_points];
        evaluate_wavefunction_batch(&positions, &params, time, &mut simd_results);

        // Compare results
        let mut max_diff = 0.0f32;
        for i in 0..num_points {
            let diff = (non_simd_results[i] - simd_results[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        // The difference should be very small (floating point precision)
        assert!(
            max_diff < 1e-5,
            "SIMD and non-SIMD results differ too much: {}",
            max_diff
        );
    }

    #[test]
    fn heavy_parallel_workload_engages_cpu_workers() {
        let workers = 8usize;
        let params = WavefunctionParams::default();
        let time = 0.125f64;
        let point_count = workers * 65_536;

        let positions: Vec<Vec3> = (0..point_count)
            .map(|i| {
                let t = i as f32 / point_count as f32;
                Vec3::new(
                    (t * 17.0).sin() * 4.0,
                    (t * 23.0).cos() * 3.0,
                    (t * 29.0).sin() * 5.0,
                )
            })
            .collect();
        let mut intensities = vec![0.0f32; point_count];
        let usage: Arc<Vec<AtomicUsize>> =
            Arc::new((0..workers).map(|_| AtomicUsize::new(0)).collect());

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("failed to build test rayon pool");

        pool.install(|| {
            const CHUNK_SIZE: usize = 1_024;
            positions
                .par_chunks(CHUNK_SIZE)
                .zip(intensities.par_chunks_mut(CHUNK_SIZE))
                .for_each(|(pos_chunk, out_chunk)| {
                    if let Some(index) = rayon::current_thread_index() {
                        if let Some(counter) = usage.get(index) {
                            counter.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    for (pos, out) in pos_chunk.iter().zip(out_chunk.iter_mut()) {
                        *out = evaluate_wavefunction(*pos, &params, time) as f32;
                    }
                });
        });

        let active_workers = usage
            .iter()
            .filter(|counter| counter.load(Ordering::Relaxed) > 0)
            .count();
        assert!(
            active_workers == workers,
            "expected {} active workers, observed {}",
            workers,
            active_workers
        );
    }
}
