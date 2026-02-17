use crate::scientific::tolerances::SampleLevel;
use std::f64::consts::PI;
use wavefunction_visualization::math::{spherical_to_cartesian, WavefunctionParams};

pub fn representative_lm_pairs(level: SampleLevel) -> Vec<(usize, isize)> {
    match level {
        SampleLevel::Pr => vec![(0, 0), (1, -1), (1, 0), (2, 1), (3, 2), (5, -3)],
        SampleLevel::Nightly => {
            let mut out = Vec::new();
            for l in 0..=5usize {
                for m in -(l as isize)..=(l as isize) {
                    out.push((l, m));
                }
            }
            out
        }
    }
}

pub fn representative_param_sets(level: SampleLevel) -> Vec<WavefunctionParams> {
    let mut params = vec![
        WavefunctionParams {
            n: 1,
            l: 0,
            m: 0,
            n2: 2,
            l2: 1,
            m2: 0,
            mix: 0.25,
            relative_phase: 0.0,
            z: 1.0,
            time_factor: 1.0,
        },
        WavefunctionParams {
            n: 2,
            l: 1,
            m: 1,
            n2: 3,
            l2: 2,
            m2: 1,
            mix: 0.4,
            relative_phase: 0.7,
            z: 1.0,
            time_factor: 5.0,
        },
        WavefunctionParams {
            n: 3,
            l: 2,
            m: -1,
            n2: 4,
            l2: 3,
            m2: 2,
            mix: 0.35,
            relative_phase: -0.8,
            z: 2.0,
            time_factor: 20.0,
        },
        WavefunctionParams {
            n: 5,
            l: 4,
            m: 1,
            n2: 6,
            l2: 5,
            m2: -2,
            mix: 0.6,
            relative_phase: 1.4,
            z: 3.0,
            time_factor: 120.0,
        },
    ];

    if level == SampleLevel::Nightly {
        params.push(WavefunctionParams {
            n: 4,
            l: 1,
            m: 0,
            n2: 6,
            l2: 4,
            m2: 2,
            mix: 0.5,
            relative_phase: PI / 3.0,
            z: 4.0,
            time_factor: 200.0,
        });
        params.push(WavefunctionParams {
            n: 2,
            l: 0,
            m: 0,
            n2: 5,
            l2: 2,
            m2: 0,
            mix: 0.8,
            relative_phase: -PI / 4.0,
            z: 6.0,
            time_factor: 80.0,
        });
    }

    params
}

pub fn representative_times(level: SampleLevel) -> Vec<f64> {
    match level {
        SampleLevel::Pr => vec![0.0, 0.17, 0.43],
        SampleLevel::Nightly => vec![0.0, 0.09, 0.21, 0.43, 0.89],
    }
}

pub fn representative_positions(level: SampleLevel) -> Vec<glam::Vec3> {
    let radial = match level {
        SampleLevel::Pr => vec![0.0, 1e-6, 0.15, 0.8, 2.0, 4.5],
        SampleLevel::Nightly => vec![0.0, 1e-6, 0.05, 0.25, 0.8, 1.5, 3.0, 5.0],
    };
    let theta = match level {
        SampleLevel::Pr => vec![0.0, PI / 6.0, PI / 2.0, 5.0 * PI / 6.0, PI - 1e-4],
        SampleLevel::Nightly => vec![
            0.0,
            PI / 10.0,
            PI / 6.0,
            PI / 3.0,
            PI / 2.0,
            2.0 * PI / 3.0,
            9.0 * PI / 10.0,
            PI - 1e-4,
        ],
    };
    let phi = match level {
        SampleLevel::Pr => vec![0.0, PI / 4.0, PI / 2.0, PI, 1.75 * PI],
        SampleLevel::Nightly => vec![
            0.0,
            PI / 8.0,
            PI / 4.0,
            PI / 2.0,
            3.0 * PI / 4.0,
            PI,
            1.5 * PI,
            1.875 * PI,
        ],
    };

    let mut out = Vec::with_capacity(radial.len() * theta.len() * phi.len());
    for &r in &radial {
        for &t in &theta {
            for &p in &phi {
                out.push(spherical_to_cartesian(r, t, p));
            }
        }
    }
    out
}

pub fn integration_theta_phi_grid(
    theta_samples: usize,
    phi_samples: usize,
) -> Vec<(f64, f64, f64)> {
    let theta_samples = theta_samples.max(1);
    let phi_samples = phi_samples.max(1);
    let dtheta = PI / theta_samples as f64;
    let dphi = 2.0 * PI / phi_samples as f64;

    let mut out = Vec::with_capacity(theta_samples * phi_samples);
    for i in 0..theta_samples {
        let theta = (i as f64 + 0.5) * dtheta;
        let sin_theta = theta.sin();
        for j in 0..phi_samples {
            let phi = (j as f64 + 0.5) * dphi;
            let weight = sin_theta * dtheta * dphi;
            out.push((theta, phi, weight));
        }
    }

    out
}
