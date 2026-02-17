#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleLevel {
    Pr,
    Nightly,
}

impl SampleLevel {
    pub fn from_env() -> Self {
        match std::env::var("TRIPPY_BALL_SCI_SAMPLE_LEVEL")
            .unwrap_or_else(|_| "pr".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "nightly" => Self::Nightly,
            _ => Self::Pr,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScientificTolerances {
    pub angular_normalization_abs: f64,
    pub angular_orthogonality_abs: f64,
    pub radial_normalization_abs: f64,
    pub density_reference_rel: f64,
    pub density_reference_abs: f64,
    pub stationary_rel: f64,
    pub cpu_gpu_rel: f32,
    pub cpu_gpu_abs: f32,
    pub theta_samples: usize,
    pub phi_samples: usize,
    pub radial_steps: usize,
}

pub fn sample_level() -> SampleLevel {
    SampleLevel::from_env()
}

pub fn current_tolerances() -> ScientificTolerances {
    match sample_level() {
        SampleLevel::Pr => ScientificTolerances {
            angular_normalization_abs: 2.5e-2,
            angular_orthogonality_abs: 3.0e-2,
            radial_normalization_abs: 2.5e-2,
            density_reference_rel: 8.0e-2,
            density_reference_abs: 1.0e-4,
            stationary_rel: 8.0e-5,
            cpu_gpu_rel: 0.12,
            cpu_gpu_abs: 2.0e-4,
            theta_samples: 64,
            phi_samples: 96,
            radial_steps: 20_000,
        },
        SampleLevel::Nightly => ScientificTolerances {
            angular_normalization_abs: 1.2e-2,
            angular_orthogonality_abs: 1.2e-2,
            radial_normalization_abs: 1.2e-2,
            density_reference_rel: 5.0e-2,
            density_reference_abs: 7.5e-5,
            stationary_rel: 3.0e-5,
            cpu_gpu_rel: 0.08,
            cpu_gpu_abs: 1.0e-4,
            theta_samples: 128,
            phi_samples: 192,
            radial_steps: 50_000,
        },
    }
}
