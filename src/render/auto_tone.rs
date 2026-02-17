#[derive(Debug, Copy, Clone)]
pub struct AutoToneOutput {
    pub exposure_black: f32,
    pub exposure_white: f32,
    pub gamma: f32,
    pub contrast: f32,
    pub brightness: f32,
    pub luminance_threshold: f32,
}

impl Default for AutoToneOutput {
    fn default() -> Self {
        Self {
            exposure_black: 0.0,
            exposure_white: 1.0,
            gamma: 1.0,
            contrast: 1.2,
            brightness: 1.0,
            luminance_threshold: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AutoToneController {
    current: AutoToneOutput,
    has_valid: bool,
}

impl Default for AutoToneController {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoToneController {
    const TAU_SECONDS: f32 = 0.9;
    const SHOCK_RATIO: f32 = 4.0;
    const EPSILON: f32 = 1e-6;

    pub fn new() -> Self {
        Self {
            current: AutoToneOutput::default(),
            has_valid: false,
        }
    }

    pub fn update(&mut self, sample: &[f32], dt_seconds: f32) -> AutoToneOutput {
        let mut filtered = sample
            .iter()
            .copied()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .collect::<Vec<_>>();

        if filtered.is_empty() {
            return self.current;
        }

        filtered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q02 = quantile_sorted(&filtered, 0.02);
        let _q50 = quantile_sorted(&filtered, 0.50);
        let q995 = quantile_sorted(&filtered, 0.995);

        let black_target = q02;
        let mut white_target = q995.max(black_target + Self::EPSILON);
        if !black_target.is_finite() || !white_target.is_finite() {
            return self.current;
        }

        if white_target < black_target + Self::EPSILON {
            white_target = black_target + Self::EPSILON;
        }

        let mut threshold_target = black_target + 0.04 * (white_target - black_target);
        threshold_target = threshold_target.clamp(black_target, white_target);

        if !self.has_valid {
            self.current.exposure_black = black_target;
            self.current.exposure_white = white_target;
            self.current.luminance_threshold = threshold_target;
            self.has_valid = true;
            return self.current;
        }

        let dt = dt_seconds.max(1e-4);
        let base_alpha = 1.0 - (-dt / Self::TAU_SECONDS).exp();
        let shock = is_shock(self.current.exposure_white, white_target)
            || is_shock(self.current.exposure_black.max(Self::EPSILON), black_target.max(Self::EPSILON));
        let alpha = if shock {
            (base_alpha * 3.0).min(0.85)
        } else {
            base_alpha
        };

        self.current.exposure_black =
            smooth(self.current.exposure_black, black_target, alpha).max(0.0);
        self.current.exposure_white = smooth(self.current.exposure_white, white_target, alpha)
            .max(self.current.exposure_black + Self::EPSILON);
        self.current.luminance_threshold = smooth(
            self.current.luminance_threshold,
            threshold_target,
            alpha,
        )
        .clamp(self.current.exposure_black, self.current.exposure_white);

        self.current
    }

}

fn quantile_sorted(sorted: &[f32], quantile: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let q = quantile.clamp(0.0, 1.0);
    let last = (sorted.len() - 1) as f32;
    let position = q * last;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let t = position - lo as f32;
    sorted[lo] * (1.0 - t) + sorted[hi] * t
}

fn smooth(current: f32, target: f32, alpha: f32) -> f32 {
    current + alpha * (target - current)
}

fn is_shock(current: f32, target: f32) -> bool {
    let current = current.max(1e-9);
    let target = target.max(1e-9);
    let ratio = (target / current).max(current / target);
    ratio > AutoToneController::SHOCK_RATIO
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_track_distribution() {
        let mut controller = AutoToneController::new();
        let sample: Vec<f32> = (0..10_000).map(|i| i as f32 / 10_000.0).collect();
        let out = controller.update(&sample, 0.016);
        assert!(out.exposure_black >= 0.015 && out.exposure_black <= 0.03);
        assert!(out.exposure_white >= 0.98);
    }

    #[test]
    fn ignores_invalid_data() {
        let mut controller = AutoToneController::new();
        let initial = controller.current;
        let out = controller.update(&[f32::NAN, f32::INFINITY, -1.0], 0.016);
        assert_eq!(out.exposure_black, initial.exposure_black);
        assert_eq!(out.exposure_white, initial.exposure_white);
    }

    #[test]
    fn smoothing_converges_without_oscillation() {
        let mut controller = AutoToneController::new();
        let sample: Vec<f32> = (0..5000).map(|i| i as f32 / 5000.0).collect();
        let mut prev = controller.update(&sample, 0.016).exposure_white;
        for _ in 0..20 {
            let curr = controller.update(&sample, 0.016).exposure_white;
            assert!(curr <= 1.001);
            assert!((curr - prev).abs() < 0.2);
            prev = curr;
        }
    }

    #[test]
    fn shock_response_accelerates_large_jumps() {
        let mut controller = AutoToneController::new();
        let low: Vec<f32> = (0..5000).map(|i| i as f32 / 50000.0).collect();
        let high: Vec<f32> = (0..5000).map(|i| i as f32 / 10.0).collect();

        let before = controller.update(&low, 0.016).exposure_white;
        let after = controller.update(&high, 0.016).exposure_white;
        assert!(after > before * 1.5);
    }
}
