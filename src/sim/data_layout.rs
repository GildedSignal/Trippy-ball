use glam::Vec3;
use std::f32::consts::PI;
use std::sync::LazyLock;

const MIN_RENDER_POINTS: usize = 1_000;
pub const DEFAULT_MAX_RENDER_POINTS: usize = 5_000_000;
pub const HARD_MAX_RENDER_POINTS: usize = 20_000_000;
const MAX_RENDER_POINTS_ENV: &str = "TRIPPY_BALL_MAX_RENDER_POINTS";

static MAX_RENDER_POINTS: LazyLock<usize> = LazyLock::new(|| {
    resolve_max_render_points(std::env::var(MAX_RENDER_POINTS_ENV).ok().as_deref())
});

pub fn effective_point_count(requested: usize) -> usize {
    requested.clamp(MIN_RENDER_POINTS, max_render_points())
}

pub fn max_render_points() -> usize {
    *MAX_RENDER_POINTS
}

fn resolve_max_render_points(env_value: Option<&str>) -> usize {
    let configured = env_value.and_then(|raw| raw.trim().parse::<usize>().ok());
    configured
        .unwrap_or(DEFAULT_MAX_RENDER_POINTS)
        .clamp(MIN_RENDER_POINTS, HARD_MAX_RENDER_POINTS)
}

pub fn generate_positions(count: usize, max_radius: f32) -> Vec<Vec3> {
    let mut positions = Vec::with_capacity(count);
    for index in 0..count {
        positions.push(procedural_position(index, count, max_radius));
    }
    positions
}

fn procedural_position(index: usize, total: usize, max_radius: f32) -> Vec3 {
    let idx = index as u64 + 1;
    let u = radical_inverse(2, idx);
    let v = radical_inverse(3, idx);
    let w = radical_inverse(5, idx);

    let bias = ((index as f32 + 0.5) / total.max(1) as f32).clamp(0.0, 1.0);

    let r = (0.65 * w + 0.35 * bias).cbrt() * max_radius;
    let theta = (1.0 - 2.0 * u).acos();
    let phi = 2.0 * PI * v;

    let sin_theta = theta.sin();
    let x = r * sin_theta * phi.cos();
    let y = r * sin_theta * phi.sin();
    let z = r * theta.cos();

    Vec3::new(x, y, z)
}

fn radical_inverse(base: u64, mut index: u64) -> f32 {
    let inv_base = 1.0 / base as f32;
    let mut reversed_digits = 0.0f32;
    let mut inv = inv_base;

    while index > 0 {
        let digit = index % base;
        reversed_digits += digit as f32 * inv;
        index /= base;
        inv *= inv_base;
    }

    reversed_digits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_max_render_points_defaults_when_missing() {
        assert_eq!(
            resolve_max_render_points(None),
            DEFAULT_MAX_RENDER_POINTS.clamp(MIN_RENDER_POINTS, HARD_MAX_RENDER_POINTS)
        );
    }

    #[test]
    fn resolve_max_render_points_clamps_low_and_high_values() {
        assert_eq!(resolve_max_render_points(Some("10")), MIN_RENDER_POINTS);
        assert_eq!(
            resolve_max_render_points(Some("999999999")),
            HARD_MAX_RENDER_POINTS
        );
    }

    #[test]
    fn resolve_max_render_points_ignores_invalid_values() {
        assert_eq!(
            resolve_max_render_points(Some("not-a-number")),
            DEFAULT_MAX_RENDER_POINTS
        );
    }
}
