// Radial Functions Implementation
//
// This file contains the implementation of radial functions for the wavefunction.
// For the MVP, we focus on the Gaussian radial function, which is simpler and
// more visually appealing than Bessel functions.
// The implementation is optimized for performance using SIMD instructions where possible.

#[cfg(all(target_arch = "x86_64", test))]
use std::arch::x86_64::*;

// Gaussian radial function: R(r) = e^(-alpha * r^2)
// This is not a true hydrogen radial function but works well for visualization
#[cfg(test)]
pub fn gaussian_radial(r: f64, alpha: f64) -> f64 {
    (-alpha * r * r).exp()
}

fn factorial_usize(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, value| acc * value as f64)
}

fn associated_laguerre(n: usize, k: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 1.0 + k as f64 - x;
    }

    let mut l_nm2 = 1.0;
    let mut l_nm1 = 1.0 + k as f64 - x;
    for i in 2..=n {
        let i_f = i as f64;
        let k_f = k as f64;
        let term1 = (2.0 * i_f - 1.0 + k_f - x) * l_nm1;
        let term2 = (i_f - 1.0 + k_f) * l_nm2;
        let l_n = (term1 - term2) / i_f;
        l_nm2 = l_nm1;
        l_nm1 = l_n;
    }
    l_nm1
}

// Hydrogenic radial term in atomic-length units (r_a0 = r / a0).
// Supports hydrogen-like ions via nuclear charge Z.
pub fn hydrogenic_radial(n: usize, l: usize, r_a0: f64, z: f64) -> f64 {
    if n == 0 || l >= n || z <= 0.0 {
        return 0.0;
    }

    let rho = 2.0 * z * r_a0 / n as f64;
    let prefactor = (2.0 * z / n as f64).powf(1.5);
    let num = factorial_usize(n - l - 1);
    let den = 2.0 * n as f64 * factorial_usize(n + l);
    let norm = prefactor * (num / den).sqrt();

    let laguerre = associated_laguerre(n - l - 1, 2 * l + 1, rho);
    norm * (-rho * 0.5).exp() * rho.powi(l as i32) * laguerre
}

// Batch calculation of Gaussian radial function for multiple points
// This is more efficient than calculating each point individually
#[cfg(test)]
pub fn gaussian_radial_batch(r_values: &[f64], alpha: f64) -> Vec<f64> {
    let mut results = Vec::with_capacity(r_values.len());

    for &r in r_values {
        results.push(gaussian_radial(r, alpha));
    }

    results
}

// SIMD-optimized batch calculation of Gaussian radial function
// This is only available on x86_64 platforms with AVX2 support
#[cfg(target_arch = "x86_64")]
#[cfg(test)]
pub fn gaussian_radial_batch_simd(r_values: &[f64], alpha: f64) -> Vec<f64> {
    if is_x86_feature_detected!("avx2") {
        unsafe {
            return gaussian_radial_batch_avx2(r_values, alpha);
        }
    }

    // Fallback to non-SIMD version
    gaussian_radial_batch(r_values, alpha)
}

// AVX2-optimized batch calculation of Gaussian radial function
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(test)]
unsafe fn gaussian_radial_batch_avx2(r_values: &[f64], alpha: f64) -> Vec<f64> {
    let n = r_values.len();
    let mut results = vec![0.0; n];

    // Process in chunks of 4 (AVX2 can process 4 doubles at once)
    let chunk_size = 4;
    let num_chunks = n / chunk_size;

    // Prepare constants
    let alpha_vec = _mm256_set1_pd(alpha);
    let neg_one = _mm256_set1_pd(-1.0);

    for chunk_idx in 0..num_chunks {
        let base_idx = chunk_idx * chunk_size;

        // Load 4 r values
        let r_vec = _mm256_loadu_pd(&r_values[base_idx]);

        // Calculate r^2
        let r_squared = _mm256_mul_pd(r_vec, r_vec);

        // Calculate -alpha * r^2
        let alpha_r_squared = _mm256_mul_pd(alpha_vec, r_squared);
        let neg_alpha_r_squared = _mm256_mul_pd(neg_one, alpha_r_squared);

        // Calculate exp(-alpha * r^2)
        // AVX2 doesn't have a direct exp instruction, so we'll extract and process individually
        for i in 0..4 {
            let idx = base_idx + i;
            let neg_alpha_r_squared_i = *neg_alpha_r_squared.as_ptr().add(i);
            results[idx] = neg_alpha_r_squared_i.exp();
        }
    }

    // Handle remaining points
    let remaining_start = num_chunks * chunk_size;
    for i in remaining_start..n {
        results[i] = gaussian_radial(r_values[i], alpha);
    }

    results
}

// AVX-512 optimized batch calculation of Gaussian radial function
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[cfg(test)]
pub unsafe fn gaussian_radial_batch_avx512(r_values: &[f64], alpha: f64) -> Vec<f64> {
    let n = r_values.len();
    let mut results = vec![0.0; n];

    // Process in chunks of 8 (AVX-512 can process 8 doubles at once)
    let chunk_size = 8;
    let num_chunks = n / chunk_size;

    // Prepare constants
    let alpha_vec = _mm512_set1_pd(alpha);
    let neg_one = _mm512_set1_pd(-1.0);

    for chunk_idx in 0..num_chunks {
        let base_idx = chunk_idx * chunk_size;

        // Load 8 r values
        let r_vec = _mm512_loadu_pd(&r_values[base_idx]);

        // Calculate r^2
        let r_squared = _mm512_mul_pd(r_vec, r_vec);

        // Calculate -alpha * r^2
        let alpha_r_squared = _mm512_mul_pd(alpha_vec, r_squared);
        let neg_alpha_r_squared = _mm512_mul_pd(neg_one, alpha_r_squared);

        // Calculate exp(-alpha * r^2)
        // AVX-512 doesn't have a direct exp instruction, so we'll extract and process individually
        for i in 0..8 {
            let idx = base_idx + i;
            let neg_alpha_r_squared_i = _mm512_extractf64_pd(neg_alpha_r_squared, i as i32);
            results[idx] = neg_alpha_r_squared_i.exp();
        }
    }

    // Handle remaining points
    let remaining_start = num_chunks * chunk_size;
    for i in remaining_start..n {
        results[i] = gaussian_radial(r_values[i], alpha);
    }

    results
}

// Extension trait for factorial calculation on f64
#[cfg(test)]
trait Factorial {
    fn factorial(&self) -> f64;
}

#[cfg(test)]
impl Factorial for f64 {
    fn factorial(&self) -> f64 {
        let n = *self as usize;
        (1..=n).map(|i| i as f64).product()
    }
}

// Normalization factor for the Gaussian radial function used in unit tests.
#[cfg(test)]
fn gaussian_normalization(alpha: f64, l: usize) -> f64 {
    use std::f64::consts::PI;
    let n = l + 1;
    let term1 = (2.0 * alpha).powf(n as f64 + 1.5);
    let term2 = (2.0 * n as f64).factorial();
    (term1 / term2).sqrt() * PI.powf(-0.25)
}

// For future implementation: Spherical Bessel function j_l(kr)
// This is more physically accurate but more complex to implement
#[allow(dead_code)]
pub fn spherical_bessel(l: usize, k: f64, r: f64) -> f64 {
    match l {
        0 => spherical_bessel_j0(k * r),
        1 => spherical_bessel_j1(k * r),
        _ => {
            // Recurrence relation for j_l
            let j_l_minus_1 = spherical_bessel(l - 1, k, r);
            let j_l_minus_2 = spherical_bessel(l - 2, k, r);
            ((2 * l - 1) as f64 / (k * r)) * j_l_minus_1 - j_l_minus_2
        }
    }
}

// Spherical Bessel function j_0(x) = sin(x)/x
#[allow(dead_code)]
fn spherical_bessel_j0(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        // Limit as x approaches 0
        1.0
    } else {
        x.sin() / x
    }
}

// Spherical Bessel function j_1(x) = sin(x)/(x^2) - cos(x)/x
#[allow(dead_code)]
fn spherical_bessel_j1(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        // Limit as x approaches 0
        0.0
    } else {
        x.sin() / (x * x) - x.cos() / x
    }
}

#[allow(dead_code)]
pub fn hydrogen_radial(n: usize, l: usize, r: f64, a0: f64) -> f64 {
    if a0 <= 0.0 {
        return 0.0;
    }
    hydrogenic_radial(n, l, r / a0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_gaussian_radial() {
        // Test Gaussian radial function

        // At r=0, the function should be 1.0 for any alpha
        assert!((gaussian_radial(0.0, 0.5) - 1.0).abs() < 1e-10);
        assert!((gaussian_radial(0.0, 1.0) - 1.0).abs() < 1e-10);

        // At r=1, the function should be e^(-alpha)
        assert!((gaussian_radial(1.0, 0.5) - (-0.5f64).exp()).abs() < 1e-10);
        assert!((gaussian_radial(1.0, 1.0) - (-1.0f64).exp()).abs() < 1e-10);

        // Test that the function decreases with increasing r
        let val1 = gaussian_radial(1.0, 0.5);
        let val2 = gaussian_radial(2.0, 0.5);
        assert!(val1 > val2);

        // Test that the function decreases faster with larger alpha
        let val_small_alpha = gaussian_radial(1.0, 0.1);
        let val_large_alpha = gaussian_radial(1.0, 1.0);
        assert!(val_small_alpha > val_large_alpha);
    }

    #[test]
    fn test_gaussian_radial_batch() {
        // Test batch calculation against individual calculation
        let r_values = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let alpha = 0.5;

        let batch = gaussian_radial_batch(&r_values, alpha);

        // Compare with individual calculation
        for i in 0..r_values.len() {
            let individual = gaussian_radial(r_values[i], alpha);
            assert!((batch[i] - individual).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gaussian_normalization() {
        // Test that normalization factor is positive
        assert!(gaussian_normalization(0.5, 0) > 0.0);
        assert!(gaussian_normalization(0.5, 1) > 0.0);

        // Test that normalization factor changes with l
        let norm0 = gaussian_normalization(0.5, 0);
        let norm1 = gaussian_normalization(0.5, 1);
        let norm2 = gaussian_normalization(0.5, 2);

        // Just check they're different, not necessarily increasing
        assert!(norm1 != norm0);
        assert!(norm2 != norm1);
    }

    #[test]
    fn test_factorial() {
        // Test factorial extension method
        assert!((0.0.factorial() - 1.0).abs() < 1e-10);
        assert!((1.0.factorial() - 1.0).abs() < 1e-10);
        assert!((2.0.factorial() - 2.0).abs() < 1e-10);
        assert!((3.0.factorial() - 6.0).abs() < 1e-10);
        assert!((4.0.factorial() - 24.0).abs() < 1e-10);
        assert!((5.0.factorial() - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_spherical_bessel() {
        // Test spherical Bessel functions

        // j_0(0) = 1
        assert!((spherical_bessel_j0(0.0) - 1.0).abs() < 1e-10);

        // j_1(0) = 0
        assert!((spherical_bessel_j1(0.0) - 0.0).abs() < 1e-10);

        // j_0(π) = 0
        assert!((spherical_bessel_j0(PI) - 0.0).abs() < 1e-10);

        // Test recurrence relation for j_2
        let x = 1.5;
        let j0 = spherical_bessel_j0(x);
        let j1 = spherical_bessel_j1(x);
        let j2 = spherical_bessel(2, 1.0, 1.5);

        // j_2(x) = (3/x)j_1(x) - j_0(x)
        let j2_expected = (3.0 / x) * j1 - j0;
        assert!((j2 - j2_expected).abs() < 1e-10);
    }
}
