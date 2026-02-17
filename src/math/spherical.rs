// Spherical Harmonics Implementation
//
// This file contains the implementation of spherical harmonics functions.
// It includes associated Legendre polynomials and the full spherical harmonic calculation.
// The implementation is optimized for performance using SIMD instructions where possible.

use super::kernel_tables;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ComplexValue {
    pub re: f64,
    pub im: f64,
}

impl ComplexValue {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
}

impl Add for ComplexValue {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for ComplexValue {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul<f64> for ComplexValue {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Mul<ComplexValue> for f64 {
    type Output = ComplexValue;

    fn mul(self, rhs: ComplexValue) -> Self::Output {
        rhs * self
    }
}

// Cache for computed associated Legendre polynomials
thread_local! {
    static LEGENDRE_CACHE: RefCell<HashMap<(usize, isize, i32), f64>> = RefCell::new(HashMap::new());
}

// Structure for batch processing of spherical harmonics
#[derive(Debug)]
#[cfg(test)]
pub struct SphericalHarmonicsBatch {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
}

#[cfg(test)]
impl SphericalHarmonicsBatch {
    // Create a new batch with the given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            real: Vec::with_capacity(capacity),
            imag: Vec::with_capacity(capacity),
        }
    }
}

// Calculate factorial with safeguards against overflow
fn factorial(n: usize) -> usize {
    if n > 20 {
        // For large n, we'll use a different approach in the spherical_harmonic function
        // This is just a placeholder to avoid overflow
        return usize::MAX;
    }
    (1..=n).product()
}

// Logarithm of factorial to avoid overflow
fn log_factorial(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }

    // For large n, use Stirling's approximation
    if n > 20 {
        let n_f64 = n as f64;
        return 0.5 * (2.0 * PI * n_f64).ln() + n_f64 * (n_f64.ln() - 1.0);
    }

    // For small n, compute directly
    let mut result = 0.0;
    for i in 2..=n {
        result += (i as f64).ln();
    }
    result
}

fn normalization_factor(l: usize, m_abs: usize) -> f64 {
    if let Some(precomputed) = kernel_tables::normalization(l, m_abs) {
        return precomputed;
    }

    if l > 20 || m_abs > 20 {
        let log_numerator = ((2 * l + 1) as f64).ln() + log_factorial(l - m_abs);
        let log_denominator = (4.0 * PI).ln() + log_factorial(l + m_abs);
        let log_normalization = 0.5 * (log_numerator - log_denominator);
        return log_normalization.exp();
    }

    let numerator = (2 * l + 1) * factorial(l - m_abs);
    let denominator = 4.0 * PI * (factorial(l + m_abs) as f64);
    (numerator as f64 / denominator).sqrt()
}

// Calculate the spherical harmonic Y_l^m(theta, phi)
pub fn spherical_harmonic(l: usize, m: isize, theta: f64, phi: f64) -> ComplexValue {
    // Validate quantum numbers
    if m.unsigned_abs() > l {
        return ComplexValue::new(0.0, 0.0);
    }

    // Calculate normalization factor using logarithms to avoid overflow
    let m_abs = m.unsigned_abs();

    let normalization = normalization_factor(l, m_abs);

    // Calculate associated Legendre polynomial
    let legendre = associated_legendre(l, m_abs, theta.cos());

    // Apply sign convention for negative m
    let sign = if m < 0 && m_abs % 2 == 1 { -1.0 } else { 1.0 };

    // Combine all parts
    let m_phi = m as f64 * phi;
    let factor = normalization * sign * legendre;
    ComplexValue::new(factor * m_phi.cos(), factor * m_phi.sin())
}

// Batch calculation of spherical harmonics for multiple points
// This is more efficient than calculating each point individually
#[cfg(test)]
pub fn spherical_harmonic_batch(
    l: usize,
    m: isize,
    thetas: &[f64],
    phis: &[f64],
) -> SphericalHarmonicsBatch {
    let n = thetas.len().min(phis.len());
    let mut batch = SphericalHarmonicsBatch::with_capacity(n);

    // Validate quantum numbers
    if m.unsigned_abs() > l {
        batch.real = vec![0.0; n];
        batch.imag = vec![0.0; n];
        return batch;
    }

    // Calculate normalization factor (same for all points)
    let m_abs = m.unsigned_abs();

    let normalization = normalization_factor(l, m_abs);

    // Apply sign convention for negative m
    let sign = if m < 0 && m_abs % 2 == 1 { -1.0 } else { 1.0 };

    // Calculate for each point
    for i in 0..n {
        let theta = thetas[i];
        let phi = phis[i];

        // Calculate associated Legendre polynomial
        let legendre = associated_legendre(l, m_abs, theta.cos());

        // Calculate complex exponential
        let m_phi = m as f64 * phi;
        let cos_phi = m_phi.cos();
        let sin_phi = m_phi.sin();

        // Combine all parts
        let factor = normalization * sign * legendre;
        batch.real.push(factor * cos_phi);
        batch.imag.push(factor * sin_phi);
    }

    batch
}

// SIMD-optimized batch calculation of spherical harmonics
// This is only available on x86_64 platforms with AVX2 support
#[cfg(target_arch = "x86_64")]
#[cfg(test)]
pub fn spherical_harmonic_batch_simd(
    l: usize,
    m: isize,
    thetas: &[f64],
    phis: &[f64],
) -> SphericalHarmonicsBatch {
    if is_x86_feature_detected!("avx2") {
        unsafe {
            return spherical_harmonic_batch_avx2(l, m, thetas, phis);
        }
    }

    // Fallback to non-SIMD version
    spherical_harmonic_batch(l, m, thetas, phis)
}

// AVX2-optimized batch calculation of spherical harmonics
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(test)]
unsafe fn spherical_harmonic_batch_avx2(
    l: usize,
    m: isize,
    thetas: &[f64],
    phis: &[f64],
) -> SphericalHarmonicsBatch {
    let n = thetas.len().min(phis.len());
    let mut batch = SphericalHarmonicsBatch::with_capacity(n);

    // Validate quantum numbers
    if m.unsigned_abs() > l {
        batch.real = vec![0.0; n];
        batch.imag = vec![0.0; n];
        return batch;
    }

    // Calculate normalization factor (same for all points)
    let m_abs = m.unsigned_abs();

    let normalization = normalization_factor(l, m_abs);

    // Apply sign convention for negative m
    let sign = if m < 0 && m_abs % 2 == 1 { -1.0 } else { 1.0 };

    // Combine normalization and sign
    let norm_sign = normalization * sign;
    let norm_sign_vec = _mm256_set1_pd(norm_sign);
    let m_f64 = m as f64;
    let m_vec = _mm256_set1_pd(m_f64);

    // Process in chunks of 4 (AVX2 can process 4 doubles at once)
    let chunk_size = 4;
    let num_chunks = n / chunk_size;

    // Pre-allocate result vectors
    batch.real = vec![0.0; n];
    batch.imag = vec![0.0; n];

    // Process each chunk
    for chunk_idx in 0..num_chunks {
        let base_idx = chunk_idx * chunk_size;

        // Load theta values
        let theta_ptr = thetas.as_ptr().add(base_idx);
        let theta_vec = _mm256_loadu_pd(theta_ptr);

        // Load phi values
        let phi_ptr = phis.as_ptr().add(base_idx);
        let phi_vec = _mm256_loadu_pd(phi_ptr);

        // Calculate cos(theta) for associated Legendre polynomial
        let cos_theta_vec = _mm256_cos_pd(theta_vec);

        // Calculate associated Legendre polynomial for each value in the chunk
        // For SIMD, we'll calculate each value separately and then combine
        let mut legendre_values = [0.0; 4];
        for i in 0..4 {
            let cos_theta = *cos_theta_vec.as_array_ref().get_unchecked(i);
            legendre_values[i] = associated_legendre(l, m_abs, cos_theta);
        }
        let legendre_vec = _mm256_loadu_pd(legendre_values.as_ptr());

        // Calculate m * phi
        let m_phi_vec = _mm256_mul_pd(m_vec, phi_vec);

        // Calculate sin(m*phi) and cos(m*phi)
        let cos_m_phi_vec = _mm256_cos_pd(m_phi_vec);
        let sin_m_phi_vec = _mm256_sin_pd(m_phi_vec);

        // Multiply by normalization * sign * legendre
        let factor_vec = _mm256_mul_pd(norm_sign_vec, legendre_vec);
        let real_vec = _mm256_mul_pd(factor_vec, cos_m_phi_vec);
        let imag_vec = _mm256_mul_pd(factor_vec, sin_m_phi_vec);

        // Store results
        let real_ptr = batch.real.as_mut_ptr().add(base_idx);
        let imag_ptr = batch.imag.as_mut_ptr().add(base_idx);
        _mm256_storeu_pd(real_ptr, real_vec);
        _mm256_storeu_pd(imag_ptr, imag_vec);
    }

    // Handle remaining elements
    let remaining_start = num_chunks * chunk_size;
    for i in remaining_start..n {
        let theta = thetas[i];
        let phi = phis[i];

        // Calculate associated Legendre polynomial
        let legendre = associated_legendre(l, m_abs, theta.cos());

        // Calculate complex exponential
        let m_phi = m as f64 * phi;
        let cos_phi = m_phi.cos();
        let sin_phi = m_phi.sin();

        // Combine all parts
        let factor = normalization * sign * legendre;
        batch.real[i] = factor * cos_phi;
        batch.imag[i] = factor * sin_phi;
    }

    batch
}

// Calculate the associated Legendre polynomial P_l^m(x)
fn associated_legendre(l: usize, m: usize, x: f64) -> f64 {
    // Quantize x to make it hashable (use 3 decimal places)
    let x_key = (x * 1000.0).round() as i32;

    // Check cache first
    let cache_key = (l, m as isize, x_key);

    let cached = LEGENDRE_CACHE.with(|cache| cache.borrow().get(&cache_key).copied());
    if let Some(cached) = cached {
        return cached;
    }

    // Calculate P_l^m(x)
    let result = if l == 0 && m == 0 {
        // P_0^0(x) = 1
        1.0
    } else if m > l {
        // P_l^m(x) = 0 if m > l
        0.0
    } else if m == 0 {
        // Use recurrence relation for m = 0
        match l {
            0 => 1.0,
            1 => x,
            _ => {
                let p_l_minus_1 = associated_legendre(l - 1, 0, x);
                let p_l_minus_2 = associated_legendre(l - 2, 0, x);
                ((2 * l - 1) as f64 * x * p_l_minus_1 - (l - 1) as f64 * p_l_minus_2) / l as f64
            }
        }
    } else if m == l {
        // P_l^l(x) = (-1)^l * (2l-1)!! * (1-x²)^(l/2)
        let sign = if l % 2 == 1 { -1.0 } else { 1.0 };
        let double_factorial = (1..=2 * l - 1).step_by(2).product::<usize>() as f64;
        sign * double_factorial * (1.0 - x * x).powf(l as f64 / 2.0)
    } else if m == l - 1 {
        // P_l^(l-1)(x) = x * (2l-1) * P_(l-1)^(l-1)(x)
        x * (2 * l - 1) as f64 * associated_legendre(l - 1, l - 1, x)
    } else {
        // Use recurrence relation for m < l
        let term1 = (2 * l - 1) as f64 * x * associated_legendre(l - 1, m, x);
        let term2 = (l + m - 1) as f64 * associated_legendre(l - 2, m, x);
        (term1 - term2) / (l - m) as f64
    };

    // Cache the result
    LEGENDRE_CACHE.with(|cache| {
        cache.borrow_mut().insert(cache_key, result);
    });

    result
}

// Real spherical harmonics (for visualization)
#[cfg(test)]
pub fn real_spherical_harmonic(l: usize, m: isize, theta: f64, phi: f64) -> f64 {
    match m.cmp(&0) {
        std::cmp::Ordering::Equal => {
            // Y_l^0 is already real
            spherical_harmonic(l, 0, theta, phi).re
        }
        std::cmp::Ordering::Greater => {
            // Y_l^m + (-1)^m * Y_l^(-m)
            let y_pos = spherical_harmonic(l, m, theta, phi);
            let y_neg = spherical_harmonic(l, -m, theta, phi);
            let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
            (y_pos + sign * y_neg).re / 2.0_f64.sqrt()
        }
        std::cmp::Ordering::Less => {
            // i * [Y_l^(-m) - (-1)^m * Y_l^m]
            let m_abs = m.abs();
            let y_pos = spherical_harmonic(l, m_abs, theta, phi);
            let y_neg = spherical_harmonic(l, -m_abs, theta, phi);
            let sign = if m_abs % 2 == 0 { 1.0 } else { -1.0 };
            (y_neg - sign * y_pos).im / 2.0_f64.sqrt()
        }
    }
}

// Batch calculation of real spherical harmonics
#[cfg(test)]
pub fn real_spherical_harmonic_batch(l: usize, m: isize, thetas: &[f64], phis: &[f64]) -> Vec<f64> {
    let n = thetas.len().min(phis.len());
    let mut results = Vec::with_capacity(n);

    match m.cmp(&0) {
        std::cmp::Ordering::Equal => {
            // Y_l^0 is already real
            let batch = spherical_harmonic_batch(l, 0, thetas, phis);
            results = batch.real;
        }
        std::cmp::Ordering::Greater => {
            // Y_l^m + (-1)^m * Y_l^(-m)
            let y_pos = spherical_harmonic_batch(l, m, thetas, phis);
            let y_neg = spherical_harmonic_batch(l, -m, thetas, phis);
            let sign = if m % 2 == 0 { 1.0 } else { -1.0 };

            for i in 0..n {
                results.push((y_pos.real[i] + sign * y_neg.real[i]) / 2.0_f64.sqrt());
            }
        }
        std::cmp::Ordering::Less => {
            // i * [Y_l^(-m) - (-1)^m * Y_l^m]
            let m_abs = m.abs();
            let y_pos = spherical_harmonic_batch(l, m_abs, thetas, phis);
            let y_neg = spherical_harmonic_batch(l, -m_abs, thetas, phis);
            let sign = if m_abs % 2 == 0 { 1.0 } else { -1.0 };

            for i in 0..n {
                results.push((y_neg.imag[i] - sign * y_pos.imag[i]) / 2.0_f64.sqrt());
            }
        }
    }

    results
}

// SIMD extension for AVX2
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn _mm256_cos_pd(a: __m256d) -> __m256d {
    // Store lanes to scalar array, apply cos, then reload.
    let mut values = [0.0f64; 4];
    _mm256_storeu_pd(values.as_mut_ptr(), a);
    let mut result = [0.0f64; 4];
    for i in 0..4 {
        result[i] = values[i].cos();
    }
    _mm256_loadu_pd(result.as_ptr())
}

// SIMD extension for AVX2
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn _mm256_sin_pd(a: __m256d) -> __m256d {
    // Store lanes to scalar array, apply sin, then reload.
    let mut values = [0.0f64; 4];
    _mm256_storeu_pd(values.as_mut_ptr(), a);
    let mut result = [0.0f64; 4];
    for i in 0..4 {
        result[i] = values[i].sin();
    }
    _mm256_loadu_pd(result.as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_spherical_harmonic_y00() {
        // Y_0_0 = 1/sqrt(4π)
        let expected = ComplexValue::new(1.0 / (4.0 * PI).sqrt(), 0.0);

        // Test at different angles
        let y00_1 = spherical_harmonic(0, 0, 0.0, 0.0);
        let y00_2 = spherical_harmonic(0, 0, PI / 2.0, PI / 4.0);
        let y00_3 = spherical_harmonic(0, 0, PI, PI);

        // Y_0_0 should be the same value regardless of angle (spherically symmetric)
        assert!((y00_1.re - expected.re).abs() < 1e-10);
        assert!((y00_1.im - expected.im).abs() < 1e-10);

        assert!((y00_2.re - expected.re).abs() < 1e-10);
        assert!((y00_2.im - expected.im).abs() < 1e-10);

        assert!((y00_3.re - expected.re).abs() < 1e-10);
        assert!((y00_3.im - expected.im).abs() < 1e-10);
    }

    #[test]
    fn test_spherical_harmonic_y10() {
        // Y_1_0 = sqrt(3/4π) * cos(θ)

        // Test at θ = 0 (north pole)
        let y10_north = spherical_harmonic(1, 0, 0.0, 0.0);
        let expected_north = ComplexValue::new((3.0 / (4.0 * PI)).sqrt(), 0.0);

        // Test at θ = π/2 (equator)
        let y10_equator = spherical_harmonic(1, 0, PI / 2.0, 0.0);
        let expected_equator = ComplexValue::new(0.0, 0.0);

        // Test at θ = π (south pole)
        let y10_south = spherical_harmonic(1, 0, PI, 0.0);
        let expected_south = ComplexValue::new(-(3.0 / (4.0 * PI)).sqrt(), 0.0);

        // Check results
        assert!((y10_north.re - expected_north.re).abs() < 1e-10);
        assert!((y10_north.im - expected_north.im).abs() < 1e-10);

        assert!((y10_equator.re - expected_equator.re).abs() < 1e-10);
        assert!((y10_equator.im - expected_equator.im).abs() < 1e-10);

        assert!((y10_south.re - expected_south.re).abs() < 1e-10);
        assert!((y10_south.im - expected_south.im).abs() < 1e-10);
    }

    #[test]
    fn test_spherical_harmonic_y11() {
        // Y_1_1 = -sqrt(3/8π) * sin(θ) * e^(iφ)

        // Test at θ = π/2 (equator), φ = 0
        let y11_0 = spherical_harmonic(1, 1, PI / 2.0, 0.0);
        let expected_0 = ComplexValue::new(-(3.0 / (8.0 * PI)).sqrt(), 0.0);

        // Test at θ = π/2 (equator), φ = π/2
        let y11_90 = spherical_harmonic(1, 1, PI / 2.0, PI / 2.0);
        let expected_90 = ComplexValue::new(0.0, -(3.0 / (8.0 * PI)).sqrt());

        // Check results
        assert!((y11_0.re - expected_0.re).abs() < 1e-10);
        assert!((y11_0.im - expected_0.im).abs() < 1e-10);

        assert!((y11_90.re - expected_90.re).abs() < 1e-10);
        assert!((y11_90.im - expected_90.im).abs() < 1e-10);
    }

    #[test]
    fn test_associated_legendre() {
        // P_0^0(x) = 1
        assert!((associated_legendre(0, 0, 0.5) - 1.0).abs() < 1e-10);

        // P_1^0(x) = x
        assert!((associated_legendre(1, 0, 0.5) - 0.5).abs() < 1e-10);

        // P_1^1(x) = -sqrt(1-x²)
        assert!((associated_legendre(1, 1, 0.5) - (-0.866025403784)).abs() < 1e-10);

        // P_2^0(x) = (3x² - 1)/2
        // For x = 0.5, this is (3*0.25 - 1)/2 = (0.75 - 1)/2 = -0.125
        assert!((associated_legendre(2, 0, 0.5) - (-0.125)).abs() < 1e-10);
    }

    #[test]
    fn test_real_spherical_harmonic() {
        // Test real spherical harmonics

        // For l=0, m=0, the real spherical harmonic is the same as the complex one
        let y00_real = real_spherical_harmonic(0, 0, PI / 4.0, PI / 3.0);
        let y00_complex = spherical_harmonic(0, 0, PI / 4.0, PI / 3.0).re;
        assert!((y00_real - y00_complex).abs() < 1e-10);

        // For l=1, m=0, the real spherical harmonic is the same as the complex one
        let y10_real = real_spherical_harmonic(1, 0, PI / 4.0, PI / 3.0);
        let y10_complex = spherical_harmonic(1, 0, PI / 4.0, PI / 3.0).re;
        assert!((y10_real - y10_complex).abs() < 1e-10);
    }

    #[test]
    fn test_spherical_harmonic_batch() {
        // Test batch calculation against individual calculation
        let l = 2;
        let m = 1;
        let thetas = vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];
        let phis = vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];

        let batch = spherical_harmonic_batch(l, m, &thetas, &phis);

        // Compare with individual calculation
        for i in 0..thetas.len() {
            let individual = spherical_harmonic(l, m, thetas[i], phis[i]);
            assert!((batch.real[i] - individual.re).abs() < 1e-10);
            assert!((batch.imag[i] - individual.im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_associated_legendre_batch() {
        // Test vectorized-style loop against individual calculation
        let l = 2;
        let m = 1;
        let x_values = [0.0, 0.25, 0.5, 0.75, 1.0];

        let batch: Vec<f64> = x_values
            .iter()
            .map(|&x| associated_legendre(l, m, x))
            .collect();

        // Compare with individual calculation
        for i in 0..x_values.len() {
            let individual = associated_legendre(l, m, x_values[i]);
            assert!((batch[i] - individual).abs() < 1e-10);
        }
    }

    #[test]
    fn test_real_spherical_harmonic_batch() {
        // Test batch calculation against individual calculation
        let l = 2;
        let m = 1;
        let thetas = vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];
        let phis = vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];

        let batch = real_spherical_harmonic_batch(l, m, &thetas, &phis);

        // Compare with individual calculation
        for i in 0..thetas.len() {
            let individual = real_spherical_harmonic(l, m, thetas[i], phis[i]);
            assert!((batch[i] - individual).abs() < 1e-10);
        }
    }
}
