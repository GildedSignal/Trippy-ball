use num_complex::Complex64;
use statrs::function::gamma::gamma;
use std::f64::consts::PI;
use wavefunction_visualization::math::WavefunctionParams;

const RYDBERG_EV: f64 = 13.605_693_122_994;
const EV_TO_J: f64 = 1.602_176_634e-19;
const HBAR_SI: f64 = 1.054_571_817e-34;

fn factorial(n: usize) -> f64 {
    gamma(n as f64 + 1.0)
}

fn associated_legendre(l: usize, m_abs: usize, x: f64) -> f64 {
    if m_abs > l {
        return 0.0;
    }

    let x = x.clamp(-1.0, 1.0);
    let mut pmm = 1.0;
    if m_abs > 0 {
        let somx2 = (1.0 - x * x).max(0.0).sqrt();
        let mut fact = 1.0;
        for _ in 1..=m_abs {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }

    if l == m_abs {
        return pmm;
    }

    let mut pmmp1 = x * (2 * m_abs + 1) as f64 * pmm;
    if l == m_abs + 1 {
        return pmmp1;
    }

    let mut pll = 0.0;
    for ll in (m_abs + 2)..=l {
        pll = ((2 * ll - 1) as f64 * x * pmmp1 - (ll + m_abs - 1) as f64 * pmm)
            / (ll - m_abs) as f64;
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

fn normalization_factor(l: usize, m_abs: usize) -> f64 {
    let numerator = (2 * l + 1) as f64 * factorial(l - m_abs);
    let denominator = 4.0 * PI * factorial(l + m_abs);
    (numerator / denominator).sqrt()
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

fn hydrogen_energy_joule(n: usize, z: f64) -> f64 {
    let n_f = n.max(1) as f64;
    -RYDBERG_EV * z * z / (n_f * n_f) * EV_TO_J
}

pub fn spherical_harmonic(l: usize, m: isize, theta: f64, phi: f64) -> Complex64 {
    let m_abs = m.unsigned_abs();
    if m_abs > l {
        return Complex64::new(0.0, 0.0);
    }

    let norm = normalization_factor(l, m_abs);
    let legendre = associated_legendre(l, m_abs, theta.cos());
    let sign = if m < 0 && m_abs % 2 == 1 { -1.0 } else { 1.0 };
    let phase = Complex64::from_polar(1.0, m as f64 * phi);
    phase * (sign * norm * legendre)
}

pub fn hydrogenic_radial(n: usize, l: usize, r_a0: f64, z: f64) -> f64 {
    if n == 0 || l >= n || z <= 0.0 {
        return 0.0;
    }

    let n_f = n as f64;
    let rho = 2.0 * z * r_a0 / n_f;
    let lag_n = n - l - 1;
    let prefactor = (2.0 * z / n_f).powf(1.5);
    let norm = prefactor * (factorial(lag_n) / (2.0 * n_f * factorial(n + l))).sqrt();
    let laguerre = associated_laguerre(lag_n, 2 * l + 1, rho);
    norm * (-0.5 * rho).exp() * rho.max(0.0).powi(l as i32) * laguerre
}

pub fn integrate_radial_probability(n: usize, l: usize, z: f64, r_max: f64, steps: usize) -> f64 {
    let steps = steps.max(1);
    let dr = r_max / steps as f64;
    let mut integral = 0.0;

    for i in 0..steps {
        let r = (i as f64 + 0.5) * dr;
        let radial = hydrogenic_radial(n, l, r, z);
        integral += radial * radial * r * r * dr;
    }

    integral
}

pub fn evaluate_density(position: glam::Vec3, params: &WavefunctionParams, time: f64) -> f64 {
    let valid = |n: usize, l: usize, m: isize| n >= 1 && l < n && m.unsigned_abs() <= l;
    if !valid(params.n, params.l, params.m) || !valid(params.n2, params.l2, params.m2) {
        return 0.0;
    }

    let x = position.x as f64;
    let y = position.y as f64;
    let z_cart = position.z as f64;
    let r = (x * x + y * y + z_cart * z_cart).sqrt();
    let theta = if r <= 1e-10 {
        0.0
    } else {
        (z_cart / r).clamp(-1.0, 1.0).acos()
    };
    let phi = if x.abs() <= 1e-12 && y.abs() <= 1e-12 {
        0.0
    } else {
        y.atan2(x)
    };

    let z = params.z.max(1e-6);
    let y1 = spherical_harmonic(params.l, params.m, theta, phi);
    let y2 = spherical_harmonic(params.l2, params.m2, theta, phi);
    let radial_1 = hydrogenic_radial(params.n, params.l, r, z);
    let radial_2 = hydrogenic_radial(params.n2, params.l2, r, z);

    let t_phys_s = time * params.time_factor.max(0.0) * 1e-15;
    let phase1 = -hydrogen_energy_joule(params.n, z) * t_phys_s / HBAR_SI;
    let phase2 = -hydrogen_energy_joule(params.n2, z) * t_phys_s / HBAR_SI + params.relative_phase;

    let mix = params.mix.clamp(0.0, 1.0);
    let amp1 = (1.0 - mix).sqrt();
    let amp2 = mix.sqrt();
    let psi1 = y1 * radial_1 * Complex64::from_polar(1.0, phase1) * amp1;
    let psi2 = y2 * radial_2 * Complex64::from_polar(1.0, phase2) * amp2;
    let psi = psi1 + psi2;

    psi.norm_sqr()
}
