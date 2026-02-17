use std::f64::consts::PI;
use std::sync::LazyLock;

pub const MAX_L_PRECOMPUTE: usize = 5;
pub const NORMALIZATION_TABLE_FLAT_LEN: usize = (MAX_L_PRECOMPUTE + 1) * (MAX_L_PRECOMPUTE + 1);

pub static NORMALIZATION_TABLE: LazyLock<[[f64; MAX_L_PRECOMPUTE + 1]; MAX_L_PRECOMPUTE + 1]> =
    LazyLock::new(|| {
        let mut table = [[0.0; MAX_L_PRECOMPUTE + 1]; MAX_L_PRECOMPUTE + 1];

        for (l, row) in table.iter_mut().enumerate().take(MAX_L_PRECOMPUTE + 1) {
            for (m_abs, cell) in row.iter_mut().enumerate().take(l + 1) {
                let numerator = (2 * l + 1) as f64 * factorial(l - m_abs) as f64;
                let denominator = 4.0 * PI * factorial(l + m_abs) as f64;
                *cell = (numerator / denominator).sqrt();
            }
        }

        table
    });

pub static NORMALIZATION_TABLE_F32_FLAT: LazyLock<[f32; NORMALIZATION_TABLE_FLAT_LEN]> =
    LazyLock::new(|| {
        let mut flat = [0.0f32; NORMALIZATION_TABLE_FLAT_LEN];
        for l in 0..=MAX_L_PRECOMPUTE {
            for m_abs in 0..=MAX_L_PRECOMPUTE {
                flat[l * (MAX_L_PRECOMPUTE + 1) + m_abs] = NORMALIZATION_TABLE[l][m_abs] as f32;
            }
        }
        flat
    });

pub fn normalization(l: usize, m_abs: usize) -> Option<f64> {
    if l <= MAX_L_PRECOMPUTE && m_abs <= l {
        Some(NORMALIZATION_TABLE[l][m_abs])
    } else {
        None
    }
}

pub fn normalization_table_f32_flat() -> &'static [f32; NORMALIZATION_TABLE_FLAT_LEN] {
    &NORMALIZATION_TABLE_F32_FLAT
}

fn factorial(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    (1..=n).product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_table_matches_scalar_lookup_for_valid_pairs() {
        let flat = normalization_table_f32_flat();
        for l in 0..=MAX_L_PRECOMPUTE {
            for m_abs in 0..=l {
                let idx = l * (MAX_L_PRECOMPUTE + 1) + m_abs;
                let scalar = normalization(l, m_abs).unwrap() as f32;
                assert!((flat[idx] - scalar).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn flat_table_keeps_invalid_entries_zeroed() {
        let flat = normalization_table_f32_flat();
        for l in 0..=MAX_L_PRECOMPUTE {
            for m_abs in (l + 1)..=MAX_L_PRECOMPUTE {
                let idx = l * (MAX_L_PRECOMPUTE + 1) + m_abs;
                assert_eq!(flat[idx], 0.0);
            }
        }
    }
}
