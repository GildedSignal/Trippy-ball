// Native Metal compute kernel for wavefunction evaluation.

#include <metal_stdlib>
using namespace metal;

constant float PI_F = 3.14159265359;
constant uint NORMALIZATION_TABLE_WIDTH = 6u;
constant uint NORMALIZATION_TABLE_MAX_L = 5u;

struct Params {
    uint n1;
    uint l;
    int m;
    uint n2;
    uint l2;
    int m2;
    float mix;
    float relative_phase;
    float z;
    float time_factor;
    float time;
    uint point_count;
    uint start_index;
    uint compute_count;
};

inline float factorial(uint n) {
    float result = 1.0;
    for (uint i = 2u; i <= n; ++i) {
        result *= float(i);
    }
    return result;
}

inline float associated_legendre(uint l, int m, float x) {
    if (l == 0u) {
        return 1.0;
    }

    if (l == 1u) {
        if (m == 0) {
            return x;
        }
        if (m == 1 || m == -1) {
            return -sqrt(max(1.0 - x * x, 0.0));
        }
    }

    if (l == 2u) {
        if (m == 0) {
            return 0.5 * (3.0 * x * x - 1.0);
        }
        if (m == 1 || m == -1) {
            return -3.0 * x * sqrt(max(1.0 - x * x, 0.0));
        }
        if (m == 2 || m == -2) {
            return 3.0 * (1.0 - x * x);
        }
    }

    int m_abs = abs(m);
    float pmm = 1.0;
    float somx2 = sqrt(max((1.0 - x) * (1.0 + x), 0.0));
    float fact = 1.0;

    for (int i = 1; i <= m_abs; ++i) {
        pmm *= (-fact) * somx2;
        fact += 2.0;
    }

    if (l == uint(m_abs)) {
        return pmm;
    }

    float pmmp1 = x * float(2 * m_abs + 1) * pmm;
    if (l == uint(m_abs + 1)) {
        return pmmp1;
    }

    float pll = 0.0;
    for (uint ll = uint(m_abs + 2); ll <= l; ++ll) {
        float ll_f = float(ll);
        float m_abs_f = float(m_abs);
        pll = (x * (2.0 * ll_f - 1.0) * pmmp1 - (ll_f + m_abs_f - 1.0) * pmm) / (ll_f - m_abs_f);
        pmm = pmmp1;
        pmmp1 = pll;
    }

    return pll;
}

inline float normalization_from_table(uint l, uint m_abs, constant float* normalization_table) {
    if (l <= NORMALIZATION_TABLE_MAX_L && m_abs <= l) {
        return normalization_table[l * NORMALIZATION_TABLE_WIDTH + m_abs];
    }

    return sqrt(((2.0 * float(l) + 1.0) * factorial(l - m_abs)) /
                (4.0 * PI_F * factorial(l + m_abs)));
}

inline float2 spherical_harmonic(
    uint l,
    int m,
    float theta,
    float phi,
    constant float* normalization_table
) {
    uint m_abs = uint(abs(m));
    if (m_abs > l) {
        return float2(0.0, 0.0);
    }
    float norm = normalization_from_table(l, m_abs, normalization_table);

    float plm = associated_legendre(l, m, cos(theta));
    float phase = float(m) * phi;
    float real = cos(phase);
    float imag = sin(phase);
    float sign = (m < 0 && (m_abs & 1u) == 1u) ? -1.0 : 1.0;

    return float2(sign * norm * plm * real, sign * norm * plm * imag);
}

inline float associated_laguerre(uint n, uint k, float x) {
    if (n == 0u) {
        return 1.0;
    }
    if (n == 1u) {
        return 1.0 + float(k) - x;
    }
    float l_nm2 = 1.0;
    float l_nm1 = 1.0 + float(k) - x;
    for (uint i = 2u; i <= n; ++i) {
        float i_f = float(i);
        float term1 = (2.0 * i_f - 1.0 + float(k) - x) * l_nm1;
        float term2 = (i_f - 1.0 + float(k)) * l_nm2;
        float l_n = (term1 - term2) / i_f;
        l_nm2 = l_nm1;
        l_nm1 = l_n;
    }
    return l_nm1;
}

inline float hydrogenic_radial(uint n, uint l, float r_a0, float z) {
    if (n == 0u || l >= n || z <= 0.0) {
        return 0.0;
    }
    float n_f = float(n);
    float rho = 2.0 * z * r_a0 / n_f;
    uint lag_n = n - l - 1u;
    float prefactor = pow(2.0 * z / n_f, 1.5);
    float norm = prefactor * sqrt(factorial(lag_n) / (2.0 * n_f * factorial(n + l)));
    float laguerre = associated_laguerre(lag_n, 2u * l + 1u, rho);
    return norm * exp(-0.5 * rho) * pow(max(rho, 0.0), float(l)) * laguerre;
}

inline float hydrogen_energy_joule(uint n, float z) {
    const float rydberg_ev = 13.605693f;
    const float ev_to_j = 1.602176634e-19f;
    float n_f = max(float(n), 1.0);
    return -rydberg_ev * z * z / (n_f * n_f) * ev_to_j;
}

inline float2 complex_phase(float phase) {
    return float2(cos(phase), sin(phase));
}

inline float2 complex_mul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

kernel void evaluate_wavefunction(
    const device packed_float3* positions [[buffer(0)]],
    device float* intensities [[buffer(1)]],
    constant Params& params [[buffer(2)]],
    constant float* normalization_table [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.compute_count) {
        return;
    }

    uint index = params.start_index + gid;
    if (index >= params.point_count) {
        return;
    }

    float3 pos = float3(positions[index]);
    float r = length(pos);
    float theta = 0.0;
    float phi = 0.0;
    if (r > 1e-10) {
        float cos_theta = clamp(pos.z / r, -1.0, 1.0);
        theta = acos(cos_theta);
        if (fabs(pos.x) > 1e-12 || fabs(pos.y) > 1e-12) {
            phi = atan2(pos.y, pos.x);
        }
    }

    float z = max(params.z, 1e-6);
    float2 angular1 = spherical_harmonic(params.l, params.m, theta, phi, normalization_table);
    float2 angular2 = spherical_harmonic(params.l2, params.m2, theta, phi, normalization_table);
    float radial1 = hydrogenic_radial(params.n1, params.l, r, z);
    float radial2 = hydrogenic_radial(params.n2, params.l2, r, z);

    const float hbar = 1.054571817e-34f;
    float t_phys_s = params.time * max(params.time_factor, 0.0) * 1e-15f;
    float phase1 = -hydrogen_energy_joule(params.n1, z) * t_phys_s / hbar;
    float phase2 = -hydrogen_energy_joule(params.n2, z) * t_phys_s / hbar + params.relative_phase;

    float mix = clamp(params.mix, 0.0, 1.0);
    float amp1 = sqrt(1.0 - mix);
    float amp2 = sqrt(mix);
    float2 psi1 = complex_mul(float2(radial1 * angular1.x, radial1 * angular1.y), complex_phase(phase1)) * amp1;
    float2 psi2 = complex_mul(float2(radial2 * angular2.x, radial2 * angular2.y), complex_phase(phase2)) * amp2;
    float2 wave = psi1 + psi2;

    intensities[index] = wave.x * wave.x + wave.y * wave.y;
}
