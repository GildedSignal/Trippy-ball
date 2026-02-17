// Native Metal point pipeline shader.

#include <metal_stdlib>
using namespace metal;

struct Camera {
    float4x4 view_proj;
    float4 camera_pos;
};

struct ColorMapParams {
    float gamma;
    float contrast;
    float brightness;
    uint use_log_scale;
    float luminance_threshold;
    uint color_scheme;
    float exposure_black;
    float exposure_white;
};

struct VertexOut {
    float4 clip_position [[position]];
    float intensity;
    float size [[point_size]];
};

inline float3 gradient5(float t, float3 c0, float3 c1, float3 c2, float3 c3, float3 c4) {
    t = clamp(t, 0.0, 1.0);
    if (t < 0.25) {
        return mix(c0, c1, t / 0.25);
    }
    if (t < 0.5) {
        return mix(c1, c2, (t - 0.25) / 0.25);
    }
    if (t < 0.75) {
        return mix(c2, c3, (t - 0.5) / 0.25);
    }
    return mix(c3, c4, (t - 0.75) / 0.25);
}

float3 intensity_to_color(float intensity, uint scheme) {
    switch (scheme) {
        case 0u: // Viridis
            return gradient5(
                intensity,
                float3(0.267, 0.005, 0.329),
                float3(0.283, 0.141, 0.458),
                float3(0.207, 0.372, 0.553),
                float3(0.129, 0.568, 0.551),
                float3(0.741, 0.873, 0.150)
            );
        case 1u: // Plasma
            return gradient5(
                intensity,
                float3(0.050, 0.030, 0.528),
                float3(0.415, 0.001, 0.658),
                float3(0.693, 0.165, 0.565),
                float3(0.902, 0.430, 0.359),
                float3(0.940, 0.975, 0.131)
            );
        case 2u: // Inferno
            return gradient5(
                intensity,
                float3(0.001, 0.000, 0.014),
                float3(0.179, 0.038, 0.338),
                float3(0.472, 0.111, 0.428),
                float3(0.843, 0.303, 0.250),
                float3(0.988, 0.998, 0.645)
            );
        case 3u: // BlueRed
            return gradient5(
                intensity,
                float3(0.020, 0.100, 0.400),
                float3(0.100, 0.300, 0.700),
                float3(0.900, 0.900, 0.920),
                float3(0.780, 0.220, 0.220),
                float3(0.500, 0.040, 0.040)
            );
        default: // Quantum
            return gradient5(
                intensity,
                float3(0.060, 0.090, 0.190),
                float3(0.120, 0.350, 0.700),
                float3(0.200, 0.720, 0.760),
                float3(0.930, 0.780, 0.280),
                float3(1.000, 0.980, 0.850)
            );
    }
}

inline float normalize_exposure(float intensity, constant ColorMapParams& color) {
    float denom = max(color.exposure_white - color.exposure_black, 1e-6);
    float normalized = (max(intensity, 0.0) - color.exposure_black) / denom;
    return clamp(normalized, 0.0, 1.0);
}

vertex VertexOut point_vs(
    uint vertex_id [[vertex_id]],
    const device packed_float3* positions [[buffer(0)]],
    const device float* intensities [[buffer(1)]],
    constant Camera& camera [[buffer(2)]],
    constant ColorMapParams& color [[buffer(3)]],
    const device uint* visible_indices [[buffer(4)]]
) {
    uint point_index = visible_indices[vertex_id];

    VertexOut out;
    float3 position = float3(positions[point_index]);
    float source_intensity = intensities[point_index];
    out.clip_position = camera.view_proj * float4(position, 1.0);

    float mapped = normalize_exposure(source_intensity, color);
    if (color.use_log_scale != 0u) {
        mapped = log(1.0 + mapped * 20.0) / log(21.0);
    }
    mapped = pow(clamp(mapped, 0.0, 1.0), 1.0 / max(color.gamma, 1e-4));
    mapped = (mapped - 0.5) * color.contrast + 0.5 + color.brightness - 1.0;
    out.intensity = clamp(mapped, 0.0, 1.0);

    float3 to_cam = position - camera.camera_pos.xyz;
    float distance = max(length(to_cam), 1e-3);
    float base_size = 3.0 + 15.0 * pow(max(source_intensity, 0.0), 0.5);
    out.size = max(2.5, base_size / (1.0 + distance * 0.08));

    return out;
}

fragment float4 point_fs(
    VertexOut in [[stage_in]],
    constant ColorMapParams& color [[buffer(0)]]
) {
    float3 mapped_color = intensity_to_color(in.intensity, color.color_scheme);
    return float4(mapped_color, 1.0);
}
