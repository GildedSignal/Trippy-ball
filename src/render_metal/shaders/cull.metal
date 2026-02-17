#include <metal_stdlib>
using namespace metal;

struct Camera {
    float4x4 view_proj;
    float4 camera_pos;
};

struct CullParams {
    uint point_count;
    uint padding0;
    uint padding1;
    uint padding2;
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

struct CullCounter {
    atomic_uint visible_count;
};

struct DrawPrimitivesIndirectArgs {
    uint vertex_count;
    uint instance_count;
    uint vertex_start;
    uint base_instance;
};

inline bool is_inside_clip_space(float4 clip) {
    if (clip.w <= 1e-5) {
        return false;
    }

    float w = clip.w;
    const float margin = 1.02;
    return clip.x >= -w * margin && clip.x <= w * margin &&
           clip.y >= -w * margin && clip.y <= w * margin &&
           clip.z >= -w * margin && clip.z <= w * margin;
}

inline float map_intensity(float intensity, constant ColorMapParams& color) {
    float denom = max(color.exposure_white - color.exposure_black, 1e-6);
    float mapped = (max(intensity, 0.0) - color.exposure_black) / denom;
    mapped = clamp(mapped, 0.0, 1.0);
    if (color.use_log_scale != 0u) {
        mapped = log(1.0 + mapped * 20.0) / log(21.0);
    }
    mapped = pow(clamp(mapped, 0.0, 1.0), 1.0 / max(color.gamma, 1e-4));
    mapped = (mapped - 0.5) * color.contrast + 0.5 + color.brightness - 1.0;
    return clamp(mapped, 0.0, 1.0);
}

kernel void cull_points(
    const device packed_float3* positions [[buffer(0)]],
    const device float* intensities [[buffer(1)]],
    constant Camera& camera [[buffer(2)]],
    constant ColorMapParams& color [[buffer(3)]],
    constant CullParams& params [[buffer(4)]],
    device uint* visible_indices [[buffer(5)]],
    device CullCounter* counter [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.point_count) {
        return;
    }

    float3 position = float3(positions[gid]);
    float4 clip = camera.view_proj * float4(position, 1.0);
    if (!is_inside_clip_space(clip)) {
        return;
    }

    float mapped = map_intensity(intensities[gid], color);
    if (mapped < color.luminance_threshold) {
        return;
    }

    uint write_index = atomic_fetch_add_explicit(
        &counter->visible_count,
        1u,
        memory_order_relaxed
    );

    if (write_index < params.point_count) {
        visible_indices[write_index] = gid;
    }
}

kernel void finalize_cull(
    device CullCounter* counter [[buffer(0)]],
    device DrawPrimitivesIndirectArgs* args [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    uint visible_count = atomic_load_explicit(&counter->visible_count, memory_order_relaxed);
    args->vertex_count = visible_count;
    args->instance_count = 1u;
    args->vertex_start = 0u;
    args->base_instance = 0u;
}
