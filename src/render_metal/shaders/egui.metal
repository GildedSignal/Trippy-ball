#include <metal_stdlib>
using namespace metal;

struct UiUniforms {
    float2 screen_size;
    float pixels_per_point;
    float _padding;
};

struct UiVertexIn {
    float2 pos [[attribute(0)]];
    float2 uv [[attribute(1)]];
    float4 color [[attribute(2)]];
};

struct UiVertexOut {
    float4 clip_position [[position]];
    float2 uv;
    float4 color;
};

vertex UiVertexOut egui_vs(
    UiVertexIn in [[stage_in]],
    constant UiUniforms& uniforms [[buffer(1)]]
) {
    UiVertexOut out;

    float2 pos_pixels = in.pos * uniforms.pixels_per_point;
    float2 safe_size = max(uniforms.screen_size, float2(1.0, 1.0));
    float2 ndc = float2(
        (pos_pixels.x / safe_size.x) * 2.0 - 1.0,
        1.0 - (pos_pixels.y / safe_size.y) * 2.0
    );

    out.clip_position = float4(ndc, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

fragment float4 egui_fs(
    UiVertexOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    sampler tex_sampler [[sampler(0)]]
) {
    float4 sampled = tex.sample(tex_sampler, in.uv);
    return sampled * in.color;
}
