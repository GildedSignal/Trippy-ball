use crate::ui;
use crate::memory::pod::Pod;

#[cfg(target_os = "macos")]
use egui::epaint::Primitive;
#[cfg(target_os = "macos")]
use egui::{TextureFilter, TextureId, TextureOptions};
#[cfg(target_os = "macos")]
use metal::{
    Buffer, BufferRef, CommandBufferRef, DeviceRef, MTLBlendFactor, MTLBlendOperation, MTLCullMode,
    MTLIndexType, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLResourceOptions,
    MTLScissorRect, MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage,
    MTLVertexFormat, MTLVertexStepFunction, MTLViewport, RenderPassDescriptor,
};
#[cfg(target_os = "macos")]
use std::collections::HashMap;

pub struct MetalPainter {
    #[cfg(target_os = "macos")]
    state: Option<MetalPainterState>,
}

impl MetalPainter {
    pub fn new() -> Self {
        Self {
            #[cfg(target_os = "macos")]
            state: None,
        }
    }

    #[cfg(target_os = "macos")]
    pub fn paint(
        &mut self,
        device: &DeviceRef,
        command_buffer: &CommandBufferRef,
        target: &metal::TextureRef,
        frame: &ui::PreparedUiFrame,
    ) -> Result<(), String> {
        let state = self.ensure_state(device)?;
        state.apply_texture_delta(device, &frame.textures_delta)?;

        let (vertices, indices, batches) = build_batches(&frame.paint_jobs);
        if frame.size_in_pixels[0] > 0 && frame.size_in_pixels[1] > 0 && !batches.is_empty() {
            state.upload_mesh_data(&vertices, &indices)?;
            state.encode_draw(command_buffer, target, frame, &batches)?;
        }

        state.release_textures(&frame.textures_delta);
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    pub fn paint(
        &mut self,
        _device: &(),
        _command_buffer: &(),
        _target: &(),
        _frame: &ui::PreparedUiFrame,
    ) -> Result<(), String> {
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn ensure_state(&mut self, device: &DeviceRef) -> Result<&mut MetalPainterState, String> {
        if self.state.is_none() {
            self.state = Some(MetalPainterState::new(device)?);
        }
        self.state
            .as_mut()
            .ok_or_else(|| "Metal painter state initialization failed".to_string())
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct UiVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [u8; 4],
}
#[cfg(target_os = "macos")]
unsafe impl Pod for UiVertex {}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct UiUniforms {
    screen_size: [f32; 2],
    pixels_per_point: f32,
    _padding: f32,
}
#[cfg(target_os = "macos")]
unsafe impl Pod for UiUniforms {}

#[cfg(target_os = "macos")]
struct DrawBatch {
    clip_rect: egui::Rect,
    texture_id: TextureId,
    index_start: usize,
    index_count: usize,
}

#[cfg(target_os = "macos")]
struct PainterTexture {
    texture: metal::Texture,
    options: TextureOptions,
}

#[cfg(target_os = "macos")]
struct MetalPainterState {
    pipeline: metal::RenderPipelineState,
    sampler_linear: metal::SamplerState,
    sampler_nearest: metal::SamplerState,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    vertex_capacity: usize,
    index_capacity: usize,
    textures: HashMap<TextureId, PainterTexture>,
}

#[cfg(target_os = "macos")]
impl MetalPainterState {
    const INITIAL_VERTEX_BYTES: usize = 64 * 1024;
    const INITIAL_INDEX_BYTES: usize = 64 * 1024;
    const MIN_BUFFER_BYTES: usize = 256;

    fn new(device: &DeviceRef) -> Result<Self, String> {
        let compile_options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(include_str!("shaders/egui.metal"), &compile_options)
            .map_err(|err| format!("egui.metal compile failed: {err:?}"))?;
        let vs = library
            .get_function("egui_vs", None)
            .map_err(|err| format!("egui_vs not found: {err:?}"))?;
        let fs = library
            .get_function("egui_fs", None)
            .map_err(|err| format!("egui_fs not found: {err:?}"))?;

        let vertex_desc = metal::VertexDescriptor::new();
        let attributes = vertex_desc.attributes();
        let layouts = vertex_desc.layouts();

        let attr_pos = attributes
            .object_at(0)
            .ok_or_else(|| "missing egui vertex attribute 0".to_string())?;
        attr_pos.set_format(MTLVertexFormat::Float2);
        attr_pos.set_offset(0);
        attr_pos.set_buffer_index(0);

        let attr_uv = attributes
            .object_at(1)
            .ok_or_else(|| "missing egui vertex attribute 1".to_string())?;
        attr_uv.set_format(MTLVertexFormat::Float2);
        attr_uv.set_offset(8);
        attr_uv.set_buffer_index(0);

        let attr_color = attributes
            .object_at(2)
            .ok_or_else(|| "missing egui vertex attribute 2".to_string())?;
        attr_color.set_format(MTLVertexFormat::UChar4Normalized);
        attr_color.set_offset(16);
        attr_color.set_buffer_index(0);

        let layout = layouts
            .object_at(0)
            .ok_or_else(|| "missing egui vertex layout 0".to_string())?;
        layout.set_stride(std::mem::size_of::<UiVertex>() as u64);
        layout.set_step_function(MTLVertexStepFunction::PerVertex);

        let pipeline_desc = metal::RenderPipelineDescriptor::new();
        pipeline_desc.set_vertex_function(Some(&vs));
        pipeline_desc.set_fragment_function(Some(&fs));
        pipeline_desc.set_vertex_descriptor(Some(vertex_desc));

        let color = pipeline_desc
            .color_attachments()
            .object_at(0)
            .ok_or_else(|| "missing egui pipeline color attachment".to_string())?;
        color.set_pixel_format(MTLPixelFormat::BGRA8Unorm_sRGB);
        color.set_blending_enabled(true);
        color.set_rgb_blend_operation(MTLBlendOperation::Add);
        color.set_alpha_blend_operation(MTLBlendOperation::Add);
        color.set_source_rgb_blend_factor(MTLBlendFactor::One);
        color.set_source_alpha_blend_factor(MTLBlendFactor::One);
        color.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        color.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        let pipeline = device
            .new_render_pipeline_state(&pipeline_desc)
            .map_err(|err| format!("egui pipeline build failed: {err:?}"))?;

        let sampler_linear = {
            let desc = metal::SamplerDescriptor::new();
            desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
            desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
            desc.set_mip_filter(metal::MTLSamplerMipFilter::NotMipmapped);
            desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
            desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
            device.new_sampler(&desc)
        };

        let sampler_nearest = {
            let desc = metal::SamplerDescriptor::new();
            desc.set_min_filter(metal::MTLSamplerMinMagFilter::Nearest);
            desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Nearest);
            desc.set_mip_filter(metal::MTLSamplerMipFilter::NotMipmapped);
            desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
            desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
            device.new_sampler(&desc)
        };

        let vertex_capacity = Self::INITIAL_VERTEX_BYTES.max(Self::MIN_BUFFER_BYTES);
        let index_capacity = Self::INITIAL_INDEX_BYTES.max(Self::MIN_BUFFER_BYTES);
        let vertex_buffer = create_shared_buffer(device, vertex_capacity, "egui-vertex-buffer");
        let index_buffer = create_shared_buffer(device, index_capacity, "egui-index-buffer");

        Ok(Self {
            pipeline,
            sampler_linear,
            sampler_nearest,
            vertex_buffer,
            index_buffer,
            vertex_capacity,
            index_capacity,
            textures: HashMap::new(),
        })
    }

    fn apply_texture_delta(
        &mut self,
        device: &DeviceRef,
        textures_delta: &egui::TexturesDelta,
    ) -> Result<(), String> {
        for (id, image_delta) in &textures_delta.set {
            self.update_texture(device, *id, image_delta)?;
        }
        Ok(())
    }

    fn release_textures(&mut self, textures_delta: &egui::TexturesDelta) {
        for id in &textures_delta.free {
            self.textures.remove(id);
        }
    }

    fn update_texture(
        &mut self,
        device: &DeviceRef,
        id: TextureId,
        image_delta: &egui::epaint::ImageDelta,
    ) -> Result<(), String> {
        let [width, height] = image_delta.image.size();
        if width == 0 || height == 0 {
            return Ok(());
        }

        let rgba = image_to_rgba_bytes(&image_delta.image);
        let x = image_delta.pos.map(|pos| pos[0]).unwrap_or(0);
        let y = image_delta.pos.map(|pos| pos[1]).unwrap_or(0);

        if image_delta.pos.is_none() {
            let needs_recreate = match self.textures.get(&id) {
                Some(existing) => {
                    existing.texture.width() != width as u64
                        || existing.texture.height() != height as u64
                }
                None => true,
            };

            if needs_recreate {
                let texture = create_texture(device, width as u64, height as u64, id);
                self.textures.insert(
                    id,
                    PainterTexture {
                        texture,
                        options: image_delta.options,
                    },
                );
            }
        }

        let entry = self
            .textures
            .get_mut(&id)
            .ok_or_else(|| format!("missing egui texture for id {id:?}"))?;
        entry.options = image_delta.options;

        let target_width = entry.texture.width() as usize;
        let target_height = entry.texture.height() as usize;
        if x + width > target_width || y + height > target_height {
            return Err(format!(
                "egui texture update out of bounds: tex={id:?} update=({},{} {}x{}) target={}x{}",
                x, y, width, height, target_width, target_height
            ));
        }

        let region = MTLRegion::new_2d(x as u64, y as u64, width as u64, height as u64);
        entry.texture.replace_region(
            region,
            0,
            rgba.as_ptr() as *const std::ffi::c_void,
            (4 * width) as u64,
        );

        Ok(())
    }

    fn upload_mesh_data(&mut self, vertices: &[UiVertex], indices: &[u32]) -> Result<(), String> {
        let vertex_bytes = std::mem::size_of_val(vertices);
        let index_bytes = std::mem::size_of_val(indices);

        if vertex_bytes > self.vertex_capacity {
            self.vertex_capacity = align_bytes(vertex_bytes.next_power_of_two());
            self.vertex_buffer = create_shared_buffer(
                self.vertex_buffer.device(),
                self.vertex_capacity,
                "egui-vertex-buffer",
            );
        }

        if index_bytes > self.index_capacity {
            self.index_capacity = align_bytes(index_bytes.next_power_of_two());
            self.index_buffer = create_shared_buffer(
                self.index_buffer.device(),
                self.index_capacity,
                "egui-index-buffer",
            );
        }

        write_buffer(self.vertex_buffer.as_ref(), vertices)?;
        write_buffer(self.index_buffer.as_ref(), indices)?;
        Ok(())
    }

    fn encode_draw(
        &self,
        command_buffer: &CommandBufferRef,
        target: &metal::TextureRef,
        frame: &ui::PreparedUiFrame,
        batches: &[DrawBatch],
    ) -> Result<(), String> {
        let width = frame.size_in_pixels[0] as f32;
        let height = frame.size_in_pixels[1] as f32;
        if width <= 0.0 || height <= 0.0 {
            return Ok(());
        }

        let uniforms = UiUniforms {
            screen_size: [width, height],
            pixels_per_point: frame.pixels_per_point,
            _padding: 0.0,
        };

        let pass = RenderPassDescriptor::new();
        let color_attachment = pass
            .color_attachments()
            .object_at(0)
            .ok_or_else(|| "missing egui render pass color attachment".to_string())?;
        color_attachment.set_texture(Some(target));
        color_attachment.set_load_action(MTLLoadAction::Load);
        color_attachment.set_store_action(MTLStoreAction::Store);

        let encoder = command_buffer.new_render_command_encoder(pass);
        encoder.set_render_pipeline_state(&self.pipeline);
        encoder.set_cull_mode(MTLCullMode::None);
        encoder.set_viewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: width as f64,
            height: height as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        encoder.set_vertex_buffer(0, Some(self.vertex_buffer.as_ref()), 0);
        encoder.set_vertex_bytes(
            1,
            std::mem::size_of::<UiUniforms>() as u64,
            (&uniforms as *const UiUniforms).cast(),
        );

        for batch in batches {
            let Some((scissor, has_area)) =
                clip_rect_to_scissor(batch.clip_rect, frame.pixels_per_point, width, height)
            else {
                continue;
            };
            if !has_area {
                continue;
            }
            encoder.set_scissor_rect(scissor);

            let Some(texture) = self.textures.get(&batch.texture_id) else {
                continue;
            };
            encoder.set_fragment_texture(0, Some(texture.texture.as_ref()));
            encoder.set_fragment_sampler_state(
                0,
                Some(match sampler_mode(texture.options) {
                    SamplerMode::Nearest => self.sampler_nearest.as_ref(),
                    SamplerMode::Linear => self.sampler_linear.as_ref(),
                }),
            );
            encoder.draw_indexed_primitives(
                MTLPrimitiveType::Triangle,
                batch.index_count as u64,
                MTLIndexType::UInt32,
                self.index_buffer.as_ref(),
                (batch.index_start * std::mem::size_of::<u32>()) as u64,
            );
        }

        encoder.end_encoding();
        Ok(())
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
enum SamplerMode {
    Linear,
    Nearest,
}

#[cfg(target_os = "macos")]
fn sampler_mode(options: TextureOptions) -> SamplerMode {
    if options.minification == TextureFilter::Nearest
        && options.magnification == TextureFilter::Nearest
    {
        SamplerMode::Nearest
    } else {
        SamplerMode::Linear
    }
}

#[cfg(target_os = "macos")]
fn build_batches(
    paint_jobs: &[egui::ClippedPrimitive],
) -> (Vec<UiVertex>, Vec<u32>, Vec<DrawBatch>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut batches = Vec::new();

    for clipped in paint_jobs {
        let Primitive::Mesh(mesh) = &clipped.primitive else {
            continue;
        };
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            continue;
        }

        let vertex_base = vertices.len() as u32;
        let index_start = indices.len();

        vertices.extend(mesh.vertices.iter().map(|v| {
            let [r, g, b, a] = v.color.to_array();
            UiVertex {
                pos: [v.pos.x, v.pos.y],
                uv: [v.uv.x, v.uv.y],
                color: [r, g, b, a],
            }
        }));
        indices.extend(mesh.indices.iter().map(|i| vertex_base + *i));

        batches.push(DrawBatch {
            clip_rect: clipped.clip_rect,
            texture_id: mesh.texture_id,
            index_start,
            index_count: mesh.indices.len(),
        });
    }

    (vertices, indices, batches)
}

#[cfg(target_os = "macos")]
fn image_to_rgba_bytes(image_data: &egui::ImageData) -> Vec<u8> {
    match image_data {
        egui::ImageData::Color(color) => {
            let mut bytes = Vec::with_capacity(color.pixels.len() * 4);
            for pixel in &color.pixels {
                bytes.extend_from_slice(&pixel.to_array());
            }
            bytes
        }
        egui::ImageData::Font(font) => {
            let mut bytes = Vec::with_capacity(font.pixels.len() * 4);
            for pixel in font.srgba_pixels(None) {
                bytes.extend_from_slice(&pixel.to_array());
            }
            bytes
        }
    }
}

#[cfg(target_os = "macos")]
fn clip_rect_to_scissor(
    clip_rect: egui::Rect,
    pixels_per_point: f32,
    target_width: f32,
    target_height: f32,
) -> Option<(MTLScissorRect, bool)> {
    let min_x = (clip_rect.min.x * pixels_per_point)
        .floor()
        .clamp(0.0, target_width);
    let min_y = (clip_rect.min.y * pixels_per_point)
        .floor()
        .clamp(0.0, target_height);
    let max_x = (clip_rect.max.x * pixels_per_point)
        .ceil()
        .clamp(min_x, target_width);
    let max_y = (clip_rect.max.y * pixels_per_point)
        .ceil()
        .clamp(min_y, target_height);

    let width = (max_x - min_x).max(0.0) as u64;
    let height = (max_y - min_y).max(0.0) as u64;
    if width == 0 || height == 0 {
        return Some((
            MTLScissorRect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
            false,
        ));
    }

    Some((
        MTLScissorRect {
            x: min_x as u64,
            y: min_y as u64,
            width,
            height,
        },
        true,
    ))
}

#[cfg(target_os = "macos")]
fn align_bytes(bytes: usize) -> usize {
    let aligned = bytes.max(MetalPainterState::MIN_BUFFER_BYTES);
    let remainder = aligned % MetalPainterState::MIN_BUFFER_BYTES;
    if remainder == 0 {
        aligned
    } else {
        aligned + (MetalPainterState::MIN_BUFFER_BYTES - remainder)
    }
}

#[cfg(target_os = "macos")]
fn create_shared_buffer(device: &DeviceRef, bytes: usize, label: &str) -> Buffer {
    let buffer = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    buffer.set_label(label);
    buffer
}

#[cfg(target_os = "macos")]
fn create_texture(device: &DeviceRef, width: u64, height: u64, id: TextureId) -> metal::Texture {
    let desc = metal::TextureDescriptor::new();
    desc.set_texture_type(MTLTextureType::D2);
    desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm_sRGB);
    desc.set_width(width.max(1));
    desc.set_height(height.max(1));
    desc.set_depth(1);
    desc.set_mipmap_level_count(1);
    desc.set_storage_mode(MTLStorageMode::Shared);
    desc.set_usage(MTLTextureUsage::ShaderRead);

    let texture = device.new_texture(&desc);
    texture.set_label(&format!("egui-texture-{id:?}"));
    texture
}

#[cfg(target_os = "macos")]
fn write_buffer<T: Pod>(buffer: &BufferRef, data: &[T]) -> Result<(), String> {
    let byte_len = std::mem::size_of_val(data);
    if byte_len > buffer.length() as usize {
        return Err(format!(
            "buffer write overflow: {} bytes > {} bytes",
            byte_len,
            buffer.length()
        ));
    }
    if byte_len == 0 {
        return Ok(());
    }

    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buffer.contents() as *mut u8,
            byte_len,
        );
    }
    Ok(())
}
