use super::metal_painter::MetalPainter;
use super::surface::MetalSurface;
use crate::memory::pod::{self, Pod};
use crate::render::CameraUniform;
use crate::ui;
use glam::Vec3;

#[cfg(target_os = "macos")]
use super::ring_buffer::SharedBufferRing;

#[cfg(target_os = "macos")]
use std::sync::LazyLock;

#[derive(Clone, Debug)]
pub struct MetalBootstrapInfo {
    pub device_name: String,
    pub ring_slots: usize,
    pub ring_slot_bytes: usize,
    pub shadow_point_capacity: usize,
    pub shadow_point_limit: usize,
    pub surface_width: u32,
    pub surface_height: u32,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MetalColorParams {
    pub gamma: f32,
    pub contrast: f32,
    pub brightness: f32,
    pub use_log_scale: bool,
    pub color_scheme: crate::render::color::ColorScheme,
    pub luminance_threshold: f32,
    pub exposure_black: f32,
    pub exposure_white: f32,
}

impl Default for MetalColorParams {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            contrast: 1.2,
            brightness: 1.0,
            use_log_scale: true,
            color_scheme: crate::render::color::ColorScheme::Quantum,
            luminance_threshold: 0.0,
            exposure_black: 0.0,
            exposure_white: 1.0,
        }
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct ShadowFrameStats {
    pub encoded_points: usize,
    pub truncated_points: usize,
    pub visible_points: usize,
    pub gpu_culled_points: usize,
    pub gpu_culling_percentage: f32,
}

#[cfg(target_os = "macos")]
pub type ExternalIntensityBufferRef<'a> = Option<&'a metal::BufferRef>;
#[cfg(not(target_os = "macos"))]
pub type ExternalIntensityBufferRef<'a> = Option<&'a ()>;

pub struct PresentedFrameInput<'a> {
    pub surface: &'a MetalSurface,
    pub ui_painter: &'a mut MetalPainter,
    pub ui_frame: &'a ui::PreparedUiFrame,
    pub frame_id: u64,
    pub positions: &'a [Vec3],
    pub intensities: &'a [f32],
    pub external_intensity_buffer: ExternalIntensityBufferRef<'a>,
    pub camera_uniform: CameraUniform,
    pub color_params: MetalColorParams,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct MetalPositionData {
    position: [f32; 3],
}
#[cfg(target_os = "macos")]
unsafe impl Pod for MetalPositionData {}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct MetalColorUniform {
    gamma: f32,
    contrast: f32,
    brightness: f32,
    use_log_scale: u32,
    luminance_threshold: f32,
    color_scheme: u32,
    exposure_black: f32,
    exposure_white: f32,
}
#[cfg(target_os = "macos")]
unsafe impl Pod for MetalColorUniform {}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct MetalCullParams {
    point_count: u32,
    padding: [u32; 3],
}
#[cfg(target_os = "macos")]
unsafe impl Pod for MetalCullParams {}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct MetalCullCounter {
    visible_count: u32,
    padding: [u32; 3],
}
#[cfg(target_os = "macos")]
unsafe impl Pod for MetalCullCounter {}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct MetalDrawPrimitivesIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    vertex_start: u32,
    base_instance: u32,
}
#[cfg(target_os = "macos")]
unsafe impl Pod for MetalDrawPrimitivesIndirectArgs {}

#[cfg(target_os = "macos")]
pub struct MetalContext {
    device: metal::Device,
    command_queue: metal::CommandQueue,
    point_library: metal::Library,
    cull_library: metal::Library,
    point_pipeline: metal::RenderPipelineState,
    cull_pipeline: metal::ComputePipelineState,
    cull_finalize_pipeline: metal::ComputePipelineState,
    shared_ring: SharedBufferRing,
    position_buffer: metal::Buffer,
    intensity_buffer: metal::Buffer,
    visible_indices_buffers: Vec<metal::Buffer>,
    cull_params_buffer: metal::Buffer,
    cull_counter_buffers: Vec<metal::Buffer>,
    draw_indirect_buffers: Vec<metal::Buffer>,
    cull_slot_input_points: Vec<usize>,
    cull_slot_truncated_points: Vec<usize>,
    point_capacity: usize,
    shadow_point_limit: usize,
    camera_buffer: metal::Buffer,
    color_buffer: metal::Buffer,
    surface_width: u32,
    surface_height: u32,
    staging_positions: Vec<MetalPositionData>,
    staging_intensities: Vec<f32>,
    positions_dirty: bool,
    uploaded_positions: usize,
    last_shadow_stats: ShadowFrameStats,
}

#[cfg(target_os = "macos")]
impl MetalContext {
    const SHARED_RING_SLOTS: usize = 3;
    const MIN_SHADOW_POINTS: usize = 1_000;
    const DEFAULT_SHADOW_POINTS: usize = 5_000_000;
    pub const MAX_SHADOW_POINTS: usize = 5_000_000;
    const SHADOW_POINT_LIMIT_ENV: &'static str = "TRIPPY_BALL_NATIVE_SHADOW_POINTS";

    fn color_scheme_code(scheme: crate::render::color::ColorScheme) -> u32 {
        match scheme {
            crate::render::color::ColorScheme::Viridis => 0,
            crate::render::color::ColorScheme::Plasma => 1,
            crate::render::color::ColorScheme::Inferno => 2,
            crate::render::color::ColorScheme::BlueRed => 3,
            crate::render::color::ColorScheme::Quantum => 4,
        }
    }

    pub fn new(
        surface_width: u32,
        surface_height: u32,
        effective_point_count: usize,
    ) -> Result<Self, String> {
        let device = metal::Device::system_default()
            .ok_or_else(|| "Metal device unavailable".to_string())?;
        let command_queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();

        let point_library = device
            .new_library_with_source(include_str!("shaders/point.metal"), &compile_options)
            .map_err(|err| format!("point.metal compile failed: {err:?}"))?;

        let cull_library = device
            .new_library_with_source(include_str!("shaders/cull.metal"), &compile_options)
            .map_err(|err| format!("cull.metal compile failed: {err:?}"))?;

        let point_vs = point_library
            .get_function("point_vs", None)
            .map_err(|err| format!("point_vs not found: {err:?}"))?;
        let point_fs = point_library
            .get_function("point_fs", None)
            .map_err(|err| format!("point_fs not found: {err:?}"))?;

        let render_desc = metal::RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&point_vs));
        render_desc.set_fragment_function(Some(&point_fs));
        let color_attachment = render_desc
            .color_attachments()
            .object_at(0)
            .ok_or_else(|| "missing color attachment 0".to_string())?;
        color_attachment.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm_sRGB);

        let point_pipeline = device
            .new_render_pipeline_state(&render_desc)
            .map_err(|err| format!("point pipeline build failed: {err:?}"))?;

        let cull_fn = cull_library
            .get_function("cull_points", None)
            .map_err(|err| format!("cull_points not found: {err:?}"))?;
        let cull_pipeline = device
            .new_compute_pipeline_state_with_function(&cull_fn)
            .map_err(|err| format!("cull pipeline build failed: {err:?}"))?;

        let cull_finalize_fn = cull_library
            .get_function("finalize_cull", None)
            .map_err(|err| format!("finalize_cull not found: {err:?}"))?;
        let cull_finalize_pipeline = device
            .new_compute_pipeline_state_with_function(&cull_finalize_fn)
            .map_err(|err| format!("finalize cull pipeline build failed: {err:?}"))?;

        let shadow_point_limit = Self::configured_shadow_point_limit();
        let shadow_capacity = effective_point_count.max(1).min(shadow_point_limit);
        let position_buffer = Self::create_shared_buffer(
            &device,
            shadow_capacity.saturating_mul(std::mem::size_of::<MetalPositionData>()),
            "metal-shadow-positions",
        );
        let intensity_buffer = Self::create_shared_buffer(
            &device,
            shadow_capacity.saturating_mul(std::mem::size_of::<f32>()),
            "metal-shadow-intensities",
        );
        let mut visible_indices_buffers = Vec::with_capacity(Self::SHARED_RING_SLOTS);
        let mut cull_counter_buffers = Vec::with_capacity(Self::SHARED_RING_SLOTS);
        let mut draw_indirect_buffers = Vec::with_capacity(Self::SHARED_RING_SLOTS);
        for slot in 0..Self::SHARED_RING_SLOTS {
            visible_indices_buffers.push(Self::create_shared_buffer(
                &device,
                shadow_capacity.saturating_mul(std::mem::size_of::<u32>()),
                &format!("metal-visible-indices-slot-{slot}"),
            ));
            cull_counter_buffers.push(Self::create_shared_buffer(
                &device,
                std::mem::size_of::<MetalCullCounter>(),
                &format!("metal-cull-counter-slot-{slot}"),
            ));
            draw_indirect_buffers.push(Self::create_shared_buffer(
                &device,
                std::mem::size_of::<MetalDrawPrimitivesIndirectArgs>(),
                &format!("metal-draw-indirect-slot-{slot}"),
            ));
        }
        let cull_params_buffer = Self::create_shared_buffer(
            &device,
            std::mem::size_of::<MetalCullParams>(),
            "metal-cull-params",
        );
        let camera_buffer = Self::create_shared_buffer(
            &device,
            std::mem::size_of::<CameraUniform>(),
            "metal-camera-uniform",
        );
        let color_buffer = Self::create_shared_buffer(
            &device,
            std::mem::size_of::<MetalColorUniform>(),
            "metal-color-uniform",
        );

        let ring_slot_bytes = shadow_capacity.saturating_mul(std::mem::size_of::<f32>());
        let shared_ring = SharedBufferRing::new(
            &device,
            Self::SHARED_RING_SLOTS,
            ring_slot_bytes,
            "metal-shared-ring",
        )?;

        Ok(Self {
            device,
            command_queue,
            point_library,
            cull_library,
            point_pipeline,
            cull_pipeline,
            cull_finalize_pipeline,
            shared_ring,
            position_buffer,
            intensity_buffer,
            visible_indices_buffers,
            cull_params_buffer,
            cull_counter_buffers,
            draw_indirect_buffers,
            cull_slot_input_points: vec![0; Self::SHARED_RING_SLOTS],
            cull_slot_truncated_points: vec![0; Self::SHARED_RING_SLOTS],
            point_capacity: shadow_capacity,
            shadow_point_limit,
            camera_buffer,
            color_buffer,
            surface_width,
            surface_height,
            staging_positions: Vec::with_capacity(shadow_capacity),
            staging_intensities: Vec::with_capacity(shadow_capacity),
            positions_dirty: true,
            uploaded_positions: 0,
            last_shadow_stats: ShadowFrameStats::default(),
        })
    }

    pub fn resize_surface(&mut self, width: u32, height: u32) {
        self.surface_width = width.max(1);
        self.surface_height = height.max(1);
    }

    pub fn reserve_for_point_count(&mut self, effective_point_count: usize) {
        let target = effective_point_count.max(1).min(self.shadow_point_limit);
        self.ensure_point_capacity(target);
        let ring_bytes = target.saturating_mul(std::mem::size_of::<f32>());
        self.shared_ring.ensure_capacity(&self.device, ring_bytes);
    }

    pub fn mark_positions_dirty(&mut self) {
        self.positions_dirty = true;
        self.uploaded_positions = 0;
    }

    pub fn render_presented_frame(
        &mut self,
        frame: PresentedFrameInput<'_>,
    ) -> Result<ShadowFrameStats, String> {
        let PresentedFrameInput {
            surface,
            ui_painter,
            ui_frame,
            frame_id,
            positions,
            intensities,
            external_intensity_buffer,
            camera_uniform,
            color_params,
        } = frame;

        let available_points = if external_intensity_buffer.is_some() {
            positions.len()
        } else {
            positions.len().min(intensities.len())
        };
        if available_points == 0 {
            self.last_shadow_stats = ShadowFrameStats::default();
            return Ok(self.last_shadow_stats);
        }

        let encoded_points = available_points.min(self.shadow_point_limit);
        let truncated_points = available_points.saturating_sub(encoded_points);
        self.reserve_for_point_count(encoded_points);
        let render_slot = frame_id as usize % Self::SHARED_RING_SLOTS;
        self.cull_slot_input_points[render_slot] = encoded_points;
        self.cull_slot_truncated_points[render_slot] = truncated_points;

        let mut stats_encoded_points = encoded_points;
        let mut stats_truncated_points = truncated_points;
        let mut visible_points = encoded_points;
        if frame_id >= 2 {
            let stats_slot =
                (frame_id as usize + Self::SHARED_RING_SLOTS - 2) % Self::SHARED_RING_SLOTS;
            let stats_input_points = self.cull_slot_input_points[stats_slot];
            if stats_input_points > 0 {
                let args: MetalDrawPrimitivesIndirectArgs =
                    Self::read_pod(self.draw_indirect_buffers[stats_slot].as_ref())?;
                stats_encoded_points = stats_input_points;
                stats_truncated_points = self.cull_slot_truncated_points[stats_slot];
                visible_points = (args.vertex_count as usize).min(stats_input_points);
            }
        }

        if self.positions_dirty || self.uploaded_positions < encoded_points {
            if self.staging_positions.len() != encoded_points {
                self.staging_positions
                    .resize(encoded_points, pod::zeroed::<MetalPositionData>());
            }

            for (i, position) in positions.iter().copied().enumerate().take(encoded_points) {
                self.staging_positions[i] = MetalPositionData {
                    position: [position.x, position.y, position.z],
                };
            }

            Self::write_pod_slice(self.position_buffer.as_ref(), &self.staging_positions)?;
            self.positions_dirty = false;
            self.uploaded_positions = encoded_points;
        }

        let intensity_source: &metal::BufferRef =
            if let Some(shared_buffer) = external_intensity_buffer {
                shared_buffer
            } else {
                if self.staging_intensities.len() != encoded_points {
                    self.staging_intensities.resize(encoded_points, 0.0);
                }
                self.staging_intensities
                    .copy_from_slice(&intensities[..encoded_points]);
                Self::write_pod_slice(self.intensity_buffer.as_ref(), &self.staging_intensities)?;
                self.intensity_buffer.as_ref()
            };
        let required_intensity_bytes = encoded_points.saturating_mul(std::mem::size_of::<f32>());
        if (intensity_source.length() as usize) < required_intensity_bytes {
            return Err(format!(
                "intensity buffer too small: {} bytes < required {} bytes",
                intensity_source.length(),
                required_intensity_bytes
            ));
        }

        Self::write_pod(self.camera_buffer.as_ref(), &camera_uniform)?;

        let color_uniform = MetalColorUniform {
            gamma: color_params.gamma,
            contrast: color_params.contrast,
            brightness: color_params.brightness,
            use_log_scale: u32::from(color_params.use_log_scale),
            luminance_threshold: color_params.luminance_threshold.clamp(0.0, 1.0),
            color_scheme: Self::color_scheme_code(color_params.color_scheme),
            exposure_black: color_params.exposure_black.max(0.0),
            exposure_white: color_params
                .exposure_white
                .max(color_params.exposure_black + 1e-6),
        };
        Self::write_pod(self.color_buffer.as_ref(), &color_uniform)?;

        let cull_params = MetalCullParams {
            point_count: encoded_points as u32,
            padding: [0; 3],
        };
        Self::write_pod(self.cull_params_buffer.as_ref(), &cull_params)?;

        let visible_indices_buffer = self.visible_indices_buffers[render_slot].as_ref();
        let cull_counter_buffer = self.cull_counter_buffers[render_slot].as_ref();
        let draw_indirect_buffer = self.draw_indirect_buffers[render_slot].as_ref();

        let cull_counter = MetalCullCounter {
            visible_count: 0,
            padding: [0; 3],
        };
        Self::write_pod(cull_counter_buffer, &cull_counter)?;

        let draw_indirect = MetalDrawPrimitivesIndirectArgs {
            vertex_count: 0,
            instance_count: 1,
            vertex_start: 0,
            base_instance: 0,
        };
        Self::write_pod(draw_indirect_buffer, &draw_indirect)?;

        let command_buffer = self.command_queue.new_command_buffer();
        let drawable = surface
            .next_drawable()
            .ok_or_else(|| "metal layer returned no drawable".to_string())?;

        {
            let cull_encoder = command_buffer.new_compute_command_encoder();
            cull_encoder.set_compute_pipeline_state(&self.cull_pipeline);
            cull_encoder.set_buffer(0, Some(self.position_buffer.as_ref()), 0);
            cull_encoder.set_buffer(1, Some(intensity_source), 0);
            cull_encoder.set_buffer(2, Some(self.camera_buffer.as_ref()), 0);
            cull_encoder.set_buffer(3, Some(self.color_buffer.as_ref()), 0);
            cull_encoder.set_buffer(4, Some(self.cull_params_buffer.as_ref()), 0);
            cull_encoder.set_buffer(5, Some(visible_indices_buffer), 0);
            cull_encoder.set_buffer(6, Some(cull_counter_buffer), 0);

            let tg_width = self.cull_pipeline.thread_execution_width().clamp(1, 64);
            let thread_groups = metal::MTLSize {
                width: (encoded_points as u64).div_ceil(tg_width),
                height: 1,
                depth: 1,
            };
            let threads_per_group = metal::MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            };
            cull_encoder.dispatch_thread_groups(thread_groups, threads_per_group);
            cull_encoder.end_encoding();
        }

        {
            let finalize_encoder = command_buffer.new_compute_command_encoder();
            finalize_encoder.set_compute_pipeline_state(&self.cull_finalize_pipeline);
            finalize_encoder.set_buffer(0, Some(cull_counter_buffer), 0);
            finalize_encoder.set_buffer(1, Some(draw_indirect_buffer), 0);
            let single = metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            finalize_encoder.dispatch_thread_groups(single, single);
            finalize_encoder.end_encoding();
        }

        {
            let render_pass = metal::RenderPassDescriptor::new();
            let color_attachment = render_pass
                .color_attachments()
                .object_at(0)
                .ok_or_else(|| "missing render color attachment".to_string())?;
            color_attachment.set_texture(Some(drawable.texture()));
            color_attachment.set_load_action(metal::MTLLoadAction::Clear);
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
            color_attachment.set_clear_color(metal::MTLClearColor::new(0.0, 0.0, 0.0, 1.0));

            let render_encoder = command_buffer.new_render_command_encoder(render_pass);
            render_encoder.set_render_pipeline_state(&self.point_pipeline);
            render_encoder.set_vertex_buffer(0, Some(self.position_buffer.as_ref()), 0);
            render_encoder.set_vertex_buffer(1, Some(intensity_source), 0);
            render_encoder.set_vertex_buffer(2, Some(self.camera_buffer.as_ref()), 0);
            render_encoder.set_vertex_buffer(3, Some(self.color_buffer.as_ref()), 0);
            render_encoder.set_vertex_buffer(4, Some(visible_indices_buffer), 0);
            render_encoder.set_fragment_buffer(0, Some(self.color_buffer.as_ref()), 0);
            render_encoder.draw_primitives_indirect(
                metal::MTLPrimitiveType::Point,
                draw_indirect_buffer,
                0,
            );
            render_encoder.end_encoding();
        }

        ui_painter.paint(
            self.device.as_ref(),
            command_buffer,
            drawable.texture(),
            ui_frame,
        )?;

        {
            let ring_slot = self.shared_ring.acquire(frame_id);
            let blit_encoder = command_buffer.new_blit_command_encoder();
            let copy_bytes = encoded_points.saturating_mul(std::mem::size_of::<f32>());
            blit_encoder.copy_from_buffer(intensity_source, 0, ring_slot, 0, copy_bytes as u64);
            blit_encoder.end_encoding();
        }

        command_buffer.present_drawable(drawable);
        command_buffer.commit();

        let gpu_culled_points = stats_encoded_points.saturating_sub(visible_points);
        let gpu_culling_percentage = if stats_encoded_points > 0 {
            100.0 * gpu_culled_points as f32 / stats_encoded_points as f32
        } else {
            0.0
        };
        self.last_shadow_stats = ShadowFrameStats {
            encoded_points: stats_encoded_points,
            truncated_points: stats_truncated_points,
            visible_points,
            gpu_culled_points,
            gpu_culling_percentage,
        };
        Ok(self.last_shadow_stats)
    }

    pub fn bootstrap_info(&self) -> MetalBootstrapInfo {
        MetalBootstrapInfo {
            device_name: self.device.name().to_string(),
            ring_slots: self.shared_ring.slot_count(),
            ring_slot_bytes: self.shared_ring.slot_bytes(),
            shadow_point_capacity: self.point_capacity,
            shadow_point_limit: self.shadow_point_limit,
            surface_width: self.surface_width,
            surface_height: self.surface_height,
        }
    }

    pub fn shadow_point_limit(&self) -> usize {
        self.shadow_point_limit
    }

    pub fn has_native_pipeline_state(&self) -> bool {
        let _ = &self.command_queue;
        let _ = &self.point_library;
        let _ = &self.cull_library;
        let _ = &self.point_pipeline;
        let _ = &self.cull_pipeline;
        let _ = &self.cull_finalize_pipeline;
        true
    }

    pub fn device_ref(&self) -> &metal::DeviceRef {
        self.device.as_ref()
    }

    fn ensure_point_capacity(&mut self, target_points: usize) {
        if target_points <= self.point_capacity {
            return;
        }

        self.point_capacity = target_points;
        self.position_buffer = Self::create_shared_buffer(
            &self.device,
            self.point_capacity
                .saturating_mul(std::mem::size_of::<MetalPositionData>()),
            "metal-shadow-positions",
        );
        self.intensity_buffer = Self::create_shared_buffer(
            &self.device,
            self.point_capacity
                .saturating_mul(std::mem::size_of::<f32>()),
            "metal-shadow-intensities",
        );
        for slot in 0..Self::SHARED_RING_SLOTS {
            self.visible_indices_buffers[slot] = Self::create_shared_buffer(
                &self.device,
                self.point_capacity
                    .saturating_mul(std::mem::size_of::<u32>()),
                &format!("metal-visible-indices-slot-{slot}"),
            );
            self.cull_counter_buffers[slot] = Self::create_shared_buffer(
                &self.device,
                std::mem::size_of::<MetalCullCounter>(),
                &format!("metal-cull-counter-slot-{slot}"),
            );
            self.draw_indirect_buffers[slot] = Self::create_shared_buffer(
                &self.device,
                std::mem::size_of::<MetalDrawPrimitivesIndirectArgs>(),
                &format!("metal-draw-indirect-slot-{slot}"),
            );
            self.cull_slot_input_points[slot] = 0;
            self.cull_slot_truncated_points[slot] = 0;
        }
        self.positions_dirty = true;
        self.uploaded_positions = 0;
    }

    fn create_shared_buffer(device: &metal::Device, bytes: usize, label: &str) -> metal::Buffer {
        let buffer = device.new_buffer(
            bytes.max(256) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        buffer.set_label(label);
        buffer
    }

    fn configured_shadow_point_limit() -> usize {
        static LIMIT: LazyLock<usize> = LazyLock::new(|| {
            MetalContext::resolve_shadow_point_limit(
                std::env::var(MetalContext::SHADOW_POINT_LIMIT_ENV)
                    .ok()
                    .as_deref(),
            )
        });
        *LIMIT
    }

    fn resolve_shadow_point_limit(env_value: Option<&str>) -> usize {
        let configured = env_value.and_then(|raw| raw.trim().parse::<usize>().ok());
        configured
            .unwrap_or(Self::DEFAULT_SHADOW_POINTS)
            .clamp(Self::MIN_SHADOW_POINTS, Self::MAX_SHADOW_POINTS)
    }

    fn write_pod<T: Pod>(buffer: &metal::BufferRef, data: &T) -> Result<(), String> {
        Self::write_pod_slice(buffer, std::slice::from_ref(data))
    }

    fn read_pod<T: Pod>(buffer: &metal::BufferRef) -> Result<T, String> {
        let byte_len = std::mem::size_of::<T>();
        if byte_len > buffer.length() as usize {
            return Err(format!(
                "buffer read overflow: {} bytes > {} bytes",
                byte_len,
                buffer.length()
            ));
        }
        unsafe {
            let bytes = std::slice::from_raw_parts(buffer.contents() as *const u8, byte_len);
            Ok(pod::read_unaligned(bytes))
        }
    }

    fn write_pod_slice<T: Pod>(buffer: &metal::BufferRef, data: &[T]) -> Result<(), String> {
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
}

#[cfg(not(target_os = "macos"))]
pub struct MetalContext;

#[cfg(not(target_os = "macos"))]
impl MetalContext {
    pub const MAX_SHADOW_POINTS: usize = 0;

    pub fn new(
        _surface_width: u32,
        _surface_height: u32,
        _effective_point_count: usize,
    ) -> Result<Self, String> {
        Err("Native Metal context is only available on macOS".to_string())
    }

    pub fn resize_surface(&mut self, _width: u32, _height: u32) {}

    pub fn reserve_for_point_count(&mut self, _effective_point_count: usize) {}

    pub fn render_presented_frame(
        &mut self,
        _frame: PresentedFrameInput<'_>,
    ) -> Result<ShadowFrameStats, String> {
        Ok(ShadowFrameStats::default())
    }

    pub fn bootstrap_info(&self) -> MetalBootstrapInfo {
        MetalBootstrapInfo {
            device_name: "unavailable".to_string(),
            ring_slots: 0,
            ring_slot_bytes: 0,
            shadow_point_capacity: 0,
            shadow_point_limit: 0,
            surface_width: 0,
            surface_height: 0,
        }
    }

    pub fn shadow_point_limit(&self) -> usize {
        0
    }

    pub fn has_native_pipeline_state(&self) -> bool {
        false
    }

    pub fn device_ref(&self) -> &() {
        &()
    }
}
