// Metal renderer bridge.
//
// This module owns the application-facing renderer entrypoint and pins
// presentation to native Metal. Scheduling/state management stays in
// `crate::render::Renderer`.

mod context;
mod metal_painter;
mod ring_buffer;
mod surface;

use crate::math::WavefunctionParams;
use crate::render::color::ColorScheme;
use crate::render::{AutoToneOutput, BufferPoolStats, PointCloudDebugInfo};
use crate::ui;
use crate::{debug, info, warn};
use winit::event::WindowEvent;
use winit::window::Window;

#[derive(Debug)]
pub enum RenderError {
    NativeUnavailable,
    NativeFailure(String),
}

pub struct MetalRenderer {
    inner: crate::render::Renderer,
    ui_painter: metal_painter::MetalPainter,
    native_context: Option<context::MetalContext>,
    native_surface: Option<surface::MetalSurface>,
    native_positions: Vec<glam::Vec3>,
    native_color_params: context::MetalColorParams,
    frame_id: u64,
    last_truncated_points: usize,
    last_shadow_stats: context::ShadowFrameStats,
    auto_tone_enabled: bool,
}

fn slice_op_code(op: ui::SliceOperation) -> u32 {
    match op {
        ui::SliceOperation::Subtractive => 0,
        ui::SliceOperation::Additive => 1,
    }
}

impl MetalRenderer {
    pub fn new(window: &Window, point_count: usize) -> Result<Self, String> {
        info!("Creating Metal renderer bridge");
        let mut inner = crate::render::Renderer::new(window, point_count)?;
        let mut native_positions = inner.point_positions_snapshot();
        let size = window.inner_size();
        let mut native_context = match context::MetalContext::new(
            size.width,
            size.height,
            native_positions.len(),
        ) {
            Ok(ctx) => {
                let bootstrap = ctx.bootstrap_info();
                info!(
                    "Native Metal bootstrap ready: device='{}' ring_slots={} ring_slot_bytes={} shadow_capacity={} shadow_limit={} surface={}x{} pipeline_ready={}",
                    bootstrap.device_name,
                    bootstrap.ring_slots,
                    bootstrap.ring_slot_bytes,
                    bootstrap.shadow_point_capacity,
                    bootstrap.shadow_point_limit,
                    bootstrap.surface_width,
                    bootstrap.surface_height,
                    ctx.has_native_pipeline_state()
                );
                Some(ctx)
            }
            Err(err) => {
                warn!("Native Metal bootstrap unavailable: {}", err);
                None
            }
        };
        if let Some(ctx) = native_context.as_mut() {
            inner.set_runtime_effective_cap_override(Some(ctx.shadow_point_limit()));
            native_positions = inner.point_positions_snapshot();
            ctx.mark_positions_dirty();
            ctx.reserve_for_point_count(native_positions.len());
        }
        let native_surface = if let Some(ctx) = native_context.as_ref() {
            match surface::MetalSurface::new(window, ctx.device_ref()) {
                Ok(surface) => Some(surface),
                Err(err) => {
                    warn!("Native Metal surface unavailable: {}", err);
                    None
                }
            }
        } else {
            None
        };
        Ok(Self {
            inner,
            ui_painter: metal_painter::MetalPainter::new(),
            native_context,
            native_surface,
            native_positions,
            native_color_params: context::MetalColorParams::default(),
            frame_id: 0,
            last_truncated_points: 0,
            last_shadow_stats: context::ShadowFrameStats::default(),
            auto_tone_enabled: true,
        })
    }

    pub fn handle_camera_input(&mut self, event: &WindowEvent) -> bool {
        self.inner.handle_camera_input(event)
    }

    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f64) {
        self.inner.resize(width, height);
        if let Some(native_context) = self.native_context.as_mut() {
            native_context.resize_surface(width, height);
        }
        if let Some(native_surface) = self.native_surface.as_ref() {
            native_surface.resize(width, height, scale_factor);
        }
    }

    pub fn update_points(
        &mut self,
        params: &WavefunctionParams,
        time: f64,
        delta_seconds: f32,
    ) -> PointCloudDebugInfo {
        let mut debug_info = self.inner.update_points(params, time, delta_seconds);
        debug_info.native_input_points = self.last_shadow_stats.encoded_points;
        debug_info.native_visible_points = self.last_shadow_stats.visible_points;
        debug_info.native_culled_points = self.last_shadow_stats.gpu_culled_points;
        debug_info.native_culling_percentage = self.last_shadow_stats.gpu_culling_percentage;
        debug_info.native_truncated_points = self.last_shadow_stats.truncated_points;
        debug_info
    }

    pub fn set_point_count(&mut self, count: usize) {
        self.inner.set_point_count(count);
        self.native_positions = self.inner.point_positions_snapshot();
        if let Some(native_context) = self.native_context.as_mut() {
            native_context.mark_positions_dirty();
            native_context.reserve_for_point_count(self.native_positions.len());
        }
    }

    pub fn get_point_count(&self) -> usize {
        self.inner.get_point_count()
    }

    pub fn update_camera_rotation(
        &mut self,
        auto_rotation: bool,
        rotation_speed: f32,
        delta_time: f32,
    ) {
        self.inner
            .update_camera_rotation(auto_rotation, rotation_speed, delta_time);
    }

    pub fn set_color_map_params(
        &mut self,
        gamma: f32,
        contrast: f32,
        brightness: f32,
        use_log_scale: bool,
    ) {
        let auto = self.inner.auto_tone_output();
        let (
            final_gamma,
            final_contrast,
            final_brightness,
            final_threshold,
            exposure_black,
            exposure_white,
        ) = if self.auto_tone_enabled {
            (
                auto.gamma,
                auto.contrast,
                auto.brightness,
                auto.luminance_threshold,
                auto.exposure_black,
                auto.exposure_white,
            )
        } else {
            (
                gamma,
                contrast,
                brightness,
                self.native_color_params.luminance_threshold,
                0.0,
                1.0,
            )
        };

        self.native_color_params = context::MetalColorParams {
            gamma: final_gamma,
            contrast: final_contrast,
            brightness: final_brightness,
            use_log_scale,
            color_scheme: self.native_color_params.color_scheme,
            luminance_threshold: final_threshold,
            exposure_black,
            exposure_white,
            ..self.native_color_params
        };
        self.inner
            .set_color_map_params(gamma, contrast, brightness, use_log_scale);
    }

    pub fn set_luminance_threshold(&mut self, luminance_threshold: f32) {
        if !self.auto_tone_enabled {
            self.native_color_params.luminance_threshold = luminance_threshold.clamp(0.0, 1.0);
        }
    }

    pub fn set_color_scheme(&mut self, scheme: ColorScheme) {
        self.native_color_params.color_scheme = scheme;
        self.inner.set_color_scheme(scheme);
    }

    pub fn set_slice_settings(&mut self, settings: ui::SliceSettings) {
        self.native_color_params.slice_thickness = settings.thickness.max(1e-4);
        self.native_color_params.slice_plane_offsets = [
            settings.x_plane.offset,
            settings.y_plane.offset,
            settings.z_plane.offset,
            0.0,
        ];
        self.native_color_params.slice_plane_enabled = [
            u32::from(settings.x_plane.enabled),
            u32::from(settings.y_plane.enabled),
            u32::from(settings.z_plane.enabled),
            0,
        ];
        self.native_color_params.slice_plane_ops = [
            slice_op_code(settings.x_plane.operation),
            slice_op_code(settings.y_plane.operation),
            slice_op_code(settings.z_plane.operation),
            0,
        ];
        self.native_color_params.slice_has_additive = u32::from(
            (settings.x_plane.enabled
                && settings.x_plane.operation == ui::SliceOperation::Additive)
                || (settings.y_plane.enabled
                    && settings.y_plane.operation == ui::SliceOperation::Additive)
                || (settings.z_plane.enabled
                    && settings.z_plane.operation == ui::SliceOperation::Additive),
        );
    }

    pub fn get_buffer_pool_stats(&self) -> BufferPoolStats {
        self.inner.get_buffer_pool_stats()
    }

    pub fn set_auto_tone_enabled(&mut self, enabled: bool) {
        self.auto_tone_enabled = enabled;
        self.inner.set_auto_tone_enabled(enabled);
    }

    pub fn auto_tone_output(&self) -> AutoToneOutput {
        self.inner.auto_tone_output()
    }

    pub fn render(&mut self, ui_state: &mut ui::State, window: &Window) -> Result<(), RenderError> {
        let prepared_ui = ui_state.prepare_frame(window);
        let native_context = self
            .native_context
            .as_mut()
            .ok_or(RenderError::NativeUnavailable)?;
        let native_surface = self
            .native_surface
            .as_ref()
            .ok_or(RenderError::NativeUnavailable)?;

        self.frame_id = self.frame_id.saturating_add(1);
        let camera_uniform = self.inner.camera_uniform_snapshot();
        let stats = native_context
            .render_presented_frame(context::PresentedFrameInput {
                surface: native_surface,
                ui_painter: &mut self.ui_painter,
                ui_frame: &prepared_ui,
                frame_id: self.frame_id,
                positions: &self.native_positions,
                intensities: self.inner.current_intensities(),
                external_intensity_buffer: self.inner.gpu_intensity_buffer_ref(),
                camera_uniform,
                color_params: self.native_color_params,
            })
            .map_err(RenderError::NativeFailure)?;

        if stats.truncated_points > 0 {
            if stats.truncated_points != self.last_truncated_points {
                warn!(
                    "Native Metal shadow pass truncated {} points (cap={})",
                    stats.truncated_points,
                    native_context.shadow_point_limit()
                );
            }
        } else {
            debug!(
                "Native Metal shadow pass encoded {} points",
                stats.encoded_points
            );
        }
        self.last_truncated_points = stats.truncated_points;
        self.last_shadow_stats = stats;
        Ok(())
    }
}
