// UI Module
//
// This module handles the user interface for the wavefunction visualization.
// It includes sliders for adjusting parameters and displays performance metrics.

mod controls;
mod input_bridge;

use crate::debug;
use crate::math::presets::{get_all_presets, get_preset_by_name};
use crate::math::WavefunctionParams;
use crate::render::color::ColorScheme;
use crate::render::BufferPoolStats;
use crate::render::PointCloudDebugInfo;
use egui::{ClippedPrimitive, ComboBox, Context, Slider, TexturesDelta};
use input_bridge::InputBridge;
use winit::window::Window;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SliceOperation {
    Additive,
    Subtractive,
}

#[derive(Debug, Copy, Clone)]
pub struct SlicePlane {
    pub enabled: bool,
    pub offset: f32,
    pub operation: SliceOperation,
}

impl Default for SlicePlane {
    fn default() -> Self {
        Self {
            enabled: false,
            offset: 0.0,
            operation: SliceOperation::Subtractive,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SliceSettings {
    pub thickness: f32,
    pub x_plane: SlicePlane,
    pub y_plane: SlicePlane,
    pub z_plane: SlicePlane,
}

impl Default for SliceSettings {
    fn default() -> Self {
        Self {
            thickness: 0.5,
            x_plane: SlicePlane::default(),
            y_plane: SlicePlane::default(),
            z_plane: SlicePlane::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AutoToneDiagnostics {
    pub exposure_black: f32,
    pub exposure_white: f32,
    pub luminance_threshold: f32,
}

fn format_point_count(count: usize) -> String {
    const K: f64 = 1_000.0;
    const M: f64 = 1_000_000.0;
    const B: f64 = 1_000_000_000.0;

    let value = count as f64;
    if value >= B {
        let scaled = value / B;
        if scaled >= 10.0 {
            format!("{scaled:.0}B")
        } else {
            format!("{scaled:.1}B")
        }
    } else if value >= M {
        let scaled = value / M;
        if scaled >= 10.0 {
            format!("{scaled:.0}M")
        } else {
            format!("{scaled:.1}M")
        }
    } else if value >= K {
        let scaled = value / K;
        if scaled >= 10.0 {
            format!("{scaled:.0}K")
        } else {
            format!("{scaled:.1}K")
        }
    } else {
        count.to_string()
    }
}

fn parse_point_count_input(input: &str) -> Option<usize> {
    const MIN_POINTS: f64 = 1_000.0;
    const MAX_POINTS: f64 = 1_000_000_000.0;

    let cleaned = input
        .trim()
        .to_ascii_lowercase()
        .replace([' ', ',', '_'], "");

    if cleaned.is_empty() {
        return None;
    }

    let (number_part, multiplier) = match cleaned.chars().last() {
        Some('k') => (&cleaned[..cleaned.len() - 1], 1_000.0),
        Some('m') => (&cleaned[..cleaned.len() - 1], 1_000_000.0),
        Some('b') => (&cleaned[..cleaned.len() - 1], 1_000_000_000.0),
        Some(_) => (cleaned.as_str(), 1.0),
        None => return None,
    };

    if number_part.is_empty() {
        return None;
    }

    let numeric_value = number_part.parse::<f64>().ok()?;
    if !numeric_value.is_finite() || numeric_value < 0.0 {
        return None;
    }

    let clamped = (numeric_value * multiplier)
        .round()
        .clamp(MIN_POINTS, MAX_POINTS);
    Some(clamped as usize)
}

// UI data structure for passing to UI builder
struct UiData {
    params: WavefunctionParams,
    color_scheme: ColorScheme,
    use_log_scale: bool,
    gamma: f32,
    contrast: f32,
    brightness: f32,
    luminance_threshold: f32,
    auto_tone_enabled: bool,
    auto_tone_diagnostics: AutoToneDiagnostics,
    point_count: usize,
    point_count_input: String,
    fps: f64,
    debug_info: Option<PointCloudDebugInfo>,

    // Camera controls
    auto_rotation: bool,
    rotation_speed: f32,

    // Presets
    current_preset: Option<String>,
    available_presets: Vec<String>,

    // Buffer pool statistics
    buffer_pool_stats: Option<BufferPoolStats>,

    // Slice controls
    slice_settings: SliceSettings,
}

// UI output structure
#[derive(Default)]
struct UiOutput {
    params: Option<WavefunctionParams>,
    color_scheme: Option<ColorScheme>,
    use_log_scale: Option<bool>,
    gamma: Option<f32>,
    contrast: Option<f32>,
    brightness: Option<f32>,
    luminance_threshold: Option<f32>,
    auto_tone_enabled: Option<bool>,
    point_count: Option<usize>,
    auto_rotation: Option<bool>,
    rotation_speed: Option<f32>,
    selected_preset: Option<String>,
    slice_settings: Option<SliceSettings>,
}

pub struct PreparedUiFrame {
    pub paint_jobs: Vec<ClippedPrimitive>,
    pub textures_delta: TexturesDelta,
    pub size_in_pixels: [u32; 2],
    pub pixels_per_point: f32,
}

// UI state
pub struct State {
    egui_ctx: Context,
    input_bridge: InputBridge,

    // Wavefunction parameters
    params: WavefunctionParams,

    // UI state
    color_scheme: ColorScheme,
    use_log_scale: bool,
    gamma: f32,
    contrast: f32,
    brightness: f32,
    luminance_threshold: f32,
    auto_tone_enabled: bool,
    auto_tone_diagnostics: AutoToneDiagnostics,
    point_count: usize,
    point_count_input: String,
    fps: f64,
    debug_info: Option<PointCloudDebugInfo>,

    // Camera controls
    auto_rotation: bool,
    rotation_speed: f32,

    // Presets
    current_preset: Option<String>,
    available_presets: Vec<String>,

    // Buffer pool statistics
    buffer_pool_stats: Option<BufferPoolStats>,

    // Slice controls
    slice_settings: SliceSettings,
}

impl State {
    // Create a new UI state
    pub fn new() -> Self {
        let egui_ctx = Context::default();

        // Get preset names
        let available_presets = get_all_presets()
            .into_iter()
            .map(|p| p.name.to_string())
            .collect();

        Self {
            egui_ctx,
            input_bridge: InputBridge::new(),

            params: WavefunctionParams::default(),

            color_scheme: ColorScheme::Quantum,
            use_log_scale: true,
            gamma: 1.0,
            contrast: 1.2,
            brightness: 1.0,
            luminance_threshold: 0.0,
            auto_tone_enabled: true,
            auto_tone_diagnostics: AutoToneDiagnostics::default(),
            point_count: 50000,
            point_count_input: format_point_count(50000),
            fps: 0.0,
            debug_info: None,

            auto_rotation: true,
            rotation_speed: 0.2,

            current_preset: None,
            available_presets,

            // Buffer pool statistics
            buffer_pool_stats: None,
            slice_settings: SliceSettings::default(),
        }
    }

    // Get the current wavefunction parameters
    pub fn get_parameters(&self) -> WavefunctionParams {
        self.params
    }

    // Get auto-rotation settings
    pub fn get_auto_rotation(&self) -> (bool, f32) {
        (self.auto_rotation, self.rotation_speed)
    }

    // Get color mapping parameters
    pub fn get_color_params(&self) -> (f32, f32, f32, bool, f32) {
        (
            self.gamma,
            self.contrast,
            self.brightness,
            self.use_log_scale,
            self.luminance_threshold,
        )
    }

    pub fn get_auto_tone_enabled(&self) -> bool {
        self.auto_tone_enabled
    }

    // Get color scheme
    pub fn get_color_scheme(&self) -> ColorScheme {
        self.color_scheme
    }

    // Get point count
    pub fn get_point_count(&self) -> usize {
        self.point_count
    }

    pub fn get_slice_settings(&self) -> SliceSettings {
        self.slice_settings
    }

    // Update the FPS counter
    pub fn update_fps(&mut self, fps: f64) {
        self.fps = fps;
    }

    // Set debug info
    pub fn set_debug_info(&mut self, debug_info: PointCloudDebugInfo) {
        self.debug_info = Some(debug_info);
    }

    pub fn set_auto_tone_diagnostics(&mut self, diagnostics: AutoToneDiagnostics) {
        self.auto_tone_diagnostics = diagnostics;
    }

    // Handle input events
    pub fn handle_input(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        let pixels_per_point = window.scale_factor() as f32;
        self.egui_ctx.set_pixels_per_point(pixels_per_point);
        self.input_bridge.on_window_event(
            event,
            pixels_per_point,
            self.egui_ctx.wants_pointer_input(),
            self.egui_ctx.wants_keyboard_input(),
        )
    }

    // Set buffer pool statistics
    pub fn set_buffer_pool_stats(&mut self, stats: BufferPoolStats) {
        self.buffer_pool_stats = Some(stats);
    }

    pub fn prepare_frame(&mut self, window: &Window) -> PreparedUiFrame {
        let size = window.inner_size();
        let scale_factor = window.scale_factor() as f32;

        let mut ui_data = UiData {
            params: self.params,
            color_scheme: self.color_scheme,
            use_log_scale: self.use_log_scale,
            gamma: self.gamma,
            contrast: self.contrast,
            brightness: self.brightness,
            luminance_threshold: self.luminance_threshold,
            auto_tone_enabled: self.auto_tone_enabled,
            auto_tone_diagnostics: self.auto_tone_diagnostics,
            point_count: self.point_count,
            point_count_input: self.point_count_input.clone(),
            fps: self.fps,
            debug_info: self.debug_info,
            auto_rotation: self.auto_rotation,
            rotation_speed: self.rotation_speed,
            current_preset: self.current_preset.clone(),
            available_presets: self.available_presets.clone(),
            buffer_pool_stats: self.buffer_pool_stats.clone(),
            slice_settings: self.slice_settings,
        };

        let mut ui_output = UiOutput::default();
        self.egui_ctx.set_pixels_per_point(scale_factor);
        let raw_input = self.input_bridge.take_raw_input(window);
        let output = self.egui_ctx.run(raw_input, |ctx| {
            ui_output = build_ui(ctx, &mut ui_data);
        });
        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            ..
        } = output;
        self.input_bridge
            .apply_platform_output(window, &platform_output);

        self.apply_ui_changes(ui_data, ui_output);

        PreparedUiFrame {
            paint_jobs: self.egui_ctx.tessellate(shapes),
            textures_delta,
            size_in_pixels: [size.width, size.height],
            pixels_per_point: scale_factor,
        }
    }

    fn apply_ui_changes(&mut self, ui_data: UiData, ui_output: UiOutput) {
        if let Some(params) = ui_output.params {
            self.params = params;
        }

        if let Some(color_scheme) = ui_output.color_scheme {
            self.color_scheme = color_scheme;
        }

        if let Some(use_log_scale) = ui_output.use_log_scale {
            self.use_log_scale = use_log_scale;
        }

        if let Some(gamma) = ui_output.gamma {
            self.gamma = gamma;
        }

        if let Some(contrast) = ui_output.contrast {
            self.contrast = contrast;
        }

        if let Some(brightness) = ui_output.brightness {
            self.brightness = brightness;
        }

        if let Some(luminance_threshold) = ui_output.luminance_threshold {
            self.luminance_threshold = luminance_threshold;
        }

        if let Some(auto_tone_enabled) = ui_output.auto_tone_enabled {
            self.auto_tone_enabled = auto_tone_enabled;
        }

        if let Some(point_count) = ui_output.point_count {
            self.point_count = point_count;
        }

        if let Some(auto_rotation) = ui_output.auto_rotation {
            self.auto_rotation = auto_rotation;
        }

        if let Some(rotation_speed) = ui_output.rotation_speed {
            self.rotation_speed = rotation_speed;
        }

        if let Some(slice_settings) = ui_output.slice_settings {
            self.slice_settings = slice_settings;
        }

        let selected_preset = ui_output.selected_preset.clone();
        if let Some(preset_name) = selected_preset {
            self.apply_preset(&preset_name);
        }

        if ui_output.params.is_none() && ui_output.selected_preset.is_none() {
            self.params = ui_data.params;
        }

        if ui_output.color_scheme.is_none() {
            self.color_scheme = ui_data.color_scheme;
        }

        if ui_output.use_log_scale.is_none() {
            self.use_log_scale = ui_data.use_log_scale;
        }

        if ui_output.gamma.is_none() {
            self.gamma = ui_data.gamma;
        }

        if ui_output.contrast.is_none() {
            self.contrast = ui_data.contrast;
        }

        if ui_output.brightness.is_none() {
            self.brightness = ui_data.brightness;
        }

        if ui_output.luminance_threshold.is_none() {
            self.luminance_threshold = ui_data.luminance_threshold;
        }

        if ui_output.auto_tone_enabled.is_none() {
            self.auto_tone_enabled = ui_data.auto_tone_enabled;
        }

        if ui_output.point_count.is_none() {
            self.point_count = ui_data.point_count;
        }

        self.point_count_input = ui_data.point_count_input;

        if ui_output.auto_rotation.is_none() {
            self.auto_rotation = ui_data.auto_rotation;
        }

        if ui_output.rotation_speed.is_none() {
            self.rotation_speed = ui_data.rotation_speed;
        }

        if ui_output.slice_settings.is_none() {
            self.slice_settings = ui_data.slice_settings;
        }

        self.current_preset = ui_data.current_preset;
    }

    // Add a method to apply a preset
    pub fn apply_preset(&mut self, preset_name: &str) {
        if let Some(preset) = get_preset_by_name(preset_name) {
            self.params = preset.params;
            self.current_preset = Some(preset_name.to_string());
            debug!("Applied preset: {}", preset_name);
        }
    }
}

// Build the UI
fn build_ui(ctx: &Context, data: &mut UiData) -> UiOutput {
    let mut output = UiOutput::default();

    egui::Window::new("Wavefunction Controls")
        .resizable(true)
        .default_width(420.0)
        .default_height(620.0)
        .vscroll(true)
        .show(ctx, |ui| {
            ui.heading("Quantum State");

            // Presets dropdown
            ui.horizontal(|ui| {
                ui.label("Preset:");
                let mut current_preset = data
                    .current_preset
                    .clone()
                    .unwrap_or_else(|| "Custom".to_string());

                if ComboBox::from_label("")
                    .selected_text(&current_preset)
                    .show_ui(ui, |ui| {
                        let mut selected = false;

                        // Add "Custom" option
                        if ui
                            .selectable_label(current_preset == "Custom", "Custom")
                            .clicked()
                        {
                            current_preset = "Custom".to_string();
                            selected = true;
                        }

                        // Add all presets
                        for preset_name in &data.available_presets {
                            if ui
                                .selectable_label(current_preset == *preset_name, preset_name)
                                .clicked()
                            {
                                current_preset = preset_name.clone();
                                selected = true;
                            }
                        }

                        selected
                    })
                    .inner
                    .unwrap_or(false)
                    && current_preset != "Custom"
                {
                    output.selected_preset = Some(current_preset.clone());
                }

                if current_preset != "Custom"
                    && data.current_preset.as_deref() != Some(&current_preset)
                {
                    data.current_preset = Some(current_preset);
                }
            });

            ui.add_space(10.0);

            ui.label("State 1");
            ui.horizontal(|ui| {
                let mut n = data.params.n;
                if ui
                    .add(Slider::new(&mut n, 1..=6).step_by(1.0).text("n"))
                    .changed()
                {
                    let mut new_params = data.params;
                    new_params.n = n;
                    new_params.l = new_params.l.min(new_params.n.saturating_sub(1));
                    new_params.m = new_params
                        .m
                        .clamp(-(new_params.l as isize), new_params.l as isize);
                    output.params = Some(new_params);
                    data.current_preset = Some("Custom".to_string());
                }

                let mut l = data.params.l;
                if ui
                    .add(
                        Slider::new(&mut l, 0..=data.params.n.saturating_sub(1))
                            .step_by(1.0)
                            .text("l"),
                    )
                    .changed()
                {
                    let mut new_params = data.params;
                    new_params.l = l;
                    new_params.m = new_params.m.clamp(-(l as isize), l as isize);
                    output.params = Some(new_params);
                    data.current_preset = Some("Custom".to_string());
                }

                let mut m = data.params.m;
                if ui
                    .add(
                        Slider::new(&mut m, -(data.params.l as isize)..=data.params.l as isize)
                            .step_by(1.0)
                            .text("m"),
                    )
                    .changed()
                {
                    let mut new_params = data.params;
                    new_params.m = m;
                    output.params = Some(new_params);
                    data.current_preset = Some("Custom".to_string());
                }
            });

            ui.label("State 2");
            ui.horizontal(|ui| {
                let mut n2 = data.params.n2;
                if ui
                    .add(Slider::new(&mut n2, 1..=6).step_by(1.0).text("n2"))
                    .changed()
                {
                    let mut new_params = data.params;
                    new_params.n2 = n2;
                    new_params.l2 = new_params.l2.min(new_params.n2.saturating_sub(1));
                    new_params.m2 = new_params
                        .m2
                        .clamp(-(new_params.l2 as isize), new_params.l2 as isize);
                    output.params = Some(new_params);
                    data.current_preset = Some("Custom".to_string());
                }

                let mut l2 = data.params.l2;
                if ui
                    .add(
                        Slider::new(&mut l2, 0..=data.params.n2.saturating_sub(1))
                            .step_by(1.0)
                            .text("l2"),
                    )
                    .changed()
                {
                    let mut new_params = data.params;
                    new_params.l2 = l2;
                    new_params.m2 = new_params.m2.clamp(-(l2 as isize), l2 as isize);
                    output.params = Some(new_params);
                    data.current_preset = Some("Custom".to_string());
                }

                let mut m2 = data.params.m2;
                if ui
                    .add(
                        Slider::new(
                            &mut m2,
                            -(data.params.l2 as isize)..=data.params.l2 as isize,
                        )
                        .step_by(1.0)
                        .text("m2"),
                    )
                    .changed()
                {
                    let mut new_params = data.params;
                    new_params.m2 = m2;
                    output.params = Some(new_params);
                    data.current_preset = Some("Custom".to_string());
                }
            });

            ui.separator();
            ui.heading("Physical Controls");

            let mut mix = data.params.mix;
            if ui
                .add(Slider::new(&mut mix, 0.0..=1.0).text("State-2 Population"))
                .changed()
            {
                let mut new_params = data.params;
                new_params.mix = mix;
                output.params = Some(new_params);
                data.current_preset = Some("Custom".to_string());
            }

            let mut relative_phase = data.params.relative_phase;
            if ui
                .add(
                    Slider::new(
                        &mut relative_phase,
                        -std::f64::consts::PI..=std::f64::consts::PI,
                    )
                    .text("Relative Phase (rad)"),
                )
                .changed()
            {
                let mut new_params = data.params;
                new_params.relative_phase = relative_phase;
                output.params = Some(new_params);
                data.current_preset = Some("Custom".to_string());
            }

            let mut z = data.params.z;
            if ui
                .add(
                    Slider::new(&mut z, 1.0..=6.0)
                        .step_by(1.0)
                        .text("Nuclear Charge Z"),
                )
                .changed()
            {
                let mut new_params = data.params;
                new_params.z = z.round().clamp(1.0, 6.0);
                output.params = Some(new_params);
                data.current_preset = Some("Custom".to_string());
            }

            let mut time_factor = data.params.time_factor;
            if ui
                .add(Slider::new(&mut time_factor, 0.0..=250.0).text("Time Scale (fs/s)"))
                .changed()
            {
                let mut new_params = data.params;
                new_params.time_factor = time_factor;
                output.params = Some(new_params);
                data.current_preset = Some("Custom".to_string());
            }

            ui.separator();
            ui.heading("Camera Controls");

            // Auto-rotation toggle
            ui.checkbox(&mut data.auto_rotation, "Auto-Rotate Camera");

            // Rotation speed (only show if auto-rotation is enabled)
            if data.auto_rotation {
                ui.add(Slider::new(&mut data.rotation_speed, 0.05..=1.0).text("Rotation Speed"));
            }

            ui.separator();
            ui.heading("Color Mapping");
            ui.checkbox(&mut data.auto_tone_enabled, "Auto Tone & Threshold");
            output.auto_tone_enabled = Some(data.auto_tone_enabled);

            // Color scheme selection
            ui.horizontal(|ui| {
                ui.label("Color Scheme:");
                ComboBox::from_id_source("color_scheme")
                    .selected_text(format!("{:?}", data.color_scheme))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut data.color_scheme,
                            ColorScheme::Viridis,
                            "Viridis",
                        );
                        ui.selectable_value(&mut data.color_scheme, ColorScheme::Plasma, "Plasma");
                        ui.selectable_value(
                            &mut data.color_scheme,
                            ColorScheme::Inferno,
                            "Inferno",
                        );
                        ui.selectable_value(
                            &mut data.color_scheme,
                            ColorScheme::BlueRed,
                            "BlueRed",
                        );
                        ui.selectable_value(
                            &mut data.color_scheme,
                            ColorScheme::Quantum,
                            "Quantum",
                        );
                    });
            });

            // Log scale
            ui.checkbox(&mut data.use_log_scale, "Logarithmic Intensity");

            ui.add_enabled_ui(!data.auto_tone_enabled, |ui| {
                ui.add(Slider::new(&mut data.gamma, 0.1..=3.0).text("Gamma"));
                ui.add(Slider::new(&mut data.contrast, 0.1..=5.0).text("Contrast"));
                ui.add(Slider::new(&mut data.brightness, 0.1..=3.0).text("Brightness"));
                ui.add(
                    Slider::new(&mut data.luminance_threshold, 0.0..=1.0)
                        .text("Luminance Threshold"),
                );
            });

            if data.auto_tone_enabled {
                ui.label(format!(
                    "Auto Exposure Black: {:.6}",
                    data.auto_tone_diagnostics.exposure_black
                ));
                ui.label(format!(
                    "Auto Exposure White: {:.6}",
                    data.auto_tone_diagnostics.exposure_white
                ));
                ui.label(format!(
                    "Auto Threshold: {:.6}",
                    data.auto_tone_diagnostics.luminance_threshold
                ));
            }

            ui.separator();
            ui.heading("Slicing");
            ui.label("Combine X/Y/Z plane slices using additive or subtractive modes.");
            let thickness_changed = ui
                .add(
                    Slider::new(&mut data.slice_settings.thickness, 0.05..=10.0)
                        .text("Slice Thickness"),
                )
                .changed();

            let mut slice_changed = false;
            slice_changed |= slice_plane_row(ui, "X Plane", &mut data.slice_settings.x_plane);
            slice_changed |= slice_plane_row(ui, "Y Plane", &mut data.slice_settings.y_plane);
            slice_changed |= slice_plane_row(ui, "Z Plane", &mut data.slice_settings.z_plane);
            if thickness_changed || slice_changed {
                output.slice_settings = Some(data.slice_settings);
            }

            ui.separator();
            ui.heading("Runtime");
            ui.label(format!("FPS: {:.1}", data.fps));
            if let Some(debug_info) = &data.debug_info {
                ui.label(format!(
                    "Requested Point Count: {}",
                    format_point_count(debug_info.point_count)
                ));
            } else {
                ui.label(format!(
                    "Point Count: {}",
                    format_point_count(data.point_count)
                ));
            }
            if let Some(debug_info) = &data.debug_info {
                ui.label(format!(
                    "Effective Render Count: {}",
                    format_point_count(debug_info.effective_point_count)
                ));
            }
            ui.label("Compute Scheduler: Automatic (CPU+GPU cooperative)");

            if let Some(debug_info) = &data.debug_info {
                ui.label(format!("Avg Intensity: {:.6}", debug_info.avg_intensity));
                ui.label(format!(
                    "Intensity Range: {:.6} .. {:.6}",
                    debug_info.min_intensity, debug_info.max_intensity
                ));
                ui.label(format!(
                    "Zero Intensity Points: {}",
                    debug_info.zero_intensity_count
                ));
                ui.label(format!("Sampling Radius: {:.2}", debug_info.max_radius));
                ui.label(format!(
                    "Points Initialized: {}",
                    debug_info.points_initialized
                ));
                ui.label(format!(
                    "Update Cache Hits / Recomputes: {} / {}",
                    debug_info.cached_updates_count, debug_info.recalculations_count
                ));
                ui.label(format!(
                    "Culled Points: {} ({:.1}%)",
                    debug_info.culled_points_count, debug_info.culling_percentage
                ));
                ui.label(format!(
                    "Native GPU Visible/Culled: {} / {} ({:.1}%)",
                    format_point_count(debug_info.native_visible_points),
                    format_point_count(debug_info.native_culled_points),
                    debug_info.native_culling_percentage
                ));
                ui.label(format!(
                    "Native GPU Input/Truncated: {} / {}",
                    format_point_count(debug_info.native_input_points),
                    format_point_count(debug_info.native_truncated_points)
                ));
                ui.label(format!(
                    "Frame/GPU/CPU ms: {:.2} / {:.2} / {:.2}",
                    debug_info.frame_ms, debug_info.gpu_ms, debug_info.cpu_ms
                ));
                ui.label(format!(
                    "Scheduler Block Size: {}",
                    format_point_count(debug_info.scheduler_block_size)
                ));
                ui.label(format!(
                    "Scheduler Quality: {:.0}%",
                    debug_info.scheduler_quality * 100.0
                ));
                ui.label(format!(
                    "GPU Queue Depth / Stale Frames: {} / {}",
                    debug_info.queue_depth, debug_info.stale_frames
                ));
                ui.label(format!(
                    "Approx Mode: {}",
                    if debug_info.approx_mode { "ON" } else { "OFF" }
                ));
                ui.label(format!(
                    "CPU Worker Utilization: {:.0}%",
                    debug_info.cpu_worker_utilization * 100.0
                ));
            }

            if let Some(stats) = &data.buffer_pool_stats {
                ui.separator();
                ui.heading("Buffer Pool");
                ui.label(format!("Pooled Buffers: {}", stats.total_buffers));
                ui.label(format!(
                    "Pooled Memory: {:.2} MB",
                    stats.total_memory as f64 / (1024.0 * 1024.0)
                ));
                ui.label(format!(
                    "Created / Reused: {} / {}",
                    stats.created_count, stats.reused_count
                ));
                if let Some(top_usage) = stats.usage_stats.iter().max_by_key(|entry| entry.memory) {
                    ui.label(format!(
                        "Largest Pool: {:?} ({} buffers, {:.2} MB)",
                        top_usage.usage,
                        top_usage.count,
                        top_usage.memory as f64 / (1024.0 * 1024.0)
                    ));
                }
            }

            ui.separator();
            ui.heading("Point Density");
            ui.horizontal(|ui| {
                ui.label("Manual:");
                let response = ui.add(
                    egui::TextEdit::singleline(&mut data.point_count_input)
                        .hint_text("e.g. 20M or 1.4K")
                        .desired_width(140.0),
                );

                if let Some(parsed) = parse_point_count_input(&data.point_count_input) {
                    if response.changed() {
                        data.point_count = parsed;
                        output.point_count = Some(parsed);
                    }
                    ui.label(format!("= {}", format_point_count(parsed)));
                } else if response.changed() {
                    ui.label("Invalid format");
                }
            });

            let mut point_count = data.point_count;
            let point_count_response = ui.add(
                Slider::new(&mut point_count, 1_000..=1_000_000_000)
                    .logarithmic(true)
                    .text("Points")
                    .custom_formatter(|value, _| format_point_count(value.round() as usize)),
            );
            if point_count_response.changed() {
                data.point_count_input = format_point_count(point_count);
                output.point_count = Some(point_count);
            }

            // Add quantum state info
            ui.separator();
            ui.heading("Quantum State Info");
            controls::quantum_state_info(ui, data.params.n, data.params.l, data.params.m);

            // Add performance tips if needed
            controls::performance_tips(ui, data.point_count, data.fps);
        });

    output
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

fn slice_plane_row(ui: &mut egui::Ui, label: &str, plane: &mut SlicePlane) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.label(label);
            if ui.checkbox(&mut plane.enabled, "Enabled").changed() {
                changed = true;
            }
        });

        ui.add_enabled_ui(plane.enabled, |ui| {
            if ui
                .add(Slider::new(&mut plane.offset, -5.0..=5.0).text("Offset"))
                .changed()
            {
                changed = true;
            }

            ui.horizontal(|ui| {
                ui.label("Mode:");
                if ui
                    .selectable_value(&mut plane.operation, SliceOperation::Additive, "Add")
                    .changed()
                {
                    changed = true;
                }
                if ui
                    .selectable_value(
                        &mut plane.operation,
                        SliceOperation::Subtractive,
                        "Subtract",
                    )
                    .changed()
                {
                    changed = true;
                }
            });
        });
    });
    changed
}
