// Color mapping runtime state.
//
// The native Metal renderer owns GPU color application. This module keeps
// UI-facing color controls/state stable for scheduler and UI integration.

use crate::memory::pod::Pod;

// Color scheme options
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ColorScheme {
    Plasma,
    #[default]
    Viridis,
    Inferno,
    BlueRed,
    Quantum,
}

// Color mapping parameters
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ColorMapParams {
    pub gamma: f32,
    pub contrast: f32,
    pub brightness: f32,
    pub use_log_scale: u32,
    pub padding: [u32; 4],
}

unsafe impl Pod for ColorMapParams {}

impl Default for ColorMapParams {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            contrast: 1.0,
            brightness: 1.0,
            use_log_scale: 0,
            padding: [0, 0, 0, 0],
        }
    }
}

// Color map runtime state
pub struct ColorMap {
    params: ColorMapParams,
    scheme: ColorScheme,
}

impl ColorMap {
    pub fn new() -> Self {
        Self {
            params: ColorMapParams::default(),
            scheme: ColorScheme::default(),
        }
    }

    pub fn update_params(
        &mut self,
        gamma: f32,
        contrast: f32,
        brightness: f32,
        use_log_scale: bool,
    ) {
        self.params.gamma = gamma;
        self.params.contrast = contrast;
        self.params.brightness = brightness;
        self.params.use_log_scale = u32::from(use_log_scale);
    }

    #[allow(dead_code)]
    pub fn set_color_scheme(&mut self, scheme: ColorScheme) {
        self.scheme = scheme;
    }

    // Set color gradient by name
    pub fn set_gradient(&mut self, gradient_name: &str) {
        self.scheme = match gradient_name.to_ascii_lowercase().as_str() {
            "plasma" => ColorScheme::Plasma,
            "viridis" => ColorScheme::Viridis,
            "inferno" => ColorScheme::Inferno,
            "bluered" | "blue_red" => ColorScheme::BlueRed,
            "quantum" => ColorScheme::Quantum,
            _ => ColorScheme::Viridis,
        };
    }
}

impl Default for ColorMap {
    fn default() -> Self {
        Self::new()
    }
}
