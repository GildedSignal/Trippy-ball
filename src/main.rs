// Wavefunction Visualization
//
// This is the main entry point for the application.
// It initializes the window, sets up the rendering pipeline,
// and manages the main event loop.

mod benchmark;
mod math;
mod memory;
mod render;
mod render_metal;
mod sim;
mod telemetry;
mod ui;

use std::env;
use std::time::{Duration, Instant};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

// Application state
struct App {
    renderer: render_metal::MetalRenderer,
    ui_state: ui::State,
    last_render_time: Instant,
    simulation_start: Instant,
    frame_times: Vec<Duration>,
    window: winit::window::Window,
}

impl App {
    fn new(window: winit::window::Window) -> Result<Self, String> {
        // Create renderer
        let renderer = render_metal::MetalRenderer::new(&window, 50000)?;

        // Create UI state
        let ui_state = ui::State::new();

        Ok(Self {
            renderer,
            ui_state,
            last_render_time: Instant::now(),
            simulation_start: Instant::now(),
            frame_times: Vec::new(),
            window,
        })
    }

    fn handle_event(&mut self, event: &winit::event::WindowEvent) {
        // Let UI handle input first
        if self.ui_state.handle_input(&self.window, event) {
            return;
        }

        // Let camera handle input
        if self.renderer.handle_camera_input(event) {
            return;
        }

        // Handle other events
        if let WindowEvent::Resized(size) = event {
            self.renderer
                .resize(size.width, size.height, self.window.scale_factor());
        }
    }

    fn update(&mut self) {
        // Calculate FPS
        let now = Instant::now();
        let frame_time = now - self.last_render_time;
        self.last_render_time = now;

        // Update frame times list
        self.frame_times.push(frame_time);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }

        // Calculate average FPS
        let avg_frame_time: Duration =
            self.frame_times.iter().sum::<Duration>() / self.frame_times.len() as u32;
        let fps = 1.0 / avg_frame_time.as_secs_f64();

        // Update UI state
        self.ui_state.update_fps(fps);

        // Get parameters from UI
        let params = self.ui_state.get_parameters();
        let (auto_rotation, rotation_speed) = self.ui_state.get_auto_rotation();
        let (gamma, contrast, brightness, use_log_scale, luminance_threshold) =
            self.ui_state.get_color_params();
        let auto_tone_enabled = self.ui_state.get_auto_tone_enabled();
        let color_scheme = self.ui_state.get_color_scheme();
        let point_count = self.ui_state.get_point_count();
        let slice_settings = self.ui_state.get_slice_settings();

        // Update renderer
        if point_count != self.renderer.get_point_count() {
            self.renderer.set_point_count(point_count);
        }

        // Update points
        let sim_time = now.duration_since(self.simulation_start).as_secs_f64();
        let debug_info = self
            .renderer
            .update_points(&params, sim_time, frame_time.as_secs_f32());
        self.ui_state.set_debug_info(debug_info);

        // Update camera rotation
        self.renderer.update_camera_rotation(
            auto_rotation,
            rotation_speed,
            frame_time.as_secs_f32(),
        );

        // Update color mapping
        self.renderer.set_auto_tone_enabled(auto_tone_enabled);
        self.renderer
            .set_color_map_params(gamma, contrast, brightness, use_log_scale);
        if !auto_tone_enabled {
            self.renderer.set_luminance_threshold(luminance_threshold);
        }
        self.renderer.set_color_scheme(color_scheme);
        self.renderer.set_slice_settings(slice_settings);
        let auto = self.renderer.auto_tone_output();
        self.ui_state.set_auto_tone_diagnostics(ui::AutoToneDiagnostics {
            exposure_black: auto.exposure_black,
            exposure_white: auto.exposure_white,
            luminance_threshold: auto.luminance_threshold,
        });

        // Update buffer pool statistics if available
        let stats = self.renderer.get_buffer_pool_stats();
        self.ui_state.set_buffer_pool_stats(stats);
    }

    fn render(&mut self) -> Result<(), render_metal::RenderError> {
        self.renderer.render(&mut self.ui_state, &self.window)
    }
}

fn main() {
    // Initialize logging
    telemetry::init();

    if !cfg!(target_os = "macos") {
        error!("This build currently supports only macOS while Metal migration is in progress.");
        return;
    }

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(sim::scheduler::RuntimePolicy::default().cpu_threads)
        .build_global();

    // Optional benchmark modes for CLI usage
    let args: Vec<String> = env::args().skip(1).collect();
    if args
        .iter()
        .any(|arg| arg == "--benchmark-stress" || arg == "--benchmark-large")
    {
        benchmark::stress_benchmark();
        return;
    }
    if args
        .iter()
        .any(|arg| arg == "--benchmark-soak-30m" || arg == "--benchmark-soak-timed")
    {
        benchmark::soak_benchmark_30m();
        return;
    }
    if args.iter().any(|arg| arg == "--benchmark-soak") {
        benchmark::soak_benchmark();
        return;
    }
    if args.iter().any(|arg| arg == "--benchmark-cap-sweep") {
        benchmark::cap_sweep_benchmark();
        return;
    }
    if args.iter().any(|arg| arg == "--benchmark-cap-pair-sweep") {
        benchmark::cap_pair_sweep_benchmark();
        return;
    }
    if args.iter().any(|arg| arg == "--benchmark") {
        benchmark::quick_benchmark();
        return;
    }

    // Create event loop and window
    let event_loop = EventLoop::new();
    let window = match WindowBuilder::new()
        .with_title("Wavefunction Visualization")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)
    {
        Ok(window) => window,
        Err(err) => {
            error!("Failed to create window: {}", err);
            return;
        }
    };

    // Create application
    let mut app = match App::new(window) {
        Ok(app) => app,
        Err(err) => {
            error!("Failed to initialize application: {}", err);
            return;
        }
    };

    // Run event loop
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, window_id } if window_id == app.window.id() => match event {
            WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::Exit;
            }
            _ => {
                app.handle_event(&event);
            }
        },
        Event::RedrawRequested(window_id) if window_id == app.window.id() => {
            app.update();

            match app.render() {
                Ok(_) => {}
                Err(render_metal::RenderError::NativeUnavailable) => {
                    error!("Native Metal renderer is unavailable, exiting");
                    *control_flow = ControlFlow::Exit;
                }
                Err(render_metal::RenderError::NativeFailure(err)) => {
                    error!("Native Metal render failed: {}", err);
                    *control_flow = ControlFlow::Exit;
                }
            }
        }
        Event::MainEventsCleared => {
            app.window.request_redraw();
        }
        _ => {}
    });
}
