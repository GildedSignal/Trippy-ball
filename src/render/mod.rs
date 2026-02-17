// Render Module
//
// This module contains the rendering components for the wavefunction visualization.

mod auto_tone;
mod buffer_pool;
mod camera;
pub mod color;
mod gpu_wavefunction;
mod points;

pub use auto_tone::AutoToneOutput;
pub use buffer_pool::{BufferPoolStats, GlobalBufferPool};
pub use camera::{Camera, CameraUniform};
pub use color::ColorMap;
use color::ColorScheme;
pub use gpu_wavefunction::GpuWavefunctionEvaluator;
pub use points::{PointCloud, PointCloudDebugInfo};

use crate::math::{evaluate_wavefunction_batch, WavefunctionParams};
use crate::sim::{AdaptiveScheduler, RuntimePolicy, SchedulePlan, SimulationRuntimeStats};
use crate::{debug, info};
use std::sync::Arc;
use std::time::Instant;
use winit::event::WindowEvent;
use winit::window::Window;

pub struct Renderer {
    camera: Camera,
    point_cloud: PointCloud,
    color_map: ColorMap,
    gpu_evaluator: GpuWavefunctionEvaluator,
    buffer_pool: Arc<GlobalBufferPool>,
    scheduler: AdaptiveScheduler,
    runtime_stats: SimulationRuntimeStats,
    current_intensities: Vec<f32>,
    intensity_stats_sample: Vec<f32>,
    intensity_stats_tick: u64,
    stale_frames: u32,
    auto_tone_enabled: bool,
    auto_tone_controller: auto_tone::AutoToneController,
    auto_tone_output: AutoToneOutput,
}

impl Renderer {
    const INTENSITY_STATS_SAMPLE_POINTS: usize = 65_536;
    const INTENSITY_STATS_SAMPLE_INTERVAL: u64 = 4;

    pub fn new(window: &Window, point_count: usize) -> Result<Self, String> {
        let start_time = Instant::now();
        info!("Creating renderer with {} points", point_count);

        let size = window.inner_size();
        let aspect_ratio = if size.height > 0 {
            size.width as f32 / size.height as f32
        } else {
            1.0
        };
        let camera = Camera::new(aspect_ratio);

        let buffer_pool = GlobalBufferPool::new(10);

        let mut point_cloud = PointCloud::new(point_count);

        let color_map = ColorMap::new();

        let scheduler = AdaptiveScheduler::new(RuntimePolicy::default());

        let mut gpu_evaluator = GpuWavefunctionEvaluator::try_new_with_buffer_pool(
            &WavefunctionParams::default(),
            point_cloud.effective_point_count(),
            Some(buffer_pool.clone()),
        )?;

        let positions = point_cloud.get_positions();
        gpu_evaluator.update_positions(&positions);

        gpu_evaluator.compute(&WavefunctionParams::default(), 0.0);
        let initial_intensities = gpu_evaluator.read_intensities();
        point_cloud.update_intensities(&initial_intensities);
        let mut intensity_stats_sample = Vec::new();
        gpu_evaluator.copy_intensity_sample(
            Self::INTENSITY_STATS_SAMPLE_POINTS,
            &mut intensity_stats_sample,
        );

        info!("Renderer created in {:?}", start_time.elapsed());

        Ok(Self {
            camera,
            point_cloud,
            color_map,
            gpu_evaluator,
            buffer_pool,
            scheduler,
            runtime_stats: SimulationRuntimeStats::default(),
            current_intensities: initial_intensities,
            intensity_stats_sample,
            intensity_stats_tick: 0,
            stale_frames: 0,
            auto_tone_enabled: true,
            auto_tone_controller: auto_tone::AutoToneController::new(),
            auto_tone_output: AutoToneOutput::default(),
        })
    }

    pub fn handle_camera_input(&mut self, event: &WindowEvent) -> bool {
        let handled = self.camera.handle_input(event);
        if handled {
            self.camera.update_buffer();
        }
        handled
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            let aspect_ratio = width as f32 / height as f32;
            self.camera.update_aspect_ratio(aspect_ratio);
            self.camera.update_buffer();
        }
    }

    pub fn update_points(
        &mut self,
        params: &WavefunctionParams,
        time: f64,
        delta_seconds: f32,
    ) -> PointCloudDebugInfo {
        let frame_start = Instant::now();
        debug!(
            "Updating points with params: n/l/m={}/{}/{}, n2/l2/m2={}/{}/{}, mix={}, phase={}, Z={}, time_factor={}, time={}",
            params.n,
            params.l,
            params.m,
            params.n2,
            params.l2,
            params.m2,
            params.mix,
            params.relative_phase,
            params.z,
            params.time_factor,
            time
        );

        self.camera.update_buffer();

        let requested_points = self.point_cloud.requested_point_count();
        let effective_points = self.point_cloud.effective_point_count();
        if self.current_intensities.len() != effective_points {
            self.current_intensities.resize(effective_points, 0.0);
        }

        let completion = self.gpu_evaluator.poll_completed_no_readback();
        let completed_gpu_points = completion.completed_points;
        let completed_gpu_ms = completion.gpu_ms;
        let completed_any = completion.completed_any;

        let max_stale_frames = self.scheduler.policy().max_stale_frames;
        if completed_any {
            self.stale_frames = 0;
        } else {
            self.stale_frames = self.stale_frames.saturating_add(1).min(max_stale_frames);
        }

        let queue_depth = self.gpu_evaluator.in_flight_count();
        let mut plan = self
            .scheduler
            .plan(requested_points, effective_points, queue_depth);

        if self.stale_frames >= max_stale_frames {
            plan.gpu_points = 0;
            plan.cpu_points = plan.scheduled_points;
        }

        let compute_ranges = self
            .point_cloud
            .prioritized_compute_ranges(plan.block_size, plan.scheduled_points);

        let mut submitted_gpu_points = 0usize;
        let mut submitted_gpu_range: Option<(usize, usize)> = None;
        if plan.gpu_points > 0 {
            if let Some((gpu_start, gpu_len)) = compute_ranges.first().copied() {
                let gpu_count = plan.gpu_points.min(gpu_len);
                let submitted = self
                    .gpu_evaluator
                    .enqueue_compute_range_no_readback(params, time, gpu_start, gpu_count);
                if submitted {
                    submitted_gpu_points = gpu_count;
                    submitted_gpu_range = Some((gpu_start, gpu_count));
                }
            }
        }

        let cpu_target_points = plan.scheduled_points.saturating_sub(submitted_gpu_points);

        let mut cpu_ms = 0.0f64;
        let mut actual_cpu_points = 0usize;
        if cpu_target_points > 0 {
            let cpu_start = Instant::now();
            let positions = self.point_cloud.positions();

            for &(range_start, range_len) in &compute_ranges {
                if actual_cpu_points >= cpu_target_points {
                    break;
                }

                let range_end = range_start.saturating_add(range_len);
                let mut work_segments = [(range_start, range_len), (0usize, 0usize)];
                let mut segment_count = 1usize;

                if let Some((gpu_start, gpu_count)) = submitted_gpu_range {
                    let gpu_end = gpu_start.saturating_add(gpu_count);
                    if gpu_start < range_end && gpu_end > range_start {
                        segment_count = 0;
                        if gpu_start > range_start {
                            work_segments[segment_count] = (range_start, gpu_start - range_start);
                            segment_count += 1;
                        }
                        if gpu_end < range_end {
                            work_segments[segment_count] = (gpu_end, range_end - gpu_end);
                            segment_count += 1;
                        }
                    }
                }

                for &(segment_start, segment_len) in work_segments.iter().take(segment_count) {
                    if actual_cpu_points >= cpu_target_points || segment_len == 0 {
                        break;
                    }

                    let take = segment_len.min(cpu_target_points - actual_cpu_points);
                    if take == 0 {
                        continue;
                    }

                    let segment_end = segment_start + take;
                    evaluate_wavefunction_batch(
                        &positions[segment_start..segment_end],
                        params,
                        time,
                        &mut self.current_intensities[segment_start..segment_end],
                    );

                    let _ = self.gpu_evaluator.upload_intensity_range(
                        segment_start,
                        &self.current_intensities[segment_start..segment_end],
                    );
                    actual_cpu_points += take;
                }
            }

            cpu_ms = cpu_start.elapsed().as_secs_f64() * 1_000.0;
        }

        if submitted_gpu_points == 0
            && actual_cpu_points == plan.scheduled_points
            && plan.scheduled_points > 0
        {
            self.stale_frames = 0;
        }

        self.intensity_stats_tick = self.intensity_stats_tick.saturating_add(1);
        let can_sample_gpu = queue_depth == 0;
        let sample_due = self.intensity_stats_tick % Self::INTENSITY_STATS_SAMPLE_INTERVAL == 0;
        let should_refresh_stats = can_sample_gpu && (completed_any || sample_due);
        if should_refresh_stats {
            self.gpu_evaluator.copy_intensity_sample(
                Self::INTENSITY_STATS_SAMPLE_POINTS,
                &mut self.intensity_stats_sample,
            );
        }
        if !self.intensity_stats_sample.is_empty() {
            self.point_cloud
                .update_intensities(&self.intensity_stats_sample);
        } else {
            self.point_cloud
                .update_intensities(&self.current_intensities);
        }

        if self.auto_tone_enabled {
            let sample = if !self.intensity_stats_sample.is_empty() {
                &self.intensity_stats_sample
            } else {
                &self.current_intensities
            };
            self.auto_tone_output = self
                .auto_tone_controller
                .update(sample, delta_seconds.max(1e-4));
        }

        self.point_cloud.update_visibility(&self.camera);

        let frame_ms = frame_start.elapsed().as_secs_f64() * 1_000.0;

        let feedback_plan = SchedulePlan {
            gpu_points: completed_gpu_points,
            cpu_points: actual_cpu_points,
            ..plan
        };
        self.scheduler
            .record_frame(feedback_plan, completed_gpu_ms, cpu_ms, frame_ms);

        self.runtime_stats = SimulationRuntimeStats {
            frame_ms: frame_ms as f32,
            gpu_ms: completed_gpu_ms as f32,
            cpu_ms: cpu_ms as f32,
            scheduler_block_size: plan.block_size,
            scheduler_quality: plan.quality_scale,
            queue_depth: self.gpu_evaluator.in_flight_count(),
            stale_frames: self.stale_frames,
            approx_mode: plan.approx_mode
                || plan.scheduled_points < plan.effective_points
                || requested_points > effective_points,
            cpu_worker_utilization: self
                .scheduler
                .worker_utilization(actual_cpu_points, plan.block_size),
        };

        debug!("Points updated in {:.3}ms", self.runtime_stats.frame_ms);

        let mut debug_info = self.point_cloud.get_debug_info();
        debug_info.frame_ms = self.runtime_stats.frame_ms;
        debug_info.gpu_ms = self.runtime_stats.gpu_ms;
        debug_info.cpu_ms = self.runtime_stats.cpu_ms;
        debug_info.scheduler_block_size = self.runtime_stats.scheduler_block_size;
        debug_info.scheduler_quality = self.runtime_stats.scheduler_quality;
        debug_info.queue_depth = self.runtime_stats.queue_depth;
        debug_info.stale_frames = self.runtime_stats.stale_frames;
        debug_info.approx_mode = self.runtime_stats.approx_mode;
        debug_info.cpu_worker_utilization = self.runtime_stats.cpu_worker_utilization;
        debug_info
    }

    pub fn set_point_count(&mut self, count: usize) {
        self.point_cloud.set_point_count(count);
        self.sync_gpu_to_point_cloud();
    }

    pub fn set_runtime_effective_cap_override(&mut self, cap: Option<usize>) {
        self.point_cloud.set_effective_cap_override(cap);
        self.sync_gpu_to_point_cloud();
    }

    fn sync_gpu_to_point_cloud(&mut self) {
        let effective_count = self.point_cloud.effective_point_count();
        self.gpu_evaluator.resize(effective_count);

        let positions = self.point_cloud.get_positions();
        self.gpu_evaluator.update_positions(&positions);

        self.current_intensities.resize(effective_count, 0.0);
        self.intensity_stats_sample.clear();
        self.gpu_evaluator.copy_intensity_sample(
            Self::INTENSITY_STATS_SAMPLE_POINTS,
            &mut self.intensity_stats_sample,
        );
    }

    pub fn set_color_map_params(
        &mut self,
        gamma: f32,
        contrast: f32,
        brightness: f32,
        use_log_scale: bool,
    ) {
        self.color_map
            .update_params(gamma, contrast, brightness, use_log_scale);
    }

    pub fn set_auto_tone_enabled(&mut self, enabled: bool) {
        self.auto_tone_enabled = enabled;
    }

    pub fn auto_tone_output(&self) -> AutoToneOutput {
        self.auto_tone_output
    }

    pub fn set_color_gradient(&mut self, gradient_name: &str) {
        self.color_map.set_gradient(gradient_name);
    }

    pub fn update_camera_rotation(
        &mut self,
        auto_rotation: bool,
        rotation_speed: f32,
        delta_time: f32,
    ) {
        self.camera
            .update_with_rotation(auto_rotation, rotation_speed, delta_time);
    }

    pub fn get_buffer_pool_stats(&self) -> BufferPoolStats {
        self.buffer_pool.get_stats()
    }

    pub fn get_point_count(&self) -> usize {
        self.point_cloud.requested_point_count()
    }

    pub fn set_color_scheme(&mut self, scheme: ColorScheme) {
        let gradient_name = match scheme {
            ColorScheme::Plasma => "plasma",
            ColorScheme::Viridis => "viridis",
            ColorScheme::Inferno => "inferno",
            ColorScheme::BlueRed => "bluered",
            ColorScheme::Quantum => "quantum",
        };
        self.set_color_gradient(gradient_name);
    }

    pub fn point_positions_snapshot(&self) -> Vec<glam::Vec3> {
        self.point_cloud.get_positions()
    }

    pub fn current_intensities(&self) -> &[f32] {
        &self.current_intensities
    }

    #[cfg(target_os = "macos")]
    pub fn gpu_intensity_buffer_ref(&self) -> Option<&metal::BufferRef> {
        Some(self.gpu_evaluator.intensity_buffer_ref())
    }

    #[cfg(not(target_os = "macos"))]
    pub fn gpu_intensity_buffer_ref(&self) -> Option<&()> {
        None
    }

    pub fn camera_uniform_snapshot(&self) -> CameraUniform {
        self.camera.uniform()
    }
}
