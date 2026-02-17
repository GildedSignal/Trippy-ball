// GPU Wavefunction Evaluator
//
// Scheduler-facing GPU evaluator with async range dispatch semantics.
// On macOS this uses native Metal compute + shared-memory readback staging.
// Non-macOS builds keep a lightweight CPU fallback for compilation.

use crate::math::WavefunctionParams;
#[cfg(target_os = "macos")]
use crate::math::{gpu_normalization_table_f32_flat, GPU_NORMALIZATION_MAX_L};
use crate::memory::pod::Pod;
#[cfg(target_os = "macos")]
use crate::memory::pod;
#[cfg(target_os = "macos")]
use crate::{debug, info};
use crate::warn;
use glam::Vec3;
use std::collections::VecDeque;
use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::time::{Duration, Instant};

use super::buffer_pool::GlobalBufferPool;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GpuWavefunctionParams {
    pub n1: u32,
    pub l: u32,
    pub m: i32,
    pub n2: u32,
    pub l2: u32,
    pub m2: i32,
    pub mix: f32,
    pub relative_phase: f32,
    pub z: f32,
    pub time_factor: f32,
    pub time: f32,
    pub point_count: u32,
    pub start_index: u32,
    pub compute_count: u32,
}

#[cfg(target_os = "macos")]
unsafe impl Pod for GpuWavefunctionParams {}

#[cfg(target_os = "macos")]
impl From<&WavefunctionParams> for GpuWavefunctionParams {
    fn from(params: &WavefunctionParams) -> Self {
        Self {
            n1: params.n as u32,
            l: params.l as u32,
            m: params.m as i32,
            n2: params.n2 as u32,
            l2: params.l2 as u32,
            m2: params.m2 as i32,
            mix: params.mix as f32,
            relative_phase: params.relative_phase as f32,
            z: params.z as f32,
            time_factor: params.time_factor as f32,
            time: 0.0,
            point_count: 0,
            start_index: 0,
            compute_count: 0,
        }
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GpuPositionData {
    pub position: [f32; 3],
}

#[cfg(target_os = "macos")]
unsafe impl Pod for GpuPositionData {}

#[derive(Debug)]
pub struct CompletedReadback {
    pub start_index: usize,
    pub intensities: Vec<f32>,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct PollCompletedStats {
    pub completed_points: usize,
    pub gpu_ms: f64,
    pub completed_any: bool,
}

#[cfg(target_os = "macos")]
struct InFlightReadback {
    staging_buffer: metal::Buffer,
    command_buffer: metal::CommandBuffer,
    start_index: usize,
    point_count: usize,
    frame_id: u64,
}

#[cfg(target_os = "macos")]
struct InFlightDispatch {
    command_buffer: metal::CommandBuffer,
    start_index: usize,
    point_count: usize,
    submitted_at: Instant,
    frame_id: u64,
}

#[cfg(target_os = "macos")]
pub struct GpuWavefunctionEvaluator {
    device: metal::Device,
    command_queue: metal::CommandQueue,
    compute_pipeline: metal::ComputePipelineState,
    params_buffer: metal::Buffer,
    normalization_buffer: metal::Buffer,
    positions_buffer: metal::Buffer,
    intensities_buffer: metal::Buffer,
    point_count: usize,
    workgroup_size: u64,
    in_flight_dispatches: VecDeque<InFlightDispatch>,
    in_flight_readbacks: VecDeque<InFlightReadback>,
    max_in_flight: usize,
    next_frame_id: u64,
    #[allow(dead_code)]
    buffer_pool: Option<Arc<GlobalBufferPool>>,
}

#[cfg(target_os = "macos")]
impl GpuWavefunctionEvaluator {
    pub fn try_new_with_buffer_pool(
        params: &WavefunctionParams,
        point_count: usize,
        buffer_pool: Option<Arc<GlobalBufferPool>>,
    ) -> Result<Self, String> {
        let start_time = Instant::now();
        info!(
            "Creating Metal GPU wavefunction evaluator with {} points",
            point_count
        );

        let device = metal::Device::system_default()
            .ok_or_else(|| "Metal device unavailable".to_string())?;
        let command_queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(
                include_str!("../render_metal/shaders/wavefunction.metal"),
                &compile_options,
            )
            .map_err(|e| format!("wavefunction.metal compile failed: {e}"))?;
        let kernel = library
            .get_function("evaluate_wavefunction", None)
            .map_err(|e| format!("evaluate_wavefunction kernel not found: {e}"))?;
        let compute_pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Metal compute pipeline build failed: {e}"))?;

        let params_buffer = create_shared_buffer(
            &device,
            std::mem::size_of::<GpuWavefunctionParams>(),
            "metal-wavefunction-params",
        );
        let normalization_table = gpu_normalization_table_f32_flat();
        let normalization_buffer = create_shared_buffer(
            &device,
            std::mem::size_of_val(normalization_table),
            "metal-wavefunction-normalization-table",
        );
        write_pod_slice(normalization_buffer.as_ref(), normalization_table)
            .map_err(|e| format!("normalization table upload failed: {e}"))?;
        let positions_buffer = create_shared_buffer(
            &device,
            std::mem::size_of::<GpuPositionData>() * point_count.max(1),
            "metal-wavefunction-positions",
        );
        let intensities_buffer = create_shared_buffer(
            &device,
            std::mem::size_of::<f32>() * point_count.max(1),
            "metal-wavefunction-intensities",
        );

        let mut gpu_params = GpuWavefunctionParams::from(params);
        gpu_params.point_count = point_count as u32;
        write_pod(params_buffer.as_ref(), &gpu_params)
            .map_err(|e| format!("initial params upload failed: {e}"))?;

        let workgroup_size = compute_pipeline.thread_execution_width().clamp(1, 64);

        info!(
            "Metal GPU wavefunction evaluator created in {:?} (norm_max_l={})",
            start_time.elapsed(),
            GPU_NORMALIZATION_MAX_L
        );

        Ok(Self {
            device,
            command_queue,
            compute_pipeline,
            params_buffer,
            normalization_buffer,
            positions_buffer,
            intensities_buffer,
            point_count,
            workgroup_size,
            in_flight_dispatches: VecDeque::new(),
            in_flight_readbacks: VecDeque::new(),
            max_in_flight: 3,
            next_frame_id: 1,
            buffer_pool,
        })
    }

    pub fn new_with_buffer_pool(
        params: &WavefunctionParams,
        point_count: usize,
        buffer_pool: Option<Arc<GlobalBufferPool>>,
    ) -> Self {
        Self::try_new_with_buffer_pool(params, point_count, buffer_pool)
            .unwrap_or_else(|err| panic!("GpuWavefunctionEvaluator init failed: {err}"))
    }

    #[allow(dead_code)]
    pub fn point_count(&self) -> usize {
        self.point_count
    }

    pub fn in_flight_count(&self) -> usize {
        self.in_flight_dispatches.len()
    }

    pub fn update_positions(&mut self, positions: &[Vec3]) {
        let upload_count = positions.len().min(self.point_count);
        if positions.len() > self.point_count {
            warn!(
                "Truncating position upload from {} to {} points",
                positions.len(),
                upload_count
            );
        }
        if upload_count == 0 {
            return;
        }

        let start_time = Instant::now();
        debug!(
            "Updating {} positions in Metal GPU wavefunction evaluator",
            upload_count
        );

        let mut gpu_positions = vec![pod::zeroed::<GpuPositionData>(); upload_count];
        for (i, pos) in positions.iter().take(upload_count).enumerate() {
            gpu_positions[i] = GpuPositionData {
                position: [pos.x, pos.y, pos.z],
            };
        }

        if let Err(err) = write_pod_slice(self.positions_buffer.as_ref(), &gpu_positions) {
            warn!("positions upload into Metal buffer failed: {}", err);
            return;
        }

        debug!("Updated positions in {:?}", start_time.elapsed());
    }

    fn update_params_range(
        &self,
        params: &WavefunctionParams,
        time: f32,
        start_index: usize,
        compute_count: usize,
    ) -> bool {
        let mut gpu_params = GpuWavefunctionParams::from(params);
        gpu_params.time = time;
        gpu_params.point_count = self.point_count as u32;
        gpu_params.start_index = start_index as u32;
        gpu_params.compute_count = compute_count as u32;

        match write_pod(self.params_buffer.as_ref(), &gpu_params) {
            Ok(_) => true,
            Err(err) => {
                warn!("params upload into Metal buffer failed: {}", err);
                false
            }
        }
    }

    fn total_in_flight(&self) -> usize {
        self.in_flight_dispatches.len() + self.in_flight_readbacks.len()
    }

    fn encode_compute_command(
        &mut self,
        params: &WavefunctionParams,
        time: f64,
        start_index: usize,
        compute_count: usize,
    ) -> Option<metal::CommandBuffer> {
        if compute_count == 0 {
            return None;
        }

        if start_index.saturating_add(compute_count) > self.point_count {
            warn!(
                "Skipping out-of-bounds Metal GPU range start={} count={} point_count={}",
                start_index, compute_count, self.point_count
            );
            return None;
        }

        if self.total_in_flight() >= self.max_in_flight {
            return None;
        }

        if !self.update_params_range(params, time as f32, start_index, compute_count) {
            return None;
        }

        let command_buffer = self.command_queue.new_command_buffer();
        command_buffer.set_label("metal-wavefunction-command-buffer");

        {
            let compute_encoder = command_buffer.new_compute_command_encoder();
            compute_encoder.set_compute_pipeline_state(&self.compute_pipeline);
            compute_encoder.set_buffer(0, Some(self.positions_buffer.as_ref()), 0);
            compute_encoder.set_buffer(1, Some(self.intensities_buffer.as_ref()), 0);
            compute_encoder.set_buffer(2, Some(self.params_buffer.as_ref()), 0);
            compute_encoder.set_buffer(3, Some(self.normalization_buffer.as_ref()), 0);

            let thread_groups = metal::MTLSize {
                width: (compute_count as u64).div_ceil(self.workgroup_size),
                height: 1,
                depth: 1,
            };
            let threads_per_group = metal::MTLSize {
                width: self.workgroup_size,
                height: 1,
                depth: 1,
            };

            compute_encoder.dispatch_thread_groups(thread_groups, threads_per_group);
            compute_encoder.end_encoding();
        }

        Some(command_buffer.to_owned())
    }

    pub fn enqueue_compute_range(
        &mut self,
        params: &WavefunctionParams,
        time: f64,
        start_index: usize,
        compute_count: usize,
    ) -> bool {
        if compute_count == 0 {
            return true;
        }

        let Some(command_buffer) =
            self.encode_compute_command(params, time, start_index, compute_count)
        else {
            return false;
        };

        let intensity_stride = std::mem::size_of::<f32>();
        let copy_bytes = intensity_stride * compute_count;
        let source_offset = intensity_stride * start_index;

        let staging_buffer = create_shared_buffer(
            &self.device,
            copy_bytes.max(intensity_stride),
            "metal-wavefunction-staging",
        );

        {
            let blit_encoder = command_buffer.new_blit_command_encoder();
            blit_encoder.copy_from_buffer(
                self.intensities_buffer.as_ref(),
                source_offset as u64,
                staging_buffer.as_ref(),
                0,
                copy_bytes as u64,
            );
            blit_encoder.end_encoding();
        }

        command_buffer.commit();

        let frame_id = self.next_frame_id;
        self.next_frame_id = self.next_frame_id.saturating_add(1);

        self.in_flight_readbacks.push_back(InFlightReadback {
            staging_buffer,
            command_buffer: command_buffer.to_owned(),
            start_index,
            point_count: compute_count,
            frame_id,
        });

        true
    }

    pub fn enqueue_compute_range_no_readback(
        &mut self,
        params: &WavefunctionParams,
        time: f64,
        start_index: usize,
        compute_count: usize,
    ) -> bool {
        if compute_count == 0 {
            return true;
        }

        let Some(command_buffer) =
            self.encode_compute_command(params, time, start_index, compute_count)
        else {
            return false;
        };

        command_buffer.commit();

        let frame_id = self.next_frame_id;
        self.next_frame_id = self.next_frame_id.saturating_add(1);
        self.in_flight_dispatches.push_back(InFlightDispatch {
            command_buffer: command_buffer.to_owned(),
            start_index,
            point_count: compute_count,
            submitted_at: Instant::now(),
            frame_id,
        });
        true
    }

    pub fn poll_completed(&mut self) -> Vec<CompletedReadback> {
        if self.in_flight_readbacks.is_empty() {
            return Vec::new();
        }

        let mut completed = Vec::new();

        loop {
            let status = self
                .in_flight_readbacks
                .front()
                .map(|front| front.command_buffer.status());

            match status {
                Some(metal::MTLCommandBufferStatus::Completed) => {
                    let Some(readback) = self.in_flight_readbacks.pop_front() else {
                        warn!("Metal GPU readback queue underflow");
                        break;
                    };

                    let ptr = readback.staging_buffer.contents() as *const f32;
                    let intensities =
                        unsafe { std::slice::from_raw_parts(ptr, readback.point_count) }.to_vec();

                    completed.push(CompletedReadback {
                        start_index: readback.start_index,
                        intensities,
                    });
                }
                Some(metal::MTLCommandBufferStatus::Error) => {
                    let Some(readback) = self.in_flight_readbacks.pop_front() else {
                        warn!("Metal GPU readback queue underflow");
                        break;
                    };
                    warn!(
                        "Metal GPU readback failed for frame {} at start={} count={}",
                        readback.frame_id, readback.start_index, readback.point_count
                    );
                }
                _ => break,
            }
        }

        completed
    }

    pub fn poll_completed_no_readback(&mut self) -> PollCompletedStats {
        if self.in_flight_dispatches.is_empty() {
            return PollCompletedStats::default();
        }

        let mut stats = PollCompletedStats::default();

        loop {
            let status = self
                .in_flight_dispatches
                .front()
                .map(|front| front.command_buffer.status());

            match status {
                Some(metal::MTLCommandBufferStatus::Completed) => {
                    let Some(completed) = self.in_flight_dispatches.pop_front() else {
                        warn!("Metal GPU dispatch queue underflow");
                        break;
                    };
                    stats.completed_any = true;
                    stats.completed_points += completed.point_count;
                    stats.gpu_ms += completed.submitted_at.elapsed().as_secs_f64() * 1_000.0;
                }
                Some(metal::MTLCommandBufferStatus::Error) => {
                    let Some(failed) = self.in_flight_dispatches.pop_front() else {
                        warn!("Metal GPU dispatch queue underflow");
                        break;
                    };
                    warn!(
                        "Metal GPU dispatch failed for frame {} at start={} count={}",
                        failed.frame_id, failed.start_index, failed.point_count
                    );
                }
                _ => break,
            }
        }

        stats
    }

    pub fn compute(&mut self, params: &WavefunctionParams, time: f64) {
        while !self.enqueue_compute_range(params, time, 0, self.point_count) {
            if self.poll_completed().is_empty() {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }

    pub fn read_intensities(&mut self) -> Vec<f32> {
        loop {
            let completed = self.poll_completed();
            for chunk in completed {
                if chunk.start_index == 0 && chunk.intensities.len() == self.point_count {
                    return chunk.intensities;
                }
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    }

    #[allow(dead_code)]
    pub fn evaluate_wavefunction(&mut self, params: &WavefunctionParams, time: f64) -> Vec<f32> {
        self.compute(params, time);
        self.read_intensities()
    }

    pub fn upload_intensity_range(&mut self, start_index: usize, intensities: &[f32]) -> bool {
        if intensities.is_empty() {
            return true;
        }
        if start_index.saturating_add(intensities.len()) > self.point_count {
            warn!(
                "Skipping out-of-bounds CPU patch upload start={} count={} point_count={}",
                start_index,
                intensities.len(),
                self.point_count
            );
            return false;
        }

        let byte_offset = start_index.saturating_mul(std::mem::size_of::<f32>());
        if let Err(err) =
            write_pod_slice_at(self.intensities_buffer.as_ref(), byte_offset, intensities)
        {
            warn!(
                "Failed CPU patch upload start={} count={}: {}",
                start_index,
                intensities.len(),
                err
            );
            return false;
        }
        true
    }

    pub fn intensity_buffer_ref(&self) -> &metal::BufferRef {
        self.intensities_buffer.as_ref()
    }

    pub fn copy_intensity_sample(&self, sample_count: usize, out: &mut Vec<f32>) {
        out.clear();
        if self.point_count == 0 || sample_count == 0 {
            return;
        }

        let source = unsafe {
            std::slice::from_raw_parts(
                self.intensities_buffer.contents() as *const f32,
                self.point_count,
            )
        };
        let sampled_points = self.point_count.clamp(1, sample_count);
        let step = (self.point_count / sampled_points).max(1);
        out.reserve(sampled_points);

        let mut sampled = 0usize;
        let mut index = 0usize;
        while index < self.point_count && sampled < sampled_points {
            out.push(source[index]);
            sampled += 1;
            index = index.saturating_add(step);
        }
    }

    pub fn resize(&mut self, new_point_count: usize) {
        if new_point_count == self.point_count {
            return;
        }

        info!(
            "Resizing Metal GPU wavefunction evaluator from {} to {} points",
            self.point_count, new_point_count
        );

        self.in_flight_dispatches.clear();
        self.in_flight_readbacks.clear();
        self.positions_buffer = create_shared_buffer(
            &self.device,
            std::mem::size_of::<GpuPositionData>() * new_point_count.max(1),
            "metal-wavefunction-positions",
        );
        self.intensities_buffer = create_shared_buffer(
            &self.device,
            std::mem::size_of::<f32>() * new_point_count.max(1),
            "metal-wavefunction-intensities",
        );
        self.point_count = new_point_count;
    }
}

#[cfg(target_os = "macos")]
fn create_shared_buffer(device: &metal::Device, bytes: usize, label: &str) -> metal::Buffer {
    let aligned = bytes.max(256);
    let buffer = device.new_buffer(aligned as u64, metal::MTLResourceOptions::StorageModeShared);
    buffer.set_label(label);
    buffer
}

#[cfg(target_os = "macos")]
fn write_pod<T: Pod>(buffer: &metal::BufferRef, data: &T) -> Result<(), String> {
    write_pod_slice(buffer, std::slice::from_ref(data))
}

#[cfg(target_os = "macos")]
fn write_pod_slice<T: Pod>(buffer: &metal::BufferRef, data: &[T]) -> Result<(), String> {
    write_pod_slice_at(buffer, 0, data)
}

#[cfg(target_os = "macos")]
fn write_pod_slice_at<T: Pod>(
    buffer: &metal::BufferRef,
    byte_offset: usize,
    data: &[T],
) -> Result<(), String> {
    let byte_len = std::mem::size_of_val(data);
    let end = byte_offset.saturating_add(byte_len);
    if end > buffer.length() as usize {
        return Err(format!(
            "buffer write overflow: end {} bytes > {} bytes",
            end,
            buffer.length()
        ));
    }
    if byte_len == 0 {
        return Ok(());
    }

    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            (buffer.contents() as *mut u8).add(byte_offset),
            byte_len,
        );
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub struct GpuWavefunctionEvaluator {
    point_count: usize,
    positions: Vec<Vec3>,
    intensities: Vec<f32>,
    pending: VecDeque<CompletedReadback>,
    pending_dispatches: VecDeque<(usize, usize)>,
    max_in_flight: usize,
}

#[cfg(not(target_os = "macos"))]
impl GpuWavefunctionEvaluator {
    pub fn try_new_with_buffer_pool(
        params: &WavefunctionParams,
        point_count: usize,
        buffer_pool: Option<Arc<GlobalBufferPool>>,
    ) -> Result<Self, String> {
        let _ = (params, buffer_pool);
        Ok(Self::new_with_buffer_pool(params, point_count, None))
    }

    pub fn new_with_buffer_pool(
        _params: &WavefunctionParams,
        point_count: usize,
        _buffer_pool: Option<Arc<GlobalBufferPool>>,
    ) -> Self {
        Self {
            point_count,
            positions: vec![Vec3::ZERO; point_count],
            intensities: vec![0.0; point_count],
            pending: VecDeque::new(),
            pending_dispatches: VecDeque::new(),
            max_in_flight: 3,
        }
    }

    #[allow(dead_code)]
    pub fn point_count(&self) -> usize {
        self.point_count
    }

    pub fn in_flight_count(&self) -> usize {
        self.pending_dispatches.len()
    }

    pub fn update_positions(&mut self, positions: &[Vec3]) {
        let upload_count = positions.len().min(self.point_count);
        if positions.len() > self.point_count {
            warn!(
                "Truncating position upload from {} to {} points",
                positions.len(),
                upload_count
            );
        }
        self.positions[..upload_count].copy_from_slice(&positions[..upload_count]);
    }

    pub fn enqueue_compute_range(
        &mut self,
        params: &WavefunctionParams,
        time: f64,
        start_index: usize,
        compute_count: usize,
    ) -> bool {
        if compute_count == 0 {
            return true;
        }
        if self.pending.len() + self.pending_dispatches.len() >= self.max_in_flight {
            return false;
        }
        if start_index.saturating_add(compute_count) > self.point_count {
            return false;
        }

        let end = start_index + compute_count;
        let mut intensities = vec![0.0f32; compute_count];
        crate::math::evaluate_wavefunction_batch(
            &self.positions[start_index..end],
            params,
            time,
            &mut intensities,
        );
        self.pending.push_back(CompletedReadback {
            start_index,
            intensities,
        });
        true
    }

    pub fn enqueue_compute_range_no_readback(
        &mut self,
        params: &WavefunctionParams,
        time: f64,
        start_index: usize,
        compute_count: usize,
    ) -> bool {
        if compute_count == 0 {
            return true;
        }
        if self.pending.len() + self.pending_dispatches.len() >= self.max_in_flight {
            return false;
        }
        if start_index.saturating_add(compute_count) > self.point_count {
            return false;
        }

        let end = start_index + compute_count;
        crate::math::evaluate_wavefunction_batch(
            &self.positions[start_index..end],
            params,
            time,
            &mut self.intensities[start_index..end],
        );
        self.pending_dispatches
            .push_back((start_index, compute_count));
        true
    }

    pub fn poll_completed(&mut self) -> Vec<CompletedReadback> {
        self.pending.drain(..).collect()
    }

    pub fn poll_completed_into(&mut self, output: &mut [f32]) -> PollCompletedStats {
        let mut stats = PollCompletedStats::default();
        for chunk in self.pending.drain(..) {
            stats.completed_any = true;

            let start = chunk.start_index.min(output.len());
            let copy_len = output
                .len()
                .saturating_sub(start)
                .min(chunk.intensities.len());
            if copy_len > 0 {
                output[start..start + copy_len].copy_from_slice(&chunk.intensities[..copy_len]);
                stats.completed_points += copy_len;
            }
        }
        stats
    }

    pub fn poll_completed_no_readback(&mut self) -> PollCompletedStats {
        let mut stats = PollCompletedStats::default();
        for (_start, count) in self.pending_dispatches.drain(..) {
            stats.completed_any = true;
            stats.completed_points += count;
        }
        stats
    }

    pub fn compute(&mut self, params: &WavefunctionParams, time: f64) {
        while !self.enqueue_compute_range(params, time, 0, self.point_count) {
            let _ = self.poll_completed();
        }
    }

    pub fn read_intensities(&mut self) -> Vec<f32> {
        loop {
            let completed = self.poll_completed();
            for chunk in completed {
                if chunk.start_index == 0 && chunk.intensities.len() == self.point_count {
                    return chunk.intensities;
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn evaluate_wavefunction(&mut self, params: &WavefunctionParams, time: f64) -> Vec<f32> {
        self.compute(params, time);
        self.read_intensities()
    }

    pub fn upload_intensity_range(&mut self, start_index: usize, intensities: &[f32]) -> bool {
        if intensities.is_empty() {
            return true;
        }
        if start_index.saturating_add(intensities.len()) > self.point_count {
            return false;
        }
        self.intensities[start_index..start_index + intensities.len()].copy_from_slice(intensities);
        true
    }

    pub fn copy_intensity_sample(&self, sample_count: usize, out: &mut Vec<f32>) {
        out.clear();
        if self.point_count == 0 || sample_count == 0 {
            return;
        }
        let sampled_points = self.point_count.clamp(1, sample_count);
        let step = (self.point_count / sampled_points).max(1);
        out.reserve(sampled_points);

        let mut sampled = 0usize;
        let mut index = 0usize;
        while index < self.point_count && sampled < sampled_points {
            out.push(self.intensities[index]);
            sampled += 1;
            index = index.saturating_add(step);
        }
    }

    pub fn resize(&mut self, new_point_count: usize) {
        self.point_count = new_point_count;
        self.positions.resize(new_point_count, Vec3::ZERO);
        self.intensities.resize(new_point_count, 0.0);
        self.pending_dispatches.clear();
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{evaluate_wavefunction_batch, spherical_to_cartesian};
    use glam::Vec3;
    use std::f64::consts::PI;

    #[test]
    fn test_gpu_wavefunction_placeholder() {
        // Integration tested via benchmark + runtime paths.
    }

    #[test]
    fn gpu_matches_cpu_reference_samples() {
        let point_count = 256usize;
        let params = WavefunctionParams {
            n: 4,
            l: 3,
            m: -2,
            n2: 5,
            l2: 4,
            m2: -2,
            mix: 0.4,
            relative_phase: 0.2,
            z: 1.0,
            time_factor: 0.8,
        };
        let time = 0.37f64;

        let positions: Vec<Vec3> = (0..point_count)
            .map(|i| {
                let t = i as f32 / point_count as f32;
                Vec3::new(
                    (t * 13.0).sin() * 3.7,
                    (t * 7.0).cos() * 2.9,
                    (t * 5.0).sin() * 4.1,
                )
            })
            .collect();

        let mut evaluator =
            GpuWavefunctionEvaluator::new_with_buffer_pool(&params, point_count, None);
        evaluator.update_positions(&positions);
        evaluator.compute(&params, time);
        let gpu = evaluator.read_intensities();

        let mut cpu = vec![0.0f32; point_count];
        evaluate_wavefunction_batch(&positions, &params, time, &mut cpu);

        let mut mismatches = 0usize;
        let mut max_rel = 0.0f32;
        let mut max_abs = 0.0f32;
        let mut max_idx = 0usize;
        let mut sample_mismatches = Vec::new();

        for i in 0..point_count {
            let expected = cpu[i];
            let actual = gpu[i];
            let abs_err = (expected - actual).abs();
            let rel_err = abs_err / expected.abs().max(1e-6);
            if rel_err > 0.08 && abs_err > 1e-4 {
                mismatches += 1;
                if sample_mismatches.len() < 8 {
                    sample_mismatches.push((i, expected, actual, rel_err, abs_err));
                }
                if rel_err > max_rel {
                    max_rel = rel_err;
                    max_abs = abs_err;
                    max_idx = i;
                }
            }
        }

        assert!(
            mismatches == 0,
            "gpu parity mismatches={} max_idx={} expected={} actual={} rel_err={} abs_err={} samples={:?}",
            mismatches,
            max_idx,
            cpu[max_idx],
            gpu[max_idx],
            max_rel,
            max_abs,
            sample_mismatches
        );
    }

    #[test]
    fn gpu_matches_cpu_reference_lm_grid() {
        let point_count = 128usize;
        let time = 0.41f64;
        let z = 1.0f64;
        let time_factor = 0.9f64;

        let positions: Vec<Vec3> = (0..point_count)
            .map(|i| {
                let t = i as f32 / point_count as f32;
                Vec3::new(
                    (t * 11.0).sin() * 4.2,
                    (t * 17.0).cos() * 3.1,
                    (t * 23.0).sin() * 2.7,
                )
            })
            .collect();

        let mut evaluator = GpuWavefunctionEvaluator::new_with_buffer_pool(
            &WavefunctionParams::default(),
            point_count,
            None,
        );
        evaluator.update_positions(&positions);

        for l in 0..=5usize {
            for m in -(l as isize)..=(l as isize) {
                let params = WavefunctionParams {
                    n: (l + 1).max(1),
                    l,
                    m,
                    n2: (l + 2).max(2),
                    l2: (l + 1).min(5),
                    m2: m.clamp(-((l + 1).min(5) as isize), (l + 1).min(5) as isize),
                    mix: 0.35,
                    relative_phase: 0.0,
                    z,
                    time_factor,
                };

                evaluator.compute(&params, time);
                let gpu = evaluator.read_intensities();

                let mut cpu = vec![0.0f32; point_count];
                evaluate_wavefunction_batch(&positions, &params, time, &mut cpu);

                for i in 0..point_count {
                    let expected = cpu[i];
                    let actual = gpu[i];
                    let abs_err = (expected - actual).abs();
                    let rel_err = abs_err / expected.abs().max(1e-6);
                    assert!(
                        rel_err <= 0.12 || abs_err <= 2e-4,
                        "lm mismatch l={} m={} idx={} expected={} actual={} rel_err={} abs_err={}",
                        l,
                        m,
                        i,
                        expected,
                        actual,
                        rel_err,
                        abs_err
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_matches_cpu_reference_spherical_grid() {
        let radial_samples = [0.0f64, 1e-6, 0.25, 1.0, 2.5, 4.5];
        let theta_samples = [
            0.0f64,
            PI / 8.0,
            PI / 3.0,
            PI / 2.0,
            2.0 * PI / 3.0,
            PI - 1e-4,
        ];
        let phi_samples = [0.0f64, PI / 6.0, PI / 2.0, PI, 1.5 * PI];

        let mut positions =
            Vec::with_capacity(radial_samples.len() * theta_samples.len() * phi_samples.len());
        for &r in &radial_samples {
            for &theta in &theta_samples {
                for &phi in &phi_samples {
                    positions.push(spherical_to_cartesian(r, theta, phi));
                }
            }
        }

        let point_count = positions.len();
        let mut evaluator = GpuWavefunctionEvaluator::new_with_buffer_pool(
            &WavefunctionParams::default(),
            point_count,
            None,
        );
        evaluator.update_positions(&positions);

        let time_samples = [0.0f64, 0.43];
        let parameter_regimes = [(1.0f64, 0.7f64), (2.0f64, 1.25f64)];

        for &(z, time_factor) in &parameter_regimes {
            for &time in &time_samples {
                for l in 0..=5usize {
                    for m in -(l as isize)..=(l as isize) {
                        let params = WavefunctionParams {
                            n: (l + 1).max(1),
                            l,
                            m,
                            n2: (l + 2).max(2),
                            l2: (l + 1).min(5),
                            m2: m.clamp(-((l + 1).min(5) as isize), (l + 1).min(5) as isize),
                            mix: 0.35,
                            relative_phase: 0.0,
                            z,
                            time_factor,
                        };

                        evaluator.compute(&params, time);
                        let gpu = evaluator.read_intensities();

                        let mut cpu = vec![0.0f32; point_count];
                        evaluate_wavefunction_batch(&positions, &params, time, &mut cpu);

                        for i in 0..point_count {
                            let expected = cpu[i];
                            let actual = gpu[i];
                            let abs_err = (expected - actual).abs();
                            let rel_err = abs_err / expected.abs().max(1e-6);
                            assert!(
                                rel_err <= 0.12 || abs_err <= 2e-4,
                                "spherical-grid mismatch z={} time_factor={} time={} l={} m={} idx={} expected={} actual={} rel_err={} abs_err={}",
                                z,
                                time_factor,
                                time,
                                l,
                                m,
                                i,
                                expected,
                                actual,
                                rel_err,
                                abs_err
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "Requires a GPU device and is intended for local regression runs"]
    fn regression_large_position_upload_no_overrun() {
        let point_count = 1_500_000usize;
        let params = WavefunctionParams::default();

        let mut evaluator =
            GpuWavefunctionEvaluator::new_with_buffer_pool(&params, point_count, None);
        let positions = vec![Vec3::new(0.0, 0.0, 0.0); point_count];

        evaluator.update_positions(&positions);
    }
}
